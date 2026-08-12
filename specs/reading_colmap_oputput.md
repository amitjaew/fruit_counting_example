# Spec: COLMAP Dense Outputs → `Frame` Objects

Defines how to go from a completed `colmap automatic_reconstructor` (dense mode) run
to the `Frame` objects consumed by `surface_backprojection.py`.

## 1. Inputs

Given workspace root `WORKSPACE/`, after a successful dense run:

| Path | Contents | Notes |
|---|---|---|
| `WORKSPACE/dense/sparse/cameras.bin` | Undistorted intrinsics | **Use this, not `sparse/0/cameras.bin`** |
| `WORKSPACE/dense/sparse/images.bin` | Undistorted poses + registered image list | **Use this, not `sparse/0/images.bin`** |
| `WORKSPACE/dense/images/` | Undistorted images | Recommended: run YOLO on these, not raw frames |
| `WORKSPACE/dense/stereo/depth_maps/*.geometric.bin` | Per-image depth maps | Prefer `.geometric.bin` over `.photometric.bin` |

Rationale for the `dense/sparse/*` requirement: image undistortion is the first stage
of the dense pipeline, and depth maps are pixel-aligned to the undistorted images —
not the original input frames. Using the original `sparse/0/*` camera parameters here
will silently produce wrong 3D points, not an error.

## 2. Output contract

```python
frames: dict[str, Frame]   # keyed by image filename, e.g. "frame_000123.png"
```

A dict, not a list — not every input video frame is guaranteed to register in SfM,
so the key set may be a strict subset of your video frame indices. Downstream code
(YOLO detection lookup, track building) must key off the same filename convention.

**Naming convention requirement:** decide one canonical frame-naming scheme (e.g.
zero-padded `frame_%06d.png`) *before* running COLMAP, and use it both for the
frames fed into COLMAP and the frames fed into YOLO. This avoids fuzzy filename
matching between the two systems later.

## 3. Reading procedure

### 3.1 Poses + intrinsics

Preferred: `pycolmap`.

```python
import pycolmap
recon = pycolmap.Reconstruction("WORKSPACE/dense/sparse")

poses = {}
for image in recon.images.values():
    cam_from_world = image.cam_from_world          # Rigid3d: X_cam = R @ X_world + t
    R = cam_from_world.rotation.matrix()            # (3, 3)
    t = np.asarray(cam_from_world.translation)       # (3,)
    camera = recon.cameras[image.camera_id]
    fx, fy, cx, cy = camera.params                    # PINHOLE model (undistorted)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    poses[image.name] = (K, R, t)
```

`pycolmap`'s attribute names (`cam_from_world`, `.rotation.matrix()`, etc.) have
changed across versions — **verify against your installed version** (`pip show
pycolmap`) before trusting this verbatim; a wrong attribute name will fail loudly
(`AttributeError`), which is preferable to a silently wrong convention.

Fallback (more stable across versions, since it hasn't changed in years): COLMAP's
own `scripts/python/read_write_model.py`, functions `read_model()` +
`qvec2rotmat()`. Same output shape — `(K, R, t)` per image, same `X_cam = R @
X_world + t` convention. Use this path if `pycolmap` isn't installed or its API
doesn't match what's shown above.

**Camera model check:** confirm `camera.model` (or the text-format `cameras.txt`
header) is actually `PINHOLE` (4 params: fx, fy, cx, cy). If undistortion used a
different output model, `camera.params` won't map to K the way shown above —
handle that explicitly rather than assume.

### 3.2 Depth maps

1. List `WORKSPACE/dense/stereo/depth_maps/` once and confirm the exact filename
   pattern for your COLMAP build (commonly `<image_name>.geometric.bin`, but treat
   this as something to verify, not assume, since it varies slightly by version).
2. Parse with COLMAP's own `scripts/python/read_write_dense.py`, function
   `read_array()`. Do not reimplement the binary header parser.
3. **Resolution check (required, not optional):** `patch_match_stereo` has a
   `max_image_size` setting that can downsample large images before stereo, so the
   depth map's `(H, W)` is not guaranteed to equal the undistorted image's `(H,
   W)`. For every frame:
   - Compare `depth_map.shape[:2]` to the corresponding image's shape.
   - If they differ, either upsample the depth map to image resolution
     (nearest-neighbor — depth must not be interpolated across object
     boundaries), or scale `K`'s `fx, fy, cx, cy` by the resolution ratio and keep
     detections in the depth map's native resolution instead.
   - Pick one convention and assert it holds for every frame at load time — a
     silent per-frame mismatch here is the single easiest way to get subtly wrong
     3D points without any error.

### 3.3 Assembling `Frame` objects

```python
frames = {}
skipped = []
for name, (K, R, t) in poses.items():
    depth_path = depth_map_path_for(name)     # from 3.2, step 1's confirmed pattern
    if not depth_path.exists():
        skipped.append(name)                    # registered in SfM, but stereo skipped/failed it
        continue
    depth = read_array(depth_path)               # from read_write_dense.py
    depth = reconcile_resolution(depth, K, image_shape_for(name))  # from 3.2 step 3
    frames[name] = Frame(depth=depth, K=K, R=R, t=t)

log.info(f"{len(frames)} frames ready, {len(skipped)} registered-but-no-depth, "
         f"{n_video_frames - len(poses)} never registered in SfM")
```

Log and report `skipped`, not just silently drop it — the count of frames that
never make it to a `Frame` object (unregistered in SfM, or registered but missing
depth) is a real limitation of the pipeline and belongs in the write-up, not buried.

## 4. Edge cases

- **Unregistered frames.** `automatic_reconstructor` commonly drops frames (motion
  blur, textureless trunk-only views, etc.). Detections on those frames have no 3D
  anchor at all — this is exactly why the 2D-tracking step matters: a pod tracked
  across neighboring *registered* frames still gets a landmark even if a few frames
  in the middle of its track were dropped by SfM.
- **Registered but no depth.** SfM succeeded but PatchMatchStereo produced nothing
  usable for that view (rare, but happens with severe local blur/exposure jumps).
  Handle identically to unregistered — skip for 3D, rely on the track.
- **Depth map exists but the masked region is empty after filtering.**
  `backproject_mask_to_surface` already returns `None` for this. The caller should
  treat this as "no evidence from this frame" and continue, not as a fatal error
  for the whole track.

## 5. Validation (do this before running on a full video)

1. **Reprojection sanity check.** Take 5–10 sparse 3D points from the reconstruction,
   reproject them into 2–3 images using the `K, R, t` you just built
   (`u,v ~ K @ (R @ X_world + t)`, divide by z), and confirm the reprojection error
   is a few pixels at most. A transposed `R` or flipped `t` typically produces
   errors of hundreds of pixels — this check catches convention bugs immediately,
   before they show up 10 steps downstream as "the fruit count is wrong."
2. **Single-pod overlap check.** Pick one large, clearly unoccluded pod visible in 3
   widely-separated frames. Back-project its mask in each and confirm the resulting
   surface patches actually spatially overlap. If they don't, the bug is in this
   reading step (wrong camera folder, wrong resolution reconciliation, or a
   convention mismatch) — not in the clustering logic downstream.
