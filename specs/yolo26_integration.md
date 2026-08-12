# Spec: YOLO26-seg Integration

Defines how raw video frames become the per-frame detection objects consumed by
tracking and `surface_backprojection.py`.

## 1. Inputs

- The provided YOLO26-Large instance-segmentation weights (cacao classes:
  `cac-sm`, `cac-m`, `cac-l`, `cac-y`, `cac-flor`).
- Frames from `WORKSPACE/dense/images/` (the **undistorted** frames COLMAP
  produced) — not the raw video frames. Running YOLO on undistorted frames keeps
  detection pixel coordinates aligned with the depth maps without a separate
  distortion-remap step. See prior spec, §1.
- Same canonical frame-naming convention used throughout (`frame_%06d.png`).

## 2. Class handling

- `cac-flor` (flowers) excluded by default — filter at inference or post-filter by
  class id. Expose as a config flag (`include_flowers: bool`), off by default,
  since the task treats flowers as optional/ignorable, not because the model
  can't detect them.
- `cac-sm` / `cac-m` / `cac-l` / `cac-y` are all **the same physical object
  category** (a cacao pod) at different maturity/ripeness stages — they must not
  be treated as separate object identities for counting purposes. A pod
  photographed early in the orbit as `cac-sm` and later reclassified as `cac-m`
  by the detector (confidence/angle-dependent) is still one fruit. Class label is
  carried as metadata on the detection, not used as a matching key.

## 3. Per-frame output contract

```python
@dataclass
class Detection:
    frame_name: str        # matches Frame dict key
    mask: np.ndarray        # (H, W) bool, same resolution as the undistorted frame
    bbox: tuple[int, int, int, int]
    class_name: str          # cac-sm / cac-m / cac-l / cac-y (post flower-filter)
    confidence: float
    det_id: str                # unique per (frame, detection) — e.g. f"{frame_name}:{i}"
```

`detections: dict[str, list[Detection]]` keyed by `frame_name`, mirroring the
`frames` dict from the reader spec — every downstream lookup joins on this key.

## 4. Confidence / quality filtering

- Apply a confidence threshold at inference (config value, not hardcoded) —
  recommend tuning this on a held-out labeled subset rather than guessing, since
  too low → noisy detections inflate the count pre-dedup, too high → misses
  partially-occluded pods, which is exactly what the 3D fusion step is meant to
  recover.
- Discard masks below a minimum pixel-area threshold. Segmentation models
  routinely emit small spurious detections at low confidence; these are also the
  detections most likely to produce a bad surface patch after mask erosion (see
  reader spec §3.2 — patches under ~15 valid pixels are already dropped by
  `backproject_mask_to_surface`, so filtering here just avoids wasted inference
  bookkeeping, not a correctness requirement).
- Do **not** apply single-frame NMS across classes assuming one detection = one
  object globally — that's the job of the 3D/tracking stage, not this one. Keep
  per-frame instance segmentation outputs as-is (standard per-frame NMS within the
  detector is fine; cross-frame dedup is explicitly out of scope here).

## 5. Mask format requirements

- Store masks as boolean arrays at the same resolution as the undistorted frame
  actually fed to `backproject_mask_to_surface` (i.e., reconciled with whatever
  resolution decision was made in reader spec §3.2 — depth-map-native or
  full-image resolution, consistently).
- If YOLO26 outputs polygon contours rather than dense masks, rasterize to a dense
  boolean array before storage — the erosion + pixel-sampling logic in
  `backproject_mask_to_surface` expects a dense mask, not a polygon.
- Persist masks in a compact encoded form (RLE) rather than dense boolean arrays
  on disk if serializing detections between the YOLO stage and later stages as
  separate pipeline runs — dense boolean masks for thousands of frames get large
  fast. In-memory during a single run, dense is fine and simpler.

## 6. What this stage does *not* do

- No cross-frame association (that's the 2D tracker, feeding the 3D fusion stage).
- No 3D reasoning at all — this stage is purely 2D, per-frame, stateless across
  frames. Keeping it stateless makes it trivially parallelizable across frames
  (relevant for the production-scale question — this stage batches embarrassingly
  well across a GPU queue independent of the sequential COLMAP/tracking stages).

## 7. Validation before trusting this at scale

- Visualize masks overlaid on 10–15 sampled frames spanning the full orbit
  (including frames near the trunk-occlusion side) — confirm no systematic
  class confusion (e.g. `cac-y` vs background bark color) before running the full
  video through.
- Spot-check mask resolution actually matches what reader spec §3.2 assumes; a
  silent resize mismatch here breaks the pixel-alignment assumption the whole
  back-projection step depends on.
