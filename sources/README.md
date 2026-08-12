# Sources

## Video files

```
sources/L{n}.mp4
```

Where `{n}` is a sequence number (e.g., `L0.mp4`, `L1.mp4`, `L2.mp4`).
Video files are gitignored.

## Extracted frames

```
sources/L{n}/00001.png
sources/L{n}/00002.png
...
```

Each `L{n}/` directory contains all frames extracted from the corresponding video.
Frame filenames are zero-padded to 5 digits.

Extract with:

```bash
pipenv run python helpers/extract_frames.py sources/L{n}.mp4
```

## COLMAP workspace

Each `sources/L{n}/` directory is a COLMAP workspace. The counting pipeline
(`count.py`) consumes only the **sparse** reconstruction outputs — not the
dense depth maps.

```
sources/L{n}/
├── 00001.png ... 00NNN.png    # extracted video frames (input to COLMAP)
├── database.db                # COLMAP feature database (feature extraction/matching)
├── L{n}.db                    # same database, per-sequence copy
├── reconstruction.ini         # COLMAP reconstruction configuration
│
├── sparse/                    # sparse reconstruction (on raw frames)
│   └── 0/
│       ├── cameras.bin        # camera intrinsics
│       ├── images.bin         # camera poses (qvec/tvec)
│       └── points3D.bin       # triangulated 3D points + tracks
│
└── dense/                     # dense reconstruction workspace
    └── 0/
        ├── sparse/            # sparse reconstruction on UNDISTORTED frames
        │   ├── images.bin     #   poses + 2D keypoint observations  <-- USED
        │   └── points3D.bin   #   triangulated sparse 3D points      <-- USED
        ├── images/            # undistorted images (fed to YOLO)      <-- USED
        ├── stereo/
        │   └── depth_maps/    # dense depth maps (NOT used by counting)
        ├── fused.ply          # fused dense point cloud
        ├── meshed-poisson.ply # Poisson mesh
        └── run-colmap-*.sh    # dense reconstruction scripts
```

### What the pipeline reads

`count.py` loads two files via `src/colmap.py`:

| File | Contents | Purpose |
|---|---|---|
| `dense/0/sparse/images.bin` | camera poses + 2D keypoint observations | project/sample sparse points, camera trajectory |
| `dense/0/sparse/points3D.bin` | triangulated 3D points | 3D anchor for fruit identity |

The dense depth maps (`stereo/depth_maps/`) are **not** used — the sparse
reconstruction is more reliable (sub-pixel reprojection error, multi-view
triangulation) and covers close objects that stereo matching misses.

### Regenerating the workspace

The workspace is produced by COLMAP's `automatic_reconstructor` (sparse) plus
`patch_match_stereo` / `stereo_fusion` (dense). The dense scripts in
`dense/0/` (e.g. `run-colmap-geometric.sh`) document the exact commands:

```bash
$COLMAP_EXE_PATH/colmap patch_match_stereo \
  --workspace_path "." \
  --workspace_format COLMAP \
  --PatchMatchStereo.max_image_size 2000 \
  --PatchMatchStereo.geom_consistency true
```

Only the sparse reconstruction is required for counting; the dense steps
(fusion/meshing) are optional.
