# Fruit Counting

Count unique cacao pods in an orbit video using COLMAP sparse points + YOLO26
segmentation. See `SOLUTION.md` for an in depth approach description.

## Setup

```bash
pipenv install          # or: pip install -r requirements.txt
```

Requires Python >= 3.12 and a working COLMAP workspace per video (see
`sources/README.md`).

## Expected files

```
models/cacao-woord-13-yolo26l-seg-t1.pt   # YOLO weights (override with --model)
sources/L0.mp4                            # input video
sources/L0/                               # COLMAP workspace
├── 00001.png ...                         # extracted frames (input to COLMAP)
└── dense/0/
    ├── sparse/images.bin                 # camera poses + observations
    ├── sparse/points3D.bin               # triangulated 3D points
    └── images/                           # undistorted frames (input to YOLO)
```

The counting pipeline only reads `dense/0/sparse/*.bin` and
`dense/0/images/`. The dense depth maps are not used.

## Sample results

<video src="sample_results.mp4" controls width="640"></video>

`sample_results.mp4` is an annotated render of the L0 pipeline output, one
frame per fruit ID overlaid in a stable color.

## Usage

```bash
# 1. Extract frames from a video (once)
python helpers/extract_frames.py sources/L0.mp4

# 2. Run COLMAP on sources/L0/ (external):
#    point the workspace at sources/L0/ (where the frames are) and run the
#    automatic reconstructor with dense enabled. It writes sparse/0/ and
#    dense/0/ into that directory.

# 3. Count fruits (prints one integer to stdout)
python count.py L0

# 4. Inspect results
python view.py L0           # 2D overlay viewer
python view_fruits.py L0    # 3D per-fruit viewer
```

Add `--cli` to either viewer to force non-GUI mode. Run `python count.py -h`
for tuning options (confidence, clustering thresholds, etc.).
