# Solution: Counting Unique Cacao Fruits

## 1. Problem

A video orbits a cacao tree, and we want the number of *unique* pods on it.

Detection is easy (YOLO26 does that). The hard part is identity. Each pod
shows up in dozens of frames from different angles, at different scales, and
with different ripeness labels. `cac-sm`, `cac-m`, `cac-l`, and `cac-y` are
all the same physical object at different stages, not separate classes. If we
just summed detections we'd count every pod 10 to 30 times.

The real task is cross-frame association: collapse thousands of 2D detections
into the correct number of 3D objects.

## 2. Core Idea

COLMAP's sparse reconstruction gives us a 3D model for free, before any dense
work happens. It produces a camera pose per frame, a set of triangulated 3D
points, and a record of which 3D point each 2D keypoint observes.

Those sparse points are pinned to the tree surface. A pod sits on a fixed spot,
so the points around it keep stable world coordinates as the camera moves.

This gives a clean rule: **two detections that see the same 3D points are the
same fruit.** We never compare masks across viewpoints (they look nothing
alike). We compare the underlying 3D points instead. Sparse points beat the
dense depth maps here because they're more reliable: sub-pixel reprojection
error and they still cover close objects that stereo matching misses.

## 3. Pipeline

```
count.py
  1. load COLMAP sparse output
  2. run YOLO26 segmentation
  3. per-frame dedup (IoU)
  4. sample sparse points inside each mask
  5. Union-Find cluster by shared points
  6. 2D-IoU fallback for unanchored detections
  7. centroid re-merge (fix gap fragments)
  8. DBSCAN split (fix over-merged pairs)
  9. resolve single-frame candidates
  10. relabel by first appearance
  11. bake results

view.py         2D overlay viewer
view_fruits.py  3D per-fruit viewer
```

`count.py` prints exactly one integer to stdout and diagnostics to stderr. The
viewers only read the baked output and never recompute.

## 4. Stages

**COLMAP** (`src/colmap.py`) parses `images.bin` and `points3D.bin` by hand
(no `pycolmap`). It returns frame poses, world points, and per-frame 2D
observations that link pixels to 3D point IDs.

**YOLO26** (`src/detector.py`) segments every undistorted frame, keeps the
fruit classes, rasterizes polygon masks, drops tiny masks, and tags each
detection as `frame:idx`. Class labels are metadata only, never a matching key.

**Per-frame dedup** (`count.py`) removes masks with IoU > 0.5 in the same
frame. This is the only place we assume one detection equals one object, and
only within a single frame.

**Sampling** (`point_ids_in_mask`) looks up which 3D points fall inside each
mask, giving every detection a viewpoint-free signature: a set of world point
IDs.

**Union-Find clustering** (`cluster_by_point_ids`) merges detections that
share at least 2 points. Instead of comparing all pairs (O(n²) at ~2500
detections), it builds an inverted index `point -> detections` and counts shared
points per pair in linear time, then unions greedily.

The **same-frame constraint** matters here. One background point can land
inside two different pods' masks in different frames, so weak sharing isn't
proof. But two detections in the same frame can't be the same pod. Each cluster
tracks its frame set, and a merge is rejected if the two clusters already share
a frame. This stops the most common over-merge: one shared point gluing two
neighbouring pods together.

**2D-IoU fallback** (`count.py`) handles detections with no sparse points
inside them. It links each to the highest-IoU detection in the previous frame
(overlap > 0.15). Adjacent frames look almost identical, so this local match is
reliable. Unlinked detections start a new landmark.

**Centroid re-merge** (`merge_by_centroid`) fixes pods split by occlusion gaps.
It computes the median 3D position of each cluster's points (median shrugs off
outliers) and merges clusters whose centroids are within 0.4 m, respecting the
same-frame constraint.

**DBSCAN split** (`split_overmerged`) fixes the opposite failure, two pods at
different depths swallowed into one cluster. For a correct pod, the median
position of each detection is stable; for a merged pair, they form two blobs.
DBSCAN on those medians finds the blobs, and if two real ones are far apart the
smaller is carved off into a new fruit id.

**Single-frame candidates** (`resolve_candidates`) are suspicious by nature.
This stage discards `cac-sm` detections, ones with no points, and ones below a
confidence floor. Survivors try to merge into the nearest confirmed fruit (via
median nearest-neighbour distance) and are kept as new fruits if nothing
correlates.

**Relabeling** (`count.py`) reorders fruit IDs by first appearance so they're
stable and easy to read.

**Baking** (`src/results.py`) splits output into an `.npz` (3D point patches
and RLE-encoded masks, which keeps thousands of masks small) and a `.json`
(frame order, detection metadata, `landmark_of`, the count, params). A partial
cache stores the YOLO detections and sampled points so tuning clustering
doesn't rerun inference.

**Viewers** colour masks by stable fruit ID (`view.py`) and plot each fruit's
3D point patch with the camera positions that saw it (`view_fruits.py`).

## 5. Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--min-shared-points` | 2 | shared 3D points needed to union two detections |
| `--centroid-merge-dist` | 0.4 m | re-merge distance for gap-fragmented fruits |
| `--single-frame-min-conf` | 0.25 | keep single-frame detections above this |
| `--correlation-eps` | 0.3 m | max distance to merge into a confirmed fruit |
| `--split-eps` | 0.15 m | DBSCAN radius for over-merge detection |
| `--split-min-pts` | 5 | DBSCAN min neighbours |
| `--split-min-cluster-size` | 50 | detections needed for a real sub-fruit |
| `--split-min-dist` | 0.5 m | min separation to trigger a split |
| `--conf` | 0.4 | YOLO confidence threshold |
| `--min-area` | 100 px | min mask area |
