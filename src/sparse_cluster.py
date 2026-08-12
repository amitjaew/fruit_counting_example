"""
sparse_cluster.py

Clusters fruit detections by shared COLMAP sparse point IDs, with a
same-frame constraint.

A sparse 3D point appears as a 2D keypoint in ~13 frames. Detections that
share many point IDs are likely the same fruit. However, a single trunk
feature point can fall inside different fruits' masks in different frames,
so weak sharing (1 point) is unreliable and must be gated by a same-frame
constraint: two detections in the same frame are always different fruits.

Algorithm: count shared point IDs per detection pair, process pairs in
order of decreasing shared count, and union via Union-Find while tracking
the set of frames each cluster occupies -- a pair is skipped if its clusters
already share any frame.
"""

from collections import defaultdict

import numpy as np


def cluster_by_point_ids(
    det_points: dict[str, set],
    frame_of: dict[str, str],
    min_shared_points: int = 2,
) -> tuple[dict[str, int], int]:
    det_ids = list(det_points.keys())
    n = len(det_ids)
    if n == 0:
        return {}, 0

    idx = {det_id: i for i, det_id in enumerate(det_ids)}

    point_to_dets: dict[int, list[int]] = defaultdict(list)
    for det_id, pids in det_points.items():
        di = idx[det_id]
        for pid in pids:
            point_to_dets[pid].append(di)

    pair_counts: dict[tuple, int] = defaultdict(int)
    for pid, dets in point_to_dets.items():
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                if a > b:
                    a, b = b, a
                pair_counts[(a, b)] += 1

    pairs = sorted(pair_counts.items(), key=lambda kv: -kv[1])

    parent = list(range(n))
    frame_sets = [set() for _ in range(n)]
    for det_id in det_ids:
        di = idx[det_id]
        frame_sets[di] = {frame_of.get(det_id, det_id.split(":")[0])}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), count in pairs:
        if count < min_shared_points:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if frame_sets[ra] & frame_sets[rb]:
            continue
        parent[rb] = ra
        frame_sets[ra] |= frame_sets[rb]

    roots: dict[int, int] = {}
    landmark_of: dict[str, int] = {}
    for det_id in det_ids:
        di = idx[det_id]
        r = find(di)
        if r not in roots:
            roots[r] = len(roots)
        landmark_of[det_id] = roots[r]

    return landmark_of, len(roots)


def merge_by_centroid(
    landmark_of: dict[str, int],
    det_points: dict[str, set],
    frame_of: dict[str, str],
    points3d: dict[int, tuple],
    centroid_merge_dist: float = 0.25,
) -> dict[str, int]:
    """
    Re-merge clusters whose sparse-point centroids are close but which
    shared no point IDs (long occlusion gaps / viewpoint changes).

    Respects the same-frame constraint via per-cluster frame sets.
    """
    lid_pts: dict[int, list] = defaultdict(list)
    lid_frames: dict[int, set] = defaultdict(set)
    for det_id, lid in landmark_of.items():
        pids = det_points.get(det_id, set())
        pts = np.array([points3d[p] for p in pids if p in points3d], dtype=np.float64)
        if len(pts):
            lid_pts[lid].append(pts)
        lid_frames[lid].add(frame_of.get(det_id, det_id.split(":")[0]))

    centroids = {}
    for lid, plist in lid_pts.items():
        centroids[lid] = np.median(np.vstack(plist), axis=0)

    lids = sorted(centroids.keys())
    pairs = []
    for i in range(len(lids)):
        for j in range(i + 1, len(lids)):
            d = float(np.linalg.norm(centroids[lids[i]] - centroids[lids[j]]))
            pairs.append((d, lids[i], lids[j]))
    pairs.sort()

    remap: dict[int, int] = {}
    for d, a, b in pairs:
        if d > centroid_merge_dist:
            break
        while a in remap:
            a = remap[a]
        while b in remap:
            b = remap[b]
        if a == b:
            continue
        if lid_frames[a] & lid_frames[b]:
            continue
        remap[b] = a
        lid_frames[a] |= lid_frames[b]

    new_lm = {}
    for det_id, lid in landmark_of.items():
        while lid in remap:
            lid = remap[lid]
        new_lm[det_id] = lid
    return new_lm
