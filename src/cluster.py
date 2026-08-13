"""
cluster.py

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
from scipy.spatial import cKDTree


def _dbscan(pts, eps, min_samples):
    """Minimal DBSCAN via KD-tree connected components. Returns labels array."""
    n = len(pts)
    if n == 0:
        return np.array([], dtype=int)
    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, eps)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_samples:
            continue
        queue = [i]
        labels[i] = cluster_id
        while queue:
            cur = queue.pop()
            for nb in neighbors[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    labels[nb] = cluster_id
                    if len(neighbors[nb]) >= min_samples:
                        queue.append(nb)
                elif labels[nb] == -1:
                    labels[nb] = cluster_id
        cluster_id += 1
    return labels


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


def split_overmerged(
    landmark_of: dict[str, int],
    det_points: dict[str, set],
    frame_of: dict[str, str],
    points3d: dict[int, tuple],
    split_eps: float = 0.3,
    split_min_pts: int = 5,
    split_min_cluster_size: int = 50,
    split_min_dist: float = 0.5,
) -> dict[str, int]:
    """
    Split fruits whose sparse-point cloud is bimodal (e.g. two pods at
    different depths swallowed into one id). The main cluster keeps the
    fruit's id; far clusters are reassigned to a new fruit id.

    Returns a new landmark_of mapping (det_ids may move to new fruit ids).
    """
    fruit_det_pts: dict[int, list] = defaultdict(list)
    for det_id, lid in landmark_of.items():
        pids = det_points.get(det_id, set())
        pts = np.array([points3d[p] for p in pids if p in points3d], dtype=np.float64)
        if len(pts) >= split_min_pts:
            fruit_det_pts[lid].append((det_id, np.median(pts, axis=0)))

    next_id = max(landmark_of.values()) + 1 if landmark_of else 0
    det_reassign: dict[str, int] = {}

    for lid, med_list in fruit_det_pts.items():
        if len(med_list) < split_min_cluster_size * 2:
            continue

        meds = np.array([m for _, m in med_list])
        labels = _dbscan(meds, split_eps, split_min_pts)

        cluster_sizes = {}
        for lab in labels:
            if lab >= 0:
                cluster_sizes[lab] = cluster_sizes.get(lab, 0) + 1
        real = {lab for lab, size in cluster_sizes.items() if size >= split_min_cluster_size}
        if len(real) < 2:
            continue

        centroids = {lab: np.median(meds[labels == lab], axis=0) for lab in real}
        main_lab = max(real, key=lambda l: cluster_sizes[l])
        main_centroid = centroids[main_lab]
        far_labs = [
            lab for lab in real
            if lab != main_lab and np.linalg.norm(centroids[lab] - main_centroid) > split_min_dist
        ]
        if not far_labs:
            continue

        new_lid = next_id
        next_id += 1

        for i, (det_id, _) in enumerate(med_list):
            if labels[i] in far_labs:
                det_reassign[det_id] = new_lid

    new_lm = {}
    for det_id, lid in landmark_of.items():
        new_lm[det_id] = det_reassign.get(det_id, lid)
    return new_lm


def resolve_candidates(
    landmark_of: dict[str, int],
    det_points: dict[str, set],
    frame_of: dict[str, str],
    points3d: dict[int, tuple],
    det_class: dict[str, str],
    det_conf: dict[str, float],
    single_frame_min_conf: float = 0.25,
    correlation_eps: float = 0.5,
) -> dict[str, int]:
    """
    Resolve single-frame fruits: discard false positives and merge real
    fragments into confirmed (multi-frame) fruits.

    A single-frame detection is discarded if it is a small fruit (cac-sm),
    has no sparse points, or falls below the confidence floor. Survivors are
    merged into the confirmed fruit with the smallest median point distance,
    or discarded if none correlates.

    Returns a new landmark_of mapping with discarded detections removed.
    """
    lid_frames: dict[int, set] = defaultdict(set)
    for det_id, lid in landmark_of.items():
        lid_frames[lid].add(frame_of.get(det_id, det_id.split(":")[0]))

    confirmed = {lid for lid, fr in lid_frames.items() if len(fr) >= 2}

    confirmed_pts: dict[int, list] = defaultdict(list)
    for det_id, lid in landmark_of.items():
        if lid in confirmed:
            pids = det_points.get(det_id, set())
            pts = [points3d[p] for p in pids if p in points3d]
            if pts:
                confirmed_pts[lid].append(np.array(pts, dtype=np.float64))

    confirmed_trees = {
        lid: cKDTree(np.vstack(plist)) for lid, plist in confirmed_pts.items() if plist
    }

    discard: set = set()
    merge: dict[str, int] = {}

    for lid in lid_frames:
        if lid in confirmed:
            continue
        for did in [d for d, l in landmark_of.items() if l == lid]:
            cls = det_class.get(did, "")
            conf = det_conf.get(did, 0.0)
            pids = det_points.get(did, set())

            if cls == "cac-sm":
                discard.add(did)
                continue
            if not pids:
                discard.add(did)
                continue
            if conf < single_frame_min_conf:
                discard.add(did)
                continue

            pts = np.array([points3d[p] for p in pids if p in points3d], dtype=np.float64)
            if len(pts) == 0:
                discard.add(did)
                continue

            best_lid = None
            best_d = float("inf")
            for clid, tree in confirmed_trees.items():
                d = float(np.median(tree.query(pts, k=1)[0]))
                if d < best_d:
                    best_d = d
                    best_lid = clid
            if best_lid is not None and best_d < correlation_eps:
                merge[did] = best_lid
            # else: keep as a separate fruit (do not discard) -- it is a
            # non-small, adequately-confident detection with 3D anchor points.

    new_lm = {}
    for det_id, lid in landmark_of.items():
        if det_id in discard:
            continue
        new_lm[det_id] = merge.get(det_id, lid)
    return new_lm
