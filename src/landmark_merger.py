"""
landmark_merger.py

Union-Find clustering over 3D surface patches to deduplicate fruit
detections observed from multiple camera viewpoints.
"""

import numpy as np
from scipy.spatial import cKDTree


def patches_overlap(patch_a: np.ndarray, patch_b: np.ndarray,
                     eps: float, min_overlap_frac: float = 0.15) -> bool:
    """
    Same physical fruit if a meaningful FRACTION of surface points lie
    close to each other -- not just if two reduced centroids are close.
    """
    tree_a = cKDTree(patch_a)
    dists, _ = tree_a.query(patch_b, k=1)
    return (dists < eps).mean() >= min_overlap_frac


class LandmarkMerger:
    """Union-Find over per-track surface patches -> final fruit count."""

    def __init__(self, eps: float, min_overlap_frac: float = 0.15,
                 centroid_filter_dist: float = 0.3):
        self.eps = eps
        self.min_overlap_frac = min_overlap_frac
        self.centroid_filter_dist = centroid_filter_dist
        self.patches = []
        self.parent = []
        self.centroids = []
        self.frame_sets = []

    def _find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def _union(self, i, j):
        ri, rj = self._find(i), self._find(j)
        if ri != rj:
            self.parent[rj] = ri
            self.frame_sets[ri] |= self.frame_sets[rj]

    def add(self, patch: np.ndarray, frames: set | None = None) -> int:
        centroid = np.median(patch, axis=0)
        idx = len(self.patches)
        self.patches.append(patch)
        self.parent.append(idx)
        self.centroids.append(centroid)
        self.frame_sets.append(frames or set())

        for j in range(idx):
            if self.frame_sets[j] & self.frame_sets[idx]:
                continue
            dist = np.linalg.norm(self.centroids[j] - centroid)
            if dist > self.centroid_filter_dist:
                continue
            if patches_overlap(self.patches[j], patch, self.eps, self.min_overlap_frac):
                self._union(j, idx)
        return idx

    def landmark_count(self) -> int:
        return len(set(self._find(i) for i in range(len(self.patches))))

    def landmark_of(self, patch_idx: int) -> int:
        return self._find(patch_idx)
