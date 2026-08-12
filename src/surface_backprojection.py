"""
surface_backprojection.py

Given a COLMAP-reconstructed frame (camera pose, intrinsics, MVS depth map)
and a YOLO instance-segmentation mask for a detected cacao pod, this module
back-projects the *visible surface* of that pod into world-space 3D points
("surface patch") instead of collapsing the mask to a single idealized
object-center ray.

Load depth maps / poses / intrinsics with COLMAP's own helpers -- do not
hand-roll parsers for these binary formats:
  - poses & intrinsics: pycolmap, or colmap/scripts/python/read_write_model.py
  - MVS depth maps:     colmap/scripts/python/read_write_dense.py
This module only assumes you already have them as plain numpy arrays.
"""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


@dataclass
class Frame:
    depth: np.ndarray   # (H, W) float32, 0/NaN = invalid
    K: np.ndarray        # (3, 3) intrinsics
    R: np.ndarray         # (3, 3) world -> camera rotation (COLMAP convention)
    t: np.ndarray          # (3,)   world -> camera translation; X_cam = R @ X_world + t


def backproject_mask_to_surface(mask: np.ndarray, frame: "Frame",
                                 erode_px: int = 3,
                                 mad_k: float = 3.0,
                                 min_valid_px: int = 15):
    """
    mask: (H, W) bool, the instance-segmentation mask for ONE detection.
    Returns an (N, 3) array of world-space points on the pod's visible
    surface in this frame, or None if too few reliable pixels remain.
    """
    eroded = binary_erosion(mask, iterations=erode_px)
    if eroded.sum() < min_valid_px:
        eroded = mask  # don't punish small/thin pods for erosion

    vs, us = np.nonzero(eroded)
    d = frame.depth[vs, us]
    valid = np.isfinite(d) & (d > 0)
    us, vs, d = us[valid], vs[valid], d[valid]
    if len(d) < min_valid_px:
        return None

    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-6
    keep = np.abs(d - med) < mad_k * 1.4826 * mad
    us, vs, d = us[keep], vs[keep], d[keep]
    if len(d) < min_valid_px:
        return None

    K_inv = np.linalg.inv(frame.K)
    pix_h = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)
    rays = K_inv @ pix_h
    cam_pts = rays * d[None, :]
    world_pts = (frame.R.T @ (cam_pts - frame.t[:, None])).T
    return world_pts


def robust_centroid(patch: np.ndarray) -> np.ndarray:
    return np.median(patch, axis=0)


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

    def __init__(self, eps: float, min_overlap_frac: float = 0.15):
        self.eps = eps
        self.min_overlap_frac = min_overlap_frac
        self.patches = []
        self.parent = []

    def _find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def _union(self, i, j):
        ri, rj = self._find(i), self._find(j)
        if ri != rj:
            self.parent[rj] = ri

    def add(self, patch: np.ndarray) -> int:
        idx = len(self.patches)
        self.patches.append(patch)
        self.parent.append(idx)
        for j in range(idx):
            if patches_overlap(self.patches[j], patch, self.eps, self.min_overlap_frac):
                self._union(j, idx)
        return idx

    def landmark_count(self) -> int:
        return len(set(self._find(i) for i in range(len(self.patches))))
