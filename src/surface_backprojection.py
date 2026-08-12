"""
surface_backprojection.py

Given a COLMAP-reconstructed frame (camera pose, intrinsics, MVS depth map)
and a YOLO instance-segmentation mask for a detected cacao pod, this module
back-projects the *visible surface* of that pod into world-space 3D points
("surface patch") instead of collapsing the mask to a single idealized
object-center ray.
"""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_erosion


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
    valid = np.isfinite(d) & (d > 0.5) & (d < 5.0)
    us, vs, d = us[valid], vs[valid], d[valid]
    if len(d) < min_valid_px:
        return None

    bins = np.linspace(0.5, 5.0, 20)
    hist, _ = np.histogram(d, bins=bins)
    peak_bin = np.argmax(hist)
    lo, hi = bins[peak_bin], bins[min(peak_bin + 1, len(bins) - 1)]
    peak_centre = (lo + hi) / 2
    keep = np.abs(d - peak_centre) < 0.5
    us, vs, d = us[keep], vs[keep], d[keep]
    if len(d) < 5:
        return None

    K_inv = np.linalg.inv(frame.K)
    pix_h = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)
    rays = K_inv @ pix_h
    cam_pts = rays * d[None, :]
    world_pts = (frame.R.T @ (cam_pts - frame.t[:, None])).T
    return world_pts


def robust_centroid(patch: np.ndarray) -> np.ndarray:
    med_z = np.median(patch[:, 2])
    mad_z = np.median(np.abs(patch[:, 2] - med_z)) + 1e-6
    keep = np.abs(patch[:, 2] - med_z) < 2.0 * mad_z
    return np.median(patch[keep], axis=0)
