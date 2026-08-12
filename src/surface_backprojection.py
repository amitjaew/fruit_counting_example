"""
surface_backprojection.py

Data container for a COLMAP-reconstructed frame: camera pose, intrinsics
and (optional) depth map. The counting pipeline now uses COLMAP's sparse
3D points rather than the dense depth maps; this module only retains the
Frame dataclass consumed by colmap_reader.py.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    depth: np.ndarray   # (H, W) float32, 0/NaN = invalid
    K: np.ndarray        # (3, 3) intrinsics
    R: np.ndarray         # (3, 3) world -> camera rotation (COLMAP convention)
    t: np.ndarray          # (3,)   world -> camera translation; X_cam = R @ X_world + t
