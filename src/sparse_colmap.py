"""
sparse_colmap.py

Reads COLMAP sparse reconstruction outputs (points3D.bin and the 2D
keypoint observations in images.bin) to provide reliable triangulated
3D points for each frame.

The sparse reconstruction is high quality (sub-pixel reprojection error,
multi-view triangulation) unlike the dense depth maps, so these points
give a stable 3D anchor for each fruit detection.
"""

import os
import struct

import numpy as np


def read_sparse_points(points3D_path: str) -> dict[int, tuple]:
    """Return point3D_id -> (x, y, z) world position."""
    points = {}
    with open(points3D_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, os.SEEK_CUR)
            points[pid] = xyz
    return points


def read_sparse_observations(images_bin_path: str) -> dict[str, np.ndarray]:
    """
    Return frame_name -> structured array with fields (x, y, id)
    for every 2D keypoint observation in that frame.
    """
    obs = {}
    with open(images_bin_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)  # image_id
            f.read(32)  # qvec
            f.read(24)  # tvec
            f.read(4)  # camera_id
            name = b""
            while True:
                b = f.read(1)
                if b == b"\x00":
                    break
                name += b
            name = name.decode()
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            if num_points2D > 0:
                data = f.read(num_points2D * 24)
                obs[name] = np.frombuffer(data, dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            else:
                obs[name] = np.empty(0, dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
    return obs


def point_ids_in_mask(mask: np.ndarray, obs: np.ndarray) -> set:
    """Return the set of point3D_ids whose 2D observation falls inside the mask."""
    valid = obs[obs["id"] >= 0]
    if len(valid) == 0:
        return set()
    xs = valid["x"].astype(np.int32)
    ys = valid["y"].astype(np.int32)
    h, w = mask.shape
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[in_bounds], ys[in_bounds]
    ids = valid["id"][in_bounds]
    selected = mask[ys, xs]
    return {int(i) for i in ids[selected]}


def load_sparse(workspace: str, video_id: str):
    sparse_dir = os.path.join(workspace, video_id, "dense", "0", "sparse")
    points = read_sparse_points(os.path.join(sparse_dir, "points3D.bin"))
    obs = read_sparse_observations(os.path.join(sparse_dir, "images.bin"))
    return points, obs
