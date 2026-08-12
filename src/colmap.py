"""
colmap.py

Reads COLMAP sparse reconstruction outputs (camera poses, sparse 3D
points, and 2D keypoint observations) that feed the fruit-counting
pipeline.

The sparse reconstruction is high quality (sub-pixel reprojection error,
multi-view triangulation), unlike the dense depth maps, so its points give
a stable 3D anchor for each fruit detection.
"""

import os
import struct

import numpy as np


def _qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])


def _read_images_bin(path):
    """
    Return (frame_names, poses, observations) from images.bin.

    frame_names  : sorted list of registered image filenames
    poses        : frame_name -> (R, t) with X_cam = R @ X_world + t
    observations : frame_name -> structured array (x, y, id) of 2D keypoints
    """
    frame_names = []
    poses = {}
    observations = {}
    obs_dtype = [("x", "<f8"), ("y", "<f8"), ("id", "<i8")]

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4)   # image_id
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            f.read(4)   # camera_id
            name = b""
            while True:
                b = f.read(1)
                if b == b"\x00":
                    break
                name += b
            name = name.decode()

            R = _qvec2rotmat(qvec)
            t = np.array(tvec)
            frame_names.append(name)
            poses[name] = (R, t)

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            if num_points2D > 0:
                observations[name] = np.frombuffer(
                    f.read(num_points2D * 24), dtype=obs_dtype
                )
            else:
                observations[name] = np.empty(0, dtype=obs_dtype)

    frame_names.sort()
    return frame_names, poses, observations


def _read_points3d(path):
    """Return point3D_id -> (x, y, z) world position."""
    points = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3)   # rgb
            f.read(8)   # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, os.SEEK_CUR)
            points[pid] = xyz
    return points


def point_ids_in_mask(mask, obs):
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


def load_colmap(workspace, video_id):
    """
    Read all COLMAP sparse outputs for a video.

    Returns (frame_names, camera_positions, points3d, observations):
      frame_names      : sorted registered image filenames
      camera_positions : frame_name -> [x, y, z] camera world position
      points3d         : point3D_id -> (x, y, z) world position
      observations     : frame_name -> structured array (x, y, id)
    """
    sparse_dir = os.path.join(workspace, video_id, "dense", "0", "sparse")
    frame_names, poses, observations = _read_images_bin(os.path.join(sparse_dir, "images.bin"))
    points3d = _read_points3d(os.path.join(sparse_dir, "points3D.bin"))
    camera_positions = {
        fname: (-R.T @ t).tolist() for fname, (R, t) in poses.items()
    }
    return frame_names, camera_positions, points3d, observations
