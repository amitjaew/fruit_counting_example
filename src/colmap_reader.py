"""
colmap_reader.py

Reads COLMAP dense reconstruction outputs (cameras, images, depth maps)
and assembles Frame objects consumable by surface_backprojection.py.

Binary format references:
  cameras.bin: [num_cameras: uint64] [camera_id: uint32] [model_id: uint32]
               [width: uint64] [height: uint64] [params: float64[]] ...
  images.bin:  [num_images: uint64] [image_id: uint32]
               [qw,qx,qy,qz: float64] [tx,ty,tz: float64] [camera_id: uint32]
               [name: null-term string] [num_points2D: uint64] [point2D data] ...
  Depth maps:  ASCII header "{w}&{h}&1&" (no null terminator, exactly 10+ chars)
               followed by raw float32 array of size w*h.
"""

import os
import struct
from collections import namedtuple

import numpy as np

from src.surface_backprojection import Frame

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])


def _read_next_bytes(fid, num_bytes):
    return fid.read(num_bytes)


def read_cameras_bin(path: str) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params_map = {
                0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12,
            }
            num_params = num_params_map.get(model_id, 0)
            params = []
            for _ in range(num_params):
                params.append(struct.unpack("<d", f.read(8))[0])
            cameras[camera_id] = Camera(
                id=camera_id, model=model_id,
                width=width, height=height,
                params=params,
            )
    return cameras


def _qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])


def read_images_bin(path: str) -> tuple[dict[str, dict], dict[int, int]]:
    images = {}
    image_camera_map = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name_bytes = b""
            while True:
                b = f.read(1)
                if b == b"\x00":
                    break
                name_bytes += b
            name = name_bytes.decode()
            R = _qvec2rotmat(qvec)
            t = np.array(tvec)
            images[name] = {"image_id": image_id, "R": R, "t": t}
            image_camera_map[image_id] = camera_id
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points2D * 24, os.SEEK_CUR)
    return images, image_camera_map


def read_depth_map(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    amp_pos = raw.find(b"&")
    if amp_pos == -1:
        return None
    parts = raw[:amp_pos + 10].decode(errors="ignore").rstrip("&").split("&")
    if len(parts) < 2:
        return None
    w, h = int(parts[0]), int(parts[1])
    data_start = len(parts[0]) + len(parts[1]) + 2
    depth = np.frombuffer(raw[data_start:data_start + w * h * 4], dtype=np.float32).reshape(h, w)
    return depth


def build_frames(sparse_dir: str, depth_map_dir: str, image_dir: str) -> dict[str, Frame]:
    import sys

    cameras = read_cameras_bin(os.path.join(sparse_dir, "cameras.bin"))
    images, image_camera_map = read_images_bin(os.path.join(sparse_dir, "images.bin"))

    depth_map_pattern = ".geometric.bin"

    frames = {}
    skipped = []
    n_total = len(images)

    for i, (name, img_data) in enumerate(images.items()):
        depth_path = os.path.join(depth_map_dir, name + depth_map_pattern)
        depth = read_depth_map(depth_path)
        if depth is None:
            skipped.append(name)
            continue

        camera = cameras[image_camera_map[img_data["image_id"]]]
        if camera.model not in (0, 1):
            raise ValueError(f"Expected PINHOLE (0 or 1), got {camera.model} for camera {camera.id}")

        if camera.model == 1:
            fx, fy, cx, cy = camera.params
        else:
            f = camera.params[0]
            cx, cy = camera.params[1], camera.params[2]
            fx, fy = f, f

        img_path = os.path.join(image_dir, name)
        if os.path.exists(img_path):
            import cv2
            img_h, img_w = cv2.imread(img_path).shape[:2]
        else:
            img_h, img_w = int(camera.height), int(camera.width)

        dh, dw = depth.shape[:2]
        if (dh, dw) != (img_h, img_w):
            import cv2
            depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        frames[name] = Frame(depth=depth, K=K, R=img_data["R"], t=img_data["t"])

        if (i + 1) % 50 == 0 or i == n_total - 1:
            print(f"\r  loading depth maps: {i+1}/{n_total} ({len(frames)} loaded, {len(skipped)} skipped)", end="", flush=True, file=sys.stderr)

    print(file=sys.stderr)
    return frames, skipped


def load_frames(workspace: str, video_id: str) -> tuple[dict[str, Frame], list[str]]:
    sparse_dir = os.path.join(workspace, video_id, "dense", "0", "sparse")
    depth_map_dir = os.path.join(workspace, video_id, "dense", "0", "stereo", "depth_maps")
    image_dir = os.path.join(workspace, video_id, "dense", "0", "images")
    return build_frames(sparse_dir, depth_map_dir, image_dir)
