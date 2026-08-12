"""
baked_results.py

Dataclass and serialization for the pipeline's intermediate results,
shared between count.py (Mode 1) and view.py (Mode 2).
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np


def _rle_encode(mask: np.ndarray) -> np.ndarray:
    flat = mask.ravel()
    transitions = np.diff(np.concatenate([[0], flat.astype(np.int8), [0]]))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    lengths = ends - starts
    rle = np.empty(len(starts) * 2, dtype=np.uint32)
    rle[0::2] = starts
    rle[1::2] = lengths
    return rle


def _rle_decode(rle: np.ndarray, size: tuple) -> np.ndarray:
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=bool)
    for i in range(0, len(rle), 2):
        start = rle[i]
        length = rle[i + 1]
        flat[start:start + length] = True
    return flat.reshape(h, w)


@dataclass
class BakedResults:
    video_id: str
    frame_order: list[str]
    detections: dict[str, list[dict]]
    patches: dict[str, np.ndarray]
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    landmark_of: dict[str, int] = field(default_factory=dict)
    landmark_count: int = 0
    skipped_frames: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    camera_positions: dict[str, list[float]] = field(default_factory=dict)


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def save_baked(results: BakedResults, workspace: str):
    out_dir = os.path.join(workspace, "results")
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{results.video_id}.baked.npz")
    json_path = os.path.join(out_dir, f"{results.video_id}.baked.json")

    arrays = {}

    for det_id, pts in results.patches.items():
        arrays[f"pts_{det_id}"] = pts.astype(np.float32)

    mask_keys = []
    for det_id, mask in results.masks.items():
        rle = _rle_encode(mask)
        arrays[f"rle_{det_id}"] = rle
        arrays[f"rle_size_{det_id}"] = np.array(mask.shape, dtype=np.uint32)
        mask_keys.append(det_id)

    np.savez_compressed(npz_path, **arrays)

    serializable_dets = {}
    for fname, det_list in results.detections.items():
        sd_list = []
        for d in det_list:
            sd_list.append({
                k: _to_native(v) for k, v in d.items()
                if not isinstance(v, np.ndarray)
            })
        serializable_dets[fname] = sd_list

    json_data = _to_native({
        "video_id": results.video_id,
        "frame_order": results.frame_order,
        "detections": serializable_dets,
        "landmark_of": results.landmark_of,
        "landmark_count": results.landmark_count,
        "skipped_frames": results.skipped_frames,
        "params": results.params,
        "camera_positions": results.camera_positions,
        "patch_keys": sorted([k for k in arrays if k.startswith("pts_")]),
        "mask_keys": mask_keys,
    })
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)


def load_baked(workspace: str, video_id: str) -> BakedResults | None:
    npz_path = os.path.join(workspace, "results", f"{video_id}.baked.npz")
    json_path = os.path.join(workspace, "results", f"{video_id}.baked.json")

    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        json_data = json.load(f)

    npz = np.load(npz_path, allow_pickle=True)

    patches = {}
    for key in json_data.get("patch_keys", []):
        det_id = key[4:]
        patches[det_id] = npz[key]

    masks = {}
    for det_id in json_data.get("mask_keys", []):
        rle_key = f"rle_{det_id}"
        size_key = f"rle_size_{det_id}"
        if rle_key in npz and size_key in npz:
            size = tuple(npz[size_key])
            masks[det_id] = _rle_decode(npz[rle_key], size)

    camera_positions = json_data.get("camera_positions", {})

    return BakedResults(
        video_id=json_data["video_id"],
        frame_order=json_data["frame_order"],
        detections=json_data["detections"],
        patches=patches,
        masks=masks,
        landmark_of=json_data["landmark_of"],
        landmark_count=json_data["landmark_count"],
        skipped_frames=json_data["skipped_frames"],
        params=json_data["params"],
        camera_positions=camera_positions,
    )


def results_path(workspace: str, video_id: str) -> str:
    return os.path.join(workspace, "results", f"{video_id}.baked.npz")


def partial_path(workspace: str, video_id: str) -> str:
    return os.path.join(workspace, "results", f"{video_id}.partial.npz")


def save_partial(det_points, masks, detections, frame_order, skipped,
                 workspace, video_id, params):
    out_dir = os.path.join(workspace, "results")
    os.makedirs(out_dir, exist_ok=True)
    npz_path = partial_path(workspace, video_id)
    json_path = os.path.join(out_dir, f"{video_id}.partial.json")

    arrays = {}
    mask_keys = []
    for det_id, mask in masks.items():
        rle = _rle_encode(mask)
        arrays[f"rle_{det_id}"] = rle
        arrays[f"rle_size_{det_id}"] = np.array(mask.shape, dtype=np.uint32)
        mask_keys.append(det_id)

    np.savez_compressed(npz_path, **arrays)

    serializable_dets = {}
    for fname, det_list in detections.items():
        sd_list = []
        for d in det_list:
            sd_list.append({
                k: _to_native(v) for k, v in d.items()
                if not isinstance(v, np.ndarray)
            })
        serializable_dets[fname] = sd_list

    json_data = _to_native({
        "video_id": video_id,
        "frame_order": frame_order,
        "detections": serializable_dets,
        "skipped_frames": skipped,
        "params": params,
        "det_points": {det_id: sorted(pids) for det_id, pids in det_points.items()},
        "mask_keys": mask_keys,
    })
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)


def load_partial(workspace: str, video_id: str) -> dict | None:
    npz_path = partial_path(workspace, video_id)
    json_path = os.path.join(workspace, "results", f"{video_id}.partial.json")
    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    npz = np.load(npz_path, allow_pickle=True)
    masks = {}
    for det_id in data.get("mask_keys", []):
        rle_key = f"rle_{det_id}"
        size_key = f"rle_size_{det_id}"
        if rle_key in npz and size_key in npz:
            masks[det_id] = _rle_decode(npz[rle_key], tuple(npz[size_key]))
    det_points = {}
    for det_id, pids in data.get("det_points", {}).items():
        det_points[det_id] = set(int(p) for p in pids)
    return {
        "det_points": det_points,
        "masks": masks,
        "detections": data["detections"],
        "frame_order": data["frame_order"],
        "skipped_frames": data["skipped_frames"],
        "params": data["params"],
    }
