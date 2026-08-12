"""
baked_results.py

Dataclass and serialization for the pipeline's intermediate results,
shared between count.py (Mode 1) and inspect.py (Mode 2).
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BakedResults:
    video_id: str
    frame_order: list[str]
    detections: dict[str, list[dict]]
    patches: dict[str, np.ndarray]
    landmark_of: dict[str, int]
    landmark_count: int
    skipped_frames: list[str]
    params: dict
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

    patches_flat = {}
    for det_id, pts in results.patches.items():
        patches_flat[f"pts_{det_id}"] = pts.astype(np.float32)

    np.savez_compressed(npz_path, **patches_flat)

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
        "patch_keys": sorted(patches_flat.keys()),
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
    for key in json_data["patch_keys"]:
        det_id = key[4:]
        patches[det_id] = npz[key]

    camera_positions = json_data.get("camera_positions", {})

    return BakedResults(
        video_id=json_data["video_id"],
        frame_order=json_data["frame_order"],
        detections=json_data["detections"],
        patches=patches,
        landmark_of=json_data["landmark_of"],
        landmark_count=json_data["landmark_count"],
        skipped_frames=json_data["skipped_frames"],
        params=json_data["params"],
        camera_positions=camera_positions,
    )


def results_path(workspace: str, video_id: str) -> str:
    return os.path.join(workspace, "results", f"{video_id}.baked.npz")
