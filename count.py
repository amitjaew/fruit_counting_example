#!/usr/bin/env python3
"""
count.py — Mode 1: compute and bake results.

Usage:
    python count.py --workspace WORKSPACE --video-id <id> [--eps 0.02]
                    [--min-overlap-frac 0.15] [--conf 0.4] [--min-area 100]

Prints exactly one integer (the fruit count) to stdout.
All diagnostics go to stderr.
"""

import argparse
import os
import sys
import time

import numpy as np

from src.colmap_reader import load_frames
from src.yolo_detector import detect_all_frames, Detection
from src.surface_backprojection import backproject_mask_to_surface
from src.landmark_merger import LandmarkMerger
from src.tracker import build_tracks
from src.baked_results import BakedResults, save_baked


def main():
    parser = argparse.ArgumentParser(
        description="Count unique cacao fruits from video using COLMAP + YOLO + 3D back-projection."
    )
    parser.add_argument("--workspace", required=True, help="Workspace root directory")
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g. L0)")
    parser.add_argument("--eps", type=float, default=0.02,
                        help="3D clustering distance in meters (default: %(default)s)")
    parser.add_argument("--min-overlap-frac", type=float, default=0.15,
                        help="Min fraction of points that must overlap to merge (default: %(default)s)")
    parser.add_argument("--conf", type=float, default=0.4, dest="conf_threshold",
                        help="YOLO confidence threshold (default: %(default)s)")
    parser.add_argument("--min-area", type=int, default=100,
                        help="Minimum mask pixel area (default: %(default)s)")
    parser.add_argument("--include-flowers", action="store_true",
                        help="Include cac-flor detections (default: False)")
    parser.add_argument("--model", default="models/cacao-woord-13-yolo26l-seg-t1.pt",
                        help="Path to YOLO model weights (default: %(default)s)")
    args = parser.parse_args()

    workspace = args.workspace
    video_id = args.video_id

    print(f"Loading COLMAP data for {video_id}...", file=sys.stderr)
    t0 = time.time()
    frames, skipped = load_frames(workspace, video_id)
    if not frames:
        print(f"ERROR: No registered frames found for '{video_id}' in {workspace}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(frames)} frames loaded, {len(skipped)} skipped ({time.time() - t0:.1f}s)", file=sys.stderr)

    image_dir = os.path.join(workspace, video_id, "dense", "0", "images")
    if not os.path.isdir(image_dir):
        print(f"ERROR: Undistorted image directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Running YOLO detection (conf >= {args.conf_threshold})...", file=sys.stderr)
    t0 = time.time()
    yolo_detections = detect_all_frames(
        args.model, image_dir,
        conf_threshold=args.conf_threshold,
        min_area=args.min_area,
        include_flowers=args.include_flowers,
    )
    total_dets = sum(len(d) for d in yolo_detections.values())
    print(f"  {total_dets} detections across {len(yolo_detections)} frames ({time.time() - t0:.1f}s)", file=sys.stderr)

    print("Deduplicating overlapping detections...", file=sys.stderr)
    dedup_removed = 0
    detections_for_tracker: dict[str, list[dict]] = {}
    all_detections: dict[str, list[dict]] = {}
    frame_order: list[str] = []

    for fname in sorted(frames.keys()):
        frame_order.append(fname)
        frame_dets = yolo_detections.get(fname, [])
        flat_dets = []
        for d in frame_dets:
            flat_dets.append({
                "frame_name": d.frame_name,
                "mask": d.mask,
                "bbox": d.bbox,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "det_id": d.det_id,
            })

        # Deduplicate overlapping detections within the same frame
        n = len(flat_dets)
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                a, b = flat_dets[i], flat_dets[j]
                ma, mb = a["mask"], b["mask"]
                # Pad masks to same size if needed
                if ma.shape != mb.shape:
                    h = max(ma.shape[0], mb.shape[0])
                    w = max(ma.shape[1], mb.shape[1])
                    pa = np.zeros((h, w), dtype=bool)
                    pb = np.zeros((h, w), dtype=bool)
                    pa[:ma.shape[0], :ma.shape[1]] = ma
                    pb[:mb.shape[0], :mb.shape[1]] = mb
                    ma, mb = pa, pb
                inter = np.logical_and(ma, mb).sum()
                union = np.logical_or(ma, mb).sum()
                iou = inter / union if union > 0 else 0
                if iou > 0.5:
                    if a["confidence"] >= b["confidence"]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break  # this detection is removed, stop comparing it
                    dedup_removed += 1

        flat_dets = [d for d, k in zip(flat_dets, keep) if k]
        if flat_dets:
            detections_for_tracker[fname] = flat_dets
        all_detections[fname] = flat_dets

    if dedup_removed > 0:
        print(f"  Removed {dedup_removed} duplicate detections", file=sys.stderr)

    print("Building 2D tracks...", file=sys.stderr)
    t0 = time.time()
    tracks = build_tracks(detections_for_tracker, iou_threshold=0.15, max_coast_frames=30)
    print(f"  {len(tracks)} tracks ({time.time() - t0:.1f}s)", file=sys.stderr)

    print("Back-projecting and merging landmarks...", file=sys.stderr)
    t0 = time.time()
    merger = LandmarkMerger(eps=args.eps, min_overlap_frac=args.min_overlap_frac)
    rng = np.random.default_rng(42)
    patches: dict[str, np.ndarray] = {}
    landmark_of: dict[str, int] = {}
    det_lookup: dict[str, tuple[str, dict]] = {}

    for fname, det_list in all_detections.items():
        for d in det_list:
            det_lookup[d["det_id"]] = (fname, d)

    n_tracks = len(tracks)
    n_total_dets = sum(len(t.det_ids) for t in tracks)
    n_processed = 0

    for ti, track in enumerate(tracks):
        track_patches = []
        track_det_ids = []
        for det_id in track.det_ids:
            entry = det_lookup.get(det_id)
            if entry is None:
                continue
            fname, d = entry
            if det_id in patches:
                track_patches.append(patches[det_id])
                track_det_ids.append(det_id)
                n_processed += 1
                continue
            if fname not in frames:
                continue
            pts = backproject_mask_to_surface(d["mask"], frames[fname])
            if pts is not None:
                patches[det_id] = pts
                track_patches.append(pts)
                track_det_ids.append(det_id)
            n_processed += 1

        if track_patches:
            combined = np.vstack(track_patches)
            if len(combined) > 500:
                idx_sample = rng.choice(len(combined), size=500, replace=False)
                combined = combined[idx_sample]
            track_frames = {det_lookup[did][0] for did in track_det_ids if did in det_lookup}
            patch_idx = merger.add(combined, frames=track_frames)
            landmark_id = merger.landmark_of(patch_idx)
            for det_id in track_det_ids:
                landmark_of[det_id] = landmark_id

        if (ti + 1) % 10 == 0 or ti == n_tracks - 1:
            print(f"\r  det {n_processed}/{n_total_dets} | track {ti+1}/{n_tracks} | {len(patches)} patches, {merger.landmark_count()} landmarks", end="", flush=True, file=sys.stderr)

    print(file=sys.stderr)
    fruit_count = merger.landmark_count()
    bp_time = time.time() - t0
    print(f"  {fruit_count} unique fruits ({bp_time:.1f}s)", file=sys.stderr)

    camera_positions = {}
    for fname, frame in frames.items():
        cam_world = (-frame.R.T @ frame.t).tolist()
        camera_positions[fname] = [float(v) for v in cam_world]

    serializable_detections = {}
    masks_for_bake = {}
    for fname, det_list in all_detections.items():
        sd_list = []
        for d in det_list:
            sd_list.append({
                "frame_name": d["frame_name"],
                "bbox": d["bbox"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "det_id": d["det_id"],
            })
            masks_for_bake[d["det_id"]] = d["mask"]
        serializable_detections[fname] = sd_list

    results = BakedResults(
        video_id=video_id,
        frame_order=frame_order,
        detections=serializable_detections,
        patches=patches,
        masks=masks_for_bake,
        landmark_of=landmark_of,
        landmark_count=fruit_count,
        skipped_frames=skipped,
        params={
            "eps": args.eps,
            "min_overlap_frac": args.min_overlap_frac,
            "conf_threshold": args.conf_threshold,
            "min_area": args.min_area,
            "include_flowers": args.include_flowers,
        },
        camera_positions=camera_positions,
    )

    save_baked(results, workspace)
    print(f"Results saved to {workspace}/results/{video_id}.baked.*", file=sys.stderr)

    print(fruit_count)


if __name__ == "__main__":
    main()
