#!/usr/bin/env python3
"""
count.py — Mode 1: compute and bake results.

Usage:
    python count.py <id> [--workspace sources]
                    [--min-shared-points 2] [--conf 0.4] [--min-area 100]

Pipeline: YOLO -> dedup -> sample COLMAP sparse points inside each mask
-> Union-Find clustering by shared point IDs.

Caches YOLO detections and sparse point IDs after the first run. Subsequent
runs skip detection and sparse sampling.

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
from src.sparse_colmap import load_sparse, point_ids_in_mask
from src.sparse_cluster import cluster_by_point_ids, merge_by_centroid
from src.baked_results import (
    BakedResults, save_baked, load_partial, save_partial, partial_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Count unique cacao fruits from video using COLMAP sparse points + YOLO."
    )
    parser.add_argument("video_id", help="Video identifier (e.g. L0)")
    parser.add_argument("--workspace", default="sources", help="Workspace root directory (default: %(default)s)")
    parser.add_argument("--min-shared-points", type=int, default=2,
                        help="Min shared sparse point IDs to merge detections (default: %(default)s)")
    parser.add_argument("--centroid-merge-dist", type=float, default=0.4,
                        help="Sparse centroid distance to re-merge gap-fragmented fruits (default: %(default)s)")
    parser.add_argument("--conf", type=float, default=0.4, dest="conf_threshold",
                        help="YOLO confidence threshold (default: %(default)s)")
    parser.add_argument("--min-area", type=int, default=100,
                        help="Minimum mask pixel area (default: %(default)s)")
    parser.add_argument("--include-flowers", action="store_true",
                        help="Include cac-flor detections (default: False)")
    parser.add_argument("--model", default="models/cacao-woord-13-yolo26l-seg-t1.pt",
                        help="Path to YOLO model weights (default: %(default)s)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-running YOLO even if partial cache exists")
    args = parser.parse_args()

    workspace = args.workspace
    video_id = args.video_id

    # ---- Load or run YOLO + sparse sampling ----

    partial = None if args.no_cache else load_partial(workspace, video_id)

    if partial is not None:
        print(f"Using cached detections ({partial_path(workspace, video_id)})", file=sys.stderr)
        det_points = partial["det_points"]
        masks = partial["masks"]
        all_detections = partial["detections"]
        frame_order = partial["frame_order"]
        skipped = partial["skipped_frames"]
        print(f"  {len(det_points)} detections with sparse points, {len(frame_order)} frames", file=sys.stderr)
    else:
        print(f"Loading COLMAP sparse reconstruction for {video_id}...", file=sys.stderr)
        t0 = time.time()
        points3d, sparse_obs = load_sparse(workspace, video_id)
        print(f"  {len(points3d)} sparse points, {len(sparse_obs)} frames ({time.time() - t0:.1f}s)", file=sys.stderr)

        print(f"Loading camera poses for {video_id}...", file=sys.stderr)
        frames, skipped = load_frames(workspace, video_id)
        if not frames:
            print(f"ERROR: No registered frames found for '{video_id}' in {workspace}", file=sys.stderr)
            sys.exit(1)

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
                            break
                        dedup_removed += 1

            all_detections[fname] = [d for d, k in zip(flat_dets, keep) if k]

        if dedup_removed > 0:
            print(f"  Removed {dedup_removed} duplicate detections", file=sys.stderr)

        print("Sampling sparse points inside masks...", file=sys.stderr)
        t0 = time.time()
        det_points: dict[str, set] = {}
        masks_for_bake: dict[str, np.ndarray] = {}
        n_frames = len(frame_order)

        for fi, fname in enumerate(frame_order):
            obs = sparse_obs.get(fname)
            dets = all_detections.get(fname, [])
            for d in dets:
                masks_for_bake[d["det_id"]] = d["mask"]
                pids = point_ids_in_mask(d["mask"], obs)
                det_points[d["det_id"]] = pids
            if (fi + 1) % 100 == 0 or fi == n_frames - 1:
                print(f"\r  {fi + 1}/{n_frames} frames | {len(det_points)} detections sampled", end="", flush=True, file=sys.stderr)

        print(file=sys.stderr)
        n_with_points = sum(1 for pids in det_points.values() if pids)
        print(f"  {n_with_points}/{len(det_points)} detections have sparse points ({time.time() - t0:.1f}s)", file=sys.stderr)

        masks = masks_for_bake
        save_partial(
            det_points, masks, all_detections, frame_order, skipped,
            workspace, video_id,
            {"conf_threshold": args.conf_threshold, "min_area": args.min_area,
             "include_flowers": args.include_flowers},
        )
        print(f"  Cached to {partial_path(workspace, video_id)}", file=sys.stderr)

    # ---- Clustering ----

    print(f"Clustering by shared sparse point IDs...", file=sys.stderr)
    t0 = time.time()

    points3d, _ = load_sparse(workspace, video_id)

    det_frame: dict[str, str] = {}
    for fname in frame_order:
        dets = all_detections.get(fname, [])
        for d in dets:
            det_frame[d["det_id"]] = fname

    landmark_of, fruit_count = cluster_by_point_ids(
        det_points, det_frame,
        min_shared_points=args.min_shared_points,
    )
    ct = time.time() - t0
    print(f"  {fruit_count} unique fruits from sparse clustering ({ct:.1f}s)", file=sys.stderr)

    # ---- Fallback: assign detections with no sparse points via 2D IoU ----

    assigned = set(landmark_of.keys())
    unassigned = [did for did in det_frame if did not in assigned]
    if unassigned:
        print(f"  {len(unassigned)} detections have no sparse points; linking via 2D IoU...", file=sys.stderr)
        masks_all = partial["masks"] if partial else masks
        dets_by_frame: dict[str, list[str]] = {}
        for did, fname in det_frame.items():
            dets_by_frame.setdefault(fname, []).append(did)

        def mask_iou(ma, mb):
            if ma.shape != mb.shape:
                h = max(ma.shape[0], mb.shape[0])
                w = max(ma.shape[1], mb.shape[1])
                pa = np.zeros((h, w), dtype=bool)
                pb = np.zeros((h, w), dtype=bool)
                pa[:ma.shape[0], :ma.shape[1]] = ma
                pb[:mb.shape[0], :mb.shape[1]] = mb
                ma, mb = pa, pb
            u = np.logical_or(ma, mb).sum()
            return np.logical_and(ma, mb).sum() / u if u else 0.0

        frames = sorted(dets_by_frame.keys())
        for i, fname in enumerate(frames):
            cur_unassigned = [did for did in dets_by_frame[fname] if did in unassigned]
            if not cur_unassigned:
                continue
            prev_dets = dets_by_frame.get(frames[i - 1], []) if i > 0 else []
            for did in cur_unassigned:
                best_lid = None
                best_iou = 0.0
                for pd in prev_dets:
                    if pd not in landmark_of:
                        continue
                    iou = mask_iou(masks_all[did], masks_all[pd])
                    if iou > best_iou:
                        best_iou = iou
                        best_lid = landmark_of[pd]
                if best_lid is not None and best_iou > 0.15:
                    landmark_of[did] = best_lid
                else:
                    landmark_of[did] = fruit_count
                    fruit_count += 1
                unassigned.discard(did)

    # ---- Centroid re-merge for gap-fragmented fruits ----

    if args.centroid_merge_dist > 0:
        print(f"Re-merging gap fragments by sparse centroid ({args.centroid_merge_dist}m)...", file=sys.stderr)
        landmark_of = merge_by_centroid(
            landmark_of, det_points, det_frame, points3d,
            centroid_merge_dist=args.centroid_merge_dist,
        )
        fruit_count = len(set(landmark_of.values()))
        print(f"  {fruit_count} fruits after centroid re-merge", file=sys.stderr)

    # ---- Relabel by first-appearance order ----

    first_frame: dict[int, str] = {}
    for det_id, lid in landmark_of.items():
        fname = det_frame.get(det_id, det_id.split(":")[0])
        if lid not in first_frame or fname < first_frame[lid]:
            first_frame[lid] = fname
    ordered = sorted(set(landmark_of.values()), key=lambda l: first_frame[l])
    remap = {old: new for new, old in enumerate(ordered)}
    for det_id in landmark_of:
        landmark_of[det_id] = remap[landmark_of[det_id]]
    fruit_count = len(ordered)

    print(f"  Total {fruit_count} fruits ({len(landmark_of)} detections assigned)", file=sys.stderr)

    # ---- Build baked results ----

    print("Computing camera positions...", file=sys.stderr)
    frames, skipped = load_frames(workspace, video_id)
    camera_positions = {}
    for fname, frame in frames.items():
        cam_world = (-frame.R.T @ frame.t).tolist()
        camera_positions[fname] = [float(v) for v in cam_world]

    # Build sparse 3D patches for visualization (det_id -> (N,3) world points)
    print("Building sparse 3D patches for visualization...", file=sys.stderr)
    patches: dict[str, np.ndarray] = {}
    for det_id, pids in det_points.items():
        pts = np.array([points3d[p] for p in pids if p in points3d], dtype=np.float32)
        if len(pts) > 0:
            patches[det_id] = pts

    masks_for_bake = partial["masks"] if partial else masks

    serializable_detections = {}
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
            "min_shared_points": args.min_shared_points,
            "centroid_merge_dist": args.centroid_merge_dist,
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
