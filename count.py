#!/usr/bin/env python3
"""
count.py — Mode 1: compute and bake results.

Usage:
    python count.py --workspace WORKSPACE --video-id <id>
                    [--eps 0.02] [--min-overlap-frac 0.15] [--conf 0.4]

Pipeline: YOLO -> dedup -> 2D IoU tracking -> 3D surface back-projection
-> Union-Find landmark merger -> centroid re-merge for occlusion gaps.

Caches YOLO detections and 3D patches after the first run. Subsequent
runs skip detection and back-projection.

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
from src.tracker import build_tracks
from src.landmark_merger import LandmarkMerger
from src.baked_results import (
    BakedResults, save_baked, load_partial, save_partial, partial_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Count unique cacao fruits from video using COLMAP + YOLO + IoU tracking + 3D landmark merging."
    )
    parser.add_argument("--workspace", required=True, help="Workspace root directory")
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g. L0)")
    parser.add_argument("--eps", type=float, default=0.02,
                        help="3D patch overlap distance in meters (default: %(default)s)")
    parser.add_argument("--min-overlap-frac", type=float, default=0.15,
                        help="Min fraction of points that must overlap to merge (default: %(default)s)")
    parser.add_argument("--iou-threshold", type=float, default=0.15,
                        help="Min 2D mask IoU for tracking (default: %(default)s)")
    parser.add_argument("--max-coast-frames", type=int, default=30,
                        help="Frames a track survives occlusion (default: %(default)s)")
    parser.add_argument("--motion-max-dist", type=float, default=60.0,
                        help="Max pixel distance for Kalman motion fallback match (default: %(default)s)")
    parser.add_argument("--centroid-merge-dist", type=float, default=0.5,
                        help="Centroid distance to re-merge broken tracks (default: %(default)s)")
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

    # ---- Load or run YOLO + back-projection ----

    partial = None if args.no_cache else load_partial(workspace, video_id)

    if partial is not None:
        print(f"Using cached detections ({partial_path(workspace, video_id)})", file=sys.stderr)
        patches = partial["patches"]
        masks = partial["masks"]
        all_detections = partial["detections"]
        frame_order = partial["frame_order"]
        skipped = partial["skipped_frames"]
        print(f"  {len(patches)} patches, {len(masks)} masks, {len(frame_order)} frames", file=sys.stderr)
    else:
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

        print(f"Back-projecting...", file=sys.stderr)
        t0 = time.time()
        patches: dict[str, np.ndarray] = {}
        masks_for_bake: dict[str, np.ndarray] = {}
        n_frames = len(frame_order)

        for fi, fname in enumerate(frame_order):
            dets = all_detections.get(fname, [])
            for d in dets:
                if fname not in frames:
                    continue
                masks_for_bake[d["det_id"]] = d["mask"]
                pts = backproject_mask_to_surface(d["mask"], frames[fname])
                if pts is None:
                    continue
                patches[d["det_id"]] = pts
            if (fi + 1) % 50 == 0 or fi == n_frames - 1:
                print(f"\r  {fi + 1}/{n_frames} frames | {len(patches)} patches", end="", flush=True, file=sys.stderr)

        print(file=sys.stderr)
        print(f"  {len(patches)} patches back-projected ({time.time() - t0:.1f}s)", file=sys.stderr)

        masks = masks_for_bake
        save_partial(
            patches, masks, all_detections, frame_order, skipped,
            workspace, video_id,
            {"conf_threshold": args.conf_threshold, "min_area": args.min_area,
             "include_flowers": args.include_flowers},
        )
        print(f"  Cached to {partial_path(workspace, video_id)}", file=sys.stderr)

    # ---- Build tracker input (masks + bboxes) ----

    masks_all = partial["masks"] if partial else masks

    detections_for_tracker: dict[str, list[dict]] = {}
    det_id_to_frame: dict[str, str] = {}
    for fname in frame_order:
        dets = all_detections.get(fname, [])
        tdet = []
        for d in dets:
            m = masks_all.get(d["det_id"])
            if m is None:
                continue
            tdet.append({
                "frame_name": d["frame_name"],
                "mask": m,
                "bbox": d["bbox"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "det_id": d["det_id"],
            })
            det_id_to_frame[d["det_id"]] = fname
        if tdet:
            detections_for_tracker[fname] = tdet

    # ---- 2D IoU tracking ----

    print(f"Building 2D tracks (iou={args.iou_threshold}, coast={args.max_coast_frames})...", file=sys.stderr)
    t0 = time.time()
    tracks = build_tracks(
        detections_for_tracker,
        iou_threshold=args.iou_threshold,
        max_coast_frames=args.max_coast_frames,
        motion_max_dist=args.motion_max_dist,
    )
    print(f"  {len(tracks)} tracks ({time.time() - t0:.1f}s)", file=sys.stderr)

    # ---- Back-project pooled patches per track, then merge landmarks ----

    print(f"Merging landmarks (eps={args.eps}, min_overlap_frac={args.min_overlap_frac})...", file=sys.stderr)
    t0 = time.time()
    merger = LandmarkMerger(eps=args.eps, min_overlap_frac=args.min_overlap_frac)
    rng = np.random.default_rng(42)
    landmark_of: dict[str, int] = {}

    n_tracks = len(tracks)
    for ti, track in enumerate(tracks):
        track_patches = []
        for det_id in track.det_ids:
            if det_id in patches:
                track_patches.append(patches[det_id])

        if track_patches:
            combined = np.vstack(track_patches)
            if len(combined) > 500:
                combined = combined[rng.choice(len(combined), size=500, replace=False)]
            track_frames = {det_id_to_frame[did] for did in track.det_ids if did in det_id_to_frame}
            patch_idx = merger.add(combined, frames=track_frames)
            landmark_id = merger.landmark_of(patch_idx)
            for det_id in track.det_ids:
                landmark_of[det_id] = landmark_id

        if (ti + 1) % 50 == 0 or ti == n_tracks - 1:
            print(f"\r  track {ti+1}/{n_tracks} | {merger.landmark_count()} landmarks", end="", flush=True, file=sys.stderr)

    print(file=sys.stderr)
    fruit_count = merger.landmark_count()
    print(f"  {fruit_count} phase-1 landmarks ({time.time() - t0:.1f}s)", file=sys.stderr)

    # ---- Phase 2: centroid re-merge for broken tracks (long gaps) ----

    landmark_roots = set(merger.landmark_of(i) for i in range(len(merger.patches)))
    lid_centroid = {}
    lid_frames = {}
    for root in landmark_roots:
        pts_list, fset = [], set()
        for i in range(len(merger.patches)):
            if merger.landmark_of(i) == root:
                pts_list.append(merger.patches[i])
                fset |= merger.frame_sets[i]
        if pts_list:
            lid_centroid[root] = np.median(np.vstack(pts_list), axis=0)
            lid_frames[root] = fset

    pairs = []
    roots = list(landmark_roots)
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            dist = np.linalg.norm(lid_centroid[roots[i]] - lid_centroid[roots[j]])
            pairs.append((dist, i, j))
    pairs.sort()

    remap = {}
    for dist, i, j in pairs:
        if dist > args.centroid_merge_dist:
            break
        ra, rb = roots[i], roots[j]
        while ra in remap:
            ra = remap[ra]
        while rb in remap:
            rb = remap[rb]
        if ra == rb:
            continue
        if lid_frames[ra] & lid_frames[rb]:
            continue
        remap[rb] = ra
        lid_frames[ra] |= lid_frames[rb]

    if remap:
        for det_id in landmark_of:
            lid = landmark_of[det_id]
            while lid in remap:
                lid = remap[lid]
            landmark_of[det_id] = lid
        fruit_count = len(set(landmark_of.values()))
        print(f"  Merged {len(remap)} broken tracks, final: {fruit_count} fruits", file=sys.stderr)
    else:
        print(f"  No centroid merges, final: {fruit_count} fruits", file=sys.stderr)

    # ---- Assign remaining detections (in tracks with no 3D data) ----

    assigned_dets = set(landmark_of.keys())
    max_lid = max(landmark_of.values()) if landmark_of else -1
    for track in tracks:
        if any(did in patches for did in track.det_ids):
            continue  # already assigned via 3D
        max_lid += 1
        for det_id in track.det_ids:
            if det_id not in assigned_dets:
                landmark_of[det_id] = max_lid
                assigned_dets.add(det_id)
    fruit_count = max_lid + 1 if assigned_dets else fruit_count

    # Relabel by first-appearance order (fruit 0 = first seen, etc.)
    first_frame: dict[int, str] = {}
    for det_id, lid in landmark_of.items():
        fname = det_id_to_frame.get(det_id, det_id.split(":")[0])
        if lid not in first_frame or fname < first_frame[lid]:
            first_frame[lid] = fname
    ordered = sorted(set(landmark_of.values()), key=lambda l: first_frame[l])
    remap2 = {old: new for new, old in enumerate(ordered)}
    for det_id in landmark_of:
        landmark_of[det_id] = remap2[landmark_of[det_id]]
    fruit_count = len(ordered)

    print(f"  Total {fruit_count} fruits ({len(assigned_dets)} detections assigned)", file=sys.stderr)

    # ---- Build baked results ----

    print("Computing camera positions...", file=sys.stderr)
    frames, skipped = load_frames(workspace, video_id)
    camera_positions = {}
    for fname, frame in frames.items():
        cam_world = (-frame.R.T @ frame.t).tolist()
        camera_positions[fname] = [float(v) for v in cam_world]

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
            "eps": args.eps,
            "min_overlap_frac": args.min_overlap_frac,
            "iou_threshold": args.iou_threshold,
            "max_coast_frames": args.max_coast_frames,
            "motion_max_dist": args.motion_max_dist,
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
