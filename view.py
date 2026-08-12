#!/usr/bin/env python3
"""
view.py — Mode 2: interactive 3D debug viewer.

Usage:
    python view.py --workspace WORKSPACE --video-id <id>

Requires baked results from count.py to exist.
Left/Right arrows: step through frames.
Escape: exit.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from mpl_toolkits.mplot3d import Axes3D

from src.baked_results import load_baked, results_path
from src.surface_backprojection import robust_centroid


def _make_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(0.3, 0.9, size=(n, 3))


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D viewer for baked fruit landmarks.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--export-camera-path", type=str, default=None, help="Export static camera path PNG")
    args = parser.parse_args()

    if not os.path.exists(results_path(args.workspace, args.video_id)):
        print(f"No baked results for '{args.video_id}'. Run:\n"
              f"  python count.py --workspace {args.workspace} --video-id {args.video_id}\n"
              f"before using view.py.", file=sys.stderr)
        sys.exit(1)

    baked = load_baked(args.workspace, args.video_id)
    if baked is None:
        print(f"Failed to load baked results for '{args.video_id}'.", file=sys.stderr)
        sys.exit(1)

    landmark_colors = _make_palette(max(baked.landmark_count, 1))
    centroid_cache: dict[int, np.ndarray] = {}
    landmark_patches: dict[int, list[np.ndarray]] = {}

    for det_id, landmark_id in baked.landmark_of.items():
        landmark_patches.setdefault(landmark_id, []).append(baked.patches.get(det_id, np.empty((0, 3))))

    for lid, pt_list in landmark_patches.items():
        combined = np.vstack([p for p in pt_list if len(p) > 0]) if pt_list else np.empty((0, 3))
        if len(combined) > 0:
            centroid_cache[lid] = robust_centroid(combined)

    frame_order = baked.frame_order

    fig = plt.figure(figsize=(12, 9))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    current_idx = 0
    view_elev = 20
    view_azim = -60

    def get_frame_det_ids(fname: str) -> set[int]:
        ids = set()
        for det_id, lid in baked.landmark_of.items():
            if det_id.startswith(fname + ":"):
                ids.add(lid)
        return ids

    def render():
        nonlocal view_elev, view_azim
        ax.clear()
        fname = frame_order[current_idx]
        active_landmarks = get_frame_det_ids(fname)

        all_pts_by_landmark: dict[int, np.ndarray] = {}
        for det_id, lid in baked.landmark_of.items():
            pts = baked.patches.get(det_id)
            if pts is not None and len(pts) > 0:
                all_pts_by_landmark.setdefault(lid, []).append(pts)

        for lid, pt_lists in all_pts_by_landmark.items():
            combined = np.vstack(pt_lists)
            if len(combined) == 0:
                continue
            if len(combined) > 500:
                idx_sample = np.random.default_rng(lid).choice(len(combined), size=500, replace=False)
                combined = combined[idx_sample]

            color = landmark_colors[lid % len(landmark_colors)]
            is_active = lid in active_landmarks
            alpha = 0.6 if is_active else 0.12
            size = 4 if is_active else 1.5

            ax.scatter(combined[:, 0], combined[:, 1], combined[:, 2],
                       c=[color], s=size, alpha=alpha, marker="o", edgecolors="none")

            if is_active and lid in centroid_cache:
                c = centroid_cache[lid]
                ax.text(c[0], c[1], c[2], str(lid), fontsize=7, color="black")

        if fname in baked.camera_positions:
            cp = baked.camera_positions[fname]
            ax.scatter([cp[0]], [cp[1]], [cp[2]], c="red", s=80, marker="^", label=f"Camera {fname}")

        n_dets = len(get_frame_det_ids(fname))
        ax.set_title(f"{fname}  [{current_idx + 1}/{len(frame_order)}]  "
                     f"detections: {n_dets}  total fruits: {baked.landmark_count}",
                     fontsize=11)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        if hasattr(render, "_initial_view"):
            ax.view_init(elev=view_elev, azim=view_azim)
        else:
            ax.view_init(elev=20, azim=-60)
            render._initial_view = True

        fig.canvas.draw_idle()

    def on_key(event: KeyEvent):
        nonlocal current_idx, view_elev, view_azim
        if event.key == "right":
            current_idx = min(current_idx + 1, len(frame_order) - 1)
            view_elev = ax.elev
            view_azim = ax.azim
            render()
        elif event.key == "left":
            current_idx = max(current_idx - 1, 0)
            view_elev = ax.elev
            view_azim = ax.azim
            render()
        elif event.key == "escape":
            plt.close(fig)

    render()

    if args.export_camera_path:
        ax.clear()
        all_cam_positions = []
        for fname in frame_order:
            if fname in baked.camera_positions:
                all_cam_positions.append(baked.camera_positions[fname])
        if all_cam_positions:
            all_cam_positions_arr = np.array(all_cam_positions)
            ax.scatter(all_cam_positions_arr[:, 0], all_cam_positions_arr[:, 1],
                       all_cam_positions_arr[:, 2], c="red", s=10, alpha=0.7)
            ax.plot(all_cam_positions_arr[:, 0], all_cam_positions_arr[:, 1],
                    all_cam_positions_arr[:, 2], "gray", alpha=0.3, linewidth=0.5)
            ax.set_title(f"Camera path — {baked.video_id} ({len(all_cam_positions)} views)")
            fig.savefig(args.export_camera_path, dpi=150, bbox_inches="tight")
            print(f"Camera path exported to {args.export_camera_path}", file=sys.stderr)

    if not args.export_camera_path:
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()


if __name__ == "__main__":
    main()
