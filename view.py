#!/usr/bin/env python3
"""
view.py — Mode 2: 2D diagnostic overlay viewer.

Usage:
    python view.py <id> [--workspace sources]

Overlays YOLO segmentation masks onto undistorted frames with consistent
per-fruit landmark IDs and colors. Displays in a PyQt window if available,
falls back to CLI image export otherwise.

GUI controls:
    Left/Right arrows   previous / next frame
    Number keys         jump to frame (multi-digit: type then press Enter)
    O                   export all frames as annotated PNGs
    Escape / Q          quit
"""

import argparse
import os
import sys

import cv2
import numpy as np

from src.results import load_baked, results_path


def _make_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(80, 220, size=(n, 3)).astype(np.uint8)


def _mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())


def _annotate_frame(img: np.ndarray, frame_dets: list[dict], masks: dict,
                    landmark_of: dict, palette: np.ndarray) -> np.ndarray:
    out = img.copy()
    overlay = out.copy()
    h, w = img.shape[:2]

    for det in frame_dets:
        det_id = det["det_id"]
        lid = landmark_of.get(det_id)
        mask = masks.get(det_id)
        if lid is None or mask is None:
            continue

        color = palette[lid % len(palette)]
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))

        mh, mw = mask.shape
        if mh > h:
            mask = mask[:h, :]
        if mw > w:
            mask = mask[:, :w]

        overlay[mask] = color_bgr
        cx, cy = _mask_centroid(mask)
        cv2.putText(out, str(lid), (cx - 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    out = cv2.addWeighted(out, 0.45, overlay, 0.55, 0)
    return out


def _draw_info_bar(img: np.ndarray, current_idx: int, n_frames: int,
                   fname: str, n_dets: int, total_fruits: int) -> np.ndarray:
    h, w = img.shape[:2]
    bar_h = 40
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)

    text = f"[{current_idx + 1}/{n_frames}]  {fname}  |  detections: {n_dets}  |  total fruits: {total_fruits}"
    cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([bar, img])


def _render_frame(baked, all_frame_dets, palette, image_dir, idx):
    frame_order = baked.frame_order
    fname = frame_order[idx]
    img_path = os.path.join(image_dir, fname)
    if not os.path.exists(img_path):
        return None, fname, 0

    img = cv2.imread(img_path)
    if img is None:
        return None, fname, 0

    dets = all_frame_dets.get(fname, [])
    img = _annotate_frame(img, dets, baked.masks, baked.landmark_of, palette)
    img = _draw_info_bar(img, idx, len(frame_order), fname, len(dets), baked.landmark_count)
    return img, fname, len(dets)


# ---- PyQt5 GUI mode ----

def _run_gui(baked, all_frame_dets, palette, image_dir, workspace):
    from PyQt5 import QtWidgets, QtGui, QtCore

    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ["QT_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(
        QtCore.QLibraryInfo.PluginsPath
    )

    app = QtWidgets.QApplication(sys.argv)
    frame_order = baked.frame_order

    window = QtWidgets.QLabel()
    window.setWindowTitle(f"Cacao Fruit Viewer — {baked.video_id} ({baked.landmark_count} fruits)")
    window.setAlignment(QtCore.Qt.AlignCenter)
    window.setMinimumSize(600, 400)

    current_idx = 0
    digit_buffer = ""

    def show_frame(idx):
        nonlocal digit_buffer
        img, fname, n_dets = _render_frame(baked, all_frame_dets, palette, image_dir, idx)
        if img is None:
            return
        h, w, c = img.shape
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(qimg)

        screen = app.primaryScreen().availableGeometry()
        max_w = screen.width() * 0.85
        max_h = screen.height() * 0.85
        if pixmap.width() > max_w or pixmap.height() > max_h:
            pixmap = pixmap.scaled(int(max_w), int(max_h),
                                   QtCore.Qt.KeepAspectRatio,
                                   QtCore.Qt.SmoothTransformation)

        window.setPixmap(pixmap)
        window.setWindowTitle(f"[{idx + 1}/{len(frame_order)}] {fname}  "
                              f"detections: {n_dets}  |  total fruits: {baked.landmark_count}")

    def on_key(event: QtGui.QKeyEvent):
        nonlocal current_idx, digit_buffer
        key = event.key()

        if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_N):
            if current_idx < len(frame_order) - 1:
                current_idx += 1
                show_frame(current_idx)
        elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_P):
            if current_idx > 0:
                current_idx -= 1
                show_frame(current_idx)
        elif QtCore.Qt.Key_0 <= key <= QtCore.Qt.Key_9:
            digit_buffer += chr(key)
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if digit_buffer:
                idx = int(digit_buffer) - 1
                digit_buffer = ""
                if 0 <= idx < len(frame_order):
                    current_idx = idx
                    show_frame(current_idx)
        elif key in (QtCore.Qt.Key_O,):
            _export_all(baked, all_frame_dets, palette, image_dir, workspace)
        elif key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            app.quit()

    window.keyPressEvent = on_key
    show_frame(current_idx)
    window.show()
    app.exec_()


# ---- CLI fallback ----

def _export_all(baked, all_frame_dets, palette, image_dir, workspace):
    results_dir = os.path.dirname(results_path(workspace, baked.video_id))
    export_dir = os.path.join(results_dir, f"{baked.video_id}_frames")
    os.makedirs(export_dir, exist_ok=True)
    frame_order = baked.frame_order
    print(f"Exporting {len(frame_order)} annotated frames to {export_dir}/ ...",
          file=sys.stderr)
    for i, fname in enumerate(frame_order):
        img_path = os.path.join(image_dir, fname)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                dets = all_frame_dets.get(fname, [])
                img = _annotate_frame(img, dets, baked.masks, baked.landmark_of, palette)
                img = _draw_info_bar(img, i, len(frame_order), fname, len(dets), baked.landmark_count)
                cv2.imwrite(os.path.join(export_dir, fname), img)
        if (i + 1) % 50 == 0 and i > 0:
            print(f"\r  {i+1}/{len(frame_order)}", end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)
    print(f"  Exported {len(frame_order)} frames", flush=True)


def _run_cli(baked, all_frame_dets, palette, image_dir, workspace):
    frame_order = baked.frame_order
    results_dir = os.path.dirname(results_path(workspace, baked.video_id))
    frame_png = os.path.join(results_dir, f"{baked.video_id}.frame.png")
    os.makedirs(results_dir, exist_ok=True)

    current_idx = 0

    def save_frame(idx):
        img, fname, n_dets = _render_frame(baked, all_frame_dets, palette, image_dir, idx)
        if img is None:
            return
        cv2.imwrite(frame_png, img)
        print(f"\r[{idx + 1}/{len(frame_order)}] {fname}  "
              f"detections: {n_dets}  →  {frame_png}", flush=True)

    save_frame(current_idx)
    print("\nCommands: n/Enter=next  p=prev  <number>=jump  o=export_all  q=quit")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd == "" or cmd in ("n", "next"):
            if current_idx < len(frame_order) - 1:
                current_idx += 1
                save_frame(current_idx)
        elif cmd in ("p", "prev"):
            if current_idx > 0:
                current_idx -= 1
                save_frame(current_idx)
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(frame_order):
                current_idx = idx
                save_frame(current_idx)
            else:
                print(f"  Frame index out of range (1-{len(frame_order)})", flush=True)
        elif cmd in ("o", "export", "export_all"):
            _export_all(baked, all_frame_dets, palette, image_dir, workspace)

    print("Done.", file=sys.stderr)


# ---- Entry point ----

def main():
    parser = argparse.ArgumentParser(description="2D diagnostic overlay viewer for baked fruit landmarks.")
    parser.add_argument("video_id", help="Video identifier (e.g. L0)")
    parser.add_argument("--workspace", default="sources", help="Workspace root directory (default: %(default)s)")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode even if display available")
    args = parser.parse_args()

    if not os.path.exists(results_path(args.workspace, args.video_id)):
        print(f"No baked results for '{args.video_id}'. Run:\n"
              f"  python count.py {args.video_id} --workspace {args.workspace}\n"
              f"before using view.py.", file=sys.stderr)
        sys.exit(1)

    baked = load_baked(args.workspace, args.video_id)
    if baked is None:
        print(f"Failed to load baked results for '{args.video_id}'.", file=sys.stderr)
        sys.exit(1)

    image_dir = os.path.join(args.workspace, args.video_id, "dense", "0", "images")
    palette = _make_palette(max(baked.landmark_count, 1))

    all_frame_dets = {}
    for fname in baked.frame_order:
        all_frame_dets[fname] = baked.detections.get(fname, [])

    if args.cli:
        _run_cli(baked, all_frame_dets, palette, image_dir, args.workspace)
        return

    try:
        import PyQt5
        _run_gui(baked, all_frame_dets, palette, image_dir, args.workspace)
    except ImportError:
        print("PyQt5 not available, falling back to CLI mode.", file=sys.stderr)
        _run_cli(baked, all_frame_dets, palette, image_dir, args.workspace)


if __name__ == "__main__":
    main()
