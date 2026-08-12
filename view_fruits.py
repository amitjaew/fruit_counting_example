#!/usr/bin/env python3
"""
view_fruits.py — Per-fruit 3D diagnostic viewer.

Usage:
    python view_fruits.py --workspace WORKSPACE --video-id <id>

Displays a 3D scatter plot of all surface patches for one fruit at a time.
Camera positions for frames where the fruit was detected are shown as red markers.

GUI controls:
    Left/Right arrows   previous / next fruit
    Number + Enter      jump to fruit
    Escape / Q          quit

CLI mode (no display available):
    n / Enter   next fruit
    p           previous fruit
    <number>    jump to fruit
    q           quit
    Saves each fruit to results/{id}.fruit_{N}.png
"""

import os
import sys
import importlib

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def _try_backends():
    for backend, imports in (
        ("Qt5Agg", ("PyQt5",)),
        ("QtAgg", ("PyQt6", "PySide6")),
        ("TkAgg", ("tkinter",)),
        ("GTK3Agg", ("gi",)),
    ):
        try:
            for mod in imports:
                importlib.import_module(mod)
        except ImportError:
            continue
        return backend
    return None


_backend = _try_backends()
if _backend:
    import matplotlib
    matplotlib.use(_backend, force=True)
    if "Qt" in _backend:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        from PyQt5 import QtCore
        os.environ["QT_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(
            QtCore.QLibraryInfo.PluginsPath
        )
else:
    import matplotlib
    matplotlib.use("Agg")

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.baked_results import load_baked, results_path


def _make_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(80, 220, size=(n, 3)).astype(np.uint8)


def _render_fruit(ax, baked, fruit_id, palette):
    ax.clear()
    color = palette[fruit_id % len(palette)] / 255.0

    dets = [did for did, lid in baked.landmark_of.items() if lid == fruit_id]
    all_pts = []
    frames = set()
    for det_id in dets:
        pts = baked.patches.get(det_id)
        if pts is not None and len(pts) > 0:
            all_pts.append(pts)
        frames.add(det_id.split(":")[0])

    if all_pts:
        combined = np.vstack(all_pts)
        if len(combined) > 1000:
            combined = combined[np.random.default_rng(fruit_id).choice(len(combined), 1000)]
        ax.scatter(combined[:, 0], combined[:, 1], combined[:, 2],
                   c=[color], s=3, alpha=0.6, edgecolors="none")

    for fname in sorted(frames):
        cp = baked.camera_positions.get(fname)
        if cp:
            ax.scatter([cp[0]], [cp[1]], [cp[2]], c="red", s=10, alpha=0.4)

    flist = sorted(frames)
    span = f"{flist[0]}..{flist[-1]}" if flist else "none"

    ax.set_title(f"Fruit {fruit_id}/{baked.landmark_count - 1}  "
                 f"|  {len(dets)} dets  |  {len(frames)} frames  |  {span}",
                 fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect((1, 1, 1))


def _run_gui(baked, palette):
    from PyQt5 import QtWidgets, QtGui, QtCore

    app = QtWidgets.QApplication(sys.argv)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    current_idx = 0
    digit_buffer = ""
    n_fruits = baked.landmark_count

    window = QtWidgets.QLabel()
    window.setWindowTitle(f"Fruit Viewer — {baked.video_id} ({n_fruits} fruits)")
    window.setAlignment(QtCore.Qt.AlignCenter)
    window.setMinimumSize(600, 400)

    def show_fruit(idx):
        nonlocal digit_buffer
        _render_fruit(ax, baked, idx, palette)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = fig.canvas.buffer_rgba()
        qimg = QtGui.QImage(img, w, h, QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        screen = app.primaryScreen().availableGeometry()
        max_w = screen.width() * 0.85
        max_h = screen.height() * 0.85
        if pixmap.width() > max_w or pixmap.height() > max_h:
            pixmap = pixmap.scaled(int(max_w), int(max_h),
                                   QtCore.Qt.KeepAspectRatio,
                                   QtCore.Qt.SmoothTransformation)
        window.setPixmap(pixmap)
        window.setWindowTitle(f"Fruit {idx}/{n_fruits - 1}  |  "
                              f"{baked.landmark_count} fruits  |  {baked.video_id}")

    def on_key(event: QtGui.QKeyEvent):
        nonlocal current_idx, digit_buffer
        key = event.key()
        if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_N):
            if current_idx < n_fruits - 1:
                current_idx += 1
                show_fruit(current_idx)
        elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_P):
            if current_idx > 0:
                current_idx -= 1
                show_fruit(current_idx)
        elif QtCore.Qt.Key_0 <= key <= QtCore.Qt.Key_9:
            digit_buffer += chr(key)
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if digit_buffer:
                idx = int(digit_buffer) - 1
                digit_buffer = ""
                if 0 <= idx < n_fruits:
                    current_idx = idx
                    show_fruit(current_idx)
        elif key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            app.quit()

    window.keyPressEvent = on_key
    show_fruit(current_idx)
    window.show()
    app.exec_()


def _run_cli(baked, palette):
    n_fruits = baked.landmark_count
    results_dir = os.path.dirname(results_path("", baked.video_id)) or "results"
    os.makedirs(results_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    current_idx = 0

    def save_fruit(idx):
        out = os.path.join(results_dir, f"{baked.video_id}.fruit_{idx}.png")
        _render_fruit(ax, baked, idx, palette)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\r[Fruit {idx}/{n_fruits - 1}]  saved → {out}", flush=True)

    save_fruit(current_idx)
    print("\nCommands: n/Enter=next  p=prev  <number>=jump  q=quit")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd == "" or cmd in ("n", "next"):
            if current_idx < n_fruits - 1:
                current_idx += 1
                save_fruit(current_idx)
        elif cmd in ("p", "prev"):
            if current_idx > 0:
                current_idx -= 1
                save_fruit(current_idx)
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < n_fruits:
                current_idx = idx
                save_fruit(current_idx)
            else:
                print(f"  Fruit index out of range (1-{n_fruits})", flush=True)

    print("Done.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Per-fruit 3D diagnostic viewer.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--cli", action="store_true", help="Force CLI mode even if display available")
    args = parser.parse_args()

    if not os.path.exists(results_path(args.workspace, args.video_id)):
        print(f"No baked results for '{args.video_id}'. Run:\n"
              f"  python count.py --workspace {args.workspace} --video-id {args.video_id}\n"
              f"before using view_fruits.py.", file=sys.stderr)
        sys.exit(1)

    baked = load_baked(args.workspace, args.video_id)
    if baked is None:
        print(f"Failed to load baked results for '{args.video_id}'.", file=sys.stderr)
        sys.exit(1)

    palette = _make_palette(max(baked.landmark_count, 1))

    if args.cli:
        _run_cli(baked, palette)
        return

    if _backend is not None:
        _run_gui(baked, palette)
    else:
        _run_cli(baked, palette)


if __name__ == "__main__":
    main()
