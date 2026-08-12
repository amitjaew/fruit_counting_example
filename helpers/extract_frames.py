#!/usr/bin/env python3
import argparse
import os
import sys

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Extract all frames from an MP4 video into a directory of PNG images."
    )
    parser.add_argument(
        "video_path",
        help="Path to the input MP4 video (e.g., sources/L0.mp4)",
    )
    parser.add_argument(
        "-o", "--out-dir",
        default=None,
        help="Output directory (default: sources/{video_name}/)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: video file not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.video_path) or "sources",
        video_name,
    )

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting {total} frames from {args.video_path} -> {out_dir}/")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        out_path = os.path.join(out_dir, f"{idx:05d}.png")
        cv2.imwrite(out_path, frame)
        print(f"\r  Frame {idx}/{total}", end="", flush=True)

    cap.release()
    print(f"\nDone. {idx} frames saved to {out_dir}/")


if __name__ == "__main__":
    main()
