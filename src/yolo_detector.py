"""
yolo_detector.py

Runs YOLO26 instance segmentation on undistorted frames and produces
per-frame Detection objects consumed by the tracker and back-projection
pipeline.
"""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    frame_name: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    det_id: str


FRUIT_CLASSES = {"cac-sm", "cac-m", "cac-l", "cac-y"}
FLOWER_CLASS = "cac-flor"


def _make_det_id(frame_name: str, idx: int) -> str:
    return f"{frame_name}:{idx}"


def _rasterize_polygon(contour, h: int, w: int) -> np.ndarray:
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def run_detection(
    model: YOLO,
    image_path: str,
    frame_name: str,
    conf_threshold: float = 0.4,
    min_area: int = 100,
    include_flowers: bool = False,
) -> list[Detection]:
    results = model(image_path, verbose=False)
    detections = []

    for idx, r in enumerate(results):
        if r.masks is None or r.boxes is None:
            continue

        h, w = r.orig_shape[:2]
        for j, (mask_data, box_data) in enumerate(zip(r.masks, r.boxes)):
            cls_id = int(box_data.cls[0].item())
            class_name = model.names.get(cls_id, str(cls_id))
            confidence = float(box_data.conf[0].item())

            if not include_flowers and class_name == FLOWER_CLASS:
                continue
            if class_name not in FRUIT_CLASSES and class_name != FLOWER_CLASS:
                continue

            xyxy = box_data.xyxy[0].cpu().numpy()
            bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

            if mask_data.xy is not None and len(mask_data.xy) > 0:
                mask = _rasterize_polygon(mask_data.xy[0], h, w)
            else:
                continue

            if mask.sum() < min_area:
                continue

            det_id = _make_det_id(frame_name, idx)
            idx += 1
            detections.append(Detection(
                frame_name=frame_name,
                mask=mask,
                bbox=bbox,
                class_name=class_name,
                confidence=confidence,
                det_id=det_id,
            ))

    return detections


def detect_all_frames(
    model_path: str,
    image_dir: str,
    conf_threshold: float = 0.4,
    min_area: int = 100,
    include_flowers: bool = False,
) -> dict[str, list[Detection]]:
    import os
    import sys

    model = YOLO(model_path)
    detections: dict[str, list[Detection]] = {}

    all_files = sorted([
        f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    total = len(all_files)

    for i, fname in enumerate(all_files):
        img_path = os.path.join(image_dir, fname)
        frame_dets = run_detection(
            model, img_path, fname,
            conf_threshold=conf_threshold,
            min_area=min_area,
            include_flowers=include_flowers,
        )
        if frame_dets:
            detections[fname] = frame_dets

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"\r  YOLO: {i+1}/{total} frames ({len(detections)} with detections)", end="", flush=True, file=sys.stderr)

    print(file=sys.stderr)
    return detections
