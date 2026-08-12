"""
tracker.py

IoU-based 2D tracking across consecutive frames using Kalman-filtered
centroid motion model. Associates YOLO detections of the same fruit
across frames to build per-fruit tracks for subsequent 3D back-projection.

Algorithm:
  - For each frame, predict existing track positions with a Kalman filter.
  - Match detections to tracks via mask IoU.
  - Unmatched tracks: coast for N frames, then terminate.
  - Unmatched detections: spawn new tracks.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    det_ids: list[str] = field(default_factory=list)
    centroids: list[np.ndarray] = field(default_factory=list)
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    frames_since_match: int = 0
    kf_state: np.ndarray | None = None
    kf_cov: np.ndarray | None = None

    def predict_centroid(self):
        if self.kf_state is None:
            if self.centroids:
                return self.centroids[-1]
            return None
        return self.kf_state[:2].copy()

    def predict_step(self):
        if self.kf_state is None:
            return
        F = np.array([
            [1, 0, 1.0, 0],
            [0, 1, 0, 1.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        Q = np.eye(4) * 0.01
        self.kf_state = F @ self.kf_state
        self.kf_cov = F @ self.kf_cov @ F.T + Q

    def update_kalman(self, centroid: np.ndarray):
        dt = 1.0
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        Q = np.eye(4) * 0.01
        R = np.eye(2) * 5.0

        if self.kf_state is None:
            self.kf_state = np.array([centroid[0], centroid[1], 0.0, 0.0])
            self.kf_cov = np.eye(4) * 100.0
        else:
            self.kf_state = F @ self.kf_state
            self.kf_cov = F @ self.kf_cov @ F.T + Q

        z = centroid[:2]
        y = z - H @ self.kf_state
        S = H @ self.kf_cov @ H.T + R
        K = self.kf_cov @ H.T @ np.linalg.inv(S)
        self.kf_state = self.kf_state + K @ y
        self.kf_cov = (np.eye(4) - K @ H) @ self.kf_cov


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        h = max(mask_a.shape[0], mask_b.shape[0])
        w = max(mask_a.shape[1], mask_b.shape[1])
        padded_a = np.zeros((h, w), dtype=bool)
        padded_b = np.zeros((h, w), dtype=bool)
        padded_a[:mask_a.shape[0], :mask_a.shape[1]] = mask_a
        padded_b[:mask_b.shape[0], :mask_b.shape[1]] = mask_b
        mask_a, mask_b = padded_a, padded_b
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return inter / union


def _compute_cost(tracks: list[Track], detections: list[dict], frame_shape: tuple) -> np.ndarray:
    n_tracks = len(tracks)
    n_dets = len(detections)
    cost = np.ones((n_tracks, n_dets))

    for i, track in enumerate(tracks):
        if not track.masks:
            continue
        last_mask = track.masks[-1]
        coast_penalty = min(track.frames_since_match, 30) * 0.05
        for j, det in enumerate(detections):
            iou = _mask_iou(last_mask, det["mask"])
            cost[i, j] = 1.0 - iou + coast_penalty

    return cost


def _match_track_to_det(track: Track, det: dict):
    cx = (det["bbox"][0] + det["bbox"][2]) / 2
    cy = (det["bbox"][1] + det["bbox"][3]) / 2
    centroid = np.array([cx, cy])
    track.det_ids.append(det["det_id"])
    track.centroids.append(centroid)
    track.bboxes.append(det["bbox"])
    track.masks.append(det["mask"])
    track.update_kalman(centroid)
    track.frames_since_match = 0


def build_tracks(
    detections: dict[str, list[dict]],
    iou_threshold: float = 0.15,
    max_coast_frames: int = 30,
    motion_max_dist: float = 60.0,
) -> list[Track]:
    active_tracks: list[Track] = []
    finished_tracks: list[Track] = []
    next_track_id = 0

    frame_names = sorted(detections.keys())

    for fname in frame_names:
        frame_dets = detections.get(fname, [])
        matched_track_ids = set()
        matched_det_ids = set()

        # ---- Stage 1: IoU matching ----
        if active_tracks and frame_dets:
            cost = _compute_cost(active_tracks, frame_dets, (0, 0))
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                iou = _mask_iou(active_tracks[i].masks[-1], frame_dets[j]["mask"])
                if iou >= iou_threshold:
                    _match_track_to_det(active_tracks[i], frame_dets[j])
                    matched_track_ids.add(i)
                    matched_det_ids.add(j)

        # ---- Stage 2: motion matching (Kalman prediction) for fast movers ----
        remaining_dets = [j for j in range(len(frame_dets)) if j not in matched_det_ids]
        for i in range(len(active_tracks)):
            if i in matched_track_ids:
                continue
            track = active_tracks[i]
            predicted = track.predict_centroid()
            if predicted is None:
                continue
            best_j = None
            best_dist = motion_max_dist
            for j in remaining_dets:
                det = frame_dets[j]
                cx = (det["bbox"][0] + det["bbox"][2]) / 2
                cy = (det["bbox"][1] + det["bbox"][3]) / 2
                dist = float(np.hypot(predicted[0] - cx, predicted[1] - cy))
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is not None:
                _match_track_to_det(track, frame_dets[best_j])
                matched_track_ids.add(i)
                matched_det_ids.add(best_j)
                remaining_dets.remove(best_j)

        for i, track in enumerate(active_tracks):
            if i not in matched_track_ids:
                track.frames_since_match += 1
                track.predict_step()

        still_active = []
        for track in active_tracks:
            if track.frames_since_match > max_coast_frames:
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

        for j, det in enumerate(frame_dets):
            if j in matched_det_ids:
                continue
            track = Track(track_id=next_track_id)
            next_track_id += 1
            _match_track_to_det(track, det)
            active_tracks.append(track)

    finished_tracks.extend(active_tracks)
    return finished_tracks
