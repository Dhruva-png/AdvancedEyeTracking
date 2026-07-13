"""Per-frame face/eye analysis backed by MediaPipe Face Mesh."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from . import landmarks as lm
from . import metrics
from .config import TrackerConfig


@dataclass
class FrameResult:
    face_found: bool
    frame: np.ndarray
    left_center: tuple[int, int] | None = None
    right_center: tuple[int, int] | None = None
    gaze: str = "UNKNOWN"
    blink: bool = False
    x_norm: float | None = None
    y_norm: float | None = None


class EyeTracker:
    """Wraps MediaPipe FaceMesh and turns raw landmarks into gaze/blink signals.

    MediaPipe's iris-refined mesh is used instead of Haar-cascade eye
    detection: it's more accurate, more stable across head pose, and gives
    sub-pixel iris landmarks that Haar cascades simply don't provide.
    """

    def __init__(self, config: TrackerConfig):
        self.config = config
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._smooth_left: tuple[int, int] | None = None
        self._smooth_right: tuple[int, int] | None = None
        self._last_blink_time: float = 0.0
        self.blink_count: int = 0

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> "EyeTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def process(self, frame: np.ndarray) -> FrameResult:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return FrameResult(face_found=False, frame=frame)

        mesh = result.multi_face_landmarks[0].landmark

        left_eye_pts = np.array([(int(mesh[i].x * w), int(mesh[i].y * h)) for i in lm.LEFT_EYE])
        right_eye_pts = np.array([(int(mesh[i].x * w), int(mesh[i].y * h)) for i in lm.RIGHT_EYE])
        left_box = cv2.boundingRect(left_eye_pts)
        right_box = cv2.boundingRect(right_eye_pts)

        left_center = metrics.iris_center(mesh, lm.LEFT_IRIS, w, h)
        right_center = metrics.iris_center(mesh, lm.RIGHT_IRIS, w, h)

        alpha = self.config.smoothing_alpha
        self._smooth_left = metrics.smooth_point(left_center, self._smooth_left, alpha)
        self._smooth_right = metrics.smooth_point(right_center, self._smooth_right, alpha)

        left_ear = metrics.eye_aspect_ratio(left_eye_pts)
        right_ear = metrics.eye_aspect_ratio(right_eye_pts)
        blink = self._detect_blink(left_ear, right_ear)

        gaze = metrics.gaze_direction(self._smooth_left, left_box)

        avg_x = (self._smooth_left[0] + self._smooth_right[0]) / 2.0
        avg_y = (self._smooth_left[1] + self._smooth_right[1]) / 2.0
        x_norm = float(np.clip(avg_x / w, 0.0, 1.0))
        y_norm = float(np.clip(avg_y / h, 0.0, 1.0))

        self._draw_overlay(frame, left_box, right_box, gaze)

        return FrameResult(
            face_found=True,
            frame=frame,
            left_center=self._smooth_left,
            right_center=self._smooth_right,
            gaze=gaze,
            blink=blink,
            x_norm=x_norm,
            y_norm=y_norm,
        )

    def _detect_blink(self, left_ear: float, right_ear: float) -> bool:
        threshold = self.config.blink_ear_threshold
        below_threshold = left_ear < threshold and right_ear < threshold
        now = time.time()
        if below_threshold and (now - self._last_blink_time) > self.config.blink_refractory_sec:
            self._last_blink_time = now
            self.blink_count += 1
            return True
        return False

    def _draw_overlay(self, frame, left_box, right_box, gaze: str) -> None:
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        cv2.circle(frame, self._smooth_left, 4, (0, 0, 255), -1)
        cv2.circle(frame, self._smooth_right, 4, (0, 0, 255), -1)
        cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (255, 255, 255), 1)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)
        cv2.putText(frame, f"Gaze: {gaze}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
