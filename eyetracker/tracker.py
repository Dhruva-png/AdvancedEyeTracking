"""Per-frame face/eye analysis backed by MediaPipe Face Mesh.

Pure signal extraction only — no drawing. Presentation lives in `hud.py` and
`gaze_view.py` so tracking logic can be reasoned about (and tested) without a
rendering pipeline attached.
"""

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
    left_box: tuple[int, int, int, int] | None = None
    right_box: tuple[int, int, int, int] | None = None
    left_eye_pts: np.ndarray | None = None
    right_eye_pts: np.ndarray | None = None
    gaze: str = "UNKNOWN"
    raw_gaze_offset: tuple[float, float] | None = None
    blink: bool = False
    left_ear: float = 0.0
    right_ear: float = 0.0
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
        self._smooth_left_box: tuple[float, float, float, float] | None = None
        self._smooth_right_box: tuple[float, float, float, float] | None = None
        self._last_valid_gaze_offset: tuple[float, float] | None = None
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

        # eye_offset divides the iris position by the box's own width/height,
        # so per-frame jitter in the box (a few noisy landmarks each frame)
        # gets amplified into a much larger jump in the ratio than the same
        # jitter would cause in a plain pixel coordinate. Smoothing the box
        # itself (not just the iris center) removes most of that.
        box_alpha = self.config.eye_box_smoothing_alpha
        self._smooth_left_box = metrics.smooth_values(left_box, self._smooth_left_box, box_alpha)
        self._smooth_right_box = metrics.smooth_values(right_box, self._smooth_right_box, box_alpha)

        left_offset = metrics.eye_offset(self._smooth_left, self._smooth_left_box)
        right_offset = metrics.eye_offset(self._smooth_right, self._smooth_right_box)
        candidate_offset = _average_offsets(left_offset, right_offset)

        # A half-closed eye (mid-blink, squinting) gives a geometrically
        # meaningless iris-in-box ratio. Rather than feed that noise into the
        # gaze signal, freeze it at the last trustworthy reading.
        eyes_open_enough = left_ear >= self.config.gaze_valid_ear_threshold and right_ear >= self.config.gaze_valid_ear_threshold
        if candidate_offset is not None and eyes_open_enough:
            self._last_valid_gaze_offset = candidate_offset
        raw_gaze_offset = self._last_valid_gaze_offset

        gaze = metrics.classify_gaze_direction(raw_gaze_offset)

        avg_x = (self._smooth_left[0] + self._smooth_right[0]) / 2.0
        avg_y = (self._smooth_left[1] + self._smooth_right[1]) / 2.0
        x_norm = float(np.clip(avg_x / w, 0.0, 1.0))
        y_norm = float(np.clip(avg_y / h, 0.0, 1.0))

        return FrameResult(
            face_found=True,
            frame=frame,
            left_center=self._smooth_left,
            right_center=self._smooth_right,
            left_box=left_box,
            right_box=right_box,
            left_eye_pts=left_eye_pts,
            right_eye_pts=right_eye_pts,
            gaze=gaze,
            raw_gaze_offset=raw_gaze_offset,
            blink=blink,
            left_ear=left_ear,
            right_ear=right_ear,
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


def _average_offsets(
    left: tuple[float, float] | None, right: tuple[float, float] | None
) -> tuple[float, float] | None:
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0
