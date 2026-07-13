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
    raw_gaze_offset: tuple[float, float] | None = None  # head-invariant, EMA-smoothed, blink-frozen
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
        self._smooth_gaze: tuple[float, float] | None = None
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

        # Sub-pixel iris centers (float) drive the gaze feature; the int
        # versions are only for drawing.
        left_iris_f = metrics.iris_center_f(mesh, lm.LEFT_IRIS, w, h)
        right_iris_f = metrics.iris_center_f(mesh, lm.RIGHT_IRIS, w, h)

        alpha = self.config.smoothing_alpha
        self._smooth_left = metrics.smooth_point((int(left_iris_f[0]), int(left_iris_f[1])), self._smooth_left, alpha)
        self._smooth_right = metrics.smooth_point((int(right_iris_f[0]), int(right_iris_f[1])), self._smooth_right, alpha)

        left_ear = metrics.eye_aspect_ratio(left_eye_pts)
        right_ear = metrics.eye_aspect_ratio(right_eye_pts)
        blink = self._detect_blink(left_ear, right_ear)

        candidate_offset = self._compute_gaze_feature(mesh, left_iris_f, right_iris_f, w, h)

        # A half-closed eye (mid-blink, squinting) gives a geometrically
        # meaningless iris position. Rather than feed that noise into the gaze
        # signal, freeze it at the last trustworthy reading.
        eyes_open_enough = (
            left_ear >= self.config.gaze_valid_ear_threshold and right_ear >= self.config.gaze_valid_ear_threshold
        )
        if candidate_offset is not None and eyes_open_enough:
            # Light EMA on the (already-quiet) feature: kills residual jitter
            # before it reaches calibration/mapping. The heavy, adaptive
            # smoothing for the visible cursor is the One-Euro filter in
            # gaze_view; this stays light to avoid stacking lag.
            self._smooth_gaze = metrics.smooth_values(
                candidate_offset, self._smooth_gaze, self.config.gaze_feature_smoothing_alpha
            )
            self._last_valid_gaze_offset = self._smooth_gaze
        raw_gaze_offset = self._last_valid_gaze_offset

        gaze = metrics.classify_centered_gaze(raw_gaze_offset, self.config.gaze_center_dead_zone)

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

    def _compute_gaze_feature(self, mesh, left_iris_f, right_iris_f, w, h) -> tuple[float, float] | None:
        def pt(idx: int) -> np.ndarray:
            return np.array([mesh[idx].x * w, mesh[idx].y * h])

        left = metrics.normalized_eye_gaze(
            left_iris_f,
            pt(lm.LEFT_EYE_LEFT_CORNER),
            pt(lm.LEFT_EYE_RIGHT_CORNER),
            pt(lm.LEFT_EYE_TOP_LID),
            pt(lm.LEFT_EYE_BOTTOM_LID),
        )
        right = metrics.normalized_eye_gaze(
            right_iris_f,
            pt(lm.RIGHT_EYE_LEFT_CORNER),
            pt(lm.RIGHT_EYE_RIGHT_CORNER),
            pt(lm.RIGHT_EYE_TOP_LID),
            pt(lm.RIGHT_EYE_BOTTOM_LID),
        )
        return metrics.fuse_eye_gaze(left, right)

    def _detect_blink(self, left_ear: float, right_ear: float) -> bool:
        threshold = self.config.blink_ear_threshold
        below_threshold = left_ear < threshold and right_ear < threshold
        now = time.time()
        if below_threshold and (now - self._last_blink_time) > self.config.blink_refractory_sec:
            self._last_blink_time = now
            self.blink_count += 1
            return True
        return False
