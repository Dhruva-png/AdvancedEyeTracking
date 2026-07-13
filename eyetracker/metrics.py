"""Pure geometry/signal functions: no camera, no I/O, no globals — easy to unit test."""

from __future__ import annotations

import numpy as np


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Eye-aspect-ratio (EAR) from 6 ordered contour points; drops toward 0 on a blink."""
    a = np.linalg.norm(eye_pts[1] - eye_pts[5])
    b = np.linalg.norm(eye_pts[2] - eye_pts[4])
    c = np.linalg.norm(eye_pts[0] - eye_pts[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def iris_center(landmarks, iris_indices: list[int], frame_w: int, frame_h: int) -> tuple[int, int]:
    pts = np.array([(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in iris_indices])
    center = pts.mean(axis=0)
    return int(center[0]), int(center[1])


def smooth_point(new_point: tuple[int, int], prev_point: tuple[int, int] | None, alpha: float) -> tuple[int, int]:
    """Exponential moving average; returns new_point unchanged on the first call."""
    if prev_point is None:
        return new_point
    x = alpha * new_point[0] + (1 - alpha) * prev_point[0]
    y = alpha * new_point[1] + (1 - alpha) * prev_point[1]
    return int(x), int(y)


def eye_offset(eye_center: tuple[int, int], eye_box: tuple[int, int, int, int]) -> tuple[float, float] | None:
    """Continuous position of the iris within its eye box, roughly in [0, 1] on each axis.

    This is the raw signal gaze estimation is built on: `classify_gaze_direction`
    buckets it into a human-readable label, and `GazeCalibrator` maps it onto
    normalized screen coordinates. Returns None when the eye box is degenerate.
    """
    x, y = eye_center
    ex, ey, ew, eh = eye_box
    if ew == 0 or eh == 0:
        return None
    return (x - ex) / float(ew), (y - ey) / float(eh)


def classify_gaze_direction(offset: tuple[float, float] | None) -> str:
    """Bucket a continuous eye offset into UP/DOWN/LEFT/RIGHT/CENTER for display."""
    if offset is None:
        return "UNKNOWN"
    nx, ny = offset
    if nx < 0.3:
        return "LEFT"
    if nx > 0.7:
        return "RIGHT"
    if ny < 0.35:
        return "UP"
    if ny > 0.75:
        return "DOWN"
    return "CENTER"


def gaze_direction(eye_center: tuple[int, int], eye_box: tuple[int, int, int, int]) -> str:
    """Convenience wrapper: eye_offset + classify_gaze_direction in one call."""
    return classify_gaze_direction(eye_offset(eye_center, eye_box))
