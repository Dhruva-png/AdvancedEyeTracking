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


def iris_center_f(landmarks, iris_indices: list[int], frame_w: int, frame_h: int) -> np.ndarray:
    """Sub-pixel iris center as a float (x, y) array — the gaze feature needs the precision."""
    pts = np.array([(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in iris_indices])
    return pts.mean(axis=0)


def normalized_eye_gaze(
    iris: np.ndarray,
    left_corner: np.ndarray,
    right_corner: np.ndarray,
    top_lid: np.ndarray,
    bottom_lid: np.ndarray,
) -> tuple[float, float] | None:
    """Head-invariant gaze feature: iris offset from the eye center, in an
    eye-aligned frame, normalized by the corner-to-corner width.

    Why this beats iris-within-a-bounding-box:
    - Normalizing by the (stable) corner distance cancels camera distance:
      lean in or out and the ratio is unchanged.
    - Projecting onto the corner axis cancels head roll (tilt your head and
      the eye axis rotates with it, so the projected offset is invariant).
    - Eye corners are far more stable landmarks than a box fit to 6 jittery
      contour points, so the feature itself is quieter frame-to-frame.

    Returns (hx, hy) centered near 0 (hx>0 = iris toward image-right,
    hy>0 = iris toward image-bottom), or None if the eye is degenerate.
    """
    axis = right_corner - left_corner
    eye_width = float(np.linalg.norm(axis))
    if eye_width < 1e-6:
        return None
    u = axis / eye_width                       # unit vector along the eye (image-left -> image-right)
    v = np.array([-u[1], u[0]])                # perpendicular, pointing image-downward
    corner_center = (left_corner + right_corner) / 2.0
    lid_center = (top_lid + bottom_lid) / 2.0

    hx = float(np.dot(iris - corner_center, u) / eye_width)
    # Vertical is measured from the lid center (which tracks up/down gaze far
    # better than the corner center) but still normalized by the stable width.
    hy = float(np.dot(iris - lid_center, v) / eye_width)
    return hx, hy


def fuse_eye_gaze(left: tuple[float, float] | None, right: tuple[float, float] | None) -> tuple[float, float] | None:
    """Average the two eyes' gaze features, falling back to whichever is valid."""
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0


def classify_centered_gaze(offset: tuple[float, float] | None, dead_zone: float = 0.04) -> str:
    """Human-readable direction from the centered gaze feature (offset around 0)."""
    if offset is None:
        return "UNKNOWN"
    hx, hy = offset
    if abs(hx) < dead_zone and abs(hy) < dead_zone:
        return "CENTER"
    if abs(hx) >= abs(hy):
        return "RIGHT" if hx > 0 else "LEFT"
    return "DOWN" if hy > 0 else "UP"


def smooth_point(new_point: tuple[int, int], prev_point: tuple[int, int] | None, alpha: float) -> tuple[int, int]:
    """Exponential moving average for pixel coordinates; returns new_point unchanged on the first call."""
    if prev_point is None:
        return new_point
    x = alpha * new_point[0] + (1 - alpha) * prev_point[0]
    y = alpha * new_point[1] + (1 - alpha) * prev_point[1]
    return int(x), int(y)


def smooth_values(new: tuple[float, ...], prev: tuple[float, ...] | None, alpha: float) -> tuple[float, ...]:
    """Exponential moving average over an arbitrary-length tuple, kept as floats.

    Used to lightly smooth the normalized gaze feature: unlike a drawn pixel
    coordinate it feeds calibration and mapping, where sub-pixel precision
    matters, so rounding to int (like `smooth_point`) would throw away the
    precision this exists to preserve.
    """
    if prev is None:
        return tuple(float(v) for v in new)
    return tuple(alpha * n + (1 - alpha) * p for n, p in zip(new, prev))


def eye_offset(eye_center: tuple[float, float], eye_box: tuple[float, float, float, float]) -> tuple[float, float] | None:
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
