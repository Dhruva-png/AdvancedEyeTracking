"""Low-level cv2 drawing helpers shared by the camera HUD and the gaze view window.

Kept separate from tracking/business logic so visual style can change without
touching anything that affects correctness.
"""

from __future__ import annotations

import cv2
import numpy as np


def _clip_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[np.ndarray, int, int] | None:
    """Returns (view_into_img, local_x_offset, local_y_offset) clipped to image bounds, or None if empty."""
    height, width = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1], x0, y0


def draw_glow_circle(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    layers: int = 4,
    core_scale: float = 0.35,
) -> None:
    """Layered translucent circles fading outward, plus a solid bright core.

    Only the bounding-box region around the circle is copied/blended, not the
    whole frame — drawing this several times per frame (both irises, the
    gaze cursor) on a full 720p+ image would otherwise dominate frame time.
    """
    pad = radius + 3
    roi_info = _clip_roi(img, center[0] - pad, center[1] - pad, pad * 2, pad * 2)
    if roi_info is None:
        return
    roi, x0, y0 = roi_info
    local_center = (center[0] - x0, center[1] - y0)

    for i in range(layers, 0, -1):
        layer_radius = max(1, int(radius * i / layers))
        alpha = 0.08 + 0.15 * (1 - (i - 1) / layers)
        overlay = roi.copy()
        cv2.circle(overlay, local_center, layer_radius, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    core_radius = max(2, int(radius * core_scale))
    cv2.circle(roi, local_center, core_radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(roi, local_center, core_radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_rounded_panel(
    img: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.55,
    radius: int = 14,
) -> None:
    """Alpha-blended rounded-rectangle panel, used as a backdrop for HUD text.

    Blends only the panel's own bounding box rather than the whole frame.
    """
    roi_info = _clip_roi(img, top_left[0], top_left[1], size[0], size[1])
    if roi_info is None:
        return
    roi, x0, y0 = roi_info
    rw, rh = roi.shape[1], roi.shape[0]
    r = max(0, min(radius, rw // 2, rh // 2))

    overlay = roi.copy()
    cv2.rectangle(overlay, (r, 0), (rw - r, rh), color, -1)
    cv2.rectangle(overlay, (0, r), (rw, rh - r), color, -1)
    for cx, cy in [(r, r), (rw - r, r), (r, rh - r), (rw - r, rh - r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_text(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.6,
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    if shadow:
        cv2.putText(img, text, (origin[0] + 1, origin[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_meter_bar(
    img: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    fraction: float,
    fill_color: tuple[int, int, int],
    track_color: tuple[int, int, int] = (60, 55, 50),
) -> None:
    """Horizontal progress bar (e.g. EAR level), fraction clipped to [0, 1]."""
    x, y = top_left
    w, h = size
    fraction = float(np.clip(fraction, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), track_color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + int(w * fraction), y + h), fill_color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (10, 10, 10), 1, lineType=cv2.LINE_AA)


def draw_polyline(img: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int = 1, closed: bool = True) -> None:
    cv2.polylines(img, [points.astype(np.int32)], closed, color, thickness, lineType=cv2.LINE_AA)
