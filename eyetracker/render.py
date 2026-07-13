"""Low-level cv2 drawing helpers shared by the camera HUD and the gaze view window.

Kept separate from tracking/business logic so visual style can change without
touching anything that affects correctness.
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_glow_circle(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    layers: int = 4,
    core_scale: float = 0.35,
) -> None:
    """Layered translucent circles fading outward, plus a solid bright core.

    Each pass re-copies the current `img` as its base so passes composite
    correctly (largest/dimmest circle first, smallest/brightest last).
    """
    for i in range(layers, 0, -1):
        layer_radius = max(1, int(radius * i / layers))
        alpha = 0.08 + 0.15 * (1 - (i - 1) / layers)
        overlay = img.copy()
        cv2.circle(overlay, center, layer_radius, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    core_radius = max(2, int(radius * core_scale))
    cv2.circle(img, center, core_radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, center, core_radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_rounded_panel(
    img: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.55,
    radius: int = 14,
) -> None:
    """Alpha-blended rounded-rectangle panel, used as a backdrop for HUD text."""
    x, y = top_left
    w, h = size
    overlay = img.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in [(x + radius, y + radius), (x + w - radius, y + radius), (x + radius, y + h - radius), (x + w - radius, y + h - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


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
