"""'Gaze View': a virtual-monitor canvas showing a glowing circle at your
estimated on-screen gaze point.

We render our own bordered canvas rather than an OS-level always-on-top
transparent overlay — that keeps the feature cross-platform and safe (no
system-wide window-manager hooks), while still giving a literal, resizable
"what am I looking at on the screen" view you can drag onto your monitor.
The canvas is sized proportionally to your real screen resolution (capped
for smooth rendering); since the gaze point is stored as a 0-1 fraction,
resizing the window doesn't affect mapping accuracy.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from . import render, theme
from .calibration import GazeCalibrator

WINDOW_NAME = "Gaze View"
_MAX_CANVAS_WIDTH = 1280


class GazeView:
    def __init__(self, screen_resolution: tuple[int, int], trail_length: int = 24, smoothing_alpha: float = 0.35):
        w, h = screen_resolution
        if w > _MAX_CANVAS_WIDTH:
            scale = _MAX_CANVAS_WIDTH / w
            w, h = int(w * scale), int(h * scale)
        self.canvas_size = (w, h)
        self.smoothing_alpha = smoothing_alpha
        self._trail: deque[tuple[int, int]] = deque(maxlen=trail_length)
        self._smoothed: tuple[float, float] | None = None
        self._window_created = False

    def _ensure_window(self) -> None:
        if self._window_created:
            return
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *self.canvas_size)
        self._window_created = True

    def _background(self) -> np.ndarray:
        w, h = self.canvas_size
        canvas = np.full((h, w, 3), theme.BG_DEEP, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), theme.GRID_LINE, 2, lineType=cv2.LINE_AA)
        for gx in range(0, w, max(60, w // 12)):
            cv2.line(canvas, (gx, 0), (gx, h), theme.GRID_LINE, 1, lineType=cv2.LINE_AA)
        for gy in range(0, h, max(60, h // 8)):
            cv2.line(canvas, (0, gy), (w, gy), theme.GRID_LINE, 1, lineType=cv2.LINE_AA)
        return canvas

    def _to_px(self, norm_point: tuple[float, float]) -> tuple[int, int]:
        w, h = self.canvas_size
        nx, ny = norm_point
        return int(np.clip(nx, 0.0, 1.0) * (w - 1)), int(np.clip(ny, 0.0, 1.0) * (h - 1))

    def draw(self, gaze_point_norm: tuple[float, float] | None, calibrator: GazeCalibrator) -> np.ndarray:
        self._ensure_window()
        canvas = self._background()

        if calibrator.active and calibrator.current_target is not None:
            self._draw_calibration_target(canvas, calibrator)
        else:
            self._draw_gaze_cursor(canvas, gaze_point_norm, calibrator.is_calibrated)

        cv2.imshow(WINDOW_NAME, canvas)
        return canvas

    def _draw_gaze_cursor(self, canvas: np.ndarray, gaze_point_norm, is_calibrated: bool) -> None:
        if gaze_point_norm is not None:
            if self._smoothed is None:
                self._smoothed = gaze_point_norm
            else:
                a = self.smoothing_alpha
                self._smoothed = (
                    a * gaze_point_norm[0] + (1 - a) * self._smoothed[0],
                    a * gaze_point_norm[1] + (1 - a) * self._smoothed[1],
                )
            self._trail.append(self._to_px(self._smoothed))

        n = len(self._trail)
        for i, pt in enumerate(self._trail):
            fade = (i + 1) / max(n, 1)
            alpha = fade * 0.3
            radius = 3 + int(6 * fade)
            overlay = canvas.copy()
            cv2.circle(overlay, pt, radius, theme.MAGENTA, -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)

        if self._trail:
            render.draw_glow_circle(canvas, self._trail[-1], radius=24, color=theme.MAGENTA, layers=5)

        status = "CALIBRATED" if is_calibrated else "UNCALIBRATED  (press C on the camera window to calibrate)"
        color = theme.GREEN if is_calibrated else theme.AMBER
        render.draw_text(canvas, status, (16, 28), color, scale=0.6, thickness=2)

    def _draw_calibration_target(self, canvas: np.ndarray, calibrator: GazeCalibrator) -> None:
        w, h = self.canvas_size
        px = self._to_px(calibrator.current_target)
        pulse = 0.5 + 0.5 * np.sin(time.time() * 6.0)
        radius = int(14 + 8 * pulse)
        render.draw_glow_circle(canvas, px, radius=radius, color=theme.CYAN, layers=5)
        cv2.circle(canvas, px, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        idx, total = calibrator.point_index + 1, len(calibrator.points)
        render.draw_text(canvas, f"Calibrating {idx}/{total} — look at the dot and hold still", (16, 28), theme.CYAN, scale=0.6, thickness=2)
        render.draw_meter_bar(canvas, (20, h - 30), (w - 40, 12), calibrator.progress, theme.CYAN)

    def close(self) -> None:
        if self._window_created:
            cv2.destroyWindow(WINDOW_NAME)
            self._window_created = False
