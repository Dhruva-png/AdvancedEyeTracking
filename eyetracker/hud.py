"""Renders the on-camera HUD: eye contours, glowing iris markers, and a stat panel.

Kept stateful (blink-flash timing, FPS smoothing) but free of tracking logic —
it only ever reads a `FrameResult` and draws.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from . import render, theme

BLINK_FLASH_SECONDS = 0.18


class CameraHUD:
    def __init__(self):
        self._blink_flash_until = 0.0
        self._frame_times: deque[float] = deque(maxlen=30)
        self._session_start = time.time()

    def _tick_fps(self) -> float:
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def draw(self, result, blink_count: int, calibrating: bool, calibrated: bool) -> np.ndarray:
        frame = result.frame
        fps = self._tick_fps()

        if result.blink:
            self._blink_flash_until = time.time() + BLINK_FLASH_SECONDS

        if result.face_found:
            self._draw_eye(frame, result.left_eye_pts, result.left_center)
            self._draw_eye(frame, result.right_eye_pts, result.right_center)

        self._draw_stat_panel(frame, result, blink_count, fps, calibrating, calibrated)

        if time.time() < self._blink_flash_until:
            self._draw_border_flash(frame, theme.MAGENTA)

        if not result.face_found:
            self._draw_no_face_banner(frame)

        return frame

    @staticmethod
    def _draw_eye(frame: np.ndarray, eye_pts: np.ndarray | None, iris_center: tuple[int, int] | None) -> None:
        if eye_pts is not None and len(eye_pts) >= 5:
            # During a blink the 6 contour points nearly collapse onto a line;
            # fitEllipse can then return non-finite axes, which cv2.ellipse
            # rejects with a hard error. Skip the outline for that one frame
            # rather than crash the whole session over a blink.
            try:
                ellipse = cv2.fitEllipse(eye_pts.astype(np.float32))
                (_, _), (major, minor), _ = ellipse
                if np.isfinite(major) and np.isfinite(minor) and major > 0 and minor > 0:
                    cv2.ellipse(frame, ellipse, theme.CYAN_SOFT, 1, lineType=cv2.LINE_AA)
            except cv2.error:
                pass
        if iris_center is not None:
            render.draw_glow_circle(frame, iris_center, radius=10, color=theme.CYAN, layers=4)

    def _draw_stat_panel(self, frame, result, blink_count: int, fps: float, calibrating: bool, calibrated: bool) -> None:
        h, w = frame.shape[:2]
        panel_w, panel_h = 250, 150
        render.draw_rounded_panel(frame, (14, 14), (panel_w, panel_h), theme.BG_DEEP, alpha=0.6)

        x, y = 30, 42
        render.draw_text(frame, "GAZE", (x, y), theme.TEXT_MUTED, scale=0.45)
        render.draw_text(frame, result.gaze, (x, y + 24), theme.CYAN, scale=0.75, thickness=2)

        y2 = y + 56
        render.draw_text(frame, "BLINKS", (x, y2), theme.TEXT_MUTED, scale=0.45)
        render.draw_text(frame, str(blink_count), (x + 90, y2 + 2), theme.TEXT_PRIMARY, scale=0.55, thickness=2)

        y3 = y2 + 26
        avg_ear = (result.left_ear + result.right_ear) / 2.0
        ear_color = theme.AMBER if avg_ear < 0.22 else theme.GREEN
        render.draw_text(frame, "EAR", (x, y3), theme.TEXT_MUTED, scale=0.45)
        render.draw_meter_bar(frame, (x + 45, y3 - 10), (110, 10), avg_ear / 0.4, ear_color)

        y4 = y3 + 26
        render.draw_text(frame, f"FPS {fps:4.1f}", (x, y4), theme.TEXT_MUTED, scale=0.42)

        status = "CALIBRATING" if calibrating else ("CALIBRATED" if calibrated else "UNCALIBRATED")
        status_color = theme.CYAN if calibrating else (theme.GREEN if calibrated else theme.TEXT_MUTED)
        render.draw_text(frame, status, (x + 110, y4), status_color, scale=0.42)

        render.draw_text(frame, "EyeTrack Pro", (w - 150, h - 16), theme.TEXT_MUTED, scale=0.5)

        if calibrating:
            render.draw_rounded_panel(frame, (w // 2 - 170, 14), (340, 40), theme.BG_DEEP, alpha=0.6)
            render.draw_text(frame, "Calibrating: look at the dot on the Gaze View window", (w // 2 - 155, 39), theme.CYAN, scale=0.5)

    @staticmethod
    def _draw_border_flash(frame: np.ndarray, color: tuple[int, int, int]) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_no_face_banner(frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        render.draw_rounded_panel(frame, (w // 2 - 140, h // 2 - 20), (280, 40), theme.BG_DEEP, alpha=0.65)
        render.draw_text(frame, "No face detected", (w // 2 - 100, h // 2 + 6), theme.AMBER, scale=0.6, thickness=2)
