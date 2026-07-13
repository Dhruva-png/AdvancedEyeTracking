"""Capture loop orchestration: wires the tracker, heatmap, logger and dashboard together."""

from __future__ import annotations

import logging
import time
from datetime import datetime

import cv2

from .config import TrackerConfig
from .heatmap import GazeHeatmap
from .live_dashboard import LiveDashboard
from .session_logger import SessionLogger
from .tracker import EyeTracker

logger = logging.getLogger("eyetracker")

WINDOW_NAME = "Eye Tracking  |  h: dashboard   e: export   q: quit"


class Application:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.heatmap = GazeHeatmap(self.config.heatmap_grid_size, self.config.heatmap_gaussian_sigma)
        self.session_logger = SessionLogger(self.config.output_dir, self.config.output_prefix)
        self.dashboard = LiveDashboard(self.config.dashboard_window_sec)
        self._show_dashboard = False
        self._last_log_time = 0.0
        self._last_dashboard_update = 0.0

    def run(self) -> None:
        cam = cv2.VideoCapture(self.config.camera_index)
        if not cam.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.config.camera_index}. "
                "Close other apps using the webcam, or pass a different --camera index."
            )
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        logger.info("Camera opened. Controls: [h] dashboard  [e] export now  [q] quit")

        try:
            with EyeTracker(self.config) as tracker:
                self._loop(cam, tracker)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cam.release()
            cv2.destroyAllWindows()
            self.dashboard.close()
            paths = self.session_logger.export(self.heatmap)
            logger.info("Session saved: %s", paths)

    def _loop(self, cam: cv2.VideoCapture, tracker: EyeTracker) -> None:
        while True:
            ok, frame = cam.read()
            if not ok:
                logger.warning("Dropped a frame from the camera.")
                continue

            frame = cv2.flip(frame, 1)
            result = tracker.process(frame)

            if result.face_found and result.x_norm is not None:
                self.heatmap.add(result.x_norm, result.y_norm)

            self._maybe_log(result, tracker.blink_count)
            cv2.imshow(WINDOW_NAME, result.frame)

            if self._show_dashboard:
                self._maybe_update_dashboard(result.frame, tracker.blink_count)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):
                self._show_dashboard = not self._show_dashboard
                logger.info("Dashboard %s", "ON" if self._show_dashboard else "OFF")
            elif key == ord("e"):
                paths = self.session_logger.export(self.heatmap)
                logger.info("Manual export complete: %s", paths)
            elif key == ord("q"):
                logger.info("Quit requested.")
                break

    def _maybe_log(self, result, blink_count: int) -> None:
        now = time.time()
        if now - self._last_log_time < self.config.log_interval_sec:
            return
        self._last_log_time = now
        self.session_logger.record(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            left_x=result.left_center[0] if result.left_center else None,
            left_y=result.left_center[1] if result.left_center else None,
            right_x=result.right_center[0] if result.right_center else None,
            right_y=result.right_center[1] if result.right_center else None,
            gaze=result.gaze,
            blink=result.blink,
            blink_count=blink_count,
        )

    def _maybe_update_dashboard(self, frame, blink_count: int) -> None:
        now = time.time()
        if now - self._last_dashboard_update < self.config.dashboard_update_interval_sec:
            return
        self._last_dashboard_update = now
        self.dashboard.update(frame, self.heatmap.normalized(), blink_count)
