"""Capture loop orchestration: wires the tracker, HUD, calibrator, gaze view,
heatmap, logger, and live dashboard together."""

from __future__ import annotations

import logging
import time
from datetime import datetime

import cv2

from .calibration import GazeCalibrator
from .config import TrackerConfig
from .gaze_view import GazeView
from .heatmap import GazeHeatmap
from .hud import CameraHUD
from .live_dashboard import LiveDashboard
from .screen_utils import get_screen_resolution
from .session_logger import SessionLogger
from .tracker import EyeTracker

logger = logging.getLogger("eyetracker")

WINDOW_NAME = "Eye Tracking  |  h: dashboard  g: gaze view  c: calibrate  e: export  q: quit"
MAX_CONSECUTIVE_FRAME_ERRORS = 30


class Application:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.heatmap = GazeHeatmap(self.config.heatmap_grid_size, self.config.heatmap_gaussian_sigma)
        self.session_logger = SessionLogger(self.config.output_dir, self.config.output_prefix)
        self.dashboard = LiveDashboard(self.config.dashboard_window_sec)
        self.hud = CameraHUD()
        self.calibrator = GazeCalibrator(
            settle_sec=self.config.calibration_settle_sec,
            capture_sec=self.config.calibration_capture_sec,
            ridge_lambda=self.config.calibration_ridge_lambda,
            head_pose_compensation=self.config.head_pose_compensation,
            sweep_settle_sec=self.config.calibration_sweep_settle_sec,
            sweep_sec=self.config.calibration_sweep_sec,
        )
        self.gaze_view = GazeView(
            get_screen_resolution(),
            trail_length=self.config.gaze_trail_length,
            min_cutoff=self.config.gaze_cursor_min_cutoff,
            beta=self.config.gaze_cursor_beta,
        )

        self._show_dashboard = False
        self._show_gaze_view = self.config.gaze_view_enabled_by_default
        self._last_log_time = 0.0
        self._last_dashboard_update = 0.0
        self._last_gaze_view_update = 0.0

    def run(self) -> None:
        cam = cv2.VideoCapture(self.config.camera_index)
        if not cam.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.config.camera_index}. "
                "Close other apps using the webcam, or pass a different --camera index."
            )
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        logger.info("Camera opened. Controls: [h] dashboard  [g] gaze view  [c] calibrate  [e] export  [q] quit")

        try:
            tracker = EyeTracker(self.config)
        except Exception:
            cam.release()
            raise

        try:
            try:
                self._loop(cam, tracker)
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
        finally:
            tracker.close()
            cam.release()
            cv2.destroyAllWindows()
            self.dashboard.close()
            self.gaze_view.close()
            if self.session_logger.samples:
                paths = self.session_logger.export(self.heatmap)
                logger.info("Session saved: %s", paths)
            else:
                logger.info("No samples recorded; nothing to export.")

    def _loop(self, cam: cv2.VideoCapture, tracker: EyeTracker) -> None:
        consecutive_errors = 0
        while True:
            ok, frame = cam.read()
            if not ok:
                logger.warning("Dropped a frame from the camera.")
                continue

            frame = cv2.flip(frame, 1)

            # A single bad frame (e.g. a rendering edge case on an unusual
            # face pose) shouldn't kill an otherwise-fine session. Skip it,
            # but escalate if failures keep happening — that's a real bug,
            # not a one-off.
            try:
                self._process_and_render(frame, tracker)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                logger.exception("Error processing a frame (%d in a row); skipping it.", consecutive_errors)
                if consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                    logger.error("Too many consecutive frame errors; stopping.")
                    raise

            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):
                self._show_dashboard = not self._show_dashboard
                logger.info("Dashboard %s", "ON" if self._show_dashboard else "OFF")
            elif key == ord("g"):
                self._show_gaze_view = not self._show_gaze_view
                if not self._show_gaze_view:
                    self.gaze_view.close()
                logger.info("Gaze view %s", "ON" if self._show_gaze_view else "OFF")
            elif key == ord("c"):
                self.calibrator.start()
                logger.info("Calibration started: look at each highlighted dot in the Gaze View window.")
                if not self._show_gaze_view:
                    self._show_gaze_view = True
            elif key == ord("e"):
                paths = self.session_logger.export(self.heatmap)
                logger.info("Manual export complete: %s", paths)
            elif key == ord("q"):
                logger.info("Quit requested.")
                break

    def _process_and_render(self, frame, tracker: EyeTracker) -> None:
        result = tracker.process(frame)

        if result.face_found and result.x_norm is not None:
            self.heatmap.add(result.x_norm, result.y_norm)

        if self.calibrator.active:
            self.calibrator.update(result.raw_gaze_offset, result.head_pose)

        self._maybe_log(result, tracker.blink_count)

        rendered = self.hud.draw(
            result, tracker.blink_count, calibrating=self.calibrator.active, calibrated=self.calibrator.is_calibrated
        )
        cv2.imshow(WINDOW_NAME, rendered)

        if self._show_gaze_view:
            self._maybe_update_gaze_view(result)

        if self._show_dashboard:
            self._maybe_update_dashboard(result, tracker.blink_count)

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

    def _maybe_update_dashboard(self, result, blink_count: int) -> None:
        now = time.time()
        if now - self._last_dashboard_update < self.config.dashboard_update_interval_sec:
            return
        self._last_dashboard_update = now
        self.dashboard.update(result.frame, self.heatmap.normalized(), blink_count, result.gaze)

    def _maybe_update_gaze_view(self, result) -> None:
        now = time.time()
        if now - self._last_gaze_view_update < self.config.gaze_view_update_interval_sec:
            return
        self._last_gaze_view_update = now
        screen_point = (
            self.calibrator.map(result.raw_gaze_offset, result.head_pose) if not self.calibrator.active else None
        )
        self.gaze_view.draw(screen_point, self.calibrator)
