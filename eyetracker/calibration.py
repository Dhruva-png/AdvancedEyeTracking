"""Maps raw iris-in-eye-box offsets onto normalized on-screen gaze coordinates.

A webcam alone can't infer absolute gaze geometry (head distance, eye shape,
and camera angle all bias the raw eye offset differently per person). A
short 5-point calibration — look at each point, hold still — collects paired
(raw_offset -> known_screen_point) samples and fits a linear least-squares
map. This is the same trick simple gaze-tracking demos use in place of a full
3D eye model, and it's good enough to make the on-screen gaze cursor track
correctly for a single seated user.

Calibration runs on wall-clock time, not frame count: each point gets a
short "settle" window (dot just appeared, eyes are still saccading toward
it — discarded) followed by a "capture" window whose samples are collapsed
to a single median point. Feeding raw per-frame samples straight into the
least-squares fit was the original design and it was wrong — at 20-30fps a
frame-count target completes in under a second, capturing eye-movement
transients rather than a steady fixation, which produces a noisy or
outright degenerate (sometimes non-finite) transform.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

# Center first (used as the "not yet calibrated" fallback target too), then corners.
CALIBRATION_POINTS: list[tuple[float, float]] = [
    (0.5, 0.5),
    (0.08, 0.08),
    (0.92, 0.08),
    (0.08, 0.92),
    (0.92, 0.92),
]

PHASE_SETTLE = "settle"
PHASE_CAPTURE = "capture"

_MIN_FITTED_POINTS = 3  # affine fit has 3 unknowns per axis; need at least this many good points


class GazeCalibrator:
    def __init__(
        self,
        points: list[tuple[float, float]] | None = None,
        settle_sec: float = 0.6,
        capture_sec: float = 1.2,
        clock: Callable[[], float] = time.time,
    ):
        self.points = points or CALIBRATION_POINTS
        self.settle_sec = settle_sec
        self.capture_sec = capture_sec
        self._clock = clock

        self.active = False
        self.point_index = 0
        self.phase: str | None = None
        self._phase_start = 0.0
        self._capture_samples: list[tuple[float, float]] = []
        self._fitted_raw: list[tuple[float, float]] = []
        self._fitted_targets: list[tuple[float, float]] = []
        self._transform: np.ndarray | None = None  # 3x2, maps [nx, ny, 1] -> [sx, sy]

    @property
    def is_calibrated(self) -> bool:
        return self._transform is not None

    @property
    def current_target(self) -> tuple[float, float] | None:
        if not self.active or self.point_index >= len(self.points):
            return None
        return self.points[self.point_index]

    @property
    def progress(self) -> float:
        """Overall completion fraction across all calibration points, for a progress bar."""
        if not self.active:
            return 0.0
        point_fraction = 0.0
        if self.phase == PHASE_CAPTURE:
            point_fraction = min(1.0, (self._clock() - self._phase_start) / self.capture_sec)
        return (self.point_index + point_fraction) / len(self.points)

    def start(self) -> None:
        self.active = True
        self.point_index = 0
        self.phase = PHASE_SETTLE
        self._phase_start = self._clock()
        self._capture_samples = []
        self._fitted_raw = []
        self._fitted_targets = []

    def cancel(self) -> None:
        self.active = False
        self.phase = None

    def update(self, raw_offset: tuple[float, float] | None) -> None:
        """Call once per frame while `active`; advances phases on wall-clock time.

        Must be called even when `raw_offset` is None (face briefly lost) so
        the settle/capture timers keep advancing instead of stalling.
        """
        if not self.active:
            return
        elapsed = self._clock() - self._phase_start

        if self.phase == PHASE_SETTLE:
            if elapsed >= self.settle_sec:
                self.phase = PHASE_CAPTURE
                self._phase_start = self._clock()
                self._capture_samples = []
            return

        if self.phase == PHASE_CAPTURE:
            if raw_offset is not None and all(np.isfinite(raw_offset)):
                self._capture_samples.append(raw_offset)
            if elapsed >= self.capture_sec:
                self._finish_point()

    def _finish_point(self) -> None:
        if self._capture_samples:
            median = np.median(np.array(self._capture_samples), axis=0)
            self._fitted_raw.append((float(median[0]), float(median[1])))
            self._fitted_targets.append(self.points[self.point_index])

        self.point_index += 1
        if self.point_index >= len(self.points):
            self._fit()
            self.active = False
            self.phase = None
        else:
            self.phase = PHASE_SETTLE
            self._phase_start = self._clock()
            self._capture_samples = []

    def _fit(self) -> None:
        if len(self._fitted_raw) < _MIN_FITTED_POINTS:
            return  # too few usable fixations (e.g. face kept dropping out); stay uncalibrated
        raw = np.array(self._fitted_raw, dtype=float)
        target = np.array(self._fitted_targets, dtype=float)
        design = np.hstack([raw, np.ones((raw.shape[0], 1))])
        transform, *_ = np.linalg.lstsq(design, target, rcond=None)
        if np.all(np.isfinite(transform)):
            self._transform = transform

    def map(self, raw_offset: tuple[float, float] | None) -> tuple[float, float] | None:
        """Project a raw eye offset to normalized (0-1) screen coordinates."""
        if raw_offset is None or not all(np.isfinite(raw_offset)):
            return None
        if self._transform is None:
            # Uncalibrated fallback: center-biased passthrough so the cursor
            # still moves sensibly before the user calibrates.
            nx, ny = raw_offset
            return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))
        vec = np.array([raw_offset[0], raw_offset[1], 1.0])
        sx, sy = vec @ self._transform
        if not (np.isfinite(sx) and np.isfinite(sy)):
            return None
        return float(np.clip(sx, 0.0, 1.0)), float(np.clip(sy, 0.0, 1.0))
