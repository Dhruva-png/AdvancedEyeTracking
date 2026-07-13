"""Maps raw iris-in-eye-box offsets onto normalized on-screen gaze coordinates.

A webcam alone can't infer absolute gaze geometry (head distance, eye shape,
and camera angle all bias the raw eye offset differently per person). A
short 5-point calibration — look at each point, hold still — collects paired
(raw_offset -> known_screen_point) samples and fits a linear least-squares
map. This is the same trick simple gaze-tracking demos use in place of a full
3D eye model, and it's good enough to make the on-screen gaze cursor track
correctly for a single seated user.
"""

from __future__ import annotations

import numpy as np

# Center first (used as the "not yet calibrated" fallback target too), then corners.
CALIBRATION_POINTS: list[tuple[float, float]] = [
    (0.5, 0.5),
    (0.08, 0.08),
    (0.92, 0.08),
    (0.08, 0.92),
    (0.92, 0.92),
]


class GazeCalibrator:
    def __init__(self, points: list[tuple[float, float]] | None = None, samples_per_point: int = 20):
        self.points = points or CALIBRATION_POINTS
        self.samples_per_point = samples_per_point
        self.active = False
        self.point_index = 0
        self._collected = 0
        self._raw_samples: list[tuple[float, float]] = []
        self._target_samples: list[tuple[float, float]] = []
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
        done_points = self.point_index
        point_fraction = self._collected / self.samples_per_point
        return (done_points + point_fraction) / len(self.points)

    def start(self) -> None:
        self.active = True
        self.point_index = 0
        self._collected = 0
        self._raw_samples = []
        self._target_samples = []

    def cancel(self) -> None:
        self.active = False

    def add_sample(self, raw_offset: tuple[float, float] | None) -> None:
        if not self.active or raw_offset is None:
            return
        self._raw_samples.append(raw_offset)
        self._target_samples.append(self.points[self.point_index])
        self._collected += 1
        if self._collected >= self.samples_per_point:
            self._collected = 0
            self.point_index += 1
            if self.point_index >= len(self.points):
                self._fit()
                self.active = False

    def _fit(self) -> None:
        raw = np.array(self._raw_samples, dtype=float)
        target = np.array(self._target_samples, dtype=float)
        design = np.hstack([raw, np.ones((raw.shape[0], 1))])
        transform, *_ = np.linalg.lstsq(design, target, rcond=None)
        self._transform = transform

    def map(self, raw_offset: tuple[float, float] | None) -> tuple[float, float] | None:
        """Project a raw eye offset to normalized (0-1) screen coordinates."""
        if raw_offset is None:
            return None
        if self._transform is None:
            # Uncalibrated fallback: center-biased passthrough so the cursor
            # still moves sensibly before the user calibrates.
            nx, ny = raw_offset
            return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))
        vec = np.array([raw_offset[0], raw_offset[1], 1.0])
        sx, sy = vec @ self._transform
        return float(np.clip(sx, 0.0, 1.0)), float(np.clip(sy, 0.0, 1.0))
