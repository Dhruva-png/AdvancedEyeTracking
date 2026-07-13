"""Maps the head-invariant gaze feature onto normalized on-screen coordinates,
with optional head-pose (yaw/pitch) compensation.

A webcam can't infer absolute gaze geometry, so calibration learns a per-user
map from the eye feature to the screen. The eye feature (iris relative to eye
corners, normalized by eye width — see metrics.normalized_eye_gaze) already
cancels head translation, camera distance, and head roll. The one thing it
does NOT cancel is head *yaw and pitch*: turning or nodding your head moves
where "eyes centered in their sockets" lands on screen. Head-pose compensation
closes that gap.

How the compensation is learned (this is the interesting part):

  Phase 1 — 9 gaze dots, head held still. Varying the gaze while the head is
  fixed pins down the eye->screen mapping (quadratic in the eye feature).

  Phase 2 — head sweep: fixate the *center* dot and slowly move your head
  around. Because the eyes counter-rotate to hold the target (the
  vestibulo-ocular reflex), the eye feature and the head pose co-vary while
  the true gaze target stays fixed at center. That co-variation is exactly the
  data needed to separate "the eye moved" from "the head moved" and to learn
  how head pose shifts screen gaze — which a static-head calibration cannot
  observe.

At run time the model is  screen = f(eye_feature) + g(head_pose - ref_pose),
so moving your head after calibration is corrected for instead of dragging the
cursor off target. If head data is unavailable or compensation is disabled,
everything degrades gracefully to the eye-only quadratic map.

Calibration runs on wall-clock time (settle -> capture per dot); capture
samples are median-aggregated so a stray blink or saccade can't skew a point.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

# 3x3 grid, evenly spread so the quadratic fit is well-conditioned across the
# whole screen rather than extrapolating heavily from just a few points.
_EDGE = 0.08
_MID = 0.5
_FAR = 0.92
CALIBRATION_POINTS: list[tuple[float, float]] = [
    (_MID, _MID),
    (_EDGE, _EDGE), (_MID, _EDGE), (_FAR, _EDGE),
    (_EDGE, _MID), (_FAR, _MID),
    (_EDGE, _FAR), (_MID, _FAR), (_FAR, _FAR),
]
_CENTER = (_MID, _MID)

PHASE_SETTLE = "settle"
PHASE_CAPTURE = "capture"
PHASE_SWEEP = "sweep"

_MIN_FITTED_POINTS = 5      # quadratic fit has 6 unknowns per axis; floor avoids fitting pure noise
_OUTLIER_RESIDUAL_FACTOR = 3.0  # a gaze point whose fit error exceeds this * median is dropped and the fit redone
_MIN_SWEEP_SAMPLES = 8      # below this the head-pose coupling isn't observable; fall back to eye-only
_MIN_SWEEP_YAW_SPAN = 0.02  # if the head barely moved during the sweep, don't trust the coupling


def _eye_terms(nx: float, ny: float) -> list[float]:
    """Quadratic expansion of the eye feature: captures the mild nonlinearity a plain affine fit misses."""
    return [nx * nx, ny * ny, nx * ny, nx, ny]


def _design_row(eye: tuple[float, float], head_delta: tuple[float, float] | None, use_head: bool) -> list[float]:
    row = _eye_terms(eye[0], eye[1])
    if use_head:
        dyaw, dpitch = head_delta if head_delta is not None else (0.0, 0.0)
        row += [dyaw, dpitch]
    row.append(1.0)  # intercept
    return row


class GazeCalibrator:
    def __init__(
        self,
        points: list[tuple[float, float]] | None = None,
        settle_sec: float = 0.5,
        capture_sec: float = 1.0,
        ridge_lambda: float = 1e-3,
        head_pose_compensation: bool = True,
        sweep_settle_sec: float = 1.0,
        sweep_sec: float = 4.0,
        clock: Callable[[], float] = time.time,
    ):
        self.points = points or CALIBRATION_POINTS
        self.settle_sec = settle_sec
        self.capture_sec = capture_sec
        self.ridge_lambda = ridge_lambda
        self.head_pose_compensation = head_pose_compensation
        self.sweep_settle_sec = sweep_settle_sec
        self.sweep_sec = sweep_sec
        self._clock = clock

        self.active = False
        self.point_index = 0
        self.phase: str | None = None
        self._phase_start = 0.0
        self._capture: list[tuple[tuple[float, float], tuple[float, float] | None]] = []
        self._sweep: list[tuple[tuple[float, float], tuple[float, float]]] = []
        # Per gaze dot: (median eye feature, median head pose or None, target).
        self._fixations: list[tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float]]] = []

        self._transform: np.ndarray | None = None
        self._use_head = False
        self._ref_pose: tuple[float, float] | None = None

    # -- state ---------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._transform is not None

    @property
    def is_sweeping(self) -> bool:
        return self.phase == PHASE_SWEEP

    @property
    def uses_head_pose(self) -> bool:
        """True once a fit has actually incorporated head-pose compensation."""
        return self._use_head

    @property
    def current_target(self) -> tuple[float, float] | None:
        if not self.active:
            return None
        if self.phase == PHASE_SWEEP:
            return _CENTER
        if self.point_index >= len(self.points):
            return None
        return self.points[self.point_index]

    @property
    def _total_units(self) -> float:
        return len(self.points) + (1.0 if self.head_pose_compensation else 0.0)

    @property
    def progress(self) -> float:
        if not self.active:
            return 0.0
        if self.phase == PHASE_SWEEP:
            swept = min(1.0, max(0.0, (self._clock() - self._phase_start - self.sweep_settle_sec) / self.sweep_sec))
            return (len(self.points) + swept) / self._total_units
        point_fraction = 0.0
        if self.phase == PHASE_CAPTURE:
            point_fraction = min(1.0, (self._clock() - self._phase_start) / self.capture_sec)
        return (self.point_index + point_fraction) / self._total_units

    # -- driving -------------------------------------------------------------

    def start(self) -> None:
        self.active = True
        self.point_index = 0
        self.phase = PHASE_SETTLE
        self._phase_start = self._clock()
        self._capture = []
        self._sweep = []
        self._fixations = []

    def cancel(self) -> None:
        self.active = False
        self.phase = None

    def update(self, raw_offset: tuple[float, float] | None, head_pose: tuple[float, float] | None = None) -> None:
        """Call once per frame while `active`; advances phases on wall-clock time.

        Call it even when inputs are None (face briefly lost) so timers keep
        advancing instead of stalling.
        """
        if not self.active:
            return
        elapsed = self._clock() - self._phase_start
        eye_ok = raw_offset is not None and all(np.isfinite(raw_offset))
        head_ok = head_pose is not None and all(np.isfinite(head_pose))

        if self.phase == PHASE_SETTLE:
            if elapsed >= self.settle_sec:
                self.phase = PHASE_CAPTURE
                self._phase_start = self._clock()
                self._capture = []
            return

        if self.phase == PHASE_CAPTURE:
            if eye_ok:
                self._capture.append((raw_offset, head_pose if head_ok else None))
            if elapsed >= self.capture_sec:
                self._finish_point()
            return

        if self.phase == PHASE_SWEEP:
            # Ignore the settle sub-window at the start (time to read the
            # instruction and start moving); then record eye+head pairs.
            if elapsed >= self.sweep_settle_sec and eye_ok and head_ok:
                self._sweep.append((raw_offset, head_pose))
            if elapsed >= self.sweep_settle_sec + self.sweep_sec:
                self._fit()
                self.active = False
                self.phase = None

    def _finish_point(self) -> None:
        if self._capture:
            eyes = np.array([c[0] for c in self._capture], dtype=float)
            med_eye = (float(np.median(eyes[:, 0])), float(np.median(eyes[:, 1])))
            heads = np.array([c[1] for c in self._capture if c[1] is not None], dtype=float)
            med_head = (float(np.median(heads[:, 0])), float(np.median(heads[:, 1]))) if len(heads) else None
            self._fixations.append((med_eye, med_head, self.points[self.point_index]))

        self.point_index += 1
        if self.point_index < len(self.points):
            self.phase = PHASE_SETTLE
            self._phase_start = self._clock()
            self._capture = []
        elif self.head_pose_compensation:
            self.phase = PHASE_SWEEP
            self._phase_start = self._clock()
            self._sweep = []
        else:
            self._fit()
            self.active = False
            self.phase = None

    # -- fitting -------------------------------------------------------------

    def _solve_ridge(self, design: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        gram = design.T @ design + self.ridge_lambda * np.eye(design.shape[1])
        rhs = design.T @ target
        try:
            transform = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return None
        return transform if np.all(np.isfinite(transform)) else None

    def _decide_use_head(self) -> bool:
        """Head compensation is only trustworthy if the sweep actually moved the
        head enough to observe the coupling; otherwise fall back to eye-only."""
        if not self.head_pose_compensation or len(self._sweep) < _MIN_SWEEP_SAMPLES:
            return False
        heads_available = all(f[1] is not None for f in self._fixations) and len(self._fixations) > 0
        if not heads_available:
            return False
        sweep_yaw = np.array([s[1][0] for s in self._sweep])
        return float(sweep_yaw.max() - sweep_yaw.min()) >= _MIN_SWEEP_YAW_SPAN

    def _fit(self) -> None:
        if len(self._fixations) < _MIN_FITTED_POINTS:
            return  # too few usable fixations (e.g. face kept dropping out); stay uncalibrated

        use_head = self._decide_use_head()
        # Reference pose = the neutral head pose you calibrated the gaze dots at.
        if use_head:
            ref = np.median(np.array([f[1] for f in self._fixations], dtype=float), axis=0)
            ref_pose: tuple[float, float] | None = (float(ref[0]), float(ref[1]))
        else:
            ref_pose = None

        fixation_rows, fixation_targets = self._build_fixation_rows(use_head, ref_pose)
        design = np.array(fixation_rows)
        target = np.array(fixation_targets, dtype=float)

        # Sweep samples (all at the center target) constrain the head-pose
        # coupling via the eye/head co-variation under fixation.
        if use_head and self._sweep:
            sweep_rows = [
                _design_row(eye, (head[0] - ref_pose[0], head[1] - ref_pose[1]), True) for eye, head in self._sweep
            ]
            design = np.vstack([design, np.array(sweep_rows)])
            target = np.vstack([target, np.tile(np.array(_CENTER), (len(self._sweep), 1))])

        transform = self._solve_ridge(design, target)
        if transform is None:
            return

        transform = self._reject_outlier_fixation(transform, use_head, ref_pose, design, target)

        self._transform = transform
        self._use_head = use_head
        self._ref_pose = ref_pose

    def _build_fixation_rows(self, use_head: bool, ref_pose):
        rows, targets = [], []
        for eye, head, target in self._fixations:
            delta = (head[0] - ref_pose[0], head[1] - ref_pose[1]) if (use_head and head is not None) else (0.0, 0.0)
            rows.append(_design_row(eye, delta, use_head))
            targets.append(target)
        return rows, targets

    def _reject_outlier_fixation(self, transform, use_head, ref_pose, design, target):
        """Drop the single worst gaze fixation if its residual dwarfs the rest
        (a blink/glance-away at that dot) and refit everything without it."""
        n_fix = len(self._fixations)
        if n_fix <= _MIN_FITTED_POINTS:
            return transform
        fix_design = design[:n_fix]
        fix_target = target[:n_fix]
        residuals = np.linalg.norm(fix_design @ transform - fix_target, axis=1)
        median_res = float(np.median(residuals))
        worst = int(np.argmax(residuals))
        if median_res <= 0 or residuals[worst] <= _OUTLIER_RESIDUAL_FACTOR * median_res:
            return transform
        keep = np.ones(design.shape[0], dtype=bool)
        keep[worst] = False  # sweep rows sit after the fixations, so this index maps correctly
        refit = self._solve_ridge(design[keep], target[keep])
        return refit if refit is not None else transform

    # -- mapping -------------------------------------------------------------

    def map(
        self, raw_offset: tuple[float, float] | None, head_pose: tuple[float, float] | None = None
    ) -> tuple[float, float] | None:
        """Project the gaze feature (+ head pose) to normalized (0-1) screen coordinates."""
        if raw_offset is None or not all(np.isfinite(raw_offset)):
            return None
        if self._transform is None:
            # Uncalibrated fallback: the feature is centered near 0, so scale it
            # out to a rough guess so the cursor still tracks direction.
            nx, ny = raw_offset
            return float(np.clip(0.5 + nx * 3.0, 0.0, 1.0)), float(np.clip(0.5 + ny * 3.0, 0.0, 1.0))

        head_delta = (0.0, 0.0)
        if self._use_head and self._ref_pose is not None and head_pose is not None and all(np.isfinite(head_pose)):
            head_delta = (head_pose[0] - self._ref_pose[0], head_pose[1] - self._ref_pose[1])

        row = np.array(_design_row(raw_offset, head_delta, self._use_head))
        sx, sy = row @ self._transform
        if not (np.isfinite(sx) and np.isfinite(sy)):
            return None
        return float(np.clip(sx, 0.0, 1.0)), float(np.clip(sy, 0.0, 1.0))
