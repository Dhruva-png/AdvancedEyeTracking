from eyetracker.calibration import CALIBRATION_POINTS, PHASE_SETTLE, GazeCalibrator


class FakeClock:
    """Deterministic clock so time-based calibration phases can be tested without real sleeps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_calibrator(points, settle_sec=0.5, capture_sec=1.0, ridge_lambda=1e-3, head_pose_compensation=False):
    clock = FakeClock()
    calibrator = GazeCalibrator(
        points=points,
        settle_sec=settle_sec,
        capture_sec=capture_sec,
        ridge_lambda=ridge_lambda,
        head_pose_compensation=head_pose_compensation,
        clock=clock,
    )
    return calibrator, clock


def _run_full_calibration(calibrator, clock, sample_for_target):
    """Drives a calibrator through settle+capture for every point using `sample_for_target(target) -> raw_offset`."""
    calibrator.start()
    for target in calibrator.points:
        clock.advance(calibrator.settle_sec + 0.1)  # clear settle
        calibrator.update(sample_for_target(target))  # tip settle -> capture
        for _ in range(5):
            calibrator.update(sample_for_target(target))
        clock.advance(calibrator.capture_sec + 0.1)  # clear capture, finalize point
        calibrator.update(sample_for_target(target))


def test_uncalibrated_map_scales_centered_feature_to_screen():
    # The raw feature is centered near 0; uncalibrated fallback maps 0 -> screen
    # center and moves in the correct direction, clipped to [0, 1].
    calibrator, _ = _make_calibrator([(0.5, 0.5)])
    assert calibrator.is_calibrated is False
    assert calibrator.map((0.0, 0.0)) == (0.5, 0.5)
    sx, sy = calibrator.map((0.1, -0.1))
    assert sx > 0.5 and sy < 0.5
    assert calibrator.map((5.0, -5.0)) == (1.0, 0.0)  # clipped


def test_map_of_none_is_none():
    calibrator, _ = _make_calibrator([(0.5, 0.5)])
    assert calibrator.map(None) is None


def test_default_calibration_grid_has_nine_points():
    assert len(CALIBRATION_POINTS) == 9


def test_start_begins_in_settle_phase():
    calibrator, _ = _make_calibrator([(0.5, 0.5)])
    calibrator.start()
    assert calibrator.active is True
    assert calibrator.phase == PHASE_SETTLE
    assert calibrator.current_target == (0.5, 0.5)


def test_update_during_settle_does_not_record_samples():
    calibrator, clock = _make_calibrator([(0.5, 0.5)], settle_sec=0.5, capture_sec=1.0)
    calibrator.start()
    for _ in range(10):
        calibrator.update((0.9, 0.9))  # wildly off-target; must be ignored during settle
        clock.advance(0.01)
    assert calibrator.phase == PHASE_SETTLE


def test_full_grid_identity_calibration_is_recovered_exactly():
    # With the true relationship exactly linear (target == raw) and enough
    # well-spread points for the quadratic model to be well-determined,
    # an unregularized fit should recover it almost exactly.
    calibrator, clock = _make_calibrator(CALIBRATION_POINTS, settle_sec=0.5, capture_sec=1.0, ridge_lambda=0.0)
    _run_full_calibration(calibrator, clock, sample_for_target=lambda target: target)

    assert calibrator.active is False
    assert calibrator.is_calibrated is True
    sx, sy = calibrator.map((0.3, 0.7))
    assert abs(sx - 0.3) < 1e-6
    assert abs(sy - 0.7) < 1e-6


def test_outlier_sample_during_capture_is_rejected_by_median():
    calibrator, clock = _make_calibrator(CALIBRATION_POINTS, settle_sec=0.5, capture_sec=1.0, ridge_lambda=0.0)
    calibrator.start()
    for target in calibrator.points:
        clock.advance(calibrator.settle_sec + 0.1)
        calibrator.update(target)  # tip settle -> capture (this sample is discarded by design, not recorded)
        calibrator.update((target[0] + 5.0, target[1] - 5.0))  # one bad sample, recorded during capture
        for _ in range(5):
            calibrator.update(target)  # good samples, recorded during capture
        clock.advance(calibrator.capture_sec + 0.1)
        calibrator.update(target)  # crosses the capture deadline -> finalize this point

    assert calibrator.is_calibrated is True
    # 6 good samples vs. 1 outlier per point: the per-axis median should
    # land exactly on the good value regardless of how far off the outlier is.
    sx, sy = calibrator.map((0.5, 0.5))
    assert abs(sx - 0.5) < 1e-6
    assert abs(sy - 0.5) < 1e-6


def test_one_bad_calibration_point_is_dropped_and_refit():
    # 8 points map identity; 1 point is entirely corrupt (user glanced away
    # for that whole dot). Point-level outlier rejection should drop it and
    # refit, recovering an accurate mapping the corrupt point would otherwise
    # have skewed.
    def sample_for_target(target):
        if target == CALIBRATION_POINTS[4]:  # sabotage one specific dot
            return (target[0] + 0.5, target[1] - 0.5)
        return target

    calibrator, clock = _make_calibrator(CALIBRATION_POINTS, settle_sec=0.5, capture_sec=1.0, ridge_lambda=0.0)
    _run_full_calibration(calibrator, clock, sample_for_target=sample_for_target)

    assert calibrator.is_calibrated is True
    # Points other than the sabotaged one should map back near-exactly.
    for tx, ty in [(0.5, 0.5), (0.08, 0.08), (0.92, 0.92)]:
        sx, sy = calibrator.map((tx, ty))
        assert abs(sx - tx) < 1e-3
        assert abs(sy - ty) < 1e-3


def test_stays_uncalibrated_if_face_missing_through_most_of_calibration():
    points = [(0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    calibrator, clock = _make_calibrator(points, settle_sec=0.1, capture_sec=0.5)
    calibrator.start()
    for _ in points:
        clock.advance(0.2)
        calibrator.update(None)  # face never found -> no usable samples for any point
        clock.advance(0.6)
        calibrator.update(None)
    assert calibrator.active is False
    assert calibrator.is_calibrated is False


def test_update_ignored_when_not_active():
    calibrator, _ = _make_calibrator([(0.5, 0.5)])
    calibrator.update((0.5, 0.5))
    assert calibrator.is_calibrated is False


def test_progress_reports_zero_when_inactive():
    calibrator, _ = _make_calibrator([(0.5, 0.5)])
    assert calibrator.progress == 0.0


def test_current_target_none_when_inactive():
    calibrator, _ = _make_calibrator([(0.5, 0.5)])
    assert calibrator.current_target is None


def test_cancel_stops_calibration():
    calibrator, _ = _make_calibrator([(0.5, 0.5), (0.0, 0.0)])
    calibrator.start()
    calibrator.cancel()
    assert calibrator.active is False
    assert calibrator.phase is None


# --- head-pose compensation ---------------------------------------------------
#
# Synthetic ground-truth model. When you fixate a screen target while your head
# is at yaw H, the eye counter-rotates (VOR) to hold the target, so the eye
# feature is whatever's left after the head's contribution:
#     screen_x = EYE_GAIN * eye_hx + HEAD_GAIN * H + C
#  => eye_hx  = (screen_x - HEAD_GAIN * H - C) / EYE_GAIN
# Compensation should recover the true screen point across head poses; the
# eye-only model can't and drifts as the head turns.

import numpy as np  # noqa: E402

EYE_GAIN = (2.6, 3.0)
HEAD_GAIN = (0.9, 0.8)
C = (0.5, 0.5)
REF_YAW, REF_PITCH = 0.10, -0.05  # non-zero neutral head pose used for the gaze dots


def _eye_for(screen, head):
    ex = (screen[0] - HEAD_GAIN[0] * head[0] - C[0]) / EYE_GAIN[0]
    ey = (screen[1] - HEAD_GAIN[1] * head[1] - C[1]) / EYE_GAIN[1]
    return ex, ey


def _drive_head_comp_calibration(head_amplitude=0.12, seed=0):
    rng = np.random.default_rng(seed)
    clock = FakeClock()
    cal = GazeCalibrator(
        settle_sec=0.4, capture_sec=0.8, ridge_lambda=1e-4,
        head_pose_compensation=True, sweep_settle_sec=0.5, sweep_sec=3.0, clock=clock,
    )
    cal.start()
    ref_head = (REF_YAW, REF_PITCH)

    # 9 gaze dots, head held at the (non-zero) reference pose.
    for target in cal.points:
        clock.advance(cal.settle_sec + 0.05)
        cal.update(_eye_for(target, ref_head), ref_head)
        for _ in range(6):
            cal.update(_eye_for(target, ref_head), ref_head)
        clock.advance(cal.capture_sec + 0.05)
        cal.update(_eye_for(target, ref_head), ref_head)

    # Head-sweep: fixate center, head yaw/pitch swings around; eye compensates.
    assert cal.is_sweeping
    clock.advance(cal.sweep_settle_sec + 0.05)
    for i in range(40):
        head = (REF_YAW + head_amplitude * np.sin(i * 0.5), REF_PITCH + head_amplitude * np.cos(i * 0.3))
        cal.update(_eye_for(C, head), head)  # gaze fixed at center C
        clock.advance(0.05)
    clock.advance(cal.sweep_sec)
    cal.update(_eye_for(C, (REF_YAW, REF_PITCH)), (REF_YAW, REF_PITCH))
    return cal


def test_head_sweep_phase_is_entered_after_gaze_dots():
    cal = _drive_head_comp_calibration()
    assert cal.is_calibrated is True
    assert cal.uses_head_pose is True


def test_compensation_keeps_gaze_accurate_when_head_moves():
    cal = _drive_head_comp_calibration()
    # Look at several targets with the head turned well away from the reference.
    moved_head = (REF_YAW + 0.10, REF_PITCH - 0.08)
    max_err = 0.0
    for tx, ty in [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.8, 0.2)]:
        eye = _eye_for((tx, ty), moved_head)
        sx, sy = cal.map(eye, moved_head)
        max_err = max(max_err, abs(sx - tx), abs(sy - ty))
    assert max_err < 0.03  # within 3% of screen despite the head being turned


def test_eye_only_model_drifts_when_head_moves():
    # Same data, but compensation disabled -> the eye-only fit can't correct
    # for head movement, so the error is large. This proves the compensation
    # is doing real work, not that the test is trivially easy.
    clock = FakeClock()
    cal = GazeCalibrator(
        settle_sec=0.4, capture_sec=0.8, ridge_lambda=1e-4, head_pose_compensation=False, clock=clock
    )
    cal.start()
    ref_head = (REF_YAW, REF_PITCH)
    for target in cal.points:
        clock.advance(cal.settle_sec + 0.05)
        cal.update(_eye_for(target, ref_head), ref_head)
        for _ in range(6):
            cal.update(_eye_for(target, ref_head), ref_head)
        clock.advance(cal.capture_sec + 0.05)
        cal.update(_eye_for(target, ref_head), ref_head)

    assert cal.is_calibrated and not cal.uses_head_pose
    moved_head = (REF_YAW + 0.10, REF_PITCH - 0.08)
    eye = _eye_for((0.5, 0.5), moved_head)
    sx, sy = cal.map(eye, moved_head)
    assert abs(sx - 0.5) > 0.05 or abs(sy - 0.5) > 0.05  # visibly off without compensation


def test_falls_back_to_eye_only_when_head_barely_moves_during_sweep():
    # If the user doesn't actually move their head in the sweep, there's no
    # coupling to learn -> must not claim head compensation, but must still
    # produce a working eye-only calibration.
    cal = _drive_head_comp_calibration(head_amplitude=0.0)
    assert cal.is_calibrated is True
    assert cal.uses_head_pose is False
