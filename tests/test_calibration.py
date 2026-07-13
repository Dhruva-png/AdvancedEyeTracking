from eyetracker.calibration import PHASE_CAPTURE, PHASE_SETTLE, GazeCalibrator


class FakeClock:
    """Deterministic clock so time-based calibration phases can be tested without real sleeps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_calibrator(points, settle_sec=0.5, capture_sec=1.0):
    clock = FakeClock()
    calibrator = GazeCalibrator(points=points, settle_sec=settle_sec, capture_sec=capture_sec, clock=clock)
    return calibrator, clock


def test_uncalibrated_map_passes_through_clipped():
    calibrator, _ = _make_calibrator([(0.5, 0.5)])
    assert calibrator.is_calibrated is False
    assert calibrator.map((0.5, 0.5)) == (0.5, 0.5)
    assert calibrator.map((-1.0, 2.0)) == (0.0, 1.0)


def test_map_of_none_is_none():
    calibrator, _ = _make_calibrator([(0.5, 0.5)])
    assert calibrator.map(None) is None


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


def test_samples_only_recorded_during_capture_and_median_aggregated():
    points = [(0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    calibrator, clock = _make_calibrator(points, settle_sec=0.5, capture_sec=1.0)
    calibrator.start()

    for target in points:
        clock.advance(0.6)  # clear the settle window
        calibrator.update(target)  # tip over from settle -> capture
        # Feed one outlier plus several on-target samples; the median should reject the outlier.
        calibrator.update((target[0] + 5.0, target[1] + 5.0))
        for _ in range(5):
            calibrator.update(target)
        clock.advance(1.1)  # clear the capture window and finalize this point
        calibrator.update(target)

    assert calibrator.active is False
    assert calibrator.is_calibrated is True
    sx, sy = calibrator.map((0.25, 0.75))
    assert abs(sx - 0.25) < 1e-6
    assert abs(sy - 0.75) < 1e-6


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
