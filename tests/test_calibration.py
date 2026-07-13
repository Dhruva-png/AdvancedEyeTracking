from eyetracker.calibration import GazeCalibrator


def test_uncalibrated_map_passes_through_clipped():
    calibrator = GazeCalibrator()
    assert calibrator.is_calibrated is False
    assert calibrator.map((0.5, 0.5)) == (0.5, 0.5)
    assert calibrator.map((-1.0, 2.0)) == (0.0, 1.0)


def test_map_of_none_is_none():
    calibrator = GazeCalibrator()
    assert calibrator.map(None) is None


def test_calibration_flow_completes_and_fits_identity_mapping():
    points = [(0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    calibrator = GazeCalibrator(points=points, samples_per_point=3)
    calibrator.start()
    assert calibrator.active is True

    for target in points:
        for _ in range(3):
            # Raw offset == target exactly: the fitted transform should learn identity.
            calibrator.add_sample(target)

    assert calibrator.active is False
    assert calibrator.is_calibrated is True

    sx, sy = calibrator.map((0.25, 0.75))
    assert abs(sx - 0.25) < 1e-6
    assert abs(sy - 0.75) < 1e-6


def test_add_sample_ignored_when_not_active():
    calibrator = GazeCalibrator()
    calibrator.add_sample((0.5, 0.5))
    assert calibrator.is_calibrated is False


def test_add_sample_ignores_none_offset_without_advancing():
    calibrator = GazeCalibrator(samples_per_point=2)
    calibrator.start()
    calibrator.add_sample(None)
    assert calibrator.point_index == 0


def test_progress_reports_zero_when_inactive():
    calibrator = GazeCalibrator()
    assert calibrator.progress == 0.0


def test_current_target_none_when_inactive():
    calibrator = GazeCalibrator()
    assert calibrator.current_target is None
