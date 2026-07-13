import numpy as np

from eyetracker import metrics


def test_eye_aspect_ratio_open_eye_is_larger_than_closed():
    open_eye = np.array([[0, 5], [2, 0], [4, 0], [10, 5], [4, 10], [2, 10]], dtype=float)
    closed_eye = np.array([[0, 5], [2, 4.8], [4, 4.8], [10, 5], [4, 5.2], [2, 5.2]], dtype=float)

    assert metrics.eye_aspect_ratio(open_eye) > metrics.eye_aspect_ratio(closed_eye)


def test_eye_aspect_ratio_zero_width_returns_zero():
    degenerate = np.array([[5, 5], [2, 0], [4, 0], [5, 5], [4, 10], [2, 10]], dtype=float)
    assert metrics.eye_aspect_ratio(degenerate) == 0.0


def test_smooth_point_first_call_returns_input_unchanged():
    assert metrics.smooth_point((10, 20), None, alpha=0.25) == (10, 20)


def test_smooth_point_moves_toward_new_point():
    result = metrics.smooth_point((100, 100), (0, 0), alpha=0.5)
    assert result == (50, 50)


def test_gaze_direction_center():
    assert metrics.gaze_direction((50, 50), (0, 0, 100, 100)) == "CENTER"


def test_gaze_direction_left():
    assert metrics.gaze_direction((10, 50), (0, 0, 100, 100)) == "LEFT"


def test_gaze_direction_right():
    assert metrics.gaze_direction((90, 50), (0, 0, 100, 100)) == "RIGHT"


def test_gaze_direction_unknown_for_empty_box():
    assert metrics.gaze_direction((10, 10), (0, 0, 0, 0)) == "UNKNOWN"


def test_eye_offset_returns_none_for_degenerate_box():
    assert metrics.eye_offset((10, 10), (0, 0, 0, 0)) is None


def test_eye_offset_returns_fraction_within_box():
    assert metrics.eye_offset((50, 25), (0, 0, 100, 100)) == (0.5, 0.25)


def test_classify_gaze_direction_unknown_for_none():
    assert metrics.classify_gaze_direction(None) == "UNKNOWN"


def test_classify_gaze_direction_matches_offset_buckets():
    assert metrics.classify_gaze_direction((0.5, 0.5)) == "CENTER"
    assert metrics.classify_gaze_direction((0.1, 0.5)) == "LEFT"
    assert metrics.classify_gaze_direction((0.9, 0.5)) == "RIGHT"


def test_smooth_values_first_call_returns_input_as_floats():
    result = metrics.smooth_values((10, 20, 30, 40), None, alpha=0.15)
    assert result == (10.0, 20.0, 30.0, 40.0)
    assert all(isinstance(v, float) for v in result)


def test_smooth_values_moves_toward_new_value_without_rounding():
    result = metrics.smooth_values((100.0, 100.0), (0.0, 0.0), alpha=0.5)
    assert result == (50.0, 50.0)


def test_smooth_values_low_alpha_damps_jitter():
    # A small alpha should barely move on a single noisy sample -- this is
    # exactly the property `eye_box_smoothing_alpha` relies on to keep the
    # iris-in-box ratio from amplifying per-frame landmark jitter.
    result = metrics.smooth_values((10.0,), (0.0,), alpha=0.1)
    assert result[0] == 1.0
