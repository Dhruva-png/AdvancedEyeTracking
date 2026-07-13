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
    # A small alpha barely moves on a single noisy sample -- the property the
    # light gaze-feature EMA relies on to knock down per-frame landmark jitter.
    result = metrics.smooth_values((10.0,), (0.0,), alpha=0.1)
    assert result[0] == 1.0


# --- head-invariant gaze feature ---------------------------------------------

def _eye(iris, left_corner, right_corner, top, bottom):
    return (np.array(iris, float), np.array(left_corner, float), np.array(right_corner, float),
            np.array(top, float), np.array(bottom, float))


def test_normalized_eye_gaze_centered_iris_is_near_zero():
    iris, lc, rc, tp, bt = _eye((50, 50), (0, 50), (100, 50), (50, 40), (50, 60))
    hx, hy = metrics.normalized_eye_gaze(iris, lc, rc, tp, bt)
    assert abs(hx) < 1e-9
    assert abs(hy) < 1e-9


def test_normalized_eye_gaze_iris_right_of_center_gives_positive_hx():
    iris, lc, rc, tp, bt = _eye((70, 50), (0, 50), (100, 50), (50, 40), (50, 60))
    hx, _ = metrics.normalized_eye_gaze(iris, lc, rc, tp, bt)
    assert hx > 0


def test_normalized_eye_gaze_is_scale_invariant():
    # Same gaze, camera twice as close (all coordinates scaled 2x about origin):
    # the feature must be unchanged. This is the property that makes leaning in
    # or out not wreck accuracy.
    near = metrics.normalized_eye_gaze(*_eye((70, 50), (0, 50), (100, 50), (50, 40), (50, 60)))
    far = metrics.normalized_eye_gaze(*_eye((140, 100), (0, 100), (200, 100), (100, 80), (100, 120)))
    assert abs(near[0] - far[0]) < 1e-9
    assert abs(near[1] - far[1]) < 1e-9


def test_normalized_eye_gaze_is_roll_invariant():
    # Rotate the whole eye (head tilt) about its corner-center; because the
    # feature is measured in the eye-aligned frame, hx/hy must be unchanged.
    import math

    base = _eye((70, 50), (0, 50), (100, 50), (50, 40), (50, 60))
    theta = math.radians(20)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    center = np.array([50.0, 50.0])
    rotated = tuple(center + rot @ (p - center) for p in base)

    flat = metrics.normalized_eye_gaze(*base)
    tilted = metrics.normalized_eye_gaze(*rotated)
    assert abs(flat[0] - tilted[0]) < 1e-6
    assert abs(flat[1] - tilted[1]) < 1e-6


def test_normalized_eye_gaze_degenerate_eye_returns_none():
    iris, lc, rc, tp, bt = _eye((50, 50), (50, 50), (50, 50), (50, 40), (50, 60))
    assert metrics.normalized_eye_gaze(iris, lc, rc, tp, bt) is None


def test_fuse_eye_gaze_averages_and_falls_back():
    assert metrics.fuse_eye_gaze((0.0, 0.0), (0.2, 0.4)) == (0.1, 0.2)
    assert metrics.fuse_eye_gaze(None, (0.2, 0.4)) == (0.2, 0.4)
    assert metrics.fuse_eye_gaze((0.2, 0.4), None) == (0.2, 0.4)
    assert metrics.fuse_eye_gaze(None, None) is None


def test_classify_centered_gaze():
    assert metrics.classify_centered_gaze(None) == "UNKNOWN"
    assert metrics.classify_centered_gaze((0.0, 0.0)) == "CENTER"
    assert metrics.classify_centered_gaze((0.2, 0.0)) == "RIGHT"
    assert metrics.classify_centered_gaze((-0.2, 0.0)) == "LEFT"
    assert metrics.classify_centered_gaze((0.0, 0.2)) == "DOWN"
    assert metrics.classify_centered_gaze((0.0, -0.2)) == "UP"
