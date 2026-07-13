from eyetracker.filters import OneEuroFilter, OneEuroFilter2D


def test_first_sample_returns_itself():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    assert f(5.0, 0.0) == 5.0


def test_constant_input_converges_to_constant():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = None
    for i in range(200):
        out = f(7.0, i / 30.0)
    assert abs(out - 7.0) < 1e-6


def test_smooths_jitter_around_a_mean():
    # Alternating noise around 0 should be pulled toward 0, not passed through.
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = None
    for i in range(100):
        noisy = 1.0 if i % 2 == 0 else -1.0
        out = f(noisy, i / 30.0)
    assert abs(out) < 0.5  # heavily damped toward the mean


def test_beta_makes_fast_motion_more_responsive():
    # Track a fast ramp; a higher beta should lag the moving target less.
    def track(beta):
        f = OneEuroFilter(min_cutoff=1.0, beta=beta)
        out = 0.0
        for i in range(30):
            out = f(float(i), i / 30.0)  # target moves +1 per step
        return out  # closer to the true value (29) is more responsive

    laggy = track(beta=0.0)
    responsive = track(beta=1.0)
    assert responsive > laggy


def test_2d_filter_smooths_both_axes_independently():
    f = OneEuroFilter2D(min_cutoff=1.0, beta=0.0)
    out = None
    for i in range(200):
        out = f((3.0, 8.0), i / 30.0)
    assert abs(out[0] - 3.0) < 1e-6
    assert abs(out[1] - 8.0) < 1e-6


def test_nonincreasing_timestamps_do_not_crash():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.2)
    f(1.0, 5.0)
    # Same/earlier timestamp should fall back to the default dt, not divide by zero.
    assert f(2.0, 5.0) == f(2.0, 5.0) or True  # just assert no exception raised
