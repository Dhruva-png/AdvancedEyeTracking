"""Central, tunable configuration for the tracking pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    log_interval_sec: float = 0.05          # ~20 Hz logging
    smoothing_alpha: float = 0.25           # exponential smoothing for iris centers (drawing only)
    gaze_feature_smoothing_alpha: float = 0.5  # light EMA on the head-invariant gaze feature; heavy smoothing is the One-Euro cursor filter
    blink_ear_threshold: float = 0.20       # eye-aspect-ratio below this counts as a blink
    blink_refractory_sec: float = 0.15      # minimum time between counted blinks (debounce)
    gaze_valid_ear_threshold: float = 0.24  # below this (but above a full blink), freeze gaze instead of trusting a half-closed eye
    gaze_center_dead_zone: float = 0.04     # centered-feature magnitude below which the HUD label reads CENTER

    heatmap_grid_size: tuple[int, int] = (64, 36)   # (cols, rows)
    heatmap_gaussian_sigma: float = 1.2

    dashboard_update_interval_sec: float = 0.2
    dashboard_window_sec: float = 30.0

    gaze_view_enabled_by_default: bool = False    # opt in with 'g', same as the dashboard, for a lighter startup
    gaze_view_update_interval_sec: float = 0.03   # ~30 Hz, kept snappy since it's the headline visual
    gaze_trail_length: int = 24
    # One-Euro cursor filter: lower min_cutoff = steadier at rest, higher beta = snappier on saccades.
    gaze_cursor_min_cutoff: float = 1.2
    gaze_cursor_beta: float = 0.5
    calibration_settle_sec: float = 0.5           # time to look toward a new dot before we start recording
    calibration_capture_sec: float = 1.0          # recording window per dot; samples are median-aggregated
    calibration_ridge_lambda: float = 1e-3        # L2 regularization on the quadratic gaze-mapping fit

    output_dir: str = "output"
    output_prefix: str = "eye_tracking_session"

    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
