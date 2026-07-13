# Advanced Eye Tracking

Real-time eye tracking, blink detection, and gaze heatmap generation from a
single webcam — built on [MediaPipe Face Mesh](https://developers.google.com/mediapipe)
iris landmarks instead of classic Haar-cascade eye detection.

## Features

- **Iris-accurate gaze tracking** via MediaPipe's refined face mesh (468 + iris landmarks)
- **Blink detection** using eye-aspect-ratio (EAR) with debouncing so a single blink isn't double-counted
- **On-screen gaze cursor** — a glowing circle in a dedicated "Gaze View" window that follows where you're looking, backed by a 9-point calibration that fits a per-user quadratic mapping from raw eye position to screen coordinates
- **Polished camera HUD** — glowing iris markers, smooth eye outlines, a translucent stat panel (gaze, blinks, EAR meter, FPS, calibration status), and a blink flash effect
- **Live gaze heatmap** rendered as a smoothed, normalized density grid
- **Optional live dashboard** — dark-themed camera/heatmap/blink-trend/KPI-card grid, all panels updating in place for low overhead
- **Session export** to CSV (raw samples), Excel (raw samples + heatmap matrix + summary), and PNG (heatmap image)

## Architecture

```
eyetracker/
  config.py          tunable parameters (dataclass)
  landmarks.py        MediaPipe landmark index groups
  metrics.py           pure functions: EAR, smoothing, gaze offset/classification
  calibration.py       9-point calibration -> ridge-regularized quadratic raw-gaze-to-screen mapping
  heatmap.py           gaze accumulation + gaussian-smoothed rendering
  tracker.py           per-frame MediaPipe processing -> FrameResult (no drawing)
  theme.py             shared color palette (BGR for cv2, hex for matplotlib)
  render.py            low-level cv2 drawing helpers (glow circles, panels, meters)
  hud.py               camera-window overlay (stateful: blink flash, FPS)
  gaze_view.py          virtual-monitor window showing the live gaze cursor
  screen_utils.py       primary monitor resolution detection
  session_logger.py     in-memory sample buffer -> CSV/Excel/PNG export
  live_dashboard.py      optional matplotlib live view
  app.py                capture loop orchestration
main.py                 CLI entry point
```

Logic is split from presentation: `metrics.py`, `heatmap.py`, and
`calibration.py` are pure/deterministic and covered by unit tests in
`tests/`, independent of any camera or display hardware. `tracker.py`
extracts signals only — all drawing lives in `hud.py`/`gaze_view.py`.

**Design note on the gaze cursor:** rather than an OS-level always-on-top
transparent overlay (fragile, platform-specific, and requires broader window-
manager access), the gaze cursor renders onto its own resizable "Gaze View"
window sized proportionally to your real screen resolution. Drag it onto
your monitor and resize freely — the gaze point is stored as a 0–1 fraction,
so accuracy doesn't depend on the window's pixel size.

## Install

Use a dedicated virtual environment — this project pins exact versions of
`mediapipe`/`protobuf` that can conflict with other projects sharing your
global Python install.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

| Key | Action |
|-----|--------|
| `h` | Toggle the live matplotlib dashboard |
| `g` | Toggle the Gaze View window (on-screen gaze cursor) |
| `c` | Start 9-point calibration |
| `e` | Export the current session (CSV + Excel + heatmap PNG) |
| `q` | Quit and auto-export |

The Gaze View window works out of the box in an uncalibrated, best-effort
mode, but accuracy improves substantially after calibrating (`c`) for your
current seating position — recalibrate whenever you move (or if lighting
changes a lot).

**Calibrating:** for each of the 9 dots, there's a brief "get ready" pause
(the ring is dim and steady) followed by a "hold still, capturing" window
(the ring pulses) — only samples from the pulsing window count, and they're
median-aggregated per point before fitting, so a stray blink or saccade
during capture won't skew the result. It takes roughly 15-20 seconds; sit
still and keep your head in a natural, comfortable position throughout,
since the fit is calibrated to that head position/distance from the
camera specifically.

**Why 9 points and not 5:** the iris's position within its eye socket
doesn't move linearly with on-screen gaze position even for a still head,
so a 5-point *linear* fit was measurably inaccurate away from the
calibration points — it's exact at the 5 dots and increasingly wrong
in between and past them, which reads as "jumpy and inaccurate" cursor
behavior. The mapping is now a ridge-regularized *quadratic* fit
(6 coefficients per axis instead of 3), and 9 well-spread points give it
enough data to be well-conditioned across the whole screen instead of
extrapolating from just the center and 4 corners.

**Why the cursor doesn't jump as much anymore:** two upstream signal fixes,
plus a cursor-side safety net:
- The eye bounding box (used to compute where the iris sits *within* the
  eye) is now smoothed frame-to-frame. Previously only the iris center was
  smoothed while the box itself was recomputed raw every frame — since the
  gaze ratio *divides by* the box's width/height, a small amount of raw
  landmark jitter in the box was being amplified into a much larger jump
  in the ratio.
- Gaze estimation now freezes during a blink or squint (EAR below a
  "still usable" threshold) instead of computing a position from a
  half-closed eye, which is geometrically meaningless.
- As a last line of defense, the Gaze View damps any single-frame jump
  larger than a plausible saccade instead of snapping the cursor there —
  legitimate fast eye movements still arrive, just over 2-3 frames.

Optional flags:

```bash
python main.py --camera 1 --width 1920 --height 1080 --output-dir output
```

Exports are written to `output/` by default (git-ignored).

## Tests

```bash
pip install pytest
pytest
```

## Troubleshooting

**`Failed to parse: node { ... }` when MediaPipe's `FaceMesh` is constructed** —
incompatibility between `mediapipe` and `protobuf>=5`. Install from
`requirements.txt` (which pins `protobuf<5`) inside a fresh virtual
environment rather than reusing one with a newer protobuf already installed.

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`** — recent
`mediapipe` releases (0.10.15+) removed the legacy `mp.solutions.*` API this
project uses in favor of the newer Tasks API. Install the exact pinned
version from `requirements.txt` (`mediapipe==0.10.14`) rather than
`pip install mediapipe`, which grabs the latest release.

## License

[MIT](LICENSE)
