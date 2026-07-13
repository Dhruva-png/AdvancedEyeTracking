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
  landmarks.py        MediaPipe landmark index groups (contour, iris, corners, lids)
  metrics.py           pure functions: EAR, head-invariant gaze feature, classification
  filters.py           One-Euro adaptive filter for the gaze cursor
  calibration.py       9-point calibration -> ridge quadratic map + outlier rejection
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

Logic is split from presentation: `metrics.py`, `filters.py`, `heatmap.py`,
and `calibration.py` are pure/deterministic and covered by unit tests in
`tests/`, independent of any camera or display hardware. `tracker.py`
extracts signals only — all drawing lives in `hud.py`/`gaze_view.py`.

## How accurate is it, really?

Honest answer: **a single webcam cannot do pixel-perfect gaze.** Commercial
eye trackers (Tobii, EyeLink) use infrared illumination and dedicated
sensors to reach ~0.5° of visual angle, which is still ~15–30 px at a normal
desk viewing distance. A standard RGB webcam has no depth or IR signal and is
noisier, so anyone claiming pixel-perfect gaze from a laptop camera is
mistaken. This project uses every technique that meaningfully closes the gap:

- **Head-invariant gaze feature.** Gaze is measured as the iris offset from
  the eye *corners*, in an eye-aligned frame, normalized by the corner-to-
  corner distance. That makes it invariant to camera distance (lean in/out)
  and head roll (tilt), and the corners are far steadier landmarks than a box
  fit to jittery contour points.
- **Quadratic calibration with ridge regularization + outlier rejection.**
  The eye-to-screen mapping isn't linear, so a 9-point grid fits a quadratic
  (6 coefficients/axis); ridge keeps a noisy fit from blowing up at the
  edges, and one clearly-bad calibration point (a blink or glance-away) is
  detected by its residual, dropped, and the fit redone.
- **Blink/squint gating.** A half-closed eye yields a geometrically
  meaningless iris position, so the signal freezes at the last good reading
  instead of lurching.
- **One-Euro adaptive cursor filter** (Casiez et al.) — rock-steady during a
  fixation, low-lag during a saccade, with no fixed-alpha compromise.

In a simulated session with a nonlinear eye model and realistic per-frame
noise, this lands roughly **20–40 px mean error** across the screen after
calibration. On real hardware, expect that ballpark **if your head stays
where it was when you calibrated** — the biggest remaining error source is
head movement (position/distance), so recalibrate (`c`) if you shift in your
seat. Filter feel (`gaze_cursor_min_cutoff`, `gaze_cursor_beta`) and all
thresholds live in `config.py` if you want to trade steadiness vs. lag.

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
during capture won't skew the result. It takes roughly 15–20 seconds; sit
still and keep your head in a natural, comfortable position throughout,
since the fit is tied to that head position/distance from the camera.
(See *How accurate is it, really?* above for the techniques behind the
mapping and cursor smoothing.)

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
