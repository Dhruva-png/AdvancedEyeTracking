# Advanced Eye Tracking

Real-time eye tracking, blink detection, and gaze heatmap generation from a
single webcam — built on [MediaPipe Face Mesh](https://developers.google.com/mediapipe)
iris landmarks instead of classic Haar-cascade eye detection.

## Features

- **🎮 Gaze Pop mini-game** — a hands-free game you play with your eyes: look at an orb and hold your gaze to pop it, chain pops for combo multipliers, beat the clock. Press `p`. (See below.)
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
  calibration.py       9-pt + head-sweep calibration -> ridge quadratic map w/ head-pose compensation
  heatmap.py           gaze accumulation + gaussian-smoothed rendering
  tracker.py           per-frame MediaPipe processing -> FrameResult (no drawing)
  theme.py             shared color palette (BGR for cv2, hex for matplotlib)
  render.py            low-level cv2 drawing helpers (glow circles, panels, meters)
  hud.py               camera-window overlay (stateful: blink flash, FPS)
  gaze_view.py          virtual-monitor window showing the live gaze cursor
  game.py               Gaze Pop rules (pure: dwell/scoring/combo/state machine)
  game_view.py          Gaze Pop renderer (orbs, particles, HUD, score card)
  screen_utils.py       primary monitor resolution detection
  session_logger.py     in-memory sample buffer -> CSV/Excel/PNG export
  live_dashboard.py      optional matplotlib live view
  app.py                capture loop orchestration
main.py                 CLI entry point
```

Logic is split from presentation: `metrics.py`, `filters.py`, `heatmap.py`,
`calibration.py`, and `game.py` are pure/deterministic and covered by unit
tests in `tests/`, independent of any camera or display hardware. `tracker.py`
extracts signals only — all drawing lives in the `*_view`/`hud` modules.

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
- **Head-pose (yaw/pitch) compensation.** The corner-normalized feature
  cancels head *translation, depth, and roll* but not *yaw/pitch* — turn or
  nod your head and "eyes centered in their sockets" lands somewhere else on
  screen. Calibration adds a short head-sweep phase (fixate the center dot,
  move your head around): because your eyes counter-rotate to hold the target
  (the vestibulo-ocular reflex), the eye feature and head pose co-vary while
  the true target stays fixed — which is exactly the data needed to separate
  "eye moved" from "head moved." The runtime model then corrects for head
  pose relative to the calibration pose. Head pose comes from a cheap,
  intrinsics-free proxy (nose position relative to the eye corners in the
  eye-aligned frame). If you skip or barely move during the sweep, it detects
  the missing signal and cleanly falls back to the eye-only map.
- **Blink/squint gating.** A half-closed eye yields a geometrically
  meaningless iris position, so the signal freezes at the last good reading
  instead of lurching.
- **One-Euro adaptive cursor filter** (Casiez et al.) — rock-steady during a
  fixation, low-lag during a saccade, with no fixed-alpha compromise.

In a simulated session with a nonlinear eye model and realistic per-frame
noise, this lands roughly **10–40 px mean error** across the screen after
calibration. The head-pose compensation is the difference between error that
*stays* around ~11 px as you turn your head and error that climbs past
**250 px** at a 0.15 yaw offset without it (measured in the test simulation).
It is not magic — hold your head roughly within the range you swept during
calibration, and recalibrate (`c`) if you move to a very different position.
Filter feel (`gaze_cursor_min_cutoff`, `gaze_cursor_beta`), head compensation
(`head_pose_compensation`), and all thresholds live in `config.py`.

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
| `p` | Play **Gaze Pop** (the mini-game) |
| `h` | Toggle the live matplotlib dashboard |
| `g` | Toggle the Gaze View window (on-screen gaze cursor) |
| `c` | Start calibration |
| `e` | Export the current session (CSV + Excel + heatmap PNG) |
| `q` | Quit and auto-export |

The Gaze View window works out of the box in an uncalibrated, best-effort
mode, but accuracy improves substantially after calibrating (`c`) for your
current seating position — recalibrate whenever you move (or if lighting
changes a lot).

**Calibrating** (about 20–25 seconds, two stages):
1. **9 gaze dots** — for each, a brief "get ready" pause (dim steady ring)
   then a "hold still, capturing" window (pulsing ring). Only the pulsing
   window counts, and samples are median-aggregated per point, so a stray
   blink or saccade won't skew a dot. Keep your head still for this stage.
2. **Head sweep** — a magenta center dot appears: keep looking at it while
   slowly moving your head left, right, up, and down. This is what teaches
   the system to compensate for head movement (see *How accurate is it,
   really?* above). Move smoothly through a comfortable range — that range
   is where tracking will stay accurate afterward.

Optional flags:

```bash
python main.py --camera 1 --width 1920 --height 1080 --output-dir output
```

Exports are written to `output/` by default (git-ignored).

## 🎮 Gaze Pop

A tiny arcade game you play entirely with your eyes — no mouse, no keyboard
during play. Press `p` from the running app to open it in its own window.

**How to play:** glowing orbs appear on screen. **Look at an orb and hold your
gaze** — a ring fills around it, and when it completes, the orb pops for points.
Pop orbs in quick succession to build a **combo multiplier** (x2, x3, …) for
bigger scores. You have 45 seconds; the round ends on a score card showing your
points, orbs popped, best combo, and accuracy. It's fully hands-free: dwell on
the center orb to start and to replay.

**Why dwell-to-pop?** Holding your gaze for half a second is what makes webcam
gaze *feel* precise — momentary jitter or a single bad frame never completes a
dwell, so the game reads as accurate even though no webcam is pixel-perfect.
(It's also the same interaction real eye-controlled and accessibility UIs use.)
Because of that, Gaze Pop is fun even **uncalibrated**, but pressing `c` to
calibrate first makes it noticeably sharper.

**Showing it off:** calibrate (`c`), press `p`, play a round, and screenshot
the end card — that's the LinkedIn shot. To record a clip, capture the Gaze Pop
window (and optionally the camera window beside it so people can see your eyes
driving it).

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
