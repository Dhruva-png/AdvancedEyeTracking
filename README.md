# Advanced Eye Tracking

Real-time eye tracking, blink detection, and gaze heatmap generation from a
single webcam — built on [MediaPipe Face Mesh](https://developers.google.com/mediapipe)
iris landmarks instead of classic Haar-cascade eye detection.

## Features

- **Iris-accurate gaze tracking** via MediaPipe's refined face mesh (468 + iris landmarks)
- **Blink detection** using eye-aspect-ratio (EAR) with debouncing so a single blink isn't double-counted
- **Live gaze heatmap** rendered as a smoothed, normalized density grid
- **Optional live dashboard** — camera feed, heatmap, and blink-rate chart, all updating in place for low overhead
- **Session export** to CSV (raw samples), Excel (raw samples + heatmap matrix + summary), and PNG (heatmap image)

## Architecture

```
eyetracker/
  config.py         tunable parameters (dataclass)
  landmarks.py       MediaPipe landmark index groups
  metrics.py         pure functions: EAR, smoothing, gaze classification
  heatmap.py         gaze accumulation + gaussian-smoothed rendering
  tracker.py         per-frame MediaPipe processing -> FrameResult
  session_logger.py  in-memory sample buffer -> CSV/Excel/PNG export
  live_dashboard.py  optional matplotlib live view
  app.py             capture loop orchestration
main.py              CLI entry point
```

Logic is split from I/O: `metrics.py` and `heatmap.py` are pure/deterministic
and covered by unit tests in `tests/`, independent of any camera hardware.

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
| `h` | Toggle the live dashboard (camera + heatmap + blink chart) |
| `e` | Export the current session (CSV + Excel + heatmap PNG) |
| `q` | Quit and auto-export |

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
