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

```bash
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
this is a known incompatibility between `mediapipe` and `protobuf>=5`. Make
sure you installed from `requirements.txt` (which pins `protobuf<4.25`) inside
a fresh virtual environment, rather than reusing one with a newer protobuf
already installed.

## License

[MIT](LICENSE)
