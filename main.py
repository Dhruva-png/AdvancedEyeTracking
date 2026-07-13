"""CLI entry point for the eye-tracking + gaze-heatmap application."""

import argparse
import logging

from eyetracker.app import Application
from eyetracker.config import TrackerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time eye tracking with blink detection and gaze heatmap.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for exported logs/heatmaps")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    config = TrackerConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        output_dir=args.output_dir,
    )
    Application(config).run()


if __name__ == "__main__":
    main()
