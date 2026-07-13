"""Optional non-blocking matplotlib dashboard: camera feed + gaze heatmap + blink rate.

Each panel updates its existing artist (`set_data`) instead of clearing and
re-plotting from scratch every frame, which is the standard way to keep
matplotlib's `FuncAnimation`-style live updates cheap.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np


class LiveDashboard:
    def __init__(self, window_sec: float = 30.0):
        self._window_sec = window_sec
        self._fig = None
        self._ax_cam = None
        self._ax_heat = None
        self._ax_blink = None
        self._cam_img = None
        self._heat_img = None
        self._blink_line = None
        self._t0 = time.time()
        self._blink_x: list[float] = []
        self._blink_y: list[int] = []

    def _ensure_built(self, frame_rgb: np.ndarray, heatmap_vis: np.ndarray) -> None:
        if self._fig is not None:
            return
        plt.ion()
        self._fig, (self._ax_cam, self._ax_heat, self._ax_blink) = plt.subplots(1, 3, figsize=(14, 4))

        self._ax_cam.set_title("Camera")
        self._ax_cam.axis("off")
        self._cam_img = self._ax_cam.imshow(frame_rgb)

        self._ax_heat.set_title("Gaze Heatmap")
        self._ax_heat.axis("off")
        self._heat_img = self._ax_heat.imshow(heatmap_vis, cmap="inferno", origin="lower", vmin=0, vmax=1)

        self._ax_blink.set_title(f"Blinks (last {int(self._window_sec)}s)")
        self._ax_blink.set_xlabel("seconds")
        self._ax_blink.set_ylabel("blink count")
        (self._blink_line,) = self._ax_blink.plot([], [], "-o")
        self._ax_blink.set_xlim(0, self._window_sec)
        self._ax_blink.set_ylim(0, 10)

        self._fig.tight_layout()

    def update(self, frame_bgr: np.ndarray, heatmap_vis: np.ndarray, blink_count: int) -> None:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._ensure_built(frame_rgb, heatmap_vis)

        self._cam_img.set_data(frame_rgb)
        self._heat_img.set_data(heatmap_vis)

        now = time.time() - self._t0
        self._blink_x.append(now)
        self._blink_y.append(blink_count)
        cutoff = now - self._window_sec
        while self._blink_x and self._blink_x[0] < cutoff:
            self._blink_x.pop(0)
            self._blink_y.pop(0)

        self._blink_line.set_data(self._blink_x, self._blink_y)
        self._ax_blink.set_xlim(max(0, now - self._window_sec), max(self._window_sec, now))
        self._ax_blink.set_ylim(0, max(10, blink_count + 2))

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
