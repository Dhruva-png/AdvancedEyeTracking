"""Optional non-blocking matplotlib dashboard: camera, heatmap, blink trend, and KPI cards.

Each panel updates its existing artist (`set_data`/`set_text`) instead of
clearing and re-plotting from scratch every frame, which is the standard way
to keep matplotlib's `FuncAnimation`-style live updates cheap.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from . import theme


class LiveDashboard:
    def __init__(self, window_sec: float = 30.0):
        self._window_sec = window_sec
        self._fig = None
        self._ax_cam = None
        self._ax_heat = None
        self._ax_blink = None
        self._ax_stats = None
        self._cam_img = None
        self._heat_img = None
        self._blink_line = None
        self._blink_fill = None
        self._stat_texts: dict[str, plt.Text] = {}
        self._t0 = time.time()
        self._blink_x: list[float] = []
        self._blink_y: list[int] = []

    def _style_axis(self, ax, title: str) -> None:
        ax.set_facecolor(theme.MPL_PANEL)
        ax.set_title(title, color=theme.MPL_TEXT, fontsize=11, fontweight="bold", loc="left")
        for spine in ax.spines.values():
            spine.set_color(theme.MPL_GRID)
        ax.tick_params(colors=theme.MPL_MUTED, labelsize=8)

    def _ensure_built(self, frame_rgb: np.ndarray, heatmap_vis: np.ndarray) -> None:
        if self._fig is not None:
            return
        plt.ion()
        self._fig = plt.figure(figsize=(13, 7.5), facecolor=theme.MPL_BG)
        try:
            self._fig.canvas.manager.set_window_title("Eye Tracking Dashboard")
        except AttributeError:
            pass
        grid = self._fig.add_gridspec(2, 2, hspace=0.35, wspace=0.22)

        self._ax_cam = self._fig.add_subplot(grid[0, 0])
        self._ax_heat = self._fig.add_subplot(grid[0, 1])
        self._ax_blink = self._fig.add_subplot(grid[1, 0])
        self._ax_stats = self._fig.add_subplot(grid[1, 1])

        self._style_axis(self._ax_cam, "LIVE CAMERA")
        self._ax_cam.axis("off")
        self._cam_img = self._ax_cam.imshow(frame_rgb)

        self._style_axis(self._ax_heat, "GAZE HEATMAP")
        self._ax_heat.axis("off")
        self._heat_img = self._ax_heat.imshow(heatmap_vis, cmap=theme.MPL_HEATMAP_CMAP, origin="lower", vmin=0, vmax=1)

        self._style_axis(self._ax_blink, f"BLINKS  (last {int(self._window_sec)}s)")
        self._ax_blink.set_xlabel("seconds", color=theme.MPL_MUTED, fontsize=8)
        self._ax_blink.grid(color=theme.MPL_GRID, linewidth=0.6, alpha=0.6)
        (self._blink_line,) = self._ax_blink.plot([], [], color=theme.MPL_ACCENT2, linewidth=2)
        self._blink_fill = self._ax_blink.fill_between([], [], color=theme.MPL_ACCENT2, alpha=0.15)
        self._ax_blink.set_xlim(0, self._window_sec)
        self._ax_blink.set_ylim(0, 10)

        self._build_stats_panel()

        self._fig.tight_layout()

    def _build_stats_panel(self) -> None:
        ax = self._ax_stats
        self._style_axis(ax, "SESSION")
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        cards = [
            ("blinks", "TOTAL BLINKS", 0.78),
            ("duration", "DURATION", 0.55),
            ("rate", "BLINK RATE /min", 0.32),
            ("gaze", "CURRENT GAZE", 0.09),
        ]
        for key, label, y in cards:
            ax.text(0.04, y + 0.11, label, color=theme.MPL_MUTED, fontsize=9, fontweight="bold", transform=ax.transAxes)
            self._stat_texts[key] = ax.text(
                0.04, y - 0.03, "--", color=theme.MPL_ACCENT, fontsize=20, fontweight="bold", transform=ax.transAxes
            )

    def update(self, frame_bgr: np.ndarray, heatmap_vis: np.ndarray, blink_count: int, gaze_label: str = "-") -> None:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._ensure_built(frame_rgb, heatmap_vis)

        self._cam_img.set_data(frame_rgb)
        self._heat_img.set_data(heatmap_vis)

        elapsed = time.time() - self._t0
        self._blink_x.append(elapsed)
        self._blink_y.append(blink_count)
        cutoff = elapsed - self._window_sec
        while self._blink_x and self._blink_x[0] < cutoff:
            self._blink_x.pop(0)
            self._blink_y.pop(0)

        self._blink_line.set_data(self._blink_x, self._blink_y)
        if self._blink_fill is not None:
            self._blink_fill.remove()
        self._blink_fill = self._ax_blink.fill_between(self._blink_x, self._blink_y, color=theme.MPL_ACCENT2, alpha=0.15)
        self._ax_blink.set_xlim(max(0, elapsed - self._window_sec), max(self._window_sec, elapsed))
        self._ax_blink.set_ylim(0, max(10, blink_count + 2))

        minutes, seconds = divmod(int(elapsed), 60)
        rate = blink_count / (elapsed / 60.0) if elapsed > 1 else 0.0
        self._stat_texts["blinks"].set_text(str(blink_count))
        self._stat_texts["duration"].set_text(f"{minutes:02d}:{seconds:02d}")
        self._stat_texts["rate"].set_text(f"{rate:.1f}")
        self._stat_texts["gaze"].set_text(gaze_label)

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
