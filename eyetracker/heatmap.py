"""Accumulates normalized gaze points into a low-resolution grid and renders a smoothed heatmap."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


class GazeHeatmap:
    def __init__(self, grid_size: tuple[int, int], sigma: float):
        cols, rows = grid_size
        self._grid = np.zeros((rows, cols), dtype=np.float32)
        self._sigma = sigma

    def add(self, x_norm: float, y_norm: float) -> None:
        """x_norm/y_norm are gaze coordinates normalized to [0, 1]."""
        cols = self._grid.shape[1]
        rows = self._grid.shape[0]
        gx = int(np.clip(x_norm, 0.0, 1.0) * (cols - 1))
        gy = int(np.clip(y_norm, 0.0, 1.0) * (rows - 1))
        self._grid[gy, gx] += 1.0

    def raw(self) -> np.ndarray:
        return self._grid

    def normalized(self) -> np.ndarray:
        """Gaussian-smoothed, peak-normalized to [0, 1] — cheap enough to call once per render."""
        smoothed = gaussian_filter(self._grid, sigma=self._sigma)
        peak = smoothed.max()
        return smoothed / peak if peak > 0 else smoothed

    def save_png(self, path: str, cmap_name: str = "inferno") -> None:
        import matplotlib.pyplot as plt

        plt.imsave(path, self.normalized(), cmap=cmap_name)
