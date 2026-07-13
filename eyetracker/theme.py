"""Shared visual language for the camera HUD, gaze view, and dashboard.

All colors are OpenCV-style BGR tuples (not RGB) since every consumer of this
module draws with `cv2`. Matplotlib consumers convert with `bgr_to_mpl`.
"""

from __future__ import annotations

BG_DEEP = (26, 22, 18)          # near-black navy panel background
BG_PANEL = (46, 36, 26)         # slightly lighter panel fill
GRID_LINE = (58, 48, 38)

CYAN = (238, 209, 30)           # primary accent
CYAN_SOFT = (200, 170, 60)
MAGENTA = (176, 46, 214)        # secondary accent (blink flash, gaze cursor)
AMBER = (26, 178, 255)          # warnings / EAR-low
GREEN = (129, 234, 108)         # calibrated / good status
TEXT_PRIMARY = (240, 240, 245)
TEXT_MUTED = (150, 150, 160)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, kept numeric to avoid importing cv2 here

# Gaze Pop orb palette (BGR). Indices match Target.color_index.
GAME_ORB_COLORS = [
    (238, 209, 30),    # cyan
    (176, 46, 214),    # magenta
    (129, 234, 108),   # green
    (26, 178, 255),    # amber
    (255, 128, 90),    # periwinkle
]


def bgr_to_mpl(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert a BGR 0-255 tuple to an RGB 0-1 tuple for matplotlib."""
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


# Matplotlib dark-dashboard palette (hex, since matplotlib is happiest with hex/RGB).
MPL_BG = "#12100e"
MPL_PANEL = "#1c1815"
MPL_GRID = "#332c24"
MPL_ACCENT = "#f2c93c"
MPL_ACCENT2 = "#d62eb0"
MPL_GOOD = "#6cea81"
MPL_TEXT = "#f0f0f5"
MPL_MUTED = "#8a8a94"
MPL_HEATMAP_CMAP = "magma"
