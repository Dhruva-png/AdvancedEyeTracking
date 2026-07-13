"""Best-effort primary monitor resolution detection, with a safe fallback."""

from __future__ import annotations

FALLBACK_RESOLUTION = (1920, 1080)


def get_screen_resolution() -> tuple[int, int]:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        pass

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return FALLBACK_RESOLUTION
