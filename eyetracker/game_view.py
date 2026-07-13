"""Renders Gaze Pop to its own window: glowing orbs with dwell rings, particle
bursts, floating score pops, a live HUD, and a screenshot-ready title/end card.

All drawing lives here; the rules live in game.py. This class owns the cv2
window and a `GazePopGame` sized to its canvas, and exposes a single `step`
that advances the game one frame and paints it.
"""

from __future__ import annotations

import math
import random
import time

import cv2
import numpy as np

from . import render, theme
from .game import STATE_OVER, STATE_PLAYING, STATE_READY, EventType, GazePopGame

WINDOW_NAME = "Gaze Pop"
_MAX_CANVAS_WIDTH = 1280


class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "color", "size")

    def __init__(self, x, y, vx, vy, life, color, size):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = life
        self.color = color
        self.size = size


class _FloatingText:
    __slots__ = ("x", "y", "text", "life", "color")

    def __init__(self, x, y, text, color):
        self.x, self.y, self.text, self.life, self.color = x, y, text, 1.0, color


class GazePopView:
    def __init__(self, screen_resolution: tuple[int, int], config, clock=time.time):
        w, h = screen_resolution
        if w > _MAX_CANVAS_WIDTH:
            scale = _MAX_CANVAS_WIDTH / w
            w, h = int(w * scale), int(h * scale)
        self.canvas_size = (w, h)
        self.config = config
        self._clock = clock
        self.game = GazePopGame(config, w, h, rng=random.Random(), clock=clock)
        self._particles: list[_Particle] = []
        self._floats: list[_FloatingText] = []
        self._cursor: tuple[float, float] | None = None
        self._last_effect_time = clock()
        self._window_created = False
        self._static_bg = self._build_background()

    # -- window --------------------------------------------------------------

    def _ensure_window(self) -> None:
        if not self._window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, *self.canvas_size)
            self._window_created = True

    def close(self) -> None:
        if self._window_created:
            cv2.destroyWindow(WINDOW_NAME)
            self._window_created = False

    def _build_background(self) -> np.ndarray:
        w, h = self.canvas_size
        # Soft radial vignette from a deep center — cheap to precompute once.
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(cx, cy)
        vign = np.clip(1.0 - dist * 0.9, 0.25, 1.0)[:, :, None]
        base = np.array(theme.BG_PANEL, dtype=np.float32)
        deep = np.array(theme.BG_DEEP, dtype=np.float32)
        canvas = (deep + (base - deep) * vign).astype(np.uint8)
        return np.ascontiguousarray(canvas)

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.canvas_size
        return int(x * w), int(y * h)

    # -- main step -----------------------------------------------------------

    def step(self, gaze_norm: tuple[float, float] | None) -> np.ndarray:
        self._ensure_window()
        if gaze_norm is not None:
            self._cursor = gaze_norm

        events = self.game.update(gaze_norm)
        for ev in events:
            self._on_event(ev)

        canvas = self._static_bg.copy()
        self._draw_targets(canvas)
        self._update_and_draw_particles(canvas)
        self._update_and_draw_floats(canvas)
        self._draw_cursor(canvas)

        if self.game.state == STATE_READY:
            self._draw_ready_overlay(canvas)
        elif self.game.state == STATE_PLAYING:
            self._draw_hud(canvas)
        elif self.game.state == STATE_OVER:
            self._draw_gameover_card(canvas)

        cv2.imshow(WINDOW_NAME, canvas)
        return canvas

    # -- events --------------------------------------------------------------

    def _on_event(self, ev) -> None:
        if ev.type is EventType.POP:
            color = theme.GAME_ORB_COLORS[ev.color_index % len(theme.GAME_ORB_COLORS)]
            self._burst(ev.x, ev.y, color, count=22)
            label = f"+{ev.value}" + (f"  x{ev.combo}" if ev.combo > 1 else "")
            self._floats.append(_FloatingText(ev.x, ev.y, label, theme.TEXT_PRIMARY))
        elif ev.type is EventType.EXPIRE:
            color = theme.GAME_ORB_COLORS[ev.color_index % len(theme.GAME_ORB_COLORS)]
            self._burst(ev.x, ev.y, color, count=6, speed=0.06)

    def _burst(self, x: float, y: float, color, count: int, speed: float = 0.12) -> None:
        for _ in range(count):
            ang = random.uniform(0, 2 * math.pi)
            mag = random.uniform(0.2, 1.0) * speed
            self._particles.append(
                _Particle(x, y, math.cos(ang) * mag, math.sin(ang) * mag,
                          life=random.uniform(0.5, 1.0), color=color, size=random.uniform(2, 5))
            )

    # -- effects rendering ---------------------------------------------------

    def _effect_dt(self) -> float:
        now = self._clock()
        dt = min(0.1, max(0.0, now - self._last_effect_time))
        self._last_effect_time = now
        return dt

    def _update_and_draw_particles(self, canvas: np.ndarray) -> None:
        dt = self._effect_dt()
        w, h = self.canvas_size
        alive: list[_Particle] = []
        for p in self._particles:
            p.life -= dt * 1.5
            if p.life <= 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += 0.25 * dt  # gravity
            px, py = int(p.x * w), int(p.y * h)
            fade = max(0.0, min(1.0, p.life))
            col = tuple(int(c * fade) for c in p.color)
            cv2.circle(canvas, (px, py), max(1, int(p.size * fade)), col, -1, lineType=cv2.LINE_AA)
            alive.append(p)
        self._particles = alive

    def _update_and_draw_floats(self, canvas: np.ndarray) -> None:
        dt = 0.016
        alive: list[_FloatingText] = []
        for f in self._floats:
            f.life -= dt * 1.1
            if f.life <= 0:
                continue
            f.y -= 0.10 * dt
            px, py = self._to_px(f.x, f.y)
            scale = 0.7 + 0.3 * (1 - f.life)
            render.draw_text(canvas, f.text, (px - 24, py - 18), f.color, scale=scale, thickness=2)
            alive.append(f)
        self._floats = alive

    def _draw_targets(self, canvas: np.ndarray) -> None:
        for t in self.game.targets:
            color = theme.GAME_ORB_COLORS[t.color_index % len(theme.GAME_ORB_COLORS)]
            px, py = self._to_px(t.x, t.y)
            r = int(t.radius_norm * min(self.canvas_size))
            pulse = 0.85 + 0.15 * math.sin(self._clock() * 4 + t.x * 10)
            render.draw_glow_circle(canvas, (px, py), int(r * pulse), color, layers=5)
            if t.dwell > 0:
                self._draw_dwell_ring(canvas, (px, py), r + 8, t.dwell, color)
            if t.is_button:
                render.draw_text(canvas, "LOOK", (px - 26, py + 6), theme.TEXT_PRIMARY, scale=0.6, thickness=2)

    @staticmethod
    def _draw_dwell_ring(canvas, center, radius, progress, color) -> None:
        progress = max(0.0, min(1.0, progress))
        ring_color = theme.GREEN if progress > 0.85 else color
        cv2.ellipse(canvas, center, (radius, radius), -90, 0, int(360 * progress), ring_color, 4, lineType=cv2.LINE_AA)

    def _draw_cursor(self, canvas: np.ndarray) -> None:
        if self._cursor is None:
            return
        px, py = self._to_px(*self._cursor)
        cv2.circle(canvas, (px, py), 7, theme.TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 2, theme.TEXT_PRIMARY, -1, lineType=cv2.LINE_AA)

    # -- overlays ------------------------------------------------------------

    def _draw_hud(self, canvas: np.ndarray) -> None:
        w, h = self.canvas_size
        render.draw_rounded_panel(canvas, (16, 14), (230, 92), theme.BG_DEEP, alpha=0.55)
        render.draw_text(canvas, "SCORE", (32, 40), theme.TEXT_MUTED, scale=0.45)
        render.draw_text(canvas, f"{self.game.score:,}", (32, 78), theme.CYAN, scale=1.1, thickness=3)

        if self.game.combo > 1:
            render.draw_text(canvas, f"COMBO x{self.game.combo}", (w // 2 - 60, 44), theme.MAGENTA, scale=0.7, thickness=2)

        # Time bar across the top-right.
        frac = self.game.time_left / max(1e-6, self.config.game_duration_sec)
        bar_color = theme.GREEN if frac > 0.3 else theme.AMBER
        render.draw_text(canvas, f"{self.game.time_left:0.0f}s", (w - 70, 40), theme.TEXT_PRIMARY, scale=0.6, thickness=2)
        render.draw_meter_bar(canvas, (w - 250, 52), (234, 12), frac, bar_color)

    def _draw_ready_overlay(self, canvas: np.ndarray) -> None:
        w, h = self.canvas_size
        render.draw_text(canvas, "GAZE POP", (w // 2 - 150, h // 2 - 140), theme.CYAN, scale=2.2, thickness=5)
        render.draw_text(
            canvas, "Look at an orb and hold your gaze to pop it. Chain pops for combos!",
            (w // 2 - 300, h // 2 - 96), theme.TEXT_PRIMARY, scale=0.66, thickness=1,
        )
        render.draw_text(
            canvas, "Look at the orb below to start", (w // 2 - 165, h // 2 + 150),
            theme.TEXT_MUTED, scale=0.6, thickness=1,
        )
        render.draw_text(
            canvas, "Tip: press  c  first to calibrate for best accuracy",
            (w // 2 - 220, h - 30), theme.TEXT_MUTED, scale=0.5, thickness=1,
        )

    def _draw_gameover_card(self, canvas: np.ndarray) -> None:
        w, h = self.canvas_size
        # Dim the field so the card reads cleanly in a screenshot.
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), theme.BG_DEEP, -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)

        cw, ch = 520, 340
        x0, y0 = w // 2 - cw // 2, h // 2 - ch // 2
        render.draw_rounded_panel(canvas, (x0, y0), (cw, ch), theme.BG_PANEL, alpha=0.92, radius=22)

        render.draw_text(canvas, "TIME!", (w // 2 - 66, y0 + 62), theme.MAGENTA, scale=1.4, thickness=3)
        render.draw_text(canvas, f"{self.game.score:,}", (w // 2 - _text_half(f"{self.game.score:,}", 2.4), y0 + 150),
                         theme.CYAN, scale=2.4, thickness=6)
        render.draw_text(canvas, "POINTS", (w // 2 - 40, y0 + 178), theme.TEXT_MUTED, scale=0.5)

        stats = [
            ("Popped", f"{self.game.popped}"),
            ("Best combo", f"x{self.game.best_combo}"),
            ("Accuracy", f"{self.game.accuracy * 100:0.0f}%"),
        ]
        col_w = cw // 3
        for i, (label, value) in enumerate(stats):
            cx = x0 + col_w * i + col_w // 2
            render.draw_text(canvas, value, (cx - _text_half(value, 0.85), y0 + 232), theme.TEXT_PRIMARY, scale=0.85, thickness=2)
            render.draw_text(canvas, label, (cx - _text_half(label, 0.45), y0 + 258), theme.TEXT_MUTED, scale=0.45)

        render.draw_text(canvas, "Look at the orb to play again", (w // 2 - 165, y0 + ch - 20),
                         theme.TEXT_MUTED, scale=0.55)
        render.draw_text(canvas, "Gaze Pop  ·  built with Python · MediaPipe · OpenCV",
                         (w // 2 - 230, h - 24), theme.TEXT_MUTED, scale=0.5)


def _text_half(text: str, scale: float) -> int:
    """Half the pixel width of `text` at the given font scale, for centering."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    return tw // 2
