"""Gaze Pop — a hands-free mini-game you play with your eyes.

Glowing orbs drift onto the screen; *look at one and hold your gaze* and a ring
fills around it — when it completes, the orb pops for points. Popping orbs in
quick succession builds a combo multiplier. A round lasts a fixed time and ends
on a score card.

Why dwell-to-pop rather than an instant "look = click": holding your gaze for
half a second is exactly the interaction that makes webcam gaze *feel* accurate.
Momentary jitter and single-frame tracking errors don't complete the dwell, so
the game reads as precise even though the underlying signal isn't pixel-perfect.
It's also the interaction real eye-controlled/accessibility UIs use.

This module is pure game logic — no OpenCV, no window, no drawing. It advances
state from a gaze point and a clock and emits `GameEvent`s the renderer turns
into particles/sound/etc. That keeps the rules deterministic and unit-testable
(inject a fake clock + seeded RNG) independent of any display.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

STATE_READY = "ready"
STATE_PLAYING = "playing"
STATE_OVER = "over"

_MAX_DT = 0.1          # clamp per-frame dt so a stall/pause can't fast-forward the game
_DWELL_DECAY = 2.0     # dwell drains this many times faster than it fills when you look away
_START_BUTTON_RADIUS = 0.11


class EventType(Enum):
    POP = "pop"
    START = "start"
    GAME_OVER = "game_over"
    EXPIRE = "expire"


@dataclass
class GameEvent:
    type: EventType
    x: float = 0.0          # normalized [0,1]
    y: float = 0.0
    value: int = 0          # points awarded (POP)
    combo: int = 0
    color_index: int = 0


@dataclass
class Target:
    x: float                # normalized [0,1]
    y: float
    radius_norm: float      # fraction of the smaller screen dimension
    born: float
    color_index: int = 0
    dwell: float = 0.0      # 0..1 fill progress
    is_button: bool = False


_PALETTE_SIZE = 5  # renderer maps these indices to colors


class GazePopGame:
    def __init__(
        self,
        config,
        width: int,
        height: int,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self.width = width
        self.height = height
        self._rng = rng or random.Random()
        self._clock = clock

        self.state = STATE_READY
        self.targets: list[Target] = []
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.popped = 0
        self.missed = 0
        self.time_left = float(config.game_duration_sec)

        self._last_update = clock()
        self._last_spawn = 0.0
        self._last_pop_time = -999.0
        self._make_start_button()

    # -- geometry ------------------------------------------------------------

    @property
    def _min_dim(self) -> int:
        return min(self.width, self.height)

    def _contains(self, target: Target, gaze: tuple[float, float]) -> bool:
        dx = (target.x - gaze[0]) * self.width
        dy = (target.y - gaze[1]) * self.height
        r = target.radius_norm * self._min_dim
        return dx * dx + dy * dy <= r * r

    # -- lifecycle -----------------------------------------------------------

    def _make_start_button(self) -> None:
        self.targets = [Target(0.5, 0.5, _START_BUTTON_RADIUS, self._clock(), color_index=0, is_button=True)]

    def start(self) -> None:
        self.state = STATE_PLAYING
        self.targets = []
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.popped = 0
        self.missed = 0
        self.time_left = float(self.config.game_duration_sec)
        now = self._clock()
        self._last_spawn = now
        self._last_pop_time = -999.0
        # Fill the board so orbs are immediately available (no dead gap while
        # the first one spawns), then top up on the interval as they're popped.
        for _ in range(self.config.game_max_targets):
            self._spawn_target(now)

    def _end(self) -> list[GameEvent]:
        self.state = STATE_OVER
        self.targets = [Target(0.5, 0.5, _START_BUTTON_RADIUS, self._clock(), color_index=0, is_button=True)]
        return [GameEvent(EventType.GAME_OVER)]

    # -- spawning ------------------------------------------------------------

    def _spawn_target(self, now: float) -> None:
        margin = self.config.game_target_radius * 1.5
        for _ in range(20):
            x = self._rng.uniform(margin, 1 - margin)
            y = self._rng.uniform(margin, 1 - margin)
            candidate = Target(x, y, self.config.game_target_radius, now, self._rng.randrange(_PALETTE_SIZE))
            if all(not self._too_close(candidate, t) for t in self.targets):
                self.targets.append(candidate)
                self._last_spawn = now
                return

    def _too_close(self, a: Target, b: Target) -> bool:
        dx = (a.x - b.x) * self.width
        dy = (a.y - b.y) * self.height
        min_gap = (a.radius_norm + b.radius_norm) * 1.4 * self._min_dim
        return dx * dx + dy * dy < min_gap * min_gap

    # -- main update ---------------------------------------------------------

    def update(self, gaze: tuple[float, float] | None) -> list[GameEvent]:
        now = self._clock()
        dt = max(0.0, min(_MAX_DT, now - self._last_update))
        self._last_update = now

        if self.state == STATE_PLAYING:
            return self._update_playing(gaze, now, dt)
        return self._update_button(gaze, dt)  # READY / OVER: dwell the center button

    def _update_button(self, gaze: tuple[float, float] | None, dt: float) -> list[GameEvent]:
        button = self.targets[0]
        if gaze is not None and self._contains(button, gaze):
            button.dwell += dt / self.config.game_dwell_sec
        else:
            button.dwell = max(0.0, button.dwell - dt / self.config.game_dwell_sec * _DWELL_DECAY)
        if button.dwell >= 1.0:
            self.start()
            return [GameEvent(EventType.START)]
        return []

    def _update_playing(self, gaze: tuple[float, float] | None, now: float, dt: float) -> list[GameEvent]:
        events: list[GameEvent] = []

        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0.0
            return self._end()

        # Combo lapses if you go too long between pops.
        if self.combo > 0 and (now - self._last_pop_time) > self.config.game_combo_window_sec:
            self.combo = 0

        if (
            len(self.targets) < self.config.game_max_targets
            and (now - self._last_spawn) >= self.config.game_spawn_interval_sec
        ):
            self._spawn_target(now)

        hovered = self._hovered_target(gaze)
        for target in list(self.targets):
            if target is hovered:
                target.dwell += dt / self.config.game_dwell_sec
                if target.dwell >= 1.0:
                    events.append(self._pop(target, now))
            else:
                target.dwell = max(0.0, target.dwell - dt / self.config.game_dwell_sec * _DWELL_DECAY)
                if (now - target.born) >= self.config.game_target_lifetime_sec:
                    self.targets.remove(target)
                    self.missed += 1
                    self.combo = 0
                    events.append(GameEvent(EventType.EXPIRE, target.x, target.y, color_index=target.color_index))
        return events

    def _hovered_target(self, gaze: tuple[float, float] | None) -> Target | None:
        if gaze is None:
            return None
        inside = [t for t in self.targets if self._contains(t, gaze)]
        if not inside:
            return None
        # If orbs overlap under the cursor, the closest center wins the dwell.
        return min(inside, key=lambda t: (t.x - gaze[0]) ** 2 + (t.y - gaze[1]) ** 2)

    def _pop(self, target: Target, now: float) -> GameEvent:
        if (now - self._last_pop_time) <= self.config.game_combo_window_sec:
            self.combo += 1
        else:
            self.combo = 1
        self._last_pop_time = now
        self.best_combo = max(self.best_combo, self.combo)
        points = 100 * self.combo
        self.score += points
        self.popped += 1
        self.targets.remove(target)
        return GameEvent(EventType.POP, target.x, target.y, value=points, combo=self.combo, color_index=target.color_index)

    # -- helpers for the renderer -------------------------------------------

    @property
    def accuracy(self) -> float:
        total = self.popped + self.missed
        return self.popped / total if total else 1.0
