"""One-Euro adaptive low-pass filter (Casiez, Roussel & Vogel, CHI 2012).

A fixed-alpha exponential average forces a single bad tradeoff: smooth enough
to kill jitter when the eye is still, or responsive enough to keep up when the
eye moves fast — never both. The One-Euro filter adapts its cutoff to the
signal's speed: heavy smoothing at rest (jitter disappears), light smoothing
during fast movement (no lag). It is the de-facto standard for exactly this
kind of interactive-pointer smoothing, which is why it replaces the previous
fixed-alpha EMA + manual jump-damping on the gaze cursor.
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _LowPass:
    def __init__(self) -> None:
        self._value: float | None = None

    @property
    def initialized(self) -> bool:
        return self._value is not None

    def __call__(self, x: float, alpha: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = alpha * x + (1.0 - alpha) * self._value
        return self._value


class OneEuroFilter:
    """Scalar One-Euro filter.

    - min_cutoff: lower => smoother at rest (more jitter removed), but laggier.
    - beta:       higher => more responsive to fast motion (less lag on saccades).
    - d_cutoff:   cutoff for the derivative estimate; 1.0 is a sensible default.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0, default_dt: float = 1.0 / 30.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.default_dt = default_dt
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_time: float | None = None
        self._last_x: float | None = None

    def __call__(self, x: float, timestamp: float) -> float:
        if self._last_time is None or timestamp <= self._last_time:
            dt = self.default_dt
        else:
            dt = timestamp - self._last_time
        self._last_time = timestamp

        dx = 0.0 if self._last_x is None else (x - self._last_x) / dt
        self._last_x = x
        edx = self._dx(dx, _alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, _alpha(cutoff, dt))


class OneEuroFilter2D:
    """Independent One-Euro filters on x and y, sharing parameters."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0, default_dt: float = 1.0 / 30.0):
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff, default_dt)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff, default_dt)

    def __call__(self, point: tuple[float, float], timestamp: float) -> tuple[float, float]:
        return self._fx(point[0], timestamp), self._fy(point[1], timestamp)
