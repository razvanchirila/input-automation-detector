"""Detector protocol and the numeric view that every detector consumes.

Detectors never touch pydantic models directly. They read a `SessionArrays`, a
flat numpy view derived from the raw event stream. This keeps the detectors
cheap enough to run over a 10k-session corpus and guarantees that the evaluation
path and the API path see byte-identical features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from iad.models import DOWN_TYPES, UP_TYPES, EventType, Session

# A cursor stroke ends when the pointer rests for longer than this. Chosen to sit
# above typical intra-stroke sampling (8-25 ms) and below human reaction pauses.
STROKE_GAP_S = 0.12

# Strokes shorter than this carry no usable curvature information.
MIN_STROKE_POINTS = 6

# Neutral score returned when a detector has too little data to say anything.
NEUTRAL = 0.5


@dataclass(frozen=True)
class SessionArrays:
    """Numeric view of a session.

    Attributes:
        t_move, mx, my: cursor samples, ordered by time.
        t_action: timestamps of press events (key_down / mouse_down) only. This is
            the semantic "action rhythm"; cursor sampling rate is deliberately
            excluded because it is a property of the capture layer, not the actor.
        dwell_ms: press durations, in milliseconds, matched down->up per code.
        n_events: total raw events, for reporting.
    """

    t_move: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    t_action: np.ndarray
    dwell_ms: np.ndarray
    n_events: int

    def strokes(self) -> list[tuple[int, int]]:
        """Segment cursor samples into strokes by resting gaps.

        Returns half-open [start, end) index pairs with at least MIN_STROKE_POINTS
        samples. Segmentation is re-derived from timestamps rather than trusted
        from the producer, so synthetic and real sessions are treated the same.
        """
        n = self.t_move.size
        if n < MIN_STROKE_POINTS:
            return []
        gaps = np.flatnonzero(np.diff(self.t_move) > STROKE_GAP_S) + 1
        bounds = np.concatenate(([0], gaps, [n]))
        return [
            (int(a), int(b))
            for a, b in zip(bounds[:-1], bounds[1:])
            if b - a >= MIN_STROKE_POINTS
        ]


@runtime_checkable
class Detector(Protocol):
    """A single independent signal.

    Implementations must be pure and stateless: the same SessionArrays must always
    yield the same score, in [0, 1], higher meaning more bot-like.
    """

    name: str

    def score(self, s: SessionArrays) -> float: ...


def clamp01(v: float) -> float:
    if not np.isfinite(v):
        return NEUTRAL
    return float(min(1.0, max(0.0, v)))


def ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0->1 ramp: `lo` maps to 0, `hi` maps to 1. Works in both directions."""
    if hi == lo:
        return NEUTRAL
    return clamp01((value - lo) / (hi - lo))


def band_deviation(value: float, center: float, halfwidth: float) -> float:
    """Two-sided deviation from a human-typical band, scaled to [0, 1].

    Used where "unlike a human" is not the same as "smaller than a human": a path
    with no direction reversals and a path made of pure high-frequency noise are
    both non-human, in opposite directions.
    """
    if halfwidth <= 0:
        return NEUTRAL
    return clamp01(abs(value - center) / halfwidth)


def from_session(session: Session) -> SessionArrays:
    """Project a validated Session onto the numeric view the detectors read."""
    t_move: list[float] = []
    mx: list[float] = []
    my: list[float] = []
    t_action: list[float] = []
    dwell_ms: list[float] = []
    pending: dict[str, float] = {}

    for ev in session.events:
        if ev.type is EventType.MOUSE_MOVE:
            t_move.append(ev.t)
            mx.append(ev.x)  # type: ignore[arg-type]
            my.append(ev.y)  # type: ignore[arg-type]
        elif ev.type in DOWN_TYPES:
            t_action.append(ev.t)
            pending[ev.code] = ev.t  # type: ignore[index]
        elif ev.type in UP_TYPES:
            down_t = pending.pop(ev.code, None)  # type: ignore[arg-type]
            if down_t is not None:
                dwell_ms.append((ev.t - down_t) * 1000.0)

    return SessionArrays(
        t_move=np.asarray(t_move, dtype=np.float64),
        mx=np.asarray(mx, dtype=np.float64),
        my=np.asarray(my, dtype=np.float64),
        t_action=np.asarray(t_action, dtype=np.float64),
        dwell_ms=np.asarray(dwell_ms, dtype=np.float64),
        n_events=len(session.events),
    )
