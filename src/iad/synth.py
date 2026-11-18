"""Synthetic session generator.

Three populations are produced:

human
    Lognormal think-times with genuine multi-second pauses, minimum-jerk
    velocity profiles, curved reaches that overshoot the target and correct
    back, hand tremor, and broad right-skewed dwell times with a fat tail.

bot_naive
    What an unsophisticated automation script actually emits: a constant delay
    between actions, constant-velocity straight-line cursor interpolation, and
    a single hardcoded press duration.

bot_evasive
    A humanizer of the kind that ships in off-the-shelf automation libraries:
    gaussian-jittered delays with occasional injected pauses, cubic Bezier
    cursor paths with an easing velocity profile and added positional noise,
    occasional overshoot, and dwell times drawn from a lognormal that
    approximates the human one.

The evasive population is intentionally hard. Its parameters overlap the human
ones on every axis; where it differs it differs by a realistic amount (a
slightly narrower dwell distribution, less frequent overshoot, an easing curve
rather than a true minimum-jerk profile), not by an amount picked to make the
detectors look good. Every population is rounded to integer screen coordinates,
so pixel quantization is not a discriminating artifact.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from iad.detectors.base import FloatArray, SessionArrays
from iad.models import EventType, InputEvent, Session

SessionKind = Literal["human", "bot_naive", "bot_evasive"]
KINDS: tuple[SessionKind, ...] = ("human", "bot_naive", "bot_evasive")

SCREEN_W = 1920.0
SCREEN_H = 1080.0
KEY_ALPHABET = ("a", "s", "d", "w", "q", "e", "r", "f", "1", "2", "3", "space")

UNITS_MIN, UNITS_MAX = 15, 23
MOUSE_UNIT_PROB = 0.60
BURST_MIN, BURST_MAX = 3, 8


@dataclass(frozen=True)
class _Raw:
    t_move: FloatArray
    mx: FloatArray
    my: FloatArray
    t_down: FloatArray
    t_up: FloatArray
    codes: list[str]


# --------------------------------------------------------------------------- #
# Timing models
# --------------------------------------------------------------------------- #


def _think_times(rng: np.random.Generator, kind: SessionKind, n: int) -> FloatArray:
    """Inter-action delays, in seconds."""
    if kind == "human":
        median = rng.uniform(0.22, 0.45)
        d = median * np.exp(rng.normal(0.0, 0.70, n))
        pauses = rng.random(n) < 0.07
        d[pauses] += rng.uniform(1.2, 6.0, int(pauses.sum()))
        return np.clip(d, 0.04, None)

    if kind == "bot_naive":
        base = rng.uniform(0.18, 0.60)
        return np.full(n, base)

    # Lognormal rather than gaussian jitter: a gaussian around a base rate leaves
    # a narrow, obviously-unimodal log-interval histogram, which is exactly what
    # the spread sub-signal looks for. A competent humanizer avoids that.
    base = rng.uniform(0.20, 0.50)
    d = base * np.exp(rng.normal(0.0, rng.uniform(0.40, 0.55), n))
    pauses = rng.random(n) < 0.05
    d[pauses] += rng.uniform(0.9, 4.0, int(pauses.sum()))
    return np.clip(d, 0.04, None)


def _dwells(rng: np.random.Generator, kind: SessionKind, n: int) -> FloatArray:
    """Press durations, in seconds."""
    if kind == "human":
        median = rng.uniform(0.075, 0.115)
        d = median * np.exp(rng.normal(0.0, 0.45, n))
        held = rng.random(n) < 0.03
        d[held] *= rng.uniform(2.0, 4.0, int(held.sum()))
        return np.clip(d, 0.035, 0.900)

    if kind == "bot_naive":
        return np.full(n, rng.uniform(0.050, 0.120))

    # Same family and nearly the same width as the human model, including a
    # (slightly rarer) long-hold tail. What remains is a marginally tighter
    # distribution, which is a realistic amount of residual signal.
    median = rng.uniform(0.078, 0.112)
    d = median * np.exp(rng.normal(0.0, rng.uniform(0.34, 0.44), n))
    held = rng.random(n) < 0.02
    d[held] *= rng.uniform(1.8, 3.2, int(held.sum()))
    return np.clip(d, 0.040, 0.600)


# --------------------------------------------------------------------------- #
# Cursor path models
# --------------------------------------------------------------------------- #


def _progress(kind: SessionKind, n: int) -> FloatArray:
    """Where along the path each sample falls, i.e. the velocity profile."""
    s = np.linspace(0.0, 1.0, n)
    if kind == "human":
        # Minimum-jerk profile: the classic model of a human ballistic reach.
        out: FloatArray = 10 * s**3 - 15 * s**4 + 6 * s**5
        return out
    if kind == "bot_naive":
        return s
    # Smoothstep easing, what humanizer libraries typically use.
    return 3 * s**2 - 2 * s**3


def _bezier(
    p0: FloatArray, p3: FloatArray, o1: float, o2: float, u: FloatArray
) -> tuple[FloatArray, FloatArray]:
    delta = p3 - p0
    dist = float(np.hypot(*delta))
    if dist < 1e-9:
        return np.full(u.size, p0[0]), np.full(u.size, p0[1])
    normal = np.array([-delta[1], delta[0]]) / dist
    p1 = p0 + delta / 3.0 + normal * o1
    p2 = p0 + delta * 2.0 / 3.0 + normal * o2
    iu = 1.0 - u
    pts = (
        np.outer(iu**3, p0)
        + 3 * np.outer(iu**2 * u, p1)
        + 3 * np.outer(iu * u**2, p2)
        + np.outer(u**3, p3)
    )
    return pts[:, 0], pts[:, 1]


def _stroke(
    rng: np.random.Generator,
    kind: SessionKind,
    p0: FloatArray,
    p3: FloatArray,
    n_pts: int,
) -> tuple[FloatArray, FloatArray]:
    dist = float(np.hypot(*(p3 - p0)))

    if kind == "human":
        curve_sd, tremor, overshoot_p, over_lo, over_hi = (
            0.090,
            rng.uniform(0.5, 1.4),
            0.65,
            0.04,
            0.12,
        )
    elif kind == "bot_naive":
        curve_sd, tremor, overshoot_p, over_lo, over_hi = 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        curve_sd, tremor, overshoot_p, over_lo, over_hi = (
            0.082,
            rng.uniform(0.40, 1.15),
            0.50,
            0.03,
            0.09,
        )

    # Same-sign control offsets so the path is a genuine arc, not an S-curve.
    sign = 1.0 if rng.random() < 0.5 else -1.0
    o1 = sign * abs(rng.normal(0.0, curve_sd * dist))
    o2 = sign * abs(rng.normal(0.0, curve_sd * dist))

    overshoots = rng.random() < overshoot_p and n_pts >= 9
    if overshoots:
        n_corr = int(rng.integers(3, 6))
        n_main = n_pts - n_corr
        p_over = p0 + (p3 - p0) * (1.0 + rng.uniform(over_lo, over_hi))
        u = _progress(kind, n_main)
        xs, ys = _bezier(p0, p_over, o1, o2, u)
        # Corrective sub-movement back onto the target, itself slightly curved.
        uc = _progress(kind, n_corr + 1)[1:]
        cx, cy = _bezier(p_over, p3, rng.normal(0.0, 0.10 * dist * 0.08), 0.0, uc)
        xs = np.concatenate([xs, cx])
        ys = np.concatenate([ys, cy])
    else:
        u = _progress(kind, n_pts)
        xs, ys = _bezier(p0, p3, o1, o2, u)

    if tremor > 0:
        xs = xs + rng.normal(0.0, tremor, xs.size)
        ys = ys + rng.normal(0.0, tremor, ys.size)

    # Every real capture layer reports integer device pixels.
    return np.round(xs), np.round(ys)


# --------------------------------------------------------------------------- #
# Session assembly
# --------------------------------------------------------------------------- #


def _generate_raw(rng: np.random.Generator, kind: SessionKind) -> _Raw:
    n_units = int(rng.integers(UNITS_MIN, UNITS_MAX + 1))
    # Upper bound on presses: every unit is either one click or one key burst.
    max_presses = n_units * BURST_MAX
    think = _think_times(rng, kind, max_presses)
    dwell = _dwells(rng, kind, max_presses)

    t_move: list[FloatArray] = []
    mxs: list[FloatArray] = []
    mys: list[FloatArray] = []
    t_down: list[float] = []
    t_up: list[float] = []
    codes: list[str] = []

    t = 0.0
    press_i = 0
    pos = np.array([rng.uniform(0, SCREEN_W), rng.uniform(0, SCREEN_H)])
    # Move sampling period, a property of the capture layer, identical for all
    # populations so that it cannot act as a discriminating feature.
    sample_hz = rng.uniform(60.0, 125.0)

    for _ in range(n_units):
        if rng.random() < MOUSE_UNIT_PROB:
            t += float(think[press_i])
            target = np.array([rng.uniform(0, SCREEN_W), rng.uniform(0, SCREEN_H)])
            dist = float(np.hypot(*(target - pos)))
            # Fitts-like duration, then sampled at the capture rate.
            duration = 0.16 + 0.11 * np.log2(1.0 + dist / 90.0)
            n_pts = int(max(7, min(40, round(duration * sample_hz))))
            xs, ys = _stroke(rng, kind, pos, target, n_pts)
            step = duration / (n_pts - 1)
            jitter = rng.normal(0.0, 0.10 * step, n_pts) if kind != "bot_naive" else 0.0
            ts = t + np.arange(n_pts) * step + jitter
            ts = np.maximum.accumulate(ts)
            t_move.append(ts)
            mxs.append(xs)
            mys.append(ys)
            pos = np.array([xs[-1], ys[-1]])
            t = float(ts[-1])

            # Reaction delay between arriving and pressing.
            t += (
                rng.uniform(0.015, 0.070)
                if kind == "human"
                else (0.004 if kind == "bot_naive" else rng.uniform(0.010, 0.050))
            )
            t_down.append(t)
            t += float(dwell[press_i])
            t_up.append(t)
            codes.append("left")
            press_i += 1
        else:
            n_keys = int(rng.integers(BURST_MIN, BURST_MAX + 1))
            for k in range(n_keys):
                # Within a burst the rhythm is faster than between units.
                t += float(think[press_i]) * (0.45 if k > 0 else 1.0)
                t_down.append(t)
                t += float(dwell[press_i])
                t_up.append(t)
                codes.append(str(rng.choice(KEY_ALPHABET)))
                press_i += 1

    return _Raw(
        t_move=np.concatenate(t_move) if t_move else np.empty(0),
        mx=np.concatenate(mxs) if mxs else np.empty(0),
        my=np.concatenate(mys) if mys else np.empty(0),
        t_down=np.asarray(t_down, dtype=np.float64),
        t_up=np.asarray(t_up, dtype=np.float64),
        codes=codes,
    )


def _to_arrays(raw: _Raw) -> SessionArrays:
    return SessionArrays(
        t_move=raw.t_move,
        mx=raw.mx,
        my=raw.my,
        t_action=raw.t_down,
        dwell_ms=(raw.t_up - raw.t_down) * 1000.0,
        n_events=raw.t_move.size + 2 * raw.t_down.size,
    )


def _to_session(raw: _Raw, session_id: str, label: SessionKind) -> Session:
    times = np.concatenate([raw.t_move, raw.t_down, raw.t_up])
    order = np.argsort(times, kind="stable")
    n_m, n_d = raw.t_move.size, raw.t_down.size

    events: list[InputEvent] = []
    for idx in order:
        i = int(idx)
        if i < n_m:
            events.append(
                InputEvent.model_construct(
                    t=float(raw.t_move[i]),
                    type=EventType.MOUSE_MOVE,
                    x=float(raw.mx[i]),
                    y=float(raw.my[i]),
                    code=None,
                )
            )
            continue
        if i < n_m + n_d:
            j = i - n_m
            code = raw.codes[j]
            events.append(
                InputEvent.model_construct(
                    t=float(raw.t_down[j]),
                    type=EventType.MOUSE_DOWN if code == "left" else EventType.KEY_DOWN,
                    x=None,
                    y=None,
                    code=code,
                )
            )
            continue
        j = i - n_m - n_d
        code = raw.codes[j]
        events.append(
            InputEvent.model_construct(
                t=float(raw.t_up[j]),
                type=EventType.MOUSE_UP if code == "left" else EventType.KEY_UP,
                x=None,
                y=None,
                code=code,
            )
        )

    return Session.model_construct(session_id=session_id, events=events, label=label)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate_arrays(kind: SessionKind, rng: np.random.Generator) -> SessionArrays:
    """Fast path used by the evaluation harness: skips model construction."""
    return _to_arrays(_generate_raw(rng, kind))


def generate_session(
    kind: SessionKind = "human",
    seed: int | None = None,
    session_id: str | None = None,
    rng: np.random.Generator | None = None,
) -> Session:
    """Generate one fully-materialized Session. Deterministic given `seed`."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")
    if rng is None:
        rng = np.random.default_rng(seed)
    raw = _generate_raw(rng, kind)
    sid = session_id or f"{kind}-{seed if seed is not None else 'rng'}"
    return _to_session(raw, sid, kind)


def generate_corpus(
    n_sessions: int,
    seed: int,
    bot_share: float = 0.5,
    evasive_share_of_bots: float = 0.45,
) -> Iterator[tuple[SessionArrays, SessionKind]]:
    """Stream a labelled corpus.

    Streaming rather than returning a list keeps memory flat at 10k+ sessions;
    the evaluation harness only ever retains the scores.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_sessions):
        if rng.random() < bot_share:
            kind: SessionKind = (
                "bot_evasive" if rng.random() < evasive_share_of_bots else "bot_naive"
            )
        else:
            kind = "human"
        yield generate_arrays(kind, rng), kind
