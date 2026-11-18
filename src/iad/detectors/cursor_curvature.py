"""Detector 2: geometry of cursor strokes.

A human reaching for a target produces a ballistic phase that overshoots, then
one or more corrective sub-movements, with tremor superimposed throughout. The
resulting path is measurably longer than the straight line, reverses its turn
direction often, and frequently travels past the endpoint before settling.

Three sub-signals:

* straightness - end-to-end distance over travelled path length. Linear
  interpolation scores ~1.0; a human reach scores ~0.75-0.88.
* reversal rate - fraction of steps where the signed turn angle changes sign.
  This is scored as a two-sided band: a spline-smoothed bot sits well below the
  human band, a bot that fakes tremor with white noise sits well above it.
* overshoot - share of strokes whose furthest projection onto the start->end
  axis passes the endpoint. Corrective motion is expensive to fake convincingly
  and most automation simply does not produce it.

Sub-signals are aggregated across strokes first (median / mean), then scored, so
one unusual stroke cannot dominate a session.
"""

from __future__ import annotations

import numpy as np

from iad.detectors.base import (
    SessionArrays,
    band_deviation,
    clamp01,
    ramp,
)

MIN_STROKES = 3
OVERSHOOT_REL_TOL = 1.03

# Centre and half-width of the human reversal-rate band, in fraction of steps.
REVERSAL_CENTER = 0.66
REVERSAL_HALFWIDTH = 0.26

WEIGHTS = (0.40, 0.35, 0.25)


def _stroke_stats(px: np.ndarray, py: np.ndarray) -> tuple[float, float, bool] | None:
    """Return (efficiency, reversal_rate, overshot) for one stroke."""
    dx = np.diff(px)
    dy = np.diff(py)
    seg = np.hypot(dx, dy)
    keep = seg > 1e-9
    if keep.sum() < 4:
        return None
    dx, dy, seg = dx[keep], dy[keep], seg[keep]

    path_len = float(seg.sum())
    net = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
    if path_len <= 1e-9 or net <= 1e-9:
        return None
    efficiency = min(1.0, net / path_len)

    # Signed turn angle via the 2-D cross product of consecutive step vectors.
    cross = dx[:-1] * dy[1:] - dy[:-1] * dx[1:]
    dot = dx[:-1] * dx[1:] + dy[:-1] * dy[1:]
    ang = np.arctan2(cross, dot)
    # Ignore turns below the angular resolution of a pixel grid at this step size.
    significant = np.abs(ang) > 0.02
    if significant.sum() < 3:
        reversal = 0.0
    else:
        sg = np.sign(ang[significant])
        reversal = float(np.mean(sg[1:] != sg[:-1]))

    # Projection of every sample onto the unit start->end axis.
    ux = (px[-1] - px[0]) / net
    uy = (py[-1] - py[0]) / net
    proj = (px - px[0]) * ux + (py - py[0]) * uy
    overshot = bool(proj.max() > OVERSHOOT_REL_TOL * net)

    return efficiency, reversal, overshot


class CursorCurvatureDetector:
    name = "cursor_curvature"

    def score(self, s: SessionArrays) -> float | None:
        effs: list[float] = []
        revs: list[float] = []
        overs: list[bool] = []

        for a, b in s.strokes():
            st = _stroke_stats(s.mx[a:b], s.my[a:b])
            if st is None:
                continue
            effs.append(st[0])
            revs.append(st[1])
            overs.append(st[2])

        if len(effs) < MIN_STROKES:
            return None

        straightness = ramp(float(np.median(effs)), 0.885, 0.990)
        reversal = band_deviation(
            float(np.mean(revs)), REVERSAL_CENTER, REVERSAL_HALFWIDTH
        )
        overshoot = ramp(float(np.mean(overs)), 0.62, 0.06)

        parts = (straightness, reversal, overshoot)
        return clamp01(float(np.dot(parts, WEIGHTS)))

