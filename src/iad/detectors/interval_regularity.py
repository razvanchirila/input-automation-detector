"""Detector 1: regularity of the action rhythm.

Human inter-action intervals are heavy-tailed: a broad lognormal body plus
occasional multi-second pauses (reading, hesitating, being interrupted).
Automation tends to a target rate, so its intervals concentrate.

Three sub-signals, deliberately measuring different failure modes of a naive
humanizer:

* dispersion - coefficient of variation. Catches "fixed delay" and "fixed delay
  plus small jitter", but a bot that injects fake pauses can inflate it.
* spread - normalized Shannon entropy of the scale-free log-interval histogram.
  Catches the shape rather than the spread, so adding two big pauses to an
  otherwise metronomic stream does not rescue it.
* clustering - share of intervals that have a near-duplicate partner. Catches
  quantized or replayed schedules that the first two sub-signals can miss.
"""

from __future__ import annotations

import numpy as np

from iad.detectors.base import SessionArrays, clamp01, ramp

MIN_INTERVALS = 8

# Relative tolerance for calling two intervals near-duplicates.
DUP_REL_TOL = 0.03

# Log-space histogram used for the entropy sub-signal.
_LOG_BINS = np.linspace(-2.5, 2.5, 21)
_MAX_ENTROPY = float(np.log(len(_LOG_BINS) - 1))

WEIGHTS = (0.45, 0.35, 0.20)


def _dispersion_score(d: np.ndarray) -> float:
    mean = float(d.mean())
    if mean <= 0:
        return 1.0
    cv = float(d.std(ddof=1)) / mean
    # Ramp anchored on the human population (mean cv ~1.14). The lower anchor is
    # not 0: even a fixed-delay script picks up dispersion from the session
    # structure itself, because clicks and key bursts have different cadences.
    return ramp(cv, 1.10, 0.50)


def _spread_score(d: np.ndarray) -> float:
    med = float(np.median(d))
    if med <= 0:
        return 1.0
    logs = np.log(d / med)
    counts, _ = np.histogram(logs, bins=_LOG_BINS)
    total = counts.sum()
    if total == 0:
        return 1.0
    p = counts[counts > 0] / total
    entropy = float(-(p * np.log(p)).sum()) / _MAX_ENTROPY
    return ramp(entropy, 0.78, 0.38)


def _clustering_score(d: np.ndarray) -> float:
    s = np.sort(d)
    # An interval is "clustered" if its neighbour in sorted order is within
    # DUP_REL_TOL relative distance. O(n log n) and tolerant of scale.
    gaps = np.diff(s)
    scale = np.maximum(s[:-1], 1e-9)
    close = gaps <= DUP_REL_TOL * scale
    clustered = np.zeros(s.size, dtype=bool)
    clustered[:-1] |= close
    clustered[1:] |= close
    frac = float(clustered.mean())
    return ramp(frac, 0.63, 0.96)


class IntervalRegularityDetector:
    name = "interval_regularity"

    def score(self, s: SessionArrays) -> float | None:
        t = s.t_action
        if t.size < MIN_INTERVALS + 1:
            return None
        d = np.diff(t)
        d = d[d > 0]
        if d.size < MIN_INTERVALS:
            return None

        parts = (_dispersion_score(d), _spread_score(d), _clustering_score(d))
        return clamp01(float(np.dot(parts, WEIGHTS)))

