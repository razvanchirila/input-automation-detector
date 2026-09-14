"""Detector 3: shape of the press-duration distribution.

Key and button dwell times in humans are broad and right-skewed - roughly
lognormal with a fat right tail from keys held during a moment of thought.
Automation usually emits a constant hold, or samples from something too narrow.

This detector compares the *shape* of the distribution, not its mean, because
mean dwell varies enormously between people, keyboards and games. Every
sub-signal is scale-free: dwells are normalized by their own median first, so a
fast typist and a slow one look the same.

Sub-signals:

* shape - one-sample Kolmogorov-Smirnov distance between log(dwell / median) and
  a reference N(0, 0.42) fitted from the human population. Used as a shape
  distance only; because the location is estimated from the sample itself the
  statistic is not distribution-free and no p-value is computed from it.
* dispersion - coefficient of variation, the single strongest sub-signal against
  humanizers that sample from a too-narrow distribution.
* asymmetry - sample skewness against the human band. Noisy at these sample
  sizes, so it carries the smallest weight.
* quantization - share of dwell values that collide with another value at 1 ms
  resolution. Fires hard on constant or grid-snapped holds.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from iad.detectors.base import (
    NEUTRAL,
    SessionArrays,
    band_deviation,
    clamp01,
    ramp,
)

MIN_DWELLS = 8

# Log-scale sigma of the human reference distribution.
REF_LOG_SIGMA = 0.42

SKEW_CENTER = 1.70
SKEW_HALFWIDTH = 1.55

WEIGHTS = (0.35, 0.35, 0.15, 0.15)


def _shape_score(d: np.ndarray) -> float:
    med = float(np.median(d))
    if med <= 0:
        return 1.0
    z = np.log(d / med)
    ks = float(stats.kstest(z, "norm", args=(0.0, REF_LOG_SIGMA)).statistic)
    return ramp(ks, 0.115, 0.42)


def _dispersion_score(d: np.ndarray) -> float:
    mean = float(d.mean())
    if mean <= 0:
        return 1.0
    cv = float(d.std(ddof=1)) / mean
    return ramp(cv, 0.52, 0.12)


def _asymmetry_score(d: np.ndarray) -> float:
    if float(d.std(ddof=1)) < 1e-9:
        return 1.0
    return band_deviation(float(stats.skew(d)), SKEW_CENTER, SKEW_HALFWIDTH)


def _quantization_score(d: np.ndarray) -> float:
    q = np.round(d).astype(np.int64)
    _, counts = np.unique(q, return_counts=True)
    collided = int(counts[counts > 1].sum())
    return ramp(collided / q.size, 0.33, 0.95)


class DwellDistributionDetector:
    name = "dwell_distribution"

    def score(self, s: SessionArrays) -> float:
        d = s.dwell_ms
        d = d[d > 0]
        if d.size < MIN_DWELLS:
            return NEUTRAL

        parts = (
            _shape_score(d),
            _dispersion_score(d),
            _asymmetry_score(d),
            _quantization_score(d),
        )
        return clamp01(float(np.dot(parts, WEIGHTS)))

