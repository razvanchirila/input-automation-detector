"""Detector implementations. Each returns a float in [0,1]; higher = more bot-like."""

from iad.detectors.base import (
    Detector,
    SessionArrays,
    band_deviation,
    clamp01,
    from_session,
    ramp,
)
from iad.detectors.cursor_curvature import CursorCurvatureDetector
from iad.detectors.dwell_distribution import DwellDistributionDetector
from iad.detectors.interval_regularity import IntervalRegularityDetector


def default_detectors() -> list[Detector]:
    """The three shipped signals, in a stable order."""
    return [
        IntervalRegularityDetector(),
        CursorCurvatureDetector(),
        DwellDistributionDetector(),
    ]


__all__ = [
    "CursorCurvatureDetector",
    "Detector",
    "DwellDistributionDetector",
    "IntervalRegularityDetector",
    "SessionArrays",
    "band_deviation",
    "clamp01",
    "default_detectors",
    "from_session",
    "ramp",
]
