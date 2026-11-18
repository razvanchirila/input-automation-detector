"""Combine detector outputs into a single score and a verdict.

The combiner is a fixed-weight arithmetic mean over the detector scores. That is
a deliberate choice, not a placeholder:

* it is monotonic - raising any one detector can never lower the combined score,
  which is what makes the output explainable to an analyst reviewing a ban;
* it stays in [0,1] by construction, because the weights are non-negative and
  sum to one;
* nothing is fitted to the corpus except the single decision threshold, so the
  reported metrics are not a self-graded model.

Weights are hand-set from the per-detector discrimination measured in
`eval/evaluate.py` and rounded; they are not optimized on the test half.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from iad.detectors import Detector, SessionArrays, default_detectors, from_session
from iad.models import ScoreResult, Session

DEFAULT_WEIGHTS: dict[str, float] = {
    "interval_regularity": 0.34,
    "cursor_curvature": 0.36,
    "dwell_distribution": 0.30,
}

# Max-F1 operating point, chosen on the training half of the synthetic corpus.
# See eval/results.md; regenerate with `python eval/evaluate.py`.
DEFAULT_THRESHOLD = 0.1923

# Operating point for a sub-1% false-positive rate. Use this one when the
# downstream action is punitive rather than a review queue: it trades evasive-bot
# recall (0.37 vs 0.76) for 0.99 precision.
CONSERVATIVE_THRESHOLD = 0.2907


@dataclass
class Scorer:
    """Runs every detector over a session and folds the results together.

    Detectors that abstain (too little data) are left out of the weighted mean
    and the remaining weights are renormalized. When every detector abstains the
    combined score is None and no verdict is given.
    """

    detectors: list[Detector] = field(default_factory=default_detectors)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    threshold: float = DEFAULT_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        missing = [d.name for d in self.detectors if d.name not in self.weights]
        if missing:
            raise ValueError(f"no weight configured for detector(s): {missing}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative")

    def detector_scores(self, arrays: SessionArrays) -> dict[str, float | None]:
        return {d.name: d.score(arrays) for d in self.detectors}

    def combine(self, scores: Mapping[str, float | None]) -> float | None:
        """Weighted mean over the detectors that produced a score, renormalized."""
        present = {name: v for name, v in scores.items() if v is not None}
        total_w = sum(self.weights[name] for name in present)
        if total_w <= 0:
            return None
        acc = sum(self.weights[name] * value for name, value in present.items())
        return min(1.0, max(0.0, acc / total_w))

    def score_arrays(
        self, arrays: SessionArrays
    ) -> tuple[float | None, dict[str, float | None]]:
        scores = self.detector_scores(arrays)
        return self.combine(scores), scores

    def score_session(self, session: Session) -> ScoreResult:
        combined, scores = self.score_arrays(from_session(session))
        return ScoreResult(
            session_id=session.session_id,
            score=combined,
            detector_scores=scores,
            threshold=self.threshold,
            verdict=None if combined is None else combined >= self.threshold,
            insufficient_data=combined is None,
            n_events=len(session.events),
        )
