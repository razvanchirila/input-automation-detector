"""Tests for the combiner's algebraic contract.

Monotonicity and boundedness are the two properties that make the combined score
defensible to an analyst, so they are tested directly rather than inferred.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from iad.detectors.base import SessionArrays
from iad.scoring import DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, Scorer
from iad.synth import generate_session

NAMES = tuple(DEFAULT_WEIGHTS)


def test_combine_stays_in_range(scorer):
    rng = np.random.default_rng(3)
    for _ in range(500):
        vals = dict(zip(NAMES, rng.random(len(NAMES))))
        assert 0.0 <= scorer.combine(vals) <= 1.0


def test_combine_is_monotonic_in_every_detector(scorer):
    """Raising one detector can never lower the combined score."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        base = dict(zip(NAMES, rng.random(len(NAMES))))
        for name in NAMES:
            lifted = dict(base)
            lifted[name] = min(1.0, base[name] + rng.uniform(0.01, 0.4))
            assert scorer.combine(lifted) >= scorer.combine(base) - 1e-12


def test_combine_is_strictly_increasing_for_positive_weights(scorer):
    base = {n: 0.4 for n in NAMES}
    for name in NAMES:
        lifted = dict(base, **{name: 0.9})
        assert scorer.combine(lifted) > scorer.combine(base)


def test_combine_hits_both_extremes(scorer):
    assert scorer.combine({n: 0.0 for n in NAMES}) == pytest.approx(0.0)
    assert scorer.combine({n: 1.0 for n in NAMES}) == pytest.approx(1.0)


def test_combine_renormalizes_when_detectors_are_missing(scorer):
    """A subset of detectors must still produce a score on the same scale."""
    for r in (1, 2):
        for subset in itertools.combinations(NAMES, r):
            assert scorer.combine({n: 1.0 for n in subset}) == pytest.approx(1.0)
            assert scorer.combine({n: 0.0 for n in subset}) == pytest.approx(0.0)


def test_combine_ignores_abstaining_detectors(scorer):
    """None means "no data", so it must not pull the score toward any value."""
    for name in NAMES:
        scores: dict[str, float | None] = {n: 0.2 for n in NAMES}
        scores[name] = None
        assert scorer.combine(scores) == pytest.approx(0.2)


def test_combine_returns_none_when_every_detector_abstains(scorer):
    assert scorer.combine({n: None for n in NAMES}) is None
    assert scorer.combine({}) is None


def test_all_abstaining_session_gets_no_verdict():
    class Abstainer:
        name = "interval_regularity"

        def score(self, s: SessionArrays) -> float | None:
            return None

    res = Scorer(detectors=[Abstainer()]).score_session(
        generate_session(kind="bot_naive", seed=1)
    )
    assert res.score is None
    assert res.verdict is None
    assert res.insufficient_data is True


def test_weights_are_a_normalized_non_negative_set():
    assert all(w >= 0 for w in DEFAULT_WEIGHTS.values())
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_verdict_follows_threshold(scorer):
    for session_kind in ("human", "bot_naive", "bot_evasive"):
        res = scorer.score_session(generate_session(kind=session_kind, seed=12))
        assert res.score is not None
        assert res.verdict == (res.score >= res.threshold)
        assert res.threshold == DEFAULT_THRESHOLD


def test_score_session_reports_every_detector(scorer):
    res = scorer.score_session(generate_session(kind="human", seed=1))
    assert set(res.detector_scores) == set(NAMES)
    assert res.n_events > 100


def test_naive_bot_is_caught_and_human_is_not(scorer):
    """End-to-end sanity at the shipped threshold, over a sample."""
    naive = [
        scorer.score_session(generate_session(kind="bot_naive", seed=s)).verdict
        for s in range(40)
    ]
    human = [
        scorer.score_session(generate_session(kind="human", seed=s)).verdict
        for s in range(40)
    ]
    assert sum(naive) >= 38
    assert sum(human) <= 8


def test_rejects_unknown_detector_weights():
    class Rogue:
        name = "not_configured"

        def score(self, s: SessionArrays) -> float:
            return 0.0

    with pytest.raises(ValueError, match="no weight configured"):
        Scorer(detectors=[Rogue()])


def test_rejects_out_of_range_threshold():
    with pytest.raises(ValueError, match="threshold"):
        Scorer(threshold=1.5)


def test_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        Scorer(weights=dict(DEFAULT_WEIGHTS, cursor_curvature=-0.1))
