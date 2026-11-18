"""Statistical tests on the detectors.

Every assertion is over a seeded sample of sessions, not a single hand-built
case, so a detector that only works by accident on one input will fail here.
The bounds are deliberately loose: they assert the direction and a usable effect
size, not the exact numbers in eval/results.md, so that ordinary retuning does
not turn the suite red for no reason.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from iad.detectors import (
    CursorCurvatureDetector,
    Detector,
    DwellDistributionDetector,
    IntervalRegularityDetector,
    default_detectors,
    from_session,
)
from iad.detectors.base import FloatArray, SessionArrays
from iad.synth import SessionKind, generate_arrays, generate_session

N_SAMPLE = 120
SAMPLE_SEED = 4242

ALL_DETECTORS: list[Detector] = [
    IntervalRegularityDetector(),
    CursorCurvatureDetector(),
    DwellDistributionDetector(),
]


def sample_scores(
    detector: Detector, kind: SessionKind, n: int = N_SAMPLE, seed: int = SAMPLE_SEED
):
    rng = np.random.default_rng(seed)
    return np.array([detector.score(generate_arrays(kind, rng)) for _ in range(n)])


@pytest.fixture(scope="module")
def scores() -> dict[str, dict[str, FloatArray]]:
    return {
        d.name: {k: sample_scores(d, k) for k in ("human", "bot_naive", "bot_evasive")}
        for d in ALL_DETECTORS
    }


@pytest.mark.parametrize("name", [d.name for d in ALL_DETECTORS])
def test_scores_stay_in_range(name, scores):
    for arr in scores[name].values():
        assert np.all(np.isfinite(arr))
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0


@pytest.mark.parametrize("name", [d.name for d in ALL_DETECTORS])
def test_naive_bots_score_higher_than_humans(name, scores):
    """Each detector must separate naive automation from humans, on average."""
    human = scores[name]["human"]
    naive = scores[name]["bot_naive"]
    assert naive.mean() > human.mean() + 0.20, f"{name}: effect size too small"

    # Non-parametric, since the score distributions are bounded and skewed.
    stat = stats.mannwhitneyu(naive, human, alternative="greater")
    assert stat.pvalue < 1e-6, f"{name}: separation not significant"


@pytest.mark.parametrize("name", [d.name for d in ALL_DETECTORS])
def test_evasive_bots_score_at_least_as_high_as_humans(name, scores):
    """Weaker claim: the evasive population must not score *below* humans.

    Three detectors against a deliberate humanizer is a hard problem; a detector
    is allowed to be nearly blind to it, but a detector that systematically
    ranks evasive bots as more human than humans is broken.
    """
    assert scores[name]["bot_evasive"].mean() >= scores[name]["human"].mean()


def test_human_false_positive_rate_is_bounded(scores):
    """Detectors must not fire on most humans at their own midpoint."""
    for name, by_kind in scores.items():
        assert by_kind["human"].mean() < 0.35, f"{name}: too hot on humans"


def test_detector_is_deterministic():
    """Same input, same score. Detectors must be pure."""
    arrays = generate_arrays("bot_evasive", np.random.default_rng(1))
    for d in ALL_DETECTORS:
        assert d.score(arrays) == d.score(arrays)


def test_empty_session_abstains():
    """With no data a detector must abstain, not guess."""
    empty = SessionArrays(
        t_move=np.empty(0),
        mx=np.empty(0),
        my=np.empty(0),
        t_action=np.empty(0),
        dwell_ms=np.empty(0),
        n_events=0,
    )
    for d in ALL_DETECTORS:
        assert d.score(empty) is None


def test_default_detectors_have_unique_names():
    names = [d.name for d in default_detectors()]
    assert len(names) == len(set(names)) == 3


def test_stroke_segmentation_finds_separate_reaches():
    """Strokes are re-derived from resting gaps, not trusted from the producer."""
    arrays = generate_arrays("human", np.random.default_rng(9))
    strokes = arrays.strokes()
    assert len(strokes) >= 3
    for a, b in strokes:
        assert b > a
        # No stroke may contain an internal resting gap.
        assert np.all(np.diff(arrays.t_move[a:b]) <= 0.12 + 1e-9)


def test_array_fast_path_matches_pydantic_path():
    """The eval fast path and the API path must see identical features."""
    session = generate_session(kind="human", seed=77)
    via_model = from_session(session)
    direct = generate_arrays("human", np.random.default_rng(77))

    for attr in ("t_move", "mx", "my", "t_action", "dwell_ms"):
        np.testing.assert_allclose(
            getattr(via_model, attr), getattr(direct, attr), rtol=0, atol=0
        )
    assert via_model.n_events == direct.n_events == len(session.events)
