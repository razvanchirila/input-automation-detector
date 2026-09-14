"""Tests for the generator: reproducibility, validity, and population realism."""

from __future__ import annotations

import numpy as np
import pytest

from iad.detectors.base import from_session
from iad.models import DOWN_TYPES, UP_TYPES, EventType, Session
from iad.synth import KINDS, generate_arrays, generate_corpus, generate_session


@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_gives_identical_session(kind):
    a = generate_session(kind=kind, seed=99)
    b = generate_session(kind=kind, seed=99)
    assert a.model_dump() == b.model_dump()


@pytest.mark.parametrize("kind", KINDS)
def test_different_seeds_give_different_sessions(kind):
    a = generate_session(kind=kind, seed=1)
    b = generate_session(kind=kind, seed=2)
    assert a.model_dump() != b.model_dump()


def test_corpus_is_reproducible():
    def digest(seed: int):
        return [
            (kind, round(float(a.t_action.sum()), 9), a.n_events)
            for a, kind in generate_corpus(25, seed=seed)
        ]

    assert digest(11) == digest(11)
    assert digest(11) != digest(12)


def test_corpus_respects_requested_composition():
    kinds = [k for _, k in generate_corpus(4000, seed=3, bot_share=0.5,
                                           evasive_share_of_bots=0.45)]
    n = len(kinds)
    assert n == 4000
    humans = kinds.count("human")
    bots = n - humans
    assert 0.46 < humans / n < 0.54
    assert 0.40 < kinds.count("bot_evasive") / bots < 0.50


@pytest.mark.parametrize("kind", KINDS)
def test_generated_sessions_pass_full_validation(kind):
    """The fast path uses model_construct, so validity must be asserted here."""
    session = generate_session(kind=kind, seed=7)
    revalidated = Session.model_validate(session.model_dump())
    assert len(revalidated.events) == len(session.events)
    assert revalidated.label == kind


@pytest.mark.parametrize("kind", KINDS)
def test_events_are_time_ordered_and_well_formed(kind):
    session = generate_session(kind=kind, seed=8)
    times = [e.t for e in session.events]
    assert times == sorted(times)
    for ev in session.events:
        if ev.type is EventType.MOUSE_MOVE:
            assert ev.x is not None and ev.y is not None
        else:
            assert ev.code is not None


@pytest.mark.parametrize("kind", KINDS)
def test_every_press_is_released(kind):
    session = generate_session(kind=kind, seed=13)
    downs = sum(1 for e in session.events if e.type in DOWN_TYPES)
    ups = sum(1 for e in session.events if e.type in UP_TYPES)
    assert downs == ups > 10
    assert from_session(session).dwell_ms.size == downs


@pytest.mark.parametrize("kind", KINDS)
def test_sessions_are_substantial(kind):
    a = generate_arrays(kind, np.random.default_rng(21))
    assert a.t_move.size > 80
    assert a.t_action.size >= 15
    assert len(a.strokes()) >= 5
    assert np.all(a.dwell_ms > 0)


def test_naive_bot_is_degenerate_by_construction():
    """Guards the generator's own contract, so a regression is caught here."""
    a = generate_arrays("bot_naive", np.random.default_rng(31))
    assert a.dwell_ms.std() == pytest.approx(0.0, abs=1e-9)
    # Straight-line interpolation: path length equals end-to-end distance.
    s, e = a.strokes()[0]
    seg = np.hypot(np.diff(a.mx[s:e]), np.diff(a.my[s:e])).sum()
    net = np.hypot(a.mx[e - 1] - a.mx[s], a.my[e - 1] - a.my[s])
    assert net / seg > 0.99


def test_evasive_bot_overlaps_humans_on_raw_statistics():
    """The evasive population must be genuinely hard, not a relabelled bot.

    If a later change to the generator makes any of these raw statistics
    separable on its own, the reported metrics stop being honest. This test is
    the tripwire for that.
    """
    def stats_for(kind: str):
        rng = np.random.default_rng(55)
        cv_i, cv_d, eff = [], [], []
        for _ in range(80):
            a = generate_arrays(kind, rng)
            d = np.diff(a.t_action)
            cv_i.append(d.std(ddof=1) / d.mean())
            cv_d.append(a.dwell_ms.std(ddof=1) / a.dwell_ms.mean())
            s, e = a.strokes()[0]
            seg = np.hypot(np.diff(a.mx[s:e]), np.diff(a.my[s:e])).sum()
            net = np.hypot(a.mx[e - 1] - a.mx[s], a.my[e - 1] - a.my[s])
            eff.append(net / max(seg, 1e-9))
        return np.array(cv_i), np.array(cv_d), np.array(eff)

    h_cvi, h_cvd, h_eff = stats_for("human")
    e_cvi, e_cvd, e_eff = stats_for("bot_evasive")

    # Distributions must overlap: the evasive mean has to sit inside roughly one
    # standard deviation of the human mean on every axis.
    for hm, em, label in (
        (h_cvi, e_cvi, "interval cv"),
        (h_cvd, e_cvd, "dwell cv"),
        (h_eff, e_eff, "path efficiency"),
    ):
        assert abs(em.mean() - hm.mean()) < 1.6 * hm.std(), f"{label} too separable"


def test_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        generate_session(kind="bot_perfect")  # type: ignore[arg-type]
