"""API tests through the real ASGI app via TestClient (httpx transport)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iad.api import app
from iad.config import settings
from iad.scoring import DEFAULT_THRESHOLD
from iad.synth import KINDS, generate_session


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["detectors"]) == 3
    assert body["threshold"] == pytest.approx(DEFAULT_THRESHOLD)
    assert sum(body["weights"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", KINDS)
def test_synth_returns_a_scorable_session(client, kind):
    r = client.post("/synth", params={"kind": kind, "seed": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == kind
    assert len(body["events"]) > 100
    assert body["events"][0]["t"] >= 0

    scored = client.post("/score", json=body)
    assert scored.status_code == 200


def test_synth_is_reproducible_over_http(client):
    a = client.post("/synth", params={"kind": "human", "seed": 42}).json()
    b = client.post("/synth", params={"kind": "human", "seed": 42}).json()
    assert a == b


def test_score_response_shape(client):
    session = generate_session(kind="bot_naive", seed=3)
    r = client.post("/score", json=session.model_dump(mode="json"))
    assert r.status_code == 200
    body = r.json()

    assert body["session_id"] == session.session_id
    assert set(body) == {
        "session_id",
        "score",
        "detector_scores",
        "threshold",
        "verdict",
        "insufficient_data",
        "n_events",
    }
    assert body["insufficient_data"] is False
    assert 0.0 <= body["score"] <= 1.0
    assert len(body["detector_scores"]) == 3
    assert all(0.0 <= v <= 1.0 for v in body["detector_scores"].values())
    assert body["verdict"] is (body["score"] >= body["threshold"])
    assert body["n_events"] == len(session.events)


def test_score_flags_naive_bot_and_clears_human(client):
    bot = client.post(
        "/score", json=generate_session(kind="bot_naive", seed=6).model_dump(mode="json")
    ).json()
    human = client.post(
        "/score", json=generate_session(kind="human", seed=6).model_dump(mode="json")
    ).json()
    assert bot["verdict"] is True
    assert human["verdict"] is False
    assert bot["score"] > human["score"]


@pytest.mark.parametrize("kind", KINDS)
def test_demo_endpoint(client, kind):
    r = client.post("/demo", params={"kind": kind, "seed": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == kind
    assert isinstance(body["label_matches_verdict"], bool)
    assert 0.0 <= body["result"]["score"] <= 1.0


def test_score_rejects_malformed_session(client):
    r = client.post("/score", json={"session_id": "x", "events": []})
    assert r.status_code == 422


def test_score_rejects_oversized_session(client):
    limit = settings.max_events
    events = [{"t": i * 0.01, "type": "key_down", "code": "a"} for i in range(limit + 1)]
    r = client.post("/score", json={"session_id": "big", "events": events})
    assert r.status_code == 422
    assert r.json()["detail"][0]["type"] == "too_long"


def test_score_accepts_session_at_the_limit(client):
    limit = settings.max_events
    events = [{"t": i * 0.01, "type": "key_down", "code": "a"} for i in range(limit)]
    r = client.post("/score", json={"session_id": "edge", "events": events})
    assert r.status_code == 200


def test_score_rejects_unordered_events(client):
    r = client.post(
        "/score",
        json={
            "session_id": "x",
            "events": [
                {"t": 1.0, "type": "key_down", "code": "a"},
                {"t": 0.5, "type": "key_up", "code": "a"},
            ],
        },
    )
    assert r.status_code == 422


def test_score_rejects_move_without_coordinates(client):
    r = client.post(
        "/score",
        json={
            "session_id": "x",
            "events": [
                {"t": 0.0, "type": "mouse_move"},
                {"t": 0.1, "type": "mouse_move", "x": 1, "y": 2},
            ],
        },
    )
    assert r.status_code == 422


def test_synth_rejects_unknown_kind(client):
    assert client.post("/synth", params={"kind": "bot_perfect"}).status_code == 422


def _keys(n_presses: int) -> list[dict]:
    events = []
    for i in range(n_presses):
        events.append({"t": i * 0.3, "type": "key_down", "code": "a"})
        events.append({"t": i * 0.3 + 0.1, "type": "key_up", "code": "a"})
    return events


def test_tiny_session_abstains_rather_than_guessing(client):
    """Too little data must yield no verdict, not a confident one."""
    r = client.post("/score", json={"session_id": "tiny", "events": _keys(1)})
    assert r.status_code == 200
    body = r.json()
    assert body["insufficient_data"] is True
    assert body["verdict"] is None
    assert body["score"] is None
    assert all(v is None for v in body["detector_scores"].values())


@pytest.mark.parametrize("n_presses", [1, 2, 4, 7])
def test_short_sessions_are_never_flagged(client, n_presses):
    r = client.post("/score", json={"session_id": "short", "events": _keys(n_presses)})
    assert r.status_code == 200
    assert r.json()["verdict"] is not True


def test_abstaining_detectors_are_excluded_from_the_score(client):
    """Keys only: cursor_curvature abstains, the other two decide alone."""
    body = client.post(
        "/score", json={"session_id": "keys", "events": _keys(12)}
    ).json()
    scores = body["detector_scores"]
    assert scores["cursor_curvature"] is None
    assert scores["interval_regularity"] is not None
    assert scores["dwell_distribution"] is not None
    assert body["insufficient_data"] is False
    # A metronomic, fixed-dwell key stream is a bot even without cursor data.
    assert body["verdict"] is True


def test_openapi_is_served(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/score" in r.json()["paths"]
