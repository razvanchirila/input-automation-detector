"""API tests through the real ASGI app via TestClient (httpx transport)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iad.api import app
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
        "n_events",
    }
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


def test_tiny_session_abstains_rather_than_guessing(client):
    """Too little data must yield the neutral score, not a confident verdict."""
    r = client.post(
        "/score",
        json={
            "session_id": "tiny",
            "events": [
                {"t": 0.0, "type": "key_down", "code": "a"},
                {"t": 0.1, "type": "key_up", "code": "a"},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["score"] == pytest.approx(0.5)
    assert all(v == pytest.approx(0.5) for v in body["detector_scores"].values())


def test_openapi_is_served(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/score" in r.json()["paths"]
