"""Ops surface: probes, metrics, request IDs and log format."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from iad.api import app
from iad.observability import JsonFormatter, request_id_var
from iad.synth import generate_session


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _samples(
    client: TestClient,
) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    r = client.get("/metrics")
    assert r.status_code == 200
    out = {}
    for fam in text_string_to_metric_families(r.text):
        for s in fam.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def test_livez(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_once_started(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_is_503_outside_lifespan():
    with TestClient(app):
        pass
    r = TestClient(app).get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"status": "starting"}


def test_metrics_exposes_custom_series(client):
    names = {name for name, _ in _samples(client)}
    assert "iad_http_requests_total" in names
    assert "iad_http_request_duration_seconds_bucket" in names


def test_scoring_updates_verdict_and_score_metrics(client):
    def verdicts(v: str) -> float:
        return _samples(client).get(("iad_verdicts_total", (("verdict", v),)), 0.0)

    def score_count() -> float:
        return _samples(client).get(("iad_score_count", ()), 0.0)

    bot, human, insufficient, n = (
        verdicts("bot"),
        verdicts("human"),
        verdicts("insufficient"),
        score_count(),
    )
    payload = generate_session(kind="bot_naive", seed=6).model_dump(mode="json")
    assert client.post("/score", json=payload).json()["verdict"] is True
    tiny = {
        "session_id": "tiny",
        "events": [
            {"t": 0.0, "type": "key_down", "code": "a"},
            {"t": 0.1, "type": "key_up", "code": "a"},
        ],
    }
    assert client.post("/score", json=tiny).json()["insufficient_data"] is True

    assert verdicts("bot") == bot + 1
    assert verdicts("human") == human
    assert verdicts("insufficient") == insufficient + 1
    assert score_count() == n + 1


def test_http_metrics_use_route_templates(client):
    client.get("/synth", params={"kind": "human", "seed": 1})
    client.get("/does-not-exist")
    keys = {
        labels for name, labels in _samples(client) if name == "iad_http_requests_total"
    }
    assert (("method", "GET"), ("route", "/synth"), ("status", "200")) in keys
    assert (("method", "GET"), ("route", "unmatched"), ("status", "404")) in keys


def test_request_id_is_generated(client):
    rid = client.get("/livez").headers["X-Request-ID"]
    assert len(rid) == 32


def test_request_id_is_propagated(client):
    r = client.get("/livez", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize("bad", ["", "x" * 200, "has space", "new\nline"])
def test_invalid_request_id_is_replaced(client, bad):
    r = client.get("/livez", headers={"X-Request-ID": bad})
    assert r.headers["X-Request-ID"] != bad
    assert len(r.headers["X-Request-ID"]) == 32


def test_json_formatter_includes_request_id_and_extras():
    record = logging.makeLogRecord(
        {"name": "iad.access", "levelno": 20, "levelname": "INFO", "msg": "request"}
    )
    record.status = 200
    token = request_id_var.set("rid-1")
    try:
        line = JsonFormatter().format(record)
    finally:
        request_id_var.reset(token)
    body = json.loads(line)
    assert body["msg"] == "request"
    assert body["level"] == "info"
    assert body["request_id"] == "rid-1"
    assert body["status"] == 200
