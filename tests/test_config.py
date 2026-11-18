from __future__ import annotations

import pytest

from iad.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("IAD_MAX_EVENTS", raising=False)
    monkeypatch.delenv("IAD_LOG_LEVEL", raising=False)
    s = Settings.from_env()
    assert s.max_events == 10_000
    assert s.log_level == "INFO"


def test_max_events_from_env(monkeypatch):
    monkeypatch.setenv("IAD_MAX_EVENTS", "500")
    assert Settings.from_env().max_events == 500


@pytest.mark.parametrize("bad", ["1", "-3", "abc"])
def test_invalid_max_events_is_rejected(monkeypatch, bad):
    monkeypatch.setenv("IAD_MAX_EVENTS", bad)
    with pytest.raises(ValueError):
        Settings.from_env()
