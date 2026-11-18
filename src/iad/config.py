"""Runtime settings, read once from the environment at import time."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    max_events: int
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            max_events=_env_int("IAD_MAX_EVENTS", 10_000, minimum=2),
            log_level=os.environ.get("IAD_LOG_LEVEL", "INFO").upper(),
        )


settings = Settings.from_env()
