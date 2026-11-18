"""Public data model for input telemetry sessions and scoring results."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from iad.config import settings


class EventType(StrEnum):
    MOUSE_MOVE = "mouse_move"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"


DOWN_TYPES = frozenset({EventType.MOUSE_DOWN, EventType.KEY_DOWN})
UP_TYPES = frozenset({EventType.MOUSE_UP, EventType.KEY_UP})

SessionLabel = Literal["human", "bot_naive", "bot_evasive"]


class InputEvent(BaseModel):
    """A single raw input event.

    `t` is seconds since the start of the session (monotonic clock, not wall clock),
    so sessions are comparable regardless of when they were recorded.
    """

    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0, description="Seconds since session start.")
    type: EventType
    x: float | None = Field(default=None, description="Cursor x, device pixels.")
    y: float | None = Field(default=None, description="Cursor y, device pixels.")
    code: str | None = Field(
        default=None, max_length=32, description="Key name or mouse button id."
    )

    @model_validator(mode="after")
    def _check_payload(self) -> InputEvent:
        if self.type is EventType.MOUSE_MOVE:
            if self.x is None or self.y is None:
                raise ValueError("mouse_move requires x and y")
        elif self.code is None:
            raise ValueError(f"{self.type.value} requires code")
        return self


class Session(BaseModel):
    """An ordered burst of input events from one client."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    events: list[InputEvent] = Field(
        min_length=2,
        max_length=settings.max_events,
        description="Ordered events; the cap is set by IAD_MAX_EVENTS.",
    )
    label: SessionLabel | None = Field(
        default=None, description="Ground truth, present only in generated corpora."
    )

    @field_validator("events")
    @classmethod
    def _check_monotonic(cls, events: list[InputEvent]) -> list[InputEvent]:
        last = -1.0
        for ev in events:
            if ev.t < last:
                raise ValueError("events must be ordered by non-decreasing t")
            last = ev.t
        return events

    @property
    def duration_s(self) -> float:
        return self.events[-1].t - self.events[0].t


class ScoreResult(BaseModel):
    """Output of the scorer for one session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    score: float | None = Field(
        ge=0.0,
        le=1.0,
        description="Combined bot likelihood; null when no detector had enough data.",
    )
    detector_scores: dict[str, float | None] = Field(
        description=(
            "Per-detector score in [0,1]; higher means more bot-like. "
            "null means the detector abstained for lack of data."
        )
    )
    threshold: float = Field(ge=0.0, le=1.0)
    verdict: bool | None = Field(
        description="True when score >= threshold; null when insufficient_data."
    )
    insufficient_data: bool = Field(
        description="True when every detector abstained, so no verdict is given."
    )
    n_events: int = Field(ge=0)
