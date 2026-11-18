"""FastAPI surface for the detector.

    uvicorn iad.api:app --reload
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from iad import __version__
from iad.detectors import default_detectors
from iad.models import ScoreResult, Session
from iad.scoring import DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, Scorer
from iad.synth import KINDS, SessionKind, generate_session

app = FastAPI(
    title="Input Automation Detector",
    version=__version__,
    summary="Scores input-telemetry sessions for the likelihood of being automated.",
)

_scorer = Scorer()


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    detectors: list[str]
    weights: dict[str, float]
    threshold: float


@app.get("/health", response_model=Health, tags=["ops"])
def health() -> Health:
    """Liveness plus the active scoring configuration."""
    return Health(
        status="ok",
        version=__version__,
        detectors=[d.name for d in default_detectors()],
        weights=DEFAULT_WEIGHTS,
        threshold=DEFAULT_THRESHOLD,
    )


@app.post("/score", response_model=ScoreResult, tags=["scoring"])
def score(session: Session) -> ScoreResult:
    """Score one session.

    Returns the combined score, every detector's individual score, and the
    boolean verdict at the configured threshold. A detector with too little data
    abstains (null) and is left out of the combined score; if all of them abstain
    the response has `insufficient_data: true` and a null score and verdict.
    """
    try:
        return _scorer.score_session(session)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/synth", response_model=Session, tags=["scoring"])
def synth(
    kind: SessionKind = Query(
        default="human", description=f"One of {', '.join(KINDS)}."
    ),
    seed: int | None = Query(default=None, ge=0, description="Omit for a random one."),
) -> Session:
    """Generate an example session so the API can be tried without real data.

    Feed the response straight back into `POST /score`.
    """
    if kind not in KINDS:
        raise HTTPException(status_code=422, detail=f"kind must be one of {KINDS}")
    return generate_session(kind=kind, seed=seed)


class Demo(BaseModel):
    kind: SessionKind
    seed: int | None
    result: ScoreResult
    label_matches_verdict: bool | None = Field(
        description="True when the verdict agrees with the generator's label; "
        "null when no verdict was given."
    )


@app.post("/demo", response_model=Demo, tags=["scoring"])
def demo(
    kind: SessionKind = Query(default="human"),
    seed: int | None = Query(default=None, ge=0),
) -> Demo:
    """Generate and immediately score a session. Convenience wrapper for demos."""
    session = synth(kind=kind, seed=seed)
    result = _scorer.score_session(session)
    return Demo(
        kind=kind,
        seed=seed,
        result=result,
        label_matches_verdict=(
            None if result.verdict is None else result.verdict == (kind != "human")
        ),
    )
