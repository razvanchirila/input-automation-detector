"""FastAPI surface for the detector.

uvicorn iad.api:app --app-dir src
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, Query, Response
from pydantic import BaseModel, Field

from iad import __version__
from iad.config import settings
from iad.detectors import default_detectors
from iad.models import ScoreResult, Session
from iad.observability import (
    configure_logging,
    metrics_response,
    observe_request,
    record_result,
)
from iad.scoring import DEFAULT_THRESHOLD, DEFAULT_WEIGHTS, Scorer
from iad.synth import KINDS, SessionKind, generate_session

configure_logging(settings.log_level)
log = logging.getLogger("iad")

_scorer = Scorer()
_state = {"ready": False}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Score one session up front so lazy imports and first-call costs are paid
    # before /readyz reports ready.
    _scorer.score_session(generate_session(kind="human", seed=0))
    _state["ready"] = True
    log.info("ready", extra={"version": __version__, "max_events": settings.max_events})
    yield
    _state["ready"] = False
    log.info("shutting down")


app = FastAPI(
    title="Input Automation Detector",
    version=__version__,
    summary="Scores input-telemetry sessions for the likelihood of being automated.",
    lifespan=lifespan,
)
app.middleware("http")(observe_request)

KindParam = Annotated[SessionKind, Query(description=f"One of {', '.join(KINDS)}.")]
SeedParam = Annotated[int | None, Query(ge=0, description="Omit for a random one.")]


class Probe(BaseModel):
    status: Literal["ok", "starting"]


@app.get("/livez", response_model=Probe, tags=["ops"])
def livez() -> Probe:
    """Liveness: the process is up and serving HTTP. No dependencies checked."""
    return Probe(status="ok")


@app.get(
    "/readyz",
    response_model=Probe,
    tags=["ops"],
    responses={503: {"model": Probe, "description": "Still starting."}},
)
def readyz(response: Response) -> Probe:
    """Readiness: the scorer has been warmed up and the app accepts traffic."""
    if not _state["ready"]:
        response.status_code = 503
        return Probe(status="starting")
    return Probe(status="ok")


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    return metrics_response()


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    detectors: list[str]
    weights: dict[str, float]
    threshold: float


@app.get("/health", response_model=Health, tags=["ops"])
def health() -> Health:
    """Status plus the active scoring configuration."""
    return Health(
        status="ok",
        version=__version__,
        detectors=[d.name for d in default_detectors()],
        weights=DEFAULT_WEIGHTS,
        threshold=DEFAULT_THRESHOLD,
    )


def _score(session: Session) -> ScoreResult:
    result = _scorer.score_session(session)
    record_result(result)
    return result


@app.post("/score", response_model=ScoreResult, tags=["scoring"])
def score(session: Session) -> ScoreResult:
    """Score one session.

    Returns the combined score, every detector's individual score, and the
    boolean verdict at the configured threshold. A detector with too little data
    abstains (null) and is left out of the combined score; if all of them abstain
    the response has `insufficient_data: true` and a null score and verdict.
    """
    return _score(session)


@app.get("/synth", response_model=Session, tags=["scoring"])
def synth(kind: KindParam = "human", seed: SeedParam = None) -> Session:
    """Generate an example session so the API can be tried without real data.

    Feed the response straight back into `POST /score`.
    """
    return generate_session(kind=kind, seed=seed)


class Demo(BaseModel):
    kind: SessionKind
    seed: int | None
    result: ScoreResult
    label_matches_verdict: bool | None = Field(
        description="True when the verdict agrees with the generator's label; "
        "null when no verdict was given."
    )


@app.get("/demo", response_model=Demo, tags=["scoring"])
def demo(kind: KindParam = "human", seed: SeedParam = None) -> Demo:
    """Generate and immediately score a session. Convenience wrapper for demos."""
    result = _score(generate_session(kind=kind, seed=seed))
    return Demo(
        kind=kind,
        seed=seed,
        result=result,
        label_matches_verdict=(
            None if result.verdict is None else result.verdict == (kind != "human")
        ),
    )
