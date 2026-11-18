"""Structured logging, request IDs and Prometheus metrics."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    disable_created_metrics,
    generate_latest,
    multiprocess,
)
from starlette.routing import Match

from iad.models import ScoreResult

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

disable_created_metrics()  # type: ignore[no-untyped-call]

_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
if _MULTIPROC_DIR:
    os.makedirs(_MULTIPROC_DIR, exist_ok=True)

HTTP_REQUESTS = Counter(
    "iad_http_requests_total",
    "HTTP requests by method, route template and status code.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "iad_http_request_duration_seconds",
    "HTTP request latency by method and route template.",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
SCORES = Histogram(
    "iad_score",
    "Combined bot-likelihood score of scored sessions.",
    buckets=tuple(round(0.05 * i, 2) for i in range(1, 21)),
)
VERDICTS = Counter(
    "iad_verdicts_total",
    "Scoring outcomes: bot, human, or insufficient (every detector abstained).",
    ["verdict"],
)

_STD_RECORD_KEYS = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "color_message",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra= fields are merged into the object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid is not None:
            payload["request_id"] = rid
        for key, value in vars(record).items():
            if key not in _STD_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    formatter = JsonFormatter()
    formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Route uvicorn through the JSON handler; its access log is replaced by ours.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    logging.getLogger("uvicorn.access").disabled = True


access_log = logging.getLogger("iad.access")


def _route_template(request: Request) -> str:
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match is Match.FULL:
            return str(getattr(route, "path", "unmatched"))
    return "unmatched"


async def observe_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Assign a request ID, time the request, record metrics, log one line."""
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    rid = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers[REQUEST_ID_HEADER] = rid
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = _route_template(request)
        HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
        HTTP_LATENCY.labels(request.method, route).observe(elapsed)
        access_log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "route": route,
                "status": status,
                "duration_ms": round(elapsed * 1000, 2),
            },
        )
        request_id_var.reset(token)


def record_result(result: ScoreResult) -> None:
    if result.score is None:
        VERDICTS.labels("insufficient").inc()
        return
    SCORES.observe(result.score)
    VERDICTS.labels("bot" if result.verdict else "human").inc()


def metrics_response() -> Response:
    registry: CollectorRegistry = REGISTRY
    if _MULTIPROC_DIR:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
