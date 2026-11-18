# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/
RUN pip wheel --wheel-dir /wheels -r requirements.txt \
 && pip wheel --wheel-dir /wheels --no-deps . \
 && cp requirements.txt /wheels/


FROM ${PYTHON_IMAGE} AS runtime

ARG VERSION=0.1.0
ARG REVISION=unknown
LABEL org.opencontainers.image.title="input-automation-detector" \
      org.opencontainers.image.description="Scores input-telemetry sessions for the likelihood of being automated." \
      org.opencontainers.image.source="https://github.com/razvanchirila/input-automation-detector" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WEB_CONCURRENCY=1 \
    IAD_LOG_LEVEL=INFO

RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links=/wheels \
        -r /wheels/requirements.txt input-automation-detector \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=2).status == 200 else 1)"]

# Exec form so uvicorn is PID 1 and receives SIGTERM; workers come from WEB_CONCURRENCY.
CMD ["uvicorn", "iad.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--timeout-graceful-shutdown", "20", "--no-access-log"]
