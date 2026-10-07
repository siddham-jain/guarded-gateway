# syntax=docker/dockerfile:1.10
# gateway runtime image; multi-arch (linux/amd64, linux/arm64). guard model weights are not baked in:
# they live on the gg-models volume and are fetched once, at pinned revisions, on first start (docs/deploy.md).
ARG PYTHON_VERSION=3.13

FROM ghcr.io/astral-sh/uv:0.9 AS uv

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
# space-separated optional extras; "ml" adds onnxruntime, fastembed and presidio (no torch)
ARG EXTRAS="ml"
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project --no-editable \
        $(for extra in $EXTRAS; do printf -- '--extra %s ' "$extra"; done)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable \
        $(for extra in $EXTRAS; do printf -- '--extra %s ' "$extra"; done)
# token estimates need o200k_base; without the cached file an offline gateway falls back to a char heuristic
ENV TIKTOKEN_CACHE_DIR=/opt/gg/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG GIT_SHA=""
ARG BUILT_AT=""
LABEL org.opencontainers.image.title="gg" \
      org.opencontainers.image.description="OpenAI-compatible multi-provider LLM gateway" \
      org.opencontainers.image.revision="${GIT_SHA}"
RUN useradd --uid 10001 --create-home --home-dir /home/gg --shell /usr/sbin/nologin gg \
 && mkdir -p /opt/gg/models \
 && chown gg:gg /opt/gg/models
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/gg/tiktoken /opt/gg/tiktoken
COPY config ./config
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/opt/gg/tiktoken \
    HF_HOME=/opt/gg/models/.hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FORWARDED_ALLOW_IPS="172.16.0.0/12,192.168.0.0/16,10.0.0.0/8" \
    GG_ENV=prod \
    GG_LOG_FORMAT=json \
    GG_HOST=0.0.0.0 \
    GG_PORT=8000 \
    GG_CONFIG_DIR=/app/config \
    GG_GIT_SHA=${GIT_SHA} \
    GG_BUILT_AT=${BUILT_AT}
USER gg
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=300s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["gg", "serve"]
