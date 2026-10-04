# Consulting Desk — build from this folder:
#   docker compose up --build

FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    NLP_TDA_HOST=0.0.0.0 \
    NLP_TDA_PORT=38417 \
    NLP_TDA_PROJECT_ROOT=/app \
    NLP_TDA_DATA_DIR=/app/data \
    NLP_TDA_SQLITE_PATH=/app/data/master.db \
    NLP_TDA_CHROMA_PATH=/app/data/chroma \
    NLP_TDA_UPLOADS_DIR=/app/data/uploads \
    NLP_TDA_EXPORTS_DIR=/app/data/exports \
    NLP_TDA_FIXTURES_DIR=/app/fixtures/synthetic_engagement \
    NLP_TDA_USE_HASH_EMBEDDINGS=true \
    NLP_TDA_FORCE_MOCK_LLM=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

# Dependencies first (torch and friends): this layer is reused until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY fixtures ./fixtures

RUN uv sync --frozen --no-dev \
    && mkdir -p /app/data/uploads /app/data/exports /app/data/chroma /app/data/packs

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 38417

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${NLP_TDA_PORT}/api/health" || exit 1

CMD ["nlp-tda", "serve", "--host", "0.0.0.0", "--port", "38417"]
