# Consulting Desk — 004_NLP_TDA_CONSULTING

Self-contained consulting MVP: ingest Office/PDF/email packs → extract master data → review → Excel export.

Everything you need to run is **in this folder** (app source, fixtures, Docker, scripts, runtime `data/`).

## Quick start (Docker)

```bash
cd 004_NLP_TDA_CONSULTING
cp -n .env.example .env
./scripts/up.sh
```

Open:

- http://127.0.0.1:38417/ — Consulting Desk
- http://127.0.0.1:38417/review — Review records
- http://127.0.0.1:38417/export — Excel export

```bash
./scripts/logs.sh     # follow logs
./scripts/smoke.sh    # extract synthetic sample engagement
./scripts/down.sh     # stop
./scripts/up.sh --build   # force image rebuild (slow / multi-GB)
```

## Folder layout

| Path | Purpose |
|------|---------|
| `src/` | FastAPI app + UI templates |
| `fixtures/` | Synthetic engagement pack |
| `tests/` | Pytest suite |
| `Dockerfile` / `docker-compose.yml` | Container build & run |
| `scripts/` | up / down / logs / smoke |
| `data/` | SQLite, Chroma, uploads, Excel exports (persisted) |

## Local (without Docker)

```bash
cd 004_NLP_TDA_CONSULTING
uv sync
uv run nlp-tda serve --host 0.0.0.0 --port 38417
```

## Local LLM (Ollama in Docker)

A fresh `.env` uses the mock LLM + hash embeddings. For real extraction and answers:

1. `docker compose up -d` starts the `ollama` service next to the app.
2. Pull a model once (kept in the `ollama-models` volume):
   `docker compose exec ollama ollama pull qwen2.5:14b`
3. In `.env`: `NLP_TDA_FORCE_MOCK_LLM=false`, `NLP_TDA_USE_HASH_EMBEDDINGS=false`,
   `NLP_TDA_OLLAMA_MODEL=qwen2.5:14b`
4. `docker compose up -d` again so the app picks up `.env`.

Inside Docker on macOS the model runs on CPU only. For GPU speed run Ollama on the host
instead and set `NLP_TDA_OLLAMA_BASE_URL=http://host.docker.internal:11434`.
