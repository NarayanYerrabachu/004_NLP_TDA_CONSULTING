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

## Optional: host Ollama

Default container uses mock LLM + hash embeddings. For a local model:

1. On the host: `OLLAMA_HOST=0.0.0.0:11434 ollama serve` then `ollama pull qwen2.5:7b`
   (must listen on `0.0.0.0`, not only `127.0.0.1`, so Docker can reach it)
2. In `.env`: `NLP_TDA_FORCE_MOCK_LLM=false`
3. `./scripts/up.sh`

Compose defaults `NLP_TDA_OLLAMA_BASE_URL` to the Docker bridge gateway `http://172.18.0.1:11434`.
On Docker Desktop you can switch to `http://host.docker.internal:11434`.
