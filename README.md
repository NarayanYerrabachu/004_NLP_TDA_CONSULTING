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

## How a run behaves

- **Background job.** `POST /api/pipeline/run` starts the run and returns a `job_id`;
  `GET /api/pipeline/jobs/{job_id}` reports `stage`, `done` / `total` LLM calls, and the `result`
  when it is `completed` (or `error` when `failed`). One run at a time: a second start gets 409.
- **Confidence comes from evidence**, not from the LLM: `verbatim` 0.9 (the name / statement
  stands word for word in the source chunk), `cited` 0.6 (the LLM named the chunk, the wording is
  its own), `none` 0.3 (no source), +0.1 when more than one LLM call or wording produced it.
- **No mock records in real results.** With `NLP_TDA_FORCE_MOCK_LLM=false` an unreachable LLM, or
  one that answers no call, fails the run with a message. Fixture records only come from mock mode.
- **Re-runs** replace the unreviewed drafts of the same documents; accepted, edited and rejected
  records stay and are not proposed again.

## Access and limits

- The port is published on `127.0.0.1` only. To reach the app from other machines set
  `NLP_TDA_BIND=0.0.0.0` and `NLP_TDA_AUTH_PASSWORD` in `.env`; every page and API call then
  needs that password (HTTP Basic, any user name; `/api/health` stays open). Basic auth is not
  encrypted: put TLS in front for anything beyond a trusted network.
- A run reads an upload batch or the synthetic fixtures; a request cannot name a server folder.
- Uploads are limited to `NLP_TDA_UPLOAD_MAX_MB` (200) per file; a ZIP to
  `NLP_TDA_ZIP_MAX_FILES` (2000) files and `NLP_TDA_ZIP_MAX_MB` (500) unpacked.
