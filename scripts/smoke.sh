#!/usr/bin/env bash
# One-shot smoke: extract sample engagement inside the running container.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
docker compose exec nlp-tda nlp-tda run --source /app/fixtures/synthetic_engagement --hash-embeddings
echo "Done. Open /review to accept draft records."
