#!/usr/bin/env bash
# Start Consulting Desk from this project folder.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/uploads data/exports data/chroma data/packs
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

# Prefer classic builder when BuildKit overlayfs fails (some VMs)
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"

echo "Building & starting Consulting Desk…"
# Reuse an existing local image when present (full rebuild is multi-GB / slow on vfs).
# Pass --build to force: ./scripts/up.sh --build
if [[ "${1:-}" == "--build" ]]; then
  docker compose up --build -d
elif docker image inspect nlp-tda:local >/dev/null 2>&1; then
  docker compose up -d --no-build
else
  docker compose up --build -d
fi

PORT="$(grep -E '^NLP_TDA_PORT=' .env 2>/dev/null | cut -d= -f2 || true)"
PORT="${PORT:-38417}"
echo
echo "Consulting Desk → http://127.0.0.1:${PORT}/"
echo "  Review         → http://127.0.0.1:${PORT}/review"
echo "  Excel export   → http://127.0.0.1:${PORT}/export"
echo
echo "Logs:  ./scripts/logs.sh"
echo "Stop:  ./scripts/down.sh"
