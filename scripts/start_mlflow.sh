#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

exec uv run mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "file:$ROOT_DIR/mlruns" \
  --default-artifact-root "file:$ROOT_DIR/mlruns" \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
