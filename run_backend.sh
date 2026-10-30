#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR/backend"
source .venv/bin/activate
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
