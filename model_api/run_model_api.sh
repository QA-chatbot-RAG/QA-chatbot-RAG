#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
exec uvicorn main:app --host 127.0.0.1 --port 8001
