#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR/frontend"
source .venv/bin/activate
exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501
