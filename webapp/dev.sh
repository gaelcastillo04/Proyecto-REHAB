#!/usr/bin/env bash
# Levanta backend (FastAPI :8000) y frontend (Vite :5173) juntos. Ctrl+C detiene ambos.
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"
[ -x "$PY" ] || { echo "No existe ../.venv — crea el entorno del proyecto primero (ver README)."; exit 1; }
"$PY" -c "import fastapi, uvicorn" 2>/dev/null || "$PY" -m pip install -r backend/requirements.txt
[ -d frontend/node_modules ] || (cd frontend && npm install)
trap 'kill 0' EXIT
(cd backend && "../$PY" -m uvicorn main:app --reload --port 8000) &
(cd frontend && npm run dev) &
wait
