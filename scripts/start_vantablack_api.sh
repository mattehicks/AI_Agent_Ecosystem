#!/bin/bash
# Run on VantaBlack. Serves FastAPI on all interfaces :8000 (Ollama on same host: localhost:11434).
set -euo pipefail
ROOT="${ROOT:-/mnt/llm/AI_Agent_Ecosystem}"
cd "$ROOT"
if [[ -f venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi
export DISABLE_GPU_PLATFORM="${DISABLE_GPU_PLATFORM:-1}"
mkdir -p data logs
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
