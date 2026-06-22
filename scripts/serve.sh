#!/usr/bin/env bash
# Launch the FaceStack service. On motis this exports the ROCm compat lib path
# so the GPU (ROCMExecutionProvider) actually loads instead of silently falling
# back to CPU. Harmless on a CPU-only box (the dir just won't contain anything).
set -euo pipefail

cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate

export LD_LIBRARY_PATH="${COMPAT_DIR:-$HOME/rocm-compat}:${LD_LIBRARY_PATH:-}"

HOST="${FACESTACK_HOST:-0.0.0.0}"
PORT="${FACESTACK_PORT:-8000}"
WORKERS="${FACESTACK_WORKERS:-1}"  # keep 1: each worker loads its own GPU model copy

exec uvicorn facestack.service.app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
