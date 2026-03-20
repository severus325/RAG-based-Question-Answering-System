#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

APP_PYTHON_BIN="${APP_PYTHON_BIN:-/data/miniconda3/envs/ttt/bin/python}"

if [[ ! -x "${APP_PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${APP_PYTHON_BIN}" >&2
  exit 1
fi

exec "${APP_PYTHON_BIN}" -m uvicorn app.main:app \
  --host "${APP_HOST:-0.0.0.0}" \
  --port "${APP_PORT:-7862}" \
  --reload
