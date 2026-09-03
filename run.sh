#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  export PYTHON_BIN_SET_BY_USER=1
else
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]] && ! command -v uv >/dev/null 2>&1; then
  echo "Python executable not found. Activate an environment or set PYTHON_BIN=/path/to/python." >&2
  exit 127
fi

CONFIG_PATH="${CONFIG_PATH:-config.json}"
INPUT_PATH="${INPUT_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
DRY_RUN="${DRY_RUN:-}"

export PYTHON_BIN
export CONFIG_PATH
if [[ -n "${INPUT_PATH}" ]]; then
  export INPUT_PATH
fi
if [[ -n "${OUTPUT_PATH}" ]]; then
  export OUTPUT_PATH
fi
if [[ -n "${DRY_RUN}" ]]; then
  export DRY_RUN
fi

if command -v uv >/dev/null 2>&1 && [[ -z "${PYTHON_BIN_SET_BY_USER:-}" ]]; then
  PYTHONPATH="${SCRIPT_DIR}" uv run python -m pipeline.run_config_log --config "${CONFIG_PATH}" "$@"
else
  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" -m pipeline.run_config_log --config "${CONFIG_PATH}" "$@"
fi

bash run_pipeline.sh "$@"
