#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
CONFIG_PATH="${CONFIG_PATH:-config.json}"
INPUT_PATH="${INPUT_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
LOG_DIR="${LOG_DIR:-}"
STATE_DIR="${STATE_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"
PHASE_UNTIL="${PHASE_UNTIL:-}"
PHASE_FROM="${PHASE_FROM:-}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vilier_mplconfig}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/vilier_numba_cache}"

args=(--config "${CONFIG_PATH}")
if [[ -n "${INPUT_PATH}" ]]; then
  args+=(--input "${INPUT_PATH}")
fi
if [[ -n "${OUTPUT_PATH}" ]]; then
  args+=(--output "${OUTPUT_PATH}")
fi
if [[ -n "${LOG_DIR}" ]]; then
  args+=(--log-dir "${LOG_DIR}")
fi
if [[ -n "${STATE_DIR}" ]]; then
  args+=(--state-dir "${STATE_DIR}")
fi
if [[ -n "${PHASE_UNTIL}" ]]; then
  args+=(--until "${PHASE_UNTIL}")
fi
if [[ -n "${PHASE_FROM}" ]]; then
  args+=(--from "${PHASE_FROM}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi
args+=("$@")

if command -v uv >/dev/null 2>&1 && [[ -z "${PYTHON_BIN_SET_BY_USER:-}" ]]; then
  PYTHONPATH="${SCRIPT_DIR}" uv run python -m pipeline.cli "${args[@]}"
else
  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" -m pipeline.cli "${args[@]}"
fi
