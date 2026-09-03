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

BENCHMARK_INPUT="${BENCHMARK_INPUT:-${OUTPUT_PATH:-outputs}}"
BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT:-benchmarks}"

if command -v uv >/dev/null 2>&1 && [[ -z "${PYTHON_BIN_SET_BY_USER:-}" ]]; then
  PYTHONPATH="${SCRIPT_DIR}" uv run python -m pipeline.benchmark \
    --input "${BENCHMARK_INPUT}" \
    --output "${BENCHMARK_OUTPUT}" \
    "$@"
else
  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" -m pipeline.benchmark \
    --input "${BENCHMARK_INPUT}" \
    --output "${BENCHMARK_OUTPUT}" \
    "$@"
fi
