#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/sommelier/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-config.json}"
eval "$(
  "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))
asr = config.get("asr", {})
runtime = config.get("runtime", {})

def env_or_config(name, value):
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else value

runtime_backend = env_or_config("RUNTIME_BACKEND", runtime.get("backend", "local"))
kaggle = config.get("kaggle", {}) if runtime_backend == "kaggle" else asr.get("kaggle", {})

values = {
    "INPUT_PATH": env_or_config("INPUT_PATH", config.get("entrypoint", {}).get("input_path", "inputs")),
    "OUTPUT_PATH": env_or_config("OUTPUT_PATH", config.get("entrypoint", {}).get("output_path", "outputs")),
    "DRY_RUN": env_or_config("DRY_RUN", "1" if config.get("runtime", {}).get("dry_run", False) else "0"),
    "RUNTIME_BACKEND": runtime_backend,
    "ASR_ENABLED": "1" if bool(asr.get("enabled", False)) else "0",
    "ASR_BACKEND": env_or_config("ASR_BACKEND", asr.get("asr_backend", asr.get("run_backend", "local"))),
    "KAGGLE_DATASET_SLUG": env_or_config("KAGGLE_DATASET_SLUG", kaggle.get("dataset_slug", "ngocbaotrinhtuan/vilier-asr-bundle")),
    "KAGGLE_KERNEL_SLUG": env_or_config("KAGGLE_KERNEL_SLUG", kaggle.get("kernel_slug", "ngocbaotrinhtuan/vilier-phowhisper-asr")),
    "KAGGLE_ACCELERATOR": env_or_config("KAGGLE_ACCELERATOR", kaggle.get("accelerator", "NvidiaTeslaT4")),
    "KAGGLE_ASR_WORK_DIR": env_or_config("KAGGLE_ASR_WORK_DIR", asr.get("kaggle", {}).get("work_dir", ".kaggle_asr_work")),
    "KAGGLE_PIPELINE_WORK_DIR": env_or_config("KAGGLE_PIPELINE_WORK_DIR", kaggle.get("work_dir", ".kaggle_pipeline_work")),
    "KAGGLE_POLL_SECONDS": env_or_config("KAGGLE_POLL_SECONDS", str(kaggle.get("poll_seconds", 30))),
    "KAGGLE_MAX_WAIT_SECONDS": env_or_config("KAGGLE_MAX_WAIT_SECONDS", str(kaggle.get("max_wait_seconds", 21600))),
    "KAGGLE_DATASET_READY_SECONDS": env_or_config("KAGGLE_DATASET_READY_SECONDS", str(kaggle.get("dataset_ready_seconds", 240))),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

export KAGGLE_DATASET_SLUG
export KAGGLE_KERNEL_SLUG
export KAGGLE_ACCELERATOR
export KAGGLE_ASR_WORK_DIR
export KAGGLE_PIPELINE_WORK_DIR
export KAGGLE_POLL_SECONDS
export KAGGLE_MAX_WAIT_SECONDS
export KAGGLE_DATASET_READY_SECONDS

PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" -m pipeline.run_config_log \
  --config "${CONFIG_PATH}" \
  --runtime-backend "${RUNTIME_BACKEND}" \
  --asr-backend "${ASR_BACKEND}"

if [[ "${RUNTIME_BACKEND}" == "kaggle" ]]; then
  audio_info="$(
    PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" - "${INPUT_PATH}" <<'PY'
from pathlib import Path
import sys
from pipeline.audio import iter_audio_files

files = iter_audio_files(Path(sys.argv[1]).expanduser().resolve())
if len(files) != 1:
    raise SystemExit(f"runtime.backend=kaggle currently expects exactly one input audio, got {len(files)}")
print(f"audio_id={files[0].stem}")
print(f"audio_path={files[0]}")
PY
  )"
  audio_id="$(printf '%s\n' "${audio_info}" | awk -F= '/^audio_id=/{print $2}')"
  audio_path="$(printf '%s\n' "${audio_info}" | awk -F= '/^audio_path=/{print $2}')"
  if [[ -z "${audio_id}" || -z "${audio_path}" ]]; then
    echo "Could not determine Kaggle pipeline audio input:" >&2
    printf '%s\n' "${audio_info}" >&2
    exit 1
  fi

  echo "Bundle đã tạo, đang upload lên kaggle để chạy pipeline"
  kaggle_pipeline_args=(
    --audio-id "${audio_id}"
    --audio "${audio_path}"
    --config "${CONFIG_PATH}"
    --output-dir "${OUTPUT_PATH%/}/${audio_id}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    kaggle_pipeline_args+=(--dry-run)
  fi

  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" tools/kaggle_pipeline.py "${kaggle_pipeline_args[@]}"

  echo "Đã tải outputs, đang chạy các phase còn lại"
  if [[ "${ASR_ENABLED}" == "1" ]]; then
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    INPUT_PATH="${INPUT_PATH}" \
    OUTPUT_PATH="${OUTPUT_PATH}" \
    DRY_RUN="${DRY_RUN}" \
    PHASE_FROM="post_asr" \
    bash run_pipeline.sh
  fi
  exit 0
fi

if [[ "${RUNTIME_BACKEND}" != "local" ]]; then
  echo "Unsupported runtime.backend=${RUNTIME_BACKEND}; expected local or kaggle" >&2
  exit 2
fi

if [[ "${ASR_ENABLED}" != "1" ]]; then
  echo "ASR disabled in config, skipping ASR backend ${ASR_BACKEND}"
  PYTHON_BIN="${PYTHON_BIN}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  INPUT_PATH="${INPUT_PATH}" \
  OUTPUT_PATH="${OUTPUT_PATH}" \
  DRY_RUN="${DRY_RUN}" \
  bash run_pipeline.sh
  exit 0
fi

if [[ "${ASR_BACKEND}" == "local" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  INPUT_PATH="${INPUT_PATH}" \
  OUTPUT_PATH="${OUTPUT_PATH}" \
  DRY_RUN="${DRY_RUN}" \
  bash run_pipeline.sh
  exit 0
fi

if [[ "${ASR_BACKEND}" != "kaggle" ]]; then
  echo "Unsupported ASR_BACKEND=${ASR_BACKEND}; expected local or kaggle" >&2
  exit 2
fi

audio_id="$(
  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" - "${INPUT_PATH}" <<'PY'
from pathlib import Path
import sys
from pipeline.audio import iter_audio_files

files = iter_audio_files(Path(sys.argv[1]).expanduser().resolve())
if len(files) != 1:
    raise SystemExit(f"ASR_BACKEND=kaggle currently expects exactly one input audio, got {len(files)}")
print(files[0].stem)
PY
)"

PYTHON_BIN="${PYTHON_BIN}" \
CONFIG_PATH="${CONFIG_PATH}" \
INPUT_PATH="${INPUT_PATH}" \
OUTPUT_PATH="${OUTPUT_PATH}" \
DRY_RUN="${DRY_RUN}" \
PHASE_UNTIL="pre_asr" \
bash run_pipeline.sh

bundle_output="$(AUDIO_ID="${audio_id}" OUTPUT_PATH="${OUTPUT_PATH}" ./bundle.sh "${audio_id}")"
bundle_path="$(printf '%s\n' "${bundle_output}" | awk -F= '/^bundle=/{print $2}')"
if [[ -z "${bundle_path}" || ! -f "${bundle_path}" ]]; then
  echo "Could not determine ASR bundle path from bundle.sh output:" >&2
  printf '%s\n' "${bundle_output}" >&2
  exit 1
fi

echo "Bundle đã tạo, đang upload lên kaggle để lấy kết quả"

kaggle_args=(
  --audio-id "${audio_id}"
  --bundle "${bundle_path}"
  --output-dir "${OUTPUT_PATH%/}/${audio_id}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  kaggle_args+=(--dry-run)
fi

PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" tools/kaggle_asr.py "${kaggle_args[@]}"

PYTHON_BIN="${PYTHON_BIN}" \
CONFIG_PATH="${CONFIG_PATH}" \
INPUT_PATH="${INPUT_PATH}" \
OUTPUT_PATH="${OUTPUT_PATH}" \
DRY_RUN="${DRY_RUN}" \
PHASE_FROM="post_asr" \
bash run_pipeline.sh
