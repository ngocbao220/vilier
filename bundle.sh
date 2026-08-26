#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./bundle.sh <audio_id>

Environment:
  AUDIO_ID=<audio_id>              Alternative to positional audio_id.
  OUTPUT_PATH=outputs              Root containing outputs/<audio_id>.
  BUNDLE_INPUT_DIR=<path>          Override exact prepared output directory.
  BUNDLE_OUTPUT_DIR=kaggle_asr_packages
  BUNDLE_NAME=<name>.zip           Override output zip filename.

Example:
  ./bundle.sh haveasip_khanhvi_5m
EOF
}

AUDIO_ID="${1:-${AUDIO_ID:-}}"
if [[ -z "${AUDIO_ID}" ]]; then
  usage
  echo
  echo "Available prepared outputs:"
  find "${OUTPUT_PATH:-outputs}" -maxdepth 2 -name manifest.timeline.json -print 2>/dev/null \
    | sed 's#^\./##; s#/manifest.timeline.json$##; s#^outputs/##' \
    | sort || true
  exit 2
fi

OUTPUT_PATH="${OUTPUT_PATH:-outputs}"
BUNDLE_INPUT_DIR="${BUNDLE_INPUT_DIR:-${OUTPUT_PATH%/}/${AUDIO_ID}}"
BUNDLE_OUTPUT_DIR="${BUNDLE_OUTPUT_DIR:-kaggle_asr_packages}"
BUNDLE_NAME="${BUNDLE_NAME:-${AUDIO_ID}_asr_bundle.zip}"

if [[ ! -d "${BUNDLE_INPUT_DIR}" ]]; then
  echo "Missing input output directory: ${BUNDLE_INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${BUNDLE_INPUT_DIR}/manifest.timeline.json" ]]; then
  echo "Missing ${BUNDLE_INPUT_DIR}/manifest.timeline.json" >&2
  exit 1
fi
if [[ ! -f "${BUNDLE_INPUT_DIR}/vad.json" ]]; then
  echo "Missing ${BUNDLE_INPUT_DIR}/vad.json" >&2
  exit 1
fi
asr_count="0"
if [[ -d "${BUNDLE_INPUT_DIR}/asr_audio" ]]; then
  asr_count="$(find "${BUNDLE_INPUT_DIR}/asr_audio" -type f -name '*.wav' | wc -l | tr -d ' ')"
fi
vad_count="0"
if [[ -d "${BUNDLE_INPUT_DIR}/vad_audio" ]]; then
  vad_count="$(find "${BUNDLE_INPUT_DIR}/vad_audio" -type f -name '*.wav' | wc -l | tr -d ' ')"
fi
if [[ "${asr_count}" == "0" && "${vad_count}" == "0" ]]; then
  echo "No ASR or VAD WAV files found under ${BUNDLE_INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${BUNDLE_OUTPUT_DIR}"
BUNDLE_DIR_ABS="$(cd "${BUNDLE_OUTPUT_DIR}" && pwd -P)"
BUNDLE_PATH="${BUNDLE_DIR_ABS}/${BUNDLE_NAME}"
TMP_BUNDLE="${BUNDLE_PATH}.tmp"
rm -f "${TMP_BUNDLE}" "${BUNDLE_PATH}"

(
  cd "${BUNDLE_INPUT_DIR}"
  entries=(manifest.timeline.json vad.json)
  if [[ "${asr_count}" != "0" ]]; then
    entries+=(asr_audio)
  fi
  if [[ "${vad_count}" != "0" ]]; then
    entries+=(vad_audio)
  fi
  zip -qr "${TMP_BUNDLE}" "${entries[@]}"
)
mv "${TMP_BUNDLE}" "${BUNDLE_PATH}"

python - "${BUNDLE_PATH}" "${AUDIO_ID}" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

bundle_path = Path(sys.argv[1])
audio_id = sys.argv[2]
with zipfile.ZipFile(bundle_path) as zf:
    names = set(zf.namelist())
    required = {"manifest.timeline.json", "vad.json"}
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"Bundle missing required entries: {missing}")
    manifest = json.loads(zf.read("manifest.timeline.json").decode("utf-8"))
    asr_files = sorted(name for name in names if name.startswith("asr_audio/") and name.endswith(".wav"))
    vad_files = sorted(name for name in names if name.startswith("vad_audio/") and name.endswith(".wav"))
    if manifest.get("asr_segments") and not asr_files:
        raise SystemExit("Bundle has asr_segments but no asr_audio/*.wav entries")
    if not manifest.get("asr_segments") and not vad_files:
        raise SystemExit("Bundle has no vad_audio/*.wav entries")

print(f"bundle={bundle_path}")
print(f"audio_id={manifest.get('audio_id', audio_id)}")
print(f"asr_audio_files={len(asr_files)}")
print(f"vad_audio_files={len(vad_files)}")
print(f"size_mb={bundle_path.stat().st_size / 1024 / 1024:.2f}")
PY
