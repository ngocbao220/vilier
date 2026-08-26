#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_PATH="${SOURCE_PATH:-inputs/haveasip_khanhvi.mp3}"
OUTPUT_PATH="${OUTPUT_PATH:-inputs/haveasip_khanhvi_5m.wav}"
DURATION_SECONDS="${DURATION_SECONDS:-300}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
CHANNELS="${CHANNELS:-1}"
SEED="${SEED:-}"

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "Missing ffprobe. Install FFmpeg first." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Missing ffmpeg. Install FFmpeg first." >&2
  exit 1
fi

if [[ ! -f "${SOURCE_PATH}" ]]; then
  echo "Missing source audio: ${SOURCE_PATH}" >&2
  exit 1
fi

SOURCE_DURATION="$(
  ffprobe \
    -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "${SOURCE_PATH}"
)"

START_SECONDS="$(
  SOURCE_DURATION="${SOURCE_DURATION}" DURATION_SECONDS="${DURATION_SECONDS}" SEED="${SEED}" python3 - <<'PY'
import os
import random

source_duration = float(os.environ["SOURCE_DURATION"])
duration = float(os.environ["DURATION_SECONDS"])
seed = os.environ.get("SEED", "")

if source_duration < duration:
    raise SystemExit(f"Source is shorter than requested sample: {source_duration:.3f}s < {duration:.3f}s")

if seed:
    random.seed(seed)

print(f"{random.uniform(0.0, source_duration - duration):.3f}")
PY
)"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "source=${SOURCE_PATH}"
echo "source_duration_sec=${SOURCE_DURATION}"
echo "sample_start_sec=${START_SECONDS}"
echo "sample_duration_sec=${DURATION_SECONDS}"
echo "output=${OUTPUT_PATH}"

ffmpeg \
  -hide_banner \
  -loglevel error \
  -y \
  -ss "${START_SECONDS}" \
  -i "${SOURCE_PATH}" \
  -t "${DURATION_SECONDS}" \
  -ar "${SAMPLE_RATE}" \
  -ac "${CHANNELS}" \
  -c:a pcm_s16le \
  "${OUTPUT_PATH}"

ffprobe \
  -v error \
  -show_entries stream=codec_name,sample_rate,channels,bits_per_sample:format=duration \
  -of default=noprint_wrappers=1 \
  "${OUTPUT_PATH}"
