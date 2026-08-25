import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".opus", ".ogg"}


def iter_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS and ".temp" not in path.name
    )


def load_mono(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    try:
        data, source_sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return _load_mono_ffmpeg(path, sample_rate)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if source_sr != sample_rate:
        divisor = math.gcd(int(source_sr), int(sample_rate))
        data = resample_poly(data, sample_rate // divisor, source_sr // divisor).astype("float32")
    return np.asarray(data, dtype=np.float32), sample_rate


def _load_mono_ffmpeg(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    try:
        output = subprocess.check_output(command)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to decode this audio file when soundfile cannot read it") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg could not decode audio file: {path}") from exc
    return np.frombuffer(output, dtype=np.float32).copy(), sample_rate


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(waveform, -1.0, 1.0), sample_rate, subtype="PCM_16")


def slice_waveform(waveform: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_idx = max(0, int(round(start * sample_rate)))
    end_idx = min(len(waveform), int(round(end * sample_rate)))
    return waveform[start_idx:end_idx]


def duration_seconds(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)
