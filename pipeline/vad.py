from pathlib import Path
from typing import Callable

import numpy as np

from .audio import slice_waveform, write_wav
from .schema import relative_path


class SileroVadRunner:
    def __init__(self, config: dict, sample_rate: int, dry_run: bool = False):
        self.config = config
        self.sample_rate = sample_rate
        self.dry_run = dry_run
        self.model = None if dry_run else self._load_model()

    def detect(self, waveform: np.ndarray) -> list[dict]:
        if waveform.size == 0:
            return []
        if self.model is None:
            return self._energy_fallback(waveform)
        return self._silero_detect(waveform)

    def _load_model(self):
        import sys

        candidates = [
            Path(__file__).resolve().parents[2] / "sommelier" / "podcast-pipeline",
            Path(__file__).resolve().parents[3] / "sommelier" / "podcast-pipeline",
        ]
        for candidate in candidates:
            if (candidate / "models" / "silero_vad.py").exists():
                sys.path.insert(0, str(candidate))
                break

        from models import silero_vad
        import torch

        return silero_vad.SileroVAD(
            model=self.config.get("model", "silero_vad"),
            device=torch.device("cpu"),
        )

    def _silero_detect(self, waveform: np.ndarray) -> list[dict]:
        from models import silero_vad
        import librosa

        target_sr = silero_vad.SAMPLING_RATE
        vad_audio = waveform
        if self.sample_rate != target_sr:
            vad_audio = librosa.resample(waveform, orig_sr=self.sample_rate, target_sr=target_sr)

        timestamps = self.model.get_speech_timestamps(
            vad_audio,
            self.model.vad_model,
            sampling_rate=target_sr,
            threshold=float(self.config.get("threshold", 0.5)),
        )
        segments = [{"start": ts["start"] / target_sr, "end": ts["end"] / target_sr} for ts in timestamps]
        return cleanup_intervals(
            segments,
            min_duration=float(self.config.get("min_duration_seconds", 0.25)),
            merge_gap=float(self.config.get("merge_gap_seconds", 0.2)),
        )

    def _energy_fallback(self, waveform: np.ndarray) -> list[dict]:
        frame_seconds = 0.03
        frame_len = max(1, int(self.sample_rate * frame_seconds))
        energies = []
        for start in range(0, len(waveform), frame_len):
            frame = waveform[start : start + frame_len]
            energies.append(float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0)
        threshold = max(0.01, float(np.percentile(energies, 60)) * 0.8)
        raw = []
        start_time = None
        for idx, energy in enumerate(energies):
            time_s = idx * frame_seconds
            if energy >= threshold and start_time is None:
                start_time = time_s
            elif energy < threshold and start_time is not None:
                raw.append({"start": start_time, "end": time_s})
                start_time = None
        if start_time is not None:
            raw.append({"start": start_time, "end": len(waveform) / self.sample_rate})
        return cleanup_intervals(
            raw,
            min_duration=float(self.config.get("min_duration_seconds", 0.25)),
            merge_gap=float(self.config.get("merge_gap_seconds", 0.2)),
        )


def cleanup_intervals(intervals: list[dict], min_duration: float, merge_gap: float) -> list[dict]:
    ordered = sorted(({"start": float(i["start"]), "end": float(i["end"])} for i in intervals), key=lambda x: x["start"])
    cleaned = []
    for interval in ordered:
        if interval["end"] <= interval["start"]:
            continue
        if cleaned and interval["start"] - cleaned[-1]["end"] <= merge_gap:
            cleaned[-1]["end"] = max(cleaned[-1]["end"], interval["end"])
        else:
            cleaned.append(interval)
    filtered = [i for i in cleaned if i["end"] - i["start"] >= min_duration]
    for idx, interval in enumerate(filtered):
        interval["id"] = f"vad_{idx:05d}"
        interval["start"] = round(interval["start"], 3)
        interval["end"] = round(interval["end"], 3)
    return filtered


def write_vad_txt(path: Path, segments: list[dict], label: str = "speech") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            handle.write(f"{start:.3f}\t{end:.3f}\t{label}\n")


def export_vad_audio(
    waveform: np.ndarray,
    sample_rate: int,
    segments: list[dict],
    output_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    vad_audio_dir = output_dir / "vad_audio"
    vad_audio_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    total = len(segments)
    for idx, segment in enumerate(segments, start=1):
        if progress_callback is not None:
            progress_callback(idx, total, str(segment.get("id", f"vad_{idx - 1:05d}")))
        audio = slice_waveform(waveform, sample_rate, float(segment["start"]), float(segment["end"]))
        path = vad_audio_dir / f"audio_{idx}.wav"
        write_wav(path, audio, sample_rate)
        paths.append(Path(relative_path(path, output_dir)))
    return paths
