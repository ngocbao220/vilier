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
        try:
            return _load_packaged_silero_vad()
        except ModuleNotFoundError as exc:
            if exc.name != "silero_vad":
                raise

        import sys

        candidates = [
            Path(__file__).resolve().parents[2] / "sommelier" / "podcast-pipeline",
            Path(__file__).resolve().parents[3] / "sommelier" / "podcast-pipeline",
        ]
        for candidate in candidates:
            if (candidate / "models" / "silero_vad.py").exists():
                sys.path.insert(0, str(candidate))
                break

        try:
            from models import silero_vad
            import torch

            return _LocalSileroVadAdapter(
                silero_vad.SileroVAD(
                    model=self.config.get("model", "silero_vad"),
                    device=torch.device("cpu"),
                ),
                sampling_rate=silero_vad.SAMPLING_RATE,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "models":
                raise

        return _load_torchhub_silero_vad()

    def _silero_detect(self, waveform: np.ndarray) -> list[dict]:
        import librosa

        target_sr = int(getattr(self.model, "sampling_rate", 16000))
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


class _LocalSileroVadAdapter:
    def __init__(self, model, sampling_rate: int):
        self.model = model
        self.sampling_rate = sampling_rate
        self.vad_model = model.vad_model

    def get_speech_timestamps(self, audio, model, **kwargs):
        return self.model.get_speech_timestamps(audio, model, **kwargs)


class _TorchHubSileroVad:
    def __init__(self, vad_model, get_speech_timestamps, sampling_rate: int = 16000):
        self.vad_model = vad_model
        self._get_speech_timestamps = get_speech_timestamps
        self.sampling_rate = sampling_rate

    def get_speech_timestamps(self, audio, model, **kwargs):
        return self._get_speech_timestamps(audio, model, **kwargs)


def _load_torchhub_silero_vad():
    import torch

    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    return _TorchHubSileroVad(vad_model=vad_model, get_speech_timestamps=get_speech_timestamps, sampling_rate=16000)


def _load_packaged_silero_vad():
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad_model = load_silero_vad()
    return _TorchHubSileroVad(vad_model=vad_model, get_speech_timestamps=get_speech_timestamps, sampling_rate=16000)


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
