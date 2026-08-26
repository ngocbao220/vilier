from pathlib import Path

import numpy as np

from .audio import write_wav
from .devices import resolve_auto_device
from .schema import relative_path


def load_music_separator(config: dict, dry_run: bool = False, warnings: list[str] | None = None):
    if not bool(config.get("enabled", False)):
        return None
    if dry_run:
        return NoOpMusicSeparator()
    backend = str(config.get("backend", "demucs"))
    if backend != "demucs":
        raise ValueError(f"Unsupported music separation backend: {backend}")
    try:
        return DemucsMusicSeparator(
            model_name=str(config.get("model", "htdemucs")),
            device=str(config.get("device", "auto")),
            shifts=int(config.get("shifts", 1)),
            split=bool(config.get("split", True)),
            overlap=float(config.get("overlap", 0.25)),
            residual_subtract=float(config.get("residual_subtract", 0.0)),
        )
    except Exception as exc:
        if warnings is not None:
            warnings.append(str(exc))
        return None


def apply_music_separation(
    waveform: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    separator,
) -> dict:
    if separator is None:
        return {"waveform": waveform, "applied": False, "audio": ""}

    cleaned = separator.separate_music(waveform, sample_rate)
    cleaned = _match_length(np.asarray(cleaned, dtype=np.float32), len(waveform))
    path = output_dir / "music_cleaned.wav"
    write_wav(path, cleaned, sample_rate)
    return {
        "waveform": cleaned,
        "applied": True,
        "audio": relative_path(path, output_dir),
    }


class NoOpMusicSeparator:
    resolved_device = "dry-run"

    def separate_music(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.asarray(waveform, dtype=np.float32).copy()


class DemucsMusicSeparator:
    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "auto",
        shifts: int = 1,
        split: bool = True,
        overlap: float = 0.25,
        residual_subtract: float = 0.0,
    ):
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        self.device = resolve_auto_device(torch, device, warn_label="music_separation.device")
        self.shifts = shifts
        self.split = split
        self.overlap = overlap
        self.residual_subtract = max(0.0, residual_subtract)
        self.apply_model = apply_model
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def separate_music(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        import librosa
        import torch

        audio = np.asarray(waveform, dtype=np.float32)
        model_sample_rate = int(getattr(self.model, "samplerate", sample_rate))
        if sample_rate != model_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=model_sample_rate)
        stereo = np.stack([audio, audio], axis=0) if audio.ndim == 1 else audio
        tensor = torch.from_numpy(stereo).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            sources = self.apply_model(
                self.model,
                tensor,
                device=self.device,
                shifts=self.shifts,
                split=self.split,
                overlap=self.overlap,
            )
        sources_np = sources[0].detach().cpu().numpy()
        vocals = sources_np[-1]
        accompaniment = np.sum(sources_np[:-1], axis=0) if len(sources_np) > 1 else np.zeros_like(vocals)
        if vocals.ndim > 1:
            vocals = vocals.mean(axis=0)
        if accompaniment.ndim > 1:
            accompaniment = accompaniment.mean(axis=0)
        if sample_rate != model_sample_rate:
            vocals = librosa.resample(vocals, orig_sr=model_sample_rate, target_sr=sample_rate)
            accompaniment = librosa.resample(accompaniment, orig_sr=model_sample_rate, target_sr=sample_rate)
        vocals = _suppress_accompaniment(vocals, accompaniment, self.residual_subtract)
        return vocals.astype(np.float32, copy=False)


def _suppress_accompaniment(vocals: np.ndarray, accompaniment: np.ndarray, strength: float) -> np.ndarray:
    vocals = np.asarray(vocals, dtype=np.float32)
    if strength <= 0:
        return vocals
    accompaniment = _match_length(np.asarray(accompaniment, dtype=np.float32), len(vocals))
    cleaned = vocals - float(strength) * accompaniment
    return np.clip(cleaned, -1.0, 1.0).astype(np.float32, copy=False)


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) > length:
        return audio[:length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), mode="constant")
    return audio
