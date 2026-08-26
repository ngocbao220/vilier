from pathlib import Path
import importlib
import sys
import types
from typing import Callable

import numpy as np

from .schema import SpeakerSegment


def apply_overlap_separation(
    waveform: np.ndarray,
    sample_rate: int,
    segments: list[SpeakerSegment],
    separator,
    overlap_threshold: float,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    if separator is None:
        return {"segment_audio": {}, "overlap_regions": []}

    pairs = detect_overlapping_pairs(segments, overlap_threshold)
    if not pairs:
        return {"segment_audio": {}, "overlap_regions": []}

    separated_regions: dict[str, list[dict]] = {segment.id: [] for segment in segments}
    overlap_regions = []
    total = len(pairs)
    for pair_idx, pair in enumerate(pairs, start=1):
        if progress_callback is not None:
            progress_callback(pair_idx, total, f"{pair['seg1'].speaker}+{pair['seg2'].speaker}")
        start = pair["start"]
        end = pair["end"]
        start_idx = int(round(start * sample_rate))
        end_idx = int(round(end * sample_rate))
        overlap_audio = waveform[start_idx:end_idx]
        if overlap_audio.size == 0:
            continue

        src1, src2 = separator.separate(overlap_audio, sample_rate)
        src1 = _match_length(np.asarray(src1, dtype=np.float32), len(overlap_audio))
        src2 = _match_length(np.asarray(src2, dtype=np.float32), len(overlap_audio))
        seg1_audio, seg2_audio = _assign_sources_by_energy(pair["seg1"], pair["seg2"], src1, src2)
        seg1_audio = _match_target_rms(seg1_audio, _segment_non_overlap_rms(pair["seg1"], waveform, sample_rate, pairs) or _rms(overlap_audio) * 0.7)
        seg2_audio = _match_target_rms(seg2_audio, _segment_non_overlap_rms(pair["seg2"], waveform, sample_rate, pairs) or _rms(overlap_audio) * 0.7)

        separated_regions[pair["seg1"].id].append({"start": start, "end": end, "audio": seg1_audio})
        separated_regions[pair["seg2"].id].append({"start": start, "end": end, "audio": seg2_audio})
        overlap_regions.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 6),
                "speakers": [pair["seg1"].speaker, pair["seg2"].speaker],
                "segments": [pair["seg1"].id, pair["seg2"].id],
            }
        )

    segment_audio = {}
    for segment in segments:
        regions = separated_regions.get(segment.id) or []
        if not regions:
            continue
        segment_audio[segment.id] = _reconstruct_segment_audio(waveform, sample_rate, segment, regions)

    return {"segment_audio": segment_audio, "overlap_regions": overlap_regions}


def detect_overlapping_pairs(segments: list[SpeakerSegment], overlap_threshold: float) -> list[dict]:
    pairs = []
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.end, segment.speaker))
    for idx, first in enumerate(ordered):
        for second in ordered[idx + 1 :]:
            if second.start >= first.end:
                break
            if second.speaker == first.speaker:
                continue
            start = max(first.start, second.start)
            end = min(first.end, second.end)
            if end - start >= overlap_threshold:
                pairs.append({"seg1": first, "seg2": second, "start": start, "end": end, "duration": end - start})
    return pairs


def load_overlap_separator(config: dict, dry_run: bool = False, warnings: list[str] | None = None):
    if not config.get("enabled", False):
        return None
    if dry_run:
        return NoOpSeparator()
    backend = config.get("backend", "sepreformer")
    if backend != "sepreformer":
        raise ValueError(f"Unsupported overlap separation backend: {backend}")
    try:
        return SepReformerSeparator(
            _resolve_sepreformer_path(config.get("sepreformer_path", "SepReformer")),
            config.get("device", "cpu"),
            config.get("model_name", "SepReformer_Base_WSJ0"),
        )
    except Exception as exc:
        if warnings is not None:
            warnings.append(str(exc))
        return None


class NoOpSeparator:
    def separate(self, audio_segment: np.ndarray, sample_rate: int):
        return audio_segment, audio_segment


class SepReformerSeparator:
    def __init__(self, sepreformer_path: Path, device: str, model_name: str = "SepReformer_Base_WSJ0"):
        import torch
        import yaml

        device = str(device).strip().lower()
        if device == "gpu":
            device = "cuda"
        self.sepreformer_path = sepreformer_path.expanduser().resolve()
        self.model_name = _validate_model_name(model_name)
        self.device = torch.device(device)
        if not self.sepreformer_path.exists():
            raise FileNotFoundError(f"SepReformer path not found: {self.sepreformer_path}")

        original_sys_path = sys.path.copy()
        restored_modules = {}
        try:
            if str(self.sepreformer_path) not in sys.path:
                sys.path.insert(0, str(self.sepreformer_path))
            Model = _import_sepreformer_model_class(self.sepreformer_path, self.model_name, restored_modules)

            config_path = self.sepreformer_path / "models" / self.model_name / "configs.yaml"
            with config_path.open("r", encoding="utf-8") as handle:
                self.config = yaml.safe_load(handle)["config"]
            checkpoint_dir = _find_checkpoint_dir(self.sepreformer_path, self.model_name)
            checkpoints = sorted(path for path in checkpoint_dir.iterdir() if path.suffix in {".pt", ".pth"})
            if not checkpoints:
                raise FileNotFoundError(f"No SepReformer checkpoint found in {checkpoint_dir}")
            self.model = Model(**self.config["model"])
            checkpoint = torch.load(checkpoints[-1], map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
        finally:
            sys.path = original_sys_path
            for module_name in list(sys.modules):
                if module_name == "utils" or module_name.startswith("utils."):
                    del sys.modules[module_name]
            for module_name, module in restored_modules.items():
                sys.modules[module_name] = module

    def separate(self, audio_segment: np.ndarray, sample_rate: int):
        import librosa
        import torch

        if sample_rate != 8000:
            audio_8k = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=8000)
        else:
            audio_8k = audio_segment
        mixture = torch.tensor(audio_8k, dtype=torch.float32).unsqueeze(0)
        stride = self.config["model"]["module_audio_enc"]["stride"]
        remains = mixture.shape[-1] % stride
        if remains:
            mixture = torch.nn.functional.pad(mixture, (0, stride - remains), "constant", 0)
        with torch.inference_mode():
            estim_src, _ = self.model(mixture.to(self.device))
            src1 = estim_src[0][..., : len(audio_8k)].squeeze().cpu().numpy()
            src2 = estim_src[1][..., : len(audio_8k)].squeeze().cpu().numpy()
        if sample_rate != 8000:
            src1 = librosa.resample(src1, orig_sr=8000, target_sr=sample_rate)
            src2 = librosa.resample(src2, orig_sr=8000, target_sr=sample_rate)
        return _match_length(src1, len(audio_segment)), _match_length(src2, len(audio_segment))


def _resolve_sepreformer_path(configured_path: str | Path) -> Path:
    path = Path(configured_path).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, Path(__file__).resolve().parents[1] / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[-1].resolve()


def _import_sepreformer_model_class(sepreformer_path: Path, model_name: str, restored_modules: dict | None = None):
    restored_modules = restored_modules if restored_modules is not None else {}
    model_root = sepreformer_path / "models"
    utils_root = sepreformer_path / "utils"
    if not (model_root / model_name / "model.py").exists():
        raise ModuleNotFoundError(f"SepReformer model file not found: {model_root / model_name / 'model.py'}")
    if not (utils_root / "decorators.py").exists():
        raise ModuleNotFoundError(f"SepReformer utils file not found: {utils_root / 'decorators.py'}")

    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            restored_modules.setdefault(module_name, sys.modules[module_name])
            del sys.modules[module_name]

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(utils_root)]
    sys.modules["utils"] = utils_pkg

    alias = "_vilier_sepreformer_models"
    alias_pkg = sys.modules.get(alias)
    if alias_pkg is None:
        alias_pkg = types.ModuleType(alias)
        sys.modules[alias] = alias_pkg
    alias_pkg.__path__ = [str(model_root)]
    importlib.invalidate_caches()

    module = importlib.import_module(f"{alias}.{model_name}.model")
    return module.Model


def _validate_model_name(model_name: str) -> str:
    if "/" in model_name or "\\" in model_name or model_name in {"", ".", ".."}:
        raise ValueError(f"Invalid SepReformer model name: {model_name}")
    return model_name


def _find_checkpoint_dir(sepreformer_path: Path, model_name: str = "SepReformer_Base_WSJ0") -> Path:
    base_dir = sepreformer_path / "models" / _validate_model_name(model_name) / "log"
    candidates = [
        base_dir / "pretrain_weights",
        base_dir / "pretrained_weights",
        base_dir / "scratch_weights",
        base_dir / "scratch_weight",
    ]
    existing_dirs = [path for path in candidates if path.exists() and any(path.iterdir())]
    if existing_dirs:
        return existing_dirs[0]
    if candidates[0].exists():
        return candidates[0]
    if candidates[2].exists():
        return candidates[2]
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No SepReformer checkpoint directory found. Checked: {checked}")


def _assign_sources_by_energy(seg1: SpeakerSegment, seg2: SpeakerSegment, src1: np.ndarray, src2: np.ndarray):
    energy1 = float(np.sum(src1 * src1))
    energy2 = float(np.sum(src2 * src2))
    seg1_is_longer = (seg1.end - seg1.start) >= (seg2.end - seg2.start)
    src1_is_louder = energy1 >= energy2
    if seg1_is_longer == src1_is_louder:
        return src1, src2
    return src2, src1


def _reconstruct_segment_audio(waveform: np.ndarray, sample_rate: int, segment: SpeakerSegment, regions: list[dict]) -> np.ndarray:
    parts = []
    current = segment.start
    for region in sorted(regions, key=lambda item: item["start"]):
        region_start = max(float(region["start"]), segment.start)
        region_end = min(float(region["end"]), segment.end)
        if current < region_start:
            parts.append(waveform[int(round(current * sample_rate)) : int(round(region_start * sample_rate))])
        parts.append(_match_length(np.asarray(region["audio"], dtype=np.float32), int(round((region_end - region_start) * sample_rate))))
        current = region_end
    if current < segment.end:
        parts.append(waveform[int(round(current * sample_rate)) : int(round(segment.end * sample_rate))])
    if not parts:
        return waveform[int(round(segment.start * sample_rate)) : int(round(segment.end * sample_rate))].copy()
    return np.concatenate(parts).astype(np.float32, copy=False)


def _segment_non_overlap_rms(segment: SpeakerSegment, waveform: np.ndarray, sample_rate: int, pairs: list[dict]) -> float | None:
    regions = []
    for pair in pairs:
        if pair["seg1"].id == segment.id or pair["seg2"].id == segment.id:
            regions.append((pair["start"], pair["end"]))
    if not regions:
        audio = waveform[int(round(segment.start * sample_rate)) : int(round(segment.end * sample_rate))]
        return _rms_or_none(audio)
    pieces = []
    current = segment.start
    for start, end in sorted(regions):
        if current < start:
            pieces.append(waveform[int(round(current * sample_rate)) : int(round(start * sample_rate))])
        current = max(current, end)
    if current < segment.end:
        pieces.append(waveform[int(round(current * sample_rate)) : int(round(segment.end * sample_rate))])
    if not pieces:
        return None
    return _rms_or_none(np.concatenate(pieces))


def _match_target_rms(source: np.ndarray, target_rms: float | None) -> np.ndarray:
    source_rms = _rms_or_none(source)
    if source_rms is None or target_rms is None:
        return source
    return np.clip(source * (target_rms / source_rms), -1.0, 1.0)


def _rms_or_none(audio: np.ndarray) -> float | None:
    if audio.size == 0:
        return None
    rms = _rms(audio)
    return rms if rms > 1e-10 else None


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) > length:
        return audio[:length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), mode="constant")
    return audio
