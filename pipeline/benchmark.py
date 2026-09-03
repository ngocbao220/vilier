from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


METRIC_NAMES = (
    "track_count",
    "duration_sec",
    "active_track_ratio",
    "multi_active_ratio",
    "reconstruction_snr_db",
    "residual_rms",
    "clipped_sample_ratio",
    "mean_track_leakage_ratio",
    "overlap_region_count",
    "enhanced_segment_count",
)


def run_benchmark(input_dir: Path, output_dir: Path) -> Path:
    """Benchmark Vilier speaker-separated audio artifacts under an output root."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.jsonl"
    review_path = output_dir / "review_manifest.jsonl"
    rows = []
    with metrics_path.open("w", encoding="utf-8") as metrics_fp, review_path.open("w", encoding="utf-8") as review_fp:
        for run_dir in _iter_vilier_runs(input_dir):
            row = benchmark_run_dir(run_dir)
            rows.append(row)
            metrics_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            review_fp.write(json.dumps(_review_row(run_dir, row), ensure_ascii=False) + "\n")

    summary = _summary(rows, metrics_path, review_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def benchmark_run_dir(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.timeline.json"
    manifest = _read_json(manifest_path)
    sample_rate = int(manifest.get("sample_rate") or 16000)
    reference_path = _reference_audio_path(run_dir, manifest)
    reference, sample_rate = _read_audio(reference_path, sample_rate)
    tracks = _read_tracks(run_dir, manifest, sample_rate, len(reference))
    track_matrix = np.stack([track["audio"] for track in tracks], axis=0) if tracks else np.zeros((0, len(reference)), dtype=np.float32)
    active_mask = np.abs(track_matrix) > 1e-4 if tracks else np.zeros((0, len(reference)), dtype=bool)
    active_counts = active_mask.sum(axis=0) if tracks else np.zeros(len(reference), dtype=np.int16)
    reconstruction = track_matrix.sum(axis=0) if tracks else np.zeros(len(reference), dtype=np.float32)
    residual = reference - reconstruction
    resolved_config = _resolved_config(run_dir, manifest)
    overlap_summary = manifest.get("overlap_separation") or _read_optional_json(run_dir / "overlap_separation.json") or {}
    overlap_regions = list(overlap_summary.get("overlap_regions") or [])

    duration = len(reference) / float(sample_rate) if sample_rate else 0.0
    track_rows = [
        _track_metrics(track, _speaker_mask(manifest, track["speaker"], sample_rate, len(reference)), sample_rate)
        for track in tracks
    ]
    row = {
        "audio_id": str(manifest.get("audio_id") or run_dir.name),
        "run_dir": str(run_dir),
        "reference_audio": str(reference_path),
        "metric_status": "computed",
        "metric_names": list(METRIC_NAMES),
        "track_count": len(tracks),
        "duration_sec": round(duration, 6),
        "active_track_ratio": _ratio(active_counts > 0),
        "multi_active_ratio": _ratio(active_counts > 1),
        "reconstruction_snr_db": _snr_db(reference, residual),
        "residual_rms": _rms(residual),
        "clipped_sample_ratio": _ratio(np.abs(track_matrix) >= 0.999) if tracks else 0.0,
        "mean_track_leakage_ratio": _average(item.get("leakage_energy_ratio") for item in track_rows),
        "overlap_region_count": len(overlap_regions),
        "enhanced_segment_count": int(overlap_summary.get("enhanced_segment_count") or 0),
        "separation_backend": str((overlap_summary.get("backend") or _backend_from_config(resolved_config) or "none")),
        "tracks": track_rows,
        "overlap_regions": _overlap_metrics(run_dir, overlap_regions, sample_rate),
    }
    return row


def _iter_vilier_runs(input_dir: Path) -> Iterable[Path]:
    if (input_dir / "manifest.timeline.json").exists():
        yield input_dir
        return
    for manifest_path in sorted(input_dir.rglob("manifest.timeline.json")):
        yield manifest_path.parent


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _read_json(path)


def _reference_audio_path(run_dir: Path, manifest: dict) -> Path:
    music = manifest.get("music_separation") or {}
    if music.get("applied") and music.get("output_audio"):
        return _resolve_run_path(run_dir, str(music["output_audio"]))
    cleaned_path = run_dir / "music_cleaned.wav"
    if cleaned_path.exists():
        return cleaned_path
    return _resolve_run_path(run_dir, str(manifest.get("standardized_audio") or "audio.standardized.wav"))


def _read_tracks(run_dir: Path, manifest: dict, sample_rate: int, length: int) -> list[dict]:
    records = manifest.get("speakers") or []
    if not records:
        records = [{"id": path.stem, "track_wav": str(path.relative_to(run_dir))} for path in sorted((run_dir / "tracks").glob("*.wav"))]
    tracks = []
    for record in records:
        path = _resolve_run_path(run_dir, str(record.get("track_wav") or ""))
        if not path.exists():
            continue
        audio, _ = _read_audio(path, sample_rate)
        tracks.append({"speaker": str(record.get("id") or path.stem), "path": str(path), "audio": _match_length(audio, length)})
    return tracks


def _read_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(source_sr) != int(sample_rate):
        raise ValueError(f"{path} has sample_rate={source_sr}, expected {sample_rate}")
    return np.asarray(audio, dtype=np.float32), int(source_sr)


def _resolve_run_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) > length:
        return audio[:length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), mode="constant")
    return audio


def _ratio(mask: np.ndarray) -> float:
    return round(float(np.mean(mask)), 8) if mask.size else 0.0


def _rms(audio: np.ndarray) -> float:
    return round(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), 10) if audio.size else 0.0


def _snr_db(reference: np.ndarray, residual: np.ndarray) -> float | None:
    signal = float(np.mean(np.square(reference, dtype=np.float64)))
    noise = float(np.mean(np.square(residual, dtype=np.float64)))
    if signal <= 0.0:
        return None
    if noise <= 1e-12:
        return 120.0
    return round(10.0 * math.log10(signal / noise), 4)


def _speaker_mask(manifest: dict, speaker: str, sample_rate: int, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for segment in manifest.get("segments") or []:
        if str(segment.get("speaker") or "") != speaker:
            continue
        start_idx = max(0, int(round(float(segment.get("start") or 0.0) * sample_rate)))
        end_idx = min(length, int(round(float(segment.get("end") or 0.0) * sample_rate)))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = True
    return mask


def _track_metrics(track: dict, speaker_mask: np.ndarray, sample_rate: int) -> dict:
    audio = track["audio"]
    total_energy = float(np.sum(np.square(audio, dtype=np.float64)))
    has_labels = bool(speaker_mask.size and np.any(speaker_mask))
    outside_mask = ~speaker_mask if has_labels else np.zeros(len(audio), dtype=bool)
    inside_energy = float(np.sum(np.square(audio[speaker_mask], dtype=np.float64))) if speaker_mask.size else 0.0
    outside_energy = float(np.sum(np.square(audio[outside_mask], dtype=np.float64))) if outside_mask.size else 0.0
    return {
        "speaker": track["speaker"],
        "audio": track["path"],
        "rms": _rms(audio),
        "peak": round(float(np.max(np.abs(audio))), 8) if audio.size else 0.0,
        "active_ratio": _ratio(np.abs(audio) > 1e-4),
        "clipped_sample_ratio": _ratio(np.abs(audio) >= 0.999),
        "labeled_duration_sec": round(float(np.sum(speaker_mask)) / float(sample_rate), 6) if sample_rate else 0.0,
        "labeled_energy_ratio": round(inside_energy / total_energy, 8) if has_labels and total_energy > 0.0 else None,
        "leakage_energy_ratio": round(outside_energy / total_energy, 8) if has_labels and total_energy > 0.0 else None,
    }


def _overlap_metrics(run_dir: Path, overlap_regions: list[dict], sample_rate: int) -> list[dict]:
    rows = []
    for region in overlap_regions:
        mixed_value = str(region.get("mixed_audio") or "")
        mixed_path = _resolve_run_path(run_dir, mixed_value) if mixed_value else None
        separated = region.get("separated_audio") or {}
        separated_rows = {}
        separated_audio = []
        for speaker, value in sorted(separated.items()):
            if not value:
                continue
            path = _resolve_run_path(run_dir, str(value))
            if not path.exists():
                continue
            audio, _ = _read_audio(path, sample_rate)
            separated_audio.append(audio)
            separated_rows[str(speaker)] = {"audio": str(path), "rms": _rms(audio), "peak": round(float(np.max(np.abs(audio))), 8)}
        mixed_rms = None
        if mixed_path is not None and mixed_path.exists():
            mixed_audio, _ = _read_audio(mixed_path, sample_rate)
            mixed_rms = _rms(mixed_audio)
        rows.append(
            {
                "id": str(region.get("id") or ""),
                "start": region.get("start"),
                "end": region.get("end"),
                "duration": region.get("duration"),
                "mixed_audio": str(mixed_path) if mixed_path is not None and mixed_path.exists() else "",
                "mixed_rms": mixed_rms,
                "separated": separated_rows,
                "source_correlation_abs": _source_correlation_abs(separated_audio),
            }
        )
    return rows


def _source_correlation_abs(sources: list[np.ndarray]) -> float | None:
    if len(sources) < 2:
        return None
    first = _match_length(sources[0], min(len(sources[0]), len(sources[1])))
    second = _match_length(sources[1], len(first))
    if _rms(first) == 0.0 or _rms(second) == 0.0:
        return None
    value = float(abs(np.corrcoef(first, second)[0, 1]))
    if not math.isfinite(value):
        return None
    return round(value, 6)


def _resolved_config(run_dir: Path, manifest: dict) -> dict:
    run_config = manifest.get("run_config") or {}
    path = run_config.get("audio")
    if not path:
        return {}
    resolved_path = _resolve_run_path(run_dir, str(path))
    return _read_optional_json(resolved_path) or {}


def _backend_from_config(resolved_config: dict) -> str:
    config = (resolved_config.get("components") or {}).get("overlap_separation") or {}
    return str(config.get("backend") or "")


def _review_row(run_dir: Path, row: dict) -> dict:
    return {
        "audio_id": row["audio_id"],
        "run_dir": str(run_dir),
        "reference_audio": row["reference_audio"],
        "tracks": [{"speaker": item["speaker"], "audio": item["audio"]} for item in row["tracks"]],
        "overlap_regions": row["overlap_regions"][:100],
        "metrics": {name: row.get(name) for name in METRIC_NAMES},
    }


def _summary(rows: list[dict], metrics_path: Path, review_path: Path) -> dict:
    total_duration = sum(float(row.get("duration_sec") or 0.0) for row in rows)
    return {
        "samples": len(rows),
        "hours": total_duration / 3600.0,
        "metric_status": "computed" if rows else "empty",
        "metric_names": list(METRIC_NAMES),
        "average_reconstruction_snr_db": _average(row.get("reconstruction_snr_db") for row in rows),
        "average_multi_active_ratio": _average(row.get("multi_active_ratio") for row in rows),
        "average_track_leakage_ratio": _average(row.get("mean_track_leakage_ratio") for row in rows),
        "total_overlap_regions": sum(int(row.get("overlap_region_count") or 0) for row in rows),
        "backends": _backend_counts(rows),
        "metrics_jsonl": str(metrics_path),
        "review_manifest_jsonl": str(review_path),
    }


def _average(values) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 6)


def _backend_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        backend = str(row.get("separation_backend") or "none")
        counts[backend] = counts.get(backend, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Vilier separated speaker audio")
    parser.add_argument("--input", required=True, help="Vilier output root or one output/<audio_id> directory")
    parser.add_argument("--output", default="benchmarks", help="Directory for summary.json and metrics.jsonl")
    args = parser.parse_args()

    summary_path = run_benchmark(Path(args.input), Path(args.output))
    print(f"benchmark summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
