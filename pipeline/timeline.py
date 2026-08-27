import json
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import write_wav
from .schema import SpeakerSegment, SpeakerTrack, relative_path


def annotate_overlaps(segments: list[SpeakerSegment], threshold: float = 0.05) -> list[SpeakerSegment]:
    group_idx = 0
    for segment in segments:
        segment.is_overlap = False
        segment.overlap_group_id = None

    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.speaker))
    for i, first in enumerate(ordered):
        overlapping = [first]
        for second in ordered[i + 1 :]:
            if second.start >= first.end:
                break
            if second.speaker == first.speaker:
                continue
            overlap = min(first.end, second.end) - max(first.start, second.start)
            if overlap >= threshold:
                overlapping.append(second)
        if len(overlapping) > 1:
            existing = next((s.overlap_group_id for s in overlapping if s.overlap_group_id), None)
            group_id = existing or f"overlap_{group_idx:05d}"
            if existing is None:
                group_idx += 1
            for item in overlapping:
                item.is_overlap = True
                item.overlap_group_id = group_id
    return segments


def export_segments_and_tracks(
    waveform: np.ndarray,
    sample_rate: int,
    segments: list[SpeakerSegment],
    output_dir: Path,
    write_segment_wavs: bool = True,
    segment_audio_overrides: dict[str, np.ndarray] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[SpeakerTrack]:
    speakers = sorted({segment.speaker for segment in segments})
    tracks = {speaker: np.zeros_like(waveform, dtype=np.float32) for speaker in speakers}
    segment_audio_overrides = segment_audio_overrides or {}
    segment_dirs = {}
    if write_segment_wavs:
        for speaker in speakers:
            segment_dir = output_dir / "segments" / speaker
            segment_dir.mkdir(parents=True, exist_ok=True)
            segment_dirs[speaker] = segment_dir

    total = len(segments)
    for segment_idx, segment in enumerate(segments, start=1):
        if progress_callback is not None:
            progress_callback(segment_idx, total, f"{segment.speaker}:{segment.id}")
        start_idx = max(0, int(round(segment.start * sample_rate)))
        end_idx = min(len(waveform), int(round(segment.end * sample_rate)))
        audio = segment_audio_overrides.get(segment.id)
        if audio is None:
            audio = waveform[start_idx:end_idx]
        else:
            audio = _match_length(np.asarray(audio, dtype=np.float32), max(0, end_idx - start_idx))
        if write_segment_wavs:
            segment_path = segment_dirs[segment.speaker] / f"{segment.id}.wav"
            write_wav(segment_path, audio, sample_rate)
            segment.segment_wav = relative_path(segment_path, output_dir)

        if end_idx > start_idx:
            target = tracks[segment.speaker][start_idx:end_idx]
            np.add(target, audio, out=target)
            np.clip(target, -1.0, 1.0, out=target)

    track_records = []
    for speaker, track_audio in tracks.items():
        track_path = output_dir / "tracks" / f"{speaker}.wav"
        write_wav(track_path, track_audio, sample_rate)
        track_records.append(SpeakerTrack(id=speaker, track_wav=relative_path(track_path, output_dir)))
    return track_records


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) > length:
        return audio[:length]
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), mode="constant")
    return audio


def export_audacity_labels(output_dir: Path, segments: list[SpeakerSegment], vad_segments: list[dict]) -> dict:
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    vad_path = labels_dir / "vad.txt"
    _write_label_rows(vad_path, ((float(segment["start"]), float(segment["end"]), "speech") for segment in vad_segments))

    ordered_segments = sorted(segments, key=lambda segment: (segment.start, segment.end, segment.speaker))
    speakers_path = labels_dir / "speakers.txt"
    _write_label_rows(speakers_path, ((segment.start, segment.end, segment.speaker) for segment in ordered_segments))

    speaker_labels = {}
    for speaker in sorted({segment.speaker for segment in ordered_segments}):
        speaker_path = labels_dir / f"{speaker}.txt"
        _write_label_rows(
            speaker_path,
            ((segment.start, segment.end, segment.speaker) for segment in ordered_segments if segment.speaker == speaker),
        )
        speaker_labels[speaker] = relative_path(speaker_path, output_dir)

    return {
        "vad": Path(relative_path(vad_path, output_dir)),
        "speakers": Path(relative_path(speakers_path, output_dir)),
        "speaker_labels": {speaker: Path(path) for speaker, path in speaker_labels.items()},
    }


def _write_label_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for start, end, label in rows:
            handle.write(f"{float(start):.3f}\t{float(end):.3f}\t{label}\n")


def write_manifest(
    output_dir: Path,
    audio_id: str,
    source_audio: Path,
    standardized_audio: Path,
    duration: float,
    sample_rate: int,
    segments: list[SpeakerSegment],
    tracks: list[SpeakerTrack],
    vad_segments: list[dict],
    audacity_labels: dict | None = None,
    vad_audio: list[Path] | None = None,
    asr_segments: list[dict] | None = None,
    diarization_chunks: list[dict] | None = None,
    speaker_linking: dict | None = None,
    run_config: dict | None = None,
    music_separation: dict | None = None,
    overlap_separation: dict | None = None,
    transcript: list[dict] | None = None,
    state_labeling: dict | None = None,
) -> Path:
    manifest = {
        "audio_id": audio_id,
        "source_audio": relative_path(source_audio, Path.cwd()),
        "standardized_audio": relative_path(standardized_audio, output_dir),
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "speakers": [track.__dict__ for track in tracks],
        "segments": [segment.to_manifest() for segment in segments],
        "vad_segments": [_vad_to_manifest(segment) for segment in vad_segments],
        "vad_audio": [str(path) for path in vad_audio or []],
        "asr_segments": [_asr_segment_to_manifest(segment) for segment in asr_segments or []],
        "diarization_chunks": [_chunk_to_manifest(chunk) for chunk in diarization_chunks or []],
        "speaker_linking": speaker_linking or {"audio": "", "strategy": "", "link_count": 0, "embedding_count": 0},
        "run_config": run_config or {"audio": ""},
        "music_separation": music_separation or {"enabled": False, "applied": False},
        "overlap_separation": overlap_separation or {"enabled": False, "overlap_regions": []},
        "audacity_labels": _labels_to_manifest(audacity_labels or {}),
        "transcript": transcript or [],
        "state_labeling": state_labeling or {"enabled": False},
    }
    path = output_dir / "manifest.timeline.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return path


def _labels_to_manifest(labels: dict) -> dict:
    return {
        "vad": str(labels.get("vad", "")),
        "speakers": str(labels.get("speakers", "")),
        "speaker_labels": {speaker: str(path) for speaker, path in labels.get("speaker_labels", {}).items()},
    }


def _chunk_to_manifest(chunk: dict) -> dict:
    return {
        "id": str(chunk["id"]),
        "audio": str(chunk["audio"]),
        "duration": float(chunk["duration"]),
        "mapping": chunk["mapping"],
    }


def _vad_to_manifest(segment: dict) -> dict:
    start = round(float(segment["start"]), 3)
    end = round(float(segment["end"]), 3)
    payload = {
        "id": str(segment.get("id", "")),
        "start": start,
        "end": end,
        "duration": round(end - start, 6),
    }
    return payload


def _asr_segment_to_manifest(segment: dict) -> dict:
    start = round(float(segment["start"]), 3)
    end = round(float(segment["end"]), 3)
    return {
        "id": str(segment.get("id", "")),
        "speaker": str(segment.get("speaker", "")),
        "start": start,
        "end": end,
        "duration": round(end - start, 6),
        "audio": str(segment.get("audio", "")),
    }
