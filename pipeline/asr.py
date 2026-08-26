import argparse
import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import soundfile as sf

from .audio import slice_waveform, write_wav
from .devices import resolve_auto_device
from .schema import SpeakerSegment
from .schema import relative_path


class AsrRunner(Protocol):
    model_name: str
    language: str

    def transcribe(self, audio_path: Path, index: int) -> str:
        ...


class DryRunAsrRunner:
    def __init__(self, model_name: str = "dry-run", language: str = "vi"):
        self.model_name = model_name
        self.language = language
        self.resolved_device = "dry-run"

    def transcribe(self, audio_path: Path, index: int) -> str:
        return f"dry-run transcript {index}"


class PhoWhisperLocalRunner:
    def __init__(self, config: dict):
        self.model_name = str(config.get("model", "vinai/PhoWhisper-large"))
        self.language = str(config.get("language", "vi"))
        self.device = config.get("device", "auto")
        self.resolved_device = None
        self.chunk_length_seconds = config.get("chunk_length_seconds", 30.0)
        self._pipeline = None

    def transcribe(self, audio_path: Path, index: int) -> str:
        call_kwargs = {}
        if self.language:
            call_kwargs["generate_kwargs"] = {"language": self.language, "task": "transcribe"}
        if self.chunk_length_seconds:
            call_kwargs["chunk_length_s"] = float(self.chunk_length_seconds)
            call_kwargs["ignore_warning"] = True
        audio, sample_rate = self._load_audio(audio_path)
        output = self._load_pipeline()({"array": audio, "sampling_rate": sample_rate}, **call_kwargs)
        if isinstance(output, dict):
            return str(output.get("text", "")).strip()
        return str(output).strip()

    def _load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        from transformers import pipeline
        import torch

        kwargs = {
            "task": "automatic-speech-recognition",
            "model": self.model_name,
        }
        resolved_device = normalize_pipeline_device(resolve_auto_device(torch, self.device, warn_label="asr.device"))
        self.resolved_device = resolved_device
        if resolved_device != "":
            kwargs["device"] = -1 if resolved_device == "cpu" else resolved_device
        self._pipeline = pipeline(**kwargs)
        return self._pipeline


def load_asr_runner(config: dict, dry_run: bool = False) -> AsrRunner | None:
    asr_config = config.get("asr", {})
    if not bool(asr_config.get("enabled", False)):
        return None
    if dry_run:
        return DryRunAsrRunner(
            model_name=str(asr_config.get("model", "vinai/PhoWhisper-large")),
            language=str(asr_config.get("language", "vi")),
        )

    backend = str(asr_config.get("backend", "phowhisper_local"))
    if backend != "phowhisper_local":
        raise ValueError(f"Unsupported ASR backend: {backend}")
    return PhoWhisperLocalRunner(asr_config)


def normalize_pipeline_device(device):
    if isinstance(device, str):
        stripped = device.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        if stripped.lower() == "gpu":
            return 0
        return stripped
    return device


def transcribe_vad_audio(
    output_dir: Path,
    vad_segments: list[dict],
    vad_audio: list[Path],
    speaker_segments: list[SpeakerSegment],
    runner: AsrRunner,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    if len(vad_segments) != len(vad_audio):
        raise ValueError("vad_segments and vad_audio must have the same length")

    transcript = []
    total = len(vad_segments)
    for idx, (vad_segment, audio_rel_path) in enumerate(zip(vad_segments, vad_audio), start=1):
        audio_path = output_dir / audio_rel_path
        if progress_callback is not None:
            progress_callback(idx, total, str(audio_rel_path))
        start = round(float(vad_segment["start"]), 3)
        end = round(float(vad_segment["end"]), 3)
        transcript.append(
            {
                "id": f"asr_{idx - 1:05d}",
                "vad_id": str(vad_segment.get("id", "")),
                "audio": str(audio_rel_path),
                "start": start,
                "end": end,
                "duration": round(end - start, 6),
                "speaker": assign_speaker(start, end, speaker_segments),
                "text": _transcribe_one(runner, audio_path, idx, str(vad_segment.get("id", "")), str(audio_rel_path)),
                "model": runner.model_name,
                "language": runner.language,
            }
        )
    return transcript


def export_speaker_asr_audio(
    output_dir: Path,
    speaker_tracks: list,
    vad_runner,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    asr_segments = []
    global_idx = 0
    total = len(speaker_tracks)
    for track_idx, track in enumerate(speaker_tracks, start=1):
        speaker = _track_speaker(track)
        if progress_callback is not None:
            progress_callback(track_idx, total, speaker)
        track_path = output_dir / _track_audio(track)
        audio, sample_rate = sf.read(track_path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        detected = vad_runner.detect(audio)
        speaker_dir = output_dir / "asr_audio" / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for speaker_idx, segment in enumerate(detected, start=1):
            start = round(float(segment["start"]), 3)
            end = round(float(segment["end"]), 3)
            if end <= start:
                continue
            path = speaker_dir / f"audio_{speaker_idx:05d}.wav"
            write_wav(path, slice_waveform(audio, sample_rate, start, end), sample_rate)
            asr_segments.append(
                {
                    "id": f"asrseg_{global_idx:05d}",
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "duration": round(end - start, 6),
                    "audio": relative_path(path, output_dir),
                }
            )
            global_idx += 1
    return sorted(asr_segments, key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])))


def transcribe_asr_segments(
    output_dir: Path,
    asr_segments: list[dict],
    runner: AsrRunner,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    transcript = []
    total = len(asr_segments)
    for idx, segment in enumerate(asr_segments, start=1):
        audio_rel_path = Path(str(segment["audio"]))
        audio_path = output_dir / audio_rel_path
        if progress_callback is not None:
            progress_callback(idx, total, str(audio_rel_path))
        start = round(float(segment["start"]), 3)
        end = round(float(segment["end"]), 3)
        transcript.append(
            {
                "id": f"asr_{idx - 1:05d}",
                "asr_segment_id": str(segment.get("id", "")),
                "audio": str(audio_rel_path),
                "start": start,
                "end": end,
                "duration": round(end - start, 6),
                "speaker": str(segment.get("speaker", "SPEAKER_UNKNOWN")),
                "text": _transcribe_one(runner, audio_path, idx, str(segment.get("id", "")), str(audio_rel_path)),
                "model": runner.model_name,
                "language": runner.language,
            }
        )
    return transcript


def _track_speaker(track) -> str:
    if isinstance(track, dict):
        return str(track.get("id", track.get("speaker", "SPEAKER_UNKNOWN")))
    return str(getattr(track, "id", getattr(track, "speaker", "SPEAKER_UNKNOWN")))


def _track_audio(track) -> Path:
    if isinstance(track, dict):
        return Path(str(track.get("track_wav", track.get("audio", ""))))
    return Path(str(getattr(track, "track_wav", getattr(track, "audio", ""))))


def _transcribe_one(runner: AsrRunner, audio_path: Path, index: int, vad_id: str, audio_rel_path: str) -> str:
    try:
        return runner.transcribe(audio_path, index)
    except Exception as exc:
        raise type(exc)(f"ASR failed for {vad_id} at {audio_rel_path}: {exc}") from exc


def assign_speaker(start: float, end: float, segments: list[SpeakerSegment]) -> str:
    best_speaker = "SPEAKER_UNKNOWN"
    best_overlap = 0.0
    for segment in segments:
        overlap = min(end, segment.end) - max(start, segment.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment.speaker
    return best_speaker


def write_transcript_json(path: Path, transcript: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(transcript, handle, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local PhoWhisper Large ASR on one audio file")
    parser.add_argument("audio", help="Path to a 16 kHz-compatible audio file")
    parser.add_argument("--model", default="vinai/PhoWhisper-large")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    runner = PhoWhisperLocalRunner(
        {
            "model": args.model,
            "language": args.language,
            "device": args.device,
        }
    )
    print(runner.transcribe(Path(args.audio), index=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
