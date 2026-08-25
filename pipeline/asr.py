import argparse
import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import soundfile as sf

from .schema import SpeakerSegment


class AsrRunner(Protocol):
    model_name: str
    language: str

    def transcribe(self, audio_path: Path, index: int) -> str:
        ...


class DryRunAsrRunner:
    def __init__(self, model_name: str = "dry-run", language: str = "vi"):
        self.model_name = model_name
        self.language = language

    def transcribe(self, audio_path: Path, index: int) -> str:
        return f"dry-run transcript {index}"


class PhoWhisperLocalRunner:
    def __init__(self, config: dict):
        self.model_name = str(config.get("model", "vinai/PhoWhisper-large"))
        self.language = str(config.get("language", "vi"))
        self.device = config.get("device", "cpu")
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

        kwargs = {
            "task": "automatic-speech-recognition",
            "model": self.model_name,
        }
        if self.device != "":
            kwargs["device"] = -1 if self.device == "cpu" else self.device
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
