import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import slice_waveform, write_wav
from .schema import SpeakerSegment, relative_path


class SortformerDiarizer:
    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.nemo_log_level = str(config.get("nemo_log_level", "WARNING")).upper()
        self.model = None if dry_run else self._load_model()

    def diarize(self, audio_path: Path, vad_segments: list[dict]) -> list[SpeakerSegment]:
        if self.model is None:
            return self._dry_run_segments(vad_segments)

        return self._diarize_audio(audio_path)

    def diarize_chunks(self, chunks: list[dict]) -> list[SpeakerSegment]:
        if self.model is None:
            return self._dry_run_chunk_segments(chunks)

        segments = []
        min_duration = float(self.config.get("min_duration_seconds", 0.25))
        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_segments = self._diarize_audio(chunk["path"])
            segments.extend(remap_chunk_segments_to_original(chunk_segments, chunk["mapping"], chunk_idx, min_duration))
        return segments

    def _diarize_audio(self, audio_path: Path) -> list[SpeakerSegment]:
        _configure_nemo_logging(self.nemo_log_level)
        predicted_segments, _ = self.model.diarize(
            audio=str(audio_path),
            batch_size=1,
            include_tensor_outputs=True,
        )
        frame = sortformer_to_dataframe(predicted_segments)
        return dataframe_to_segments(frame, min_duration=float(self.config.get("min_duration_seconds", 0.25)))

    def _load_model(self):
        _configure_nemo_logging(self.nemo_log_level)
        from nemo.collections.asr.models import SortformerEncLabelModel

        model = SortformerEncLabelModel.from_pretrained(self.config.get("model", "nvidia/diar_sortformer_4spk-v1"))
        model.eval()
        return model

    def _dry_run_segments(self, vad_segments: list[dict]) -> list[SpeakerSegment]:
        segments = []
        for idx, vad in enumerate(vad_segments):
            speaker = f"SPEAKER_{idx % 2:02d}"
            segments.append(
                SpeakerSegment(
                    id=f"{idx:05d}_{speaker}",
                    speaker=speaker,
                    start=float(vad["start"]),
                    end=float(vad["end"]),
                )
            )
        return segments

    def _dry_run_chunk_segments(self, chunks: list[dict]) -> list[SpeakerSegment]:
        segments = []
        idx = 0
        for chunk in chunks:
            for item in chunk["mapping"]:
                speaker = f"SPEAKER_{idx % 2:02d}"
                segments.append(
                    SpeakerSegment(
                        id=f"{idx:05d}_{speaker}",
                        speaker=speaker,
                        start=float(item["source_start"]),
                        end=float(item["source_end"]),
                    )
                )
                idx += 1
        return segments


def build_diarization_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    vad_segments: list[dict],
    output_dir: Path,
    max_chunk_seconds: float,
) -> list[dict]:
    chunks_dir = output_dir / "diarization_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    max_chunk_samples = max(1, int(round(max_chunk_seconds * sample_rate)) - 1)
    chunks = []
    current_audio = []
    current_mapping = []
    current_samples = 0

    def flush() -> None:
        nonlocal current_audio, current_mapping, current_samples
        if not current_audio:
            return
        chunk_idx = len(chunks) + 1
        chunk_path = chunks_dir / f"chunk_{chunk_idx}.wav"
        chunk_audio = np.concatenate(current_audio).astype(np.float32, copy=False)
        write_wav(chunk_path, chunk_audio, sample_rate)
        chunks.append(
            {
                "id": f"chunk_{chunk_idx}",
                "path": chunk_path,
                "audio": relative_path(chunk_path, output_dir),
                "duration": round(len(chunk_audio) / sample_rate, 6),
                "mapping": current_mapping,
            }
        )
        current_audio = []
        current_mapping = []
        current_samples = 0

    for vad in vad_segments:
        source_start = float(vad["start"])
        source_end = float(vad["end"])
        utterance = slice_waveform(waveform, sample_rate, source_start, source_end)
        offset = 0
        while offset < len(utterance):
            if current_samples >= max_chunk_samples:
                flush()
            available = max_chunk_samples - current_samples
            if available <= 0:
                flush()
                available = max_chunk_samples
            take = min(len(utterance) - offset, available)
            piece = utterance[offset : offset + take]
            chunk_start = current_samples / sample_rate
            chunk_end = (current_samples + take) / sample_rate
            piece_source_start = source_start + offset / sample_rate
            piece_source_end = source_start + (offset + take) / sample_rate
            current_audio.append(piece)
            current_mapping.append(
                {
                    "vad_id": str(vad.get("id", "")),
                    "chunk_start": round(chunk_start, 6),
                    "chunk_end": round(chunk_end, 6),
                    "source_start": round(piece_source_start, 6),
                    "source_end": round(piece_source_end, 6),
                }
            )
            current_samples += take
            offset += take
            if current_samples >= max_chunk_samples:
                flush()

    flush()
    return chunks


def _configure_nemo_logging(level_name: str) -> None:
    level = getattr(logging, level_name, logging.WARNING)
    for logger_name in ("nemo", "nemo_logger", "nemo.utils", "nemo.collections.asr.parts.utils.diarization_utils"):
        logging.getLogger(logger_name).setLevel(level)
    try:
        from nemo.utils import logging as nemo_logging
    except Exception:
        return
    if hasattr(nemo_logging, "setLevel"):
        nemo_logging.setLevel(level)


def remap_chunk_segments_to_original(
    segments: list[SpeakerSegment],
    mapping: list[dict],
    chunk_idx: int,
    min_duration: float = 0.25,
) -> list[SpeakerSegment]:
    remapped = []
    for segment in segments:
        for item in mapping:
            overlap_start = max(segment.start, float(item["chunk_start"]))
            overlap_end = min(segment.end, float(item["chunk_end"]))
            if overlap_end <= overlap_start:
                continue
            source_start = float(item["source_start"]) + (overlap_start - float(item["chunk_start"]))
            source_end = float(item["source_start"]) + (overlap_end - float(item["chunk_start"]))
            source_start = round(source_start, 3)
            source_end = round(source_end, 3)
            if source_end - source_start < min_duration:
                continue
            idx = len(remapped)
            remapped.append(
                SpeakerSegment(
                    id=f"chunk_{chunk_idx:03d}_{idx:05d}_{segment.speaker}",
                    speaker=segment.speaker,
                    start=source_start,
                    end=source_end,
                )
            )
    return remapped


def sortformer_to_dataframe(predicted_segments) -> pd.DataFrame:
    lists = [x for x in predicted_segments if isinstance(x, (list, tuple))]
    if not lists:
        lists = predicted_segments
    rows = []
    for sub in lists:
        for raw_segment in sub:
            start_s, end_s, speaker_raw = raw_segment.split()
            speaker_num = int(speaker_raw.split("_")[1])
            start = float(start_s)
            end = float(end_s)
            rows.append(
                {
                    "segment": _format_segment(start, end),
                    "label": chr(ord("A") + len(rows) % 26),
                    "speaker": f"SPEAKER_{speaker_num:02d}",
                    "start": start,
                    "end": end,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
    return pd.DataFrame(rows).sort_values(["start", "end", "speaker"]).reset_index(drop=True)


def dataframe_to_segments(frame: pd.DataFrame, min_duration: float) -> list[SpeakerSegment]:
    segments = []
    for idx, row in frame.iterrows():
        start = round(float(row["start"]), 3)
        end = round(float(row["end"]), 3)
        if end - start < min_duration:
            continue
        speaker = str(row["speaker"])
        segments.append(SpeakerSegment(id=f"{idx:05d}_{speaker}", speaker=speaker, start=start, end=end))
    return segments


def _format_segment(start: float, end: float) -> str:
    def fmt(sec: float) -> str:
        td = datetime.timedelta(seconds=sec)
        hrs = td.seconds // 3600 + td.days * 24
        mins = (td.seconds // 60) % 60
        secs = td.seconds % 60
        ms = int(td.microseconds / 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

    return f"[ {fmt(start)} --> {fmt(end)}]"
