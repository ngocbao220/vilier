import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .asr import load_asr_runner, transcribe_vad_audio, write_transcript_json
from .audio import iter_audio_files, load_mono, write_wav
from .diarization import SortformerDiarizer, build_diarization_chunks
from .labeling import label_transcripts, load_labeling_runner, resolve_state_dir, write_state_outputs
from .overlap_separation import apply_overlap_separation, load_overlap_separator
from .schema import relative_path
from .timeline import annotate_overlaps, export_audacity_labels, export_segments_and_tracks, write_manifest
from .tree_log import kv, log_tree, section, write_tree_log
from .vad import SileroVadRunner, export_vad_audio, write_vad_txt
from .visualization import write_visualizations


class PipelineRunError(Exception):
    def __init__(self, audio_id: str, step: str, sections: list, original: Exception):
        super().__init__(str(original))
        self.audio_id = audio_id
        self.step = step
        self.sections = sections
        self.original = original


class ProgressBar:
    def __init__(self, total: int, enabled: bool = True, stream: TextIO | None = None):
        self.total = max(1, total)
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.completed = 0

    def start(self, audio_id: str, step: str) -> None:
        self._write(audio_id, step, "RUN", self.completed)

    def complete(self, audio_id: str, step: str) -> None:
        self.completed = min(self.completed + 1, self.total)
        self._write(audio_id, step, "DONE", self.completed)

    def fail(self, audio_id: str, step: str) -> None:
        self._write(audio_id, step, "FAIL", self.completed)

    def item(self, audio_id: str, step: str, current: int, total: int, label: str) -> None:
        if not self.enabled:
            return
        print(f"  {audio_id}/{step}: {current}/{total} {label}", file=self.stream, flush=True)

    def _write(self, audio_id: str, step: str, status: str, done: int) -> None:
        if not self.enabled:
            return
        print(format_progress_bar(done, self.total, f"{audio_id}/{step}", status), file=self.stream, flush=True)


def format_progress_bar(done: int, total: int, label: str, status: str, width: int = 24) -> str:
    total = max(1, total)
    done = min(max(0, done), total)
    filled = round(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} {status} {label}"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_log_dir(config: dict, run_date: str, log_dir_arg: str = "") -> Path:
    raw_log_dir = log_dir_arg or os.environ.get("LOG_DIR") or config.get("logging", {}).get("log_dir", "logs")
    log_dir = Path(raw_log_dir).expanduser()
    if log_dir.name != run_date:
        log_dir = log_dir / run_date
    return log_dir.resolve()


def log_path_for_audio(log_dir: Path, audio_path: Path) -> Path:
    return log_dir / audio_path.name / "pipeline.log"


def batch_log_path(log_dir: Path) -> Path:
    return log_dir / "batch.log"


def process_one(
    audio_path: Path,
    config: dict,
    output_root: Path,
    state_dir: Path,
    dry_run: bool,
    progress: ProgressBar | None = None,
) -> tuple[Path, list]:
    sample_rate = int(config["entrypoint"].get("sample_rate", 16000))
    audio_id = audio_path.stem
    output_dir = output_root / audio_id
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sections = [
        section("input", "PASS", [kv("path", audio_path), kv("output_dir", output_dir)]),
    ]
    current_step = "input"

    try:
        current_step = "preprocess"
        if progress is not None:
            progress.start(audio_id, current_step)
        waveform, sample_rate = load_mono(audio_path, sample_rate)
        standardized_path = output_dir / "audio.standardized.wav"
        write_wav(standardized_path, waveform, sample_rate)
        sections.append(
            section(
                "preprocess",
                "PASS",
                [
                    kv("sample_rate", sample_rate),
                    kv("duration_sec", len(waveform) / sample_rate),
                    kv("output", standardized_path.name),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "vad"
        if progress is not None:
            progress.start(audio_id, current_step)
        vad_config = config.get("vad", {})
        vad_runner = SileroVadRunner(vad_config, sample_rate=sample_rate, dry_run=dry_run)
        vad_segments = vad_runner.detect(waveform)
        with (output_dir / "vad.json").open("w", encoding="utf-8") as handle:
            json.dump(vad_segments, handle, ensure_ascii=False, indent=2)
        write_vad_txt(output_dir / "vad.txt", vad_segments)
        vad_audio = export_vad_audio(waveform, sample_rate, vad_segments, output_dir)
        sections.append(
            section(
                "vad",
                "PASS",
                [
                    kv("backend", vad_config.get("backend", "silero")),
                    kv("segments", len(vad_segments)),
                    kv("output", "vad.json"),
                    kv("labels", "vad.txt"),
                    kv("audio_files", len(vad_audio)),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "diarization_chunks"
        if progress is not None:
            progress.start(audio_id, current_step)
        diarization_config = config.get("diarization", {})
        max_chunk_seconds = float(diarization_config.get("max_chunk_seconds", 180.0))
        diarization_chunks = build_diarization_chunks(waveform, sample_rate, vad_segments, output_dir, max_chunk_seconds)
        sections.append(
            section(
                "diarization_chunks",
                "PASS",
                [
                    kv("chunks", len(diarization_chunks)),
                    kv("max_chunk_sec", max_chunk_seconds),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "diarization"
        if progress is not None:
            progress.start(audio_id, current_step)
        diarizer = SortformerDiarizer(diarization_config, dry_run=dry_run)
        segments = diarizer.diarize_chunks(diarization_chunks)
        segments = annotate_overlaps(segments, threshold=float(config.get("overlap", {}).get("threshold_seconds", 0.05)))
        sections.append(
            section(
                "diarization",
                "PASS",
                [
                    kv("backend", diarization_config.get("backend", "sortformer")),
                    kv("model", diarization_config.get("model", "nvidia/diar_sortformer_4spk-v1")),
                    kv("segments", len(segments)),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "overlap_separation"
        if progress is not None:
            progress.start(audio_id, current_step)
        overlap_config = config.get("overlap_separation", {})
        overlap_warnings = []
        separator = load_overlap_separator(overlap_config, dry_run=dry_run, warnings=overlap_warnings)
        overlap_result = apply_overlap_separation(
            waveform,
            sample_rate,
            segments,
            separator,
            overlap_threshold=float(overlap_config.get("overlap_threshold_seconds", config.get("overlap", {}).get("threshold_seconds", 0.05))),
        )
        overlap_attrs = [
            kv("enabled", bool(overlap_config.get("enabled", False))),
            kv("regions", len(overlap_result["overlap_regions"])),
            kv("enhanced_segments", len(overlap_result["segment_audio"])),
        ]
        if overlap_warnings:
            overlap_attrs.append(kv("warning", overlap_warnings[0]))
        sections.append(section("overlap_separation", "PASS" if separator is not None else "SKIP", overlap_attrs))
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "tracks"
        if progress is not None:
            progress.start(audio_id, current_step)
        write_segment_wavs = bool(config.get("export", {}).get("write_segment_wavs", True))
        tracks = export_segments_and_tracks(
            waveform,
            sample_rate,
            segments,
            output_dir,
            write_segment_wavs=write_segment_wavs,
            segment_audio_overrides=overlap_result["segment_audio"],
        )
        audacity_labels = export_audacity_labels(output_dir, segments, vad_segments)
        if bool(config.get("export", {}).get("write_visualizations", True)):
            write_visualizations(waveform, sample_rate, segments, tracks, output_dir)
        sections.append(
            section(
                "tracks",
                "PASS",
                [
                    kv("speakers", len(tracks)),
                    kv("write_segment_wavs", write_segment_wavs),
                    kv("labels", "labels"),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "asr"
        if progress is not None:
            progress.start(audio_id, current_step)
        transcript = []
        asr_config = config.get("asr", {})
        asr_runner = load_asr_runner(config, dry_run=dry_run)
        if asr_runner is not None:
            transcript = transcribe_vad_audio(
                output_dir,
                vad_segments,
                vad_audio,
                segments,
                asr_runner,
                progress_callback=(
                    (lambda current, total, label: progress.item(audio_id, current_step, current, total, label))
                    if progress is not None
                    else None
                ),
            )
            write_transcript_json(output_dir / "transcript.json", transcript)
        sections.append(
            section(
                "asr",
                "PASS" if asr_runner is not None else "SKIP",
                [
                    kv("enabled", bool(asr_config.get("enabled", False))),
                    kv("backend", asr_config.get("backend", "")),
                    kv("model", asr_config.get("model", "")),
                    kv("language", asr_config.get("language", "")),
                    kv("unit", "vad_audio"),
                    kv("transcripts", len(transcript)),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "state_labeling"
        if progress is not None:
            progress.start(audio_id, current_step)
        state_labeling_config = config.get("state_labeling", {})
        state_summary = {"enabled": False}
        labeling_runner = load_labeling_runner(config, dry_run=dry_run)
        if labeling_runner is not None:
            labeled_records = label_transcripts(transcript, labeling_runner)
            state_summary = write_state_outputs(audio_id, output_dir, state_dir, labeled_records)
            state_summary["labels"] = list(labeling_runner.labels)
            state_summary["model"] = labeling_runner.model_name
            transcript = labeled_records
            write_transcript_json(output_dir / "transcript.json", transcript)
        state_attrs = [
            kv("enabled", bool(state_labeling_config.get("enabled", False))),
            kv("backend", state_labeling_config.get("backend", "")),
            kv("model", state_labeling_config.get("model", "")),
            kv("labels", ",".join(state_labeling_config.get("labels", []))),
            kv("labeled", state_summary.get("labeled", 0)),
        ]
        for label, count in sorted(state_summary.get("counts", {}).items()):
            state_attrs.append(kv(label, count))
        state_attrs.append(kv("state_dir", state_summary.get("state_dir", relative_path(state_dir, Path.cwd()))))
        sections.append(section("state_labeling", "PASS" if labeling_runner is not None else "SKIP", state_attrs))
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "manifest"
        if progress is not None:
            progress.start(audio_id, current_step)
        manifest_path = write_manifest(
            output_dir=output_dir,
            audio_id=audio_id,
            source_audio=audio_path,
            standardized_audio=standardized_path,
            duration=len(waveform) / sample_rate,
            sample_rate=sample_rate,
            segments=segments,
            tracks=tracks,
            vad_segments=vad_segments,
            audacity_labels=audacity_labels,
            vad_audio=vad_audio,
            diarization_chunks=diarization_chunks,
            overlap_separation={
                "enabled": bool(separator is not None),
                "overlap_regions": overlap_result["overlap_regions"],
                "enhanced_segment_count": len(overlap_result["segment_audio"]),
            },
            transcript=transcript,
            state_labeling=state_summary,
        )
        sections.append(
            section(
                "manifest",
                "PASS",
                [
                    kv("output", manifest_path.name),
                    kv("transcript", "transcript.json" if transcript else ""),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)
        sections.append(section("done", "PASS", [kv("elapsed_sec", time.perf_counter() - started)]))
        return manifest_path, sections
    except Exception as exc:
        if progress is not None:
            progress.fail(audio_id, current_step)
        raise PipelineRunError(audio_id, current_step, sections, exc) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Vilier speaker split pipeline")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    dry_run = bool(args.dry_run or config.get("runtime", {}).get("dry_run", False))
    input_path = Path(args.input or config["entrypoint"]["input_path"]).expanduser().resolve()
    output_root = Path(args.output or config["entrypoint"]["output_path"]).expanduser().resolve()
    state_dir = resolve_state_dir(config, args.state_dir)
    files = iter_audio_files(input_path)
    if not files:
        raise FileNotFoundError(f"No audio files found in {input_path}")

    run_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = resolve_log_dir(config, run_date, args.log_dir)
    batch_items = [
        (
            "batch",
            [
                kv("input_path", input_path),
                kv("output_path", output_root),
                kv("log_dir", log_dir),
                kv("state_dir", state_dir),
                kv("dry_run", dry_run),
                kv("files", len(files)),
            ],
        )
    ]
    log_tree(
        "Vilier pipeline",
        batch_items,
    )

    success = 0
    failed = 0
    progress_enabled = bool(config.get("logging", {}).get("progress_bar", True))
    for audio_path in files:
        audio_log_path = log_path_for_audio(log_dir, audio_path)
        progress = ProgressBar(total=9, enabled=progress_enabled)
        try:
            _, sections = process_one(audio_path, config, output_root, state_dir, dry_run=dry_run, progress=progress)
            log_tree(f"audio={audio_path.stem}", sections)
            write_tree_log(audio_log_path, f"audio={audio_path.stem}", sections)
            success += 1
        except PipelineRunError as exc:
            error_sections = exc.sections + [
                section(
                    exc.step,
                    "FAIL",
                    [
                        kv("error_type", type(exc.original).__name__),
                        kv("error", exc.original),
                    ],
                )
            ]
            log_tree(
                f"audio={exc.audio_id}",
                error_sections,
                level="ERROR",
            )
            write_tree_log(audio_log_path, f"audio={exc.audio_id}", error_sections, level="ERROR")
            failed += 1
        except Exception as exc:
            error_sections = [section("pipeline", "FAIL", [kv("error_type", type(exc).__name__), kv("error", exc)])]
            log_tree(
                f"audio={audio_path.stem}",
                error_sections,
                level="ERROR",
            )
            write_tree_log(audio_log_path, f"audio={audio_path.stem}", error_sections, level="ERROR")
            failed += 1
    done_items = batch_items + [section("done", "PASS" if failed == 0 else "FAIL", [kv("success", success), kv("failed", failed)])]
    log_tree("batch done", [kv("success", success), kv("failed", failed)])
    write_tree_log(batch_log_path(log_dir), "Vilier pipeline", done_items, level="INFO" if failed == 0 else "ERROR")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
