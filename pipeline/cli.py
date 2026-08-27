import argparse
import copy
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .asr import export_speaker_asr_audio, load_asr_runner, transcribe_asr_segments, write_transcript_json
from .audio import iter_audio_files, load_mono, write_wav
from .diarization import (
    DiariZenDiarizer,
    PyannotePixitDiarizer,
    build_diarization_chunks,
    build_speaker_linking_artifact,
    load_diarizer,
    write_speaker_linking_artifact,
)
from .labeling import label_transcripts, load_labeling_runner, resolve_state_dir, write_state_outputs
from .model_options import add_model_option_arguments, apply_model_overrides
from .music import apply_music_separation, load_music_separator
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
        self.item_bars = {}

    def start(self, audio_id: str, step: str) -> None:
        self._write(audio_id, step, "RUN", self.completed)

    def complete(self, audio_id: str, step: str) -> None:
        self._close_item_bar(audio_id, step)
        self.completed = min(self.completed + 1, self.total)
        self._write(audio_id, step, "DONE", self.completed)

    def fail(self, audio_id: str, step: str) -> None:
        self._close_item_bar(audio_id, step)
        self._write(audio_id, step, "FAIL", self.completed)

    def item(self, audio_id: str, step: str, current: int, total: int, label: str) -> None:
        if not self.enabled:
            return
        tqdm_cls = _tqdm()
        if tqdm_cls is not None and total > 0:
            key = (audio_id, step)
            bar = self.item_bars.get(key)
            if bar is None:
                bar = tqdm_cls(
                    total=total,
                    desc=f"{audio_id}/{step}",
                    unit="it",
                    leave=False,
                    file=self.stream,
                    dynamic_ncols=True,
                )
                self.item_bars[key] = bar
            delta = max(0, current - int(bar.n))
            if delta:
                bar.update(delta)
            if label:
                bar.set_postfix_str(_short_label(label), refresh=True)
            return
        print(f"  {audio_id}/{step}: {current}/{total} {label}", file=self.stream, flush=True)

    def _write(self, audio_id: str, step: str, status: str, done: int) -> None:
        if not self.enabled:
            return
        print(format_progress_bar(done, self.total, f"{audio_id}/{step}", status), file=self.stream, flush=True)

    def _close_item_bar(self, audio_id: str, step: str) -> None:
        bar = self.item_bars.pop((audio_id, step), None)
        if bar is not None:
            bar.close()


def _tqdm():
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm


def _short_label(label: str, max_len: int = 48) -> str:
    label = str(label)
    if len(label) <= max_len:
        return label
    return "..." + label[-max_len + 3 :]


def format_progress_bar(done: int, total: int, label: str, status: str, width: int = 24) -> str:
    total = max(1, total)
    done = min(max(0, done), total)
    filled = round(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} {status} {label}"


def run_with_progress_heartbeat(fn, progress: ProgressBar | None, audio_id: str, step: str, label: str, interval_seconds: float):
    if progress is None or interval_seconds <= 0:
        return fn()

    stop = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not stop.wait(interval_seconds):
            elapsed = format_elapsed(time.monotonic() - started)
            progress.item(audio_id, step, 0, 1, f"{label} elapsed={elapsed}")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=0.2)


def format_elapsed(seconds: float) -> str:
    seconds_int = max(0, int(round(seconds)))
    minutes, seconds_part = divmod(seconds_int, 60)
    hours, minutes_part = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes_part:02d}m{seconds_part:02d}s"
    if minutes_part:
        return f"{minutes_part}m{seconds_part:02d}s"
    return f"{seconds_part}s"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_resolved_config(
    output_dir: Path,
    config: dict,
    *,
    input_path: Path,
    output_root: Path,
    dry_run: bool,
    sample_rate: int,
    components: dict,
) -> dict:
    payload = {
        "input_path": relative_path(input_path, Path.cwd()),
        "output_root": relative_path(output_root, Path.cwd()),
        "output_dir": relative_path(output_dir, Path.cwd()),
        "dry_run": bool(dry_run),
        "sample_rate": int(sample_rate),
        "config": copy.deepcopy(config),
        "components": components,
    }
    path = output_dir / "config.resolved.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"audio": relative_path(path, output_dir)}


def _component_summary(section: dict, enabled: bool, runner=None, model_key: str = "model") -> dict:
    return {
        "enabled": bool(enabled),
        "backend": str(section.get("backend", "")),
        "model": str(section.get(model_key, section.get("model", ""))),
        "requested_device": str(section.get("device", "")),
        "resolved_device": str(getattr(runner, "resolved_device", section.get("device", ""))),
    }


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


def state_dir_for_audio(output_dir: Path, state_dir: Path) -> Path:
    return state_dir if state_dir.is_absolute() else output_dir / state_dir


def state_dir_for_batch_log(output_root: Path, state_dir: Path, files: list[Path]) -> Path | str:
    if state_dir.is_absolute():
        return state_dir
    if len(files) == 1:
        return state_dir_for_audio(output_root / files[0].stem, state_dir)
    return str(output_root / "<input_stem>" / state_dir)


def process_one(
    audio_path: Path,
    config: dict,
    output_root: Path,
    state_dir: Path,
    dry_run: bool,
    until: str = "",
    from_phase: str = "",
    progress: ProgressBar | None = None,
) -> tuple[Path, list]:
    if from_phase == "post_asr":
        return process_post_asr(audio_path, config, output_root, state_dir, dry_run=dry_run, progress=progress)

    sample_rate = int(config["entrypoint"].get("sample_rate", 16000))
    audio_id = audio_path.stem
    output_dir = output_root / audio_id
    audio_state_dir = state_dir_for_audio(output_dir, state_dir)
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
        vad_audio = export_vad_audio(
            waveform,
            sample_rate,
            vad_segments,
            output_dir,
            progress_callback=(
                (lambda current, total, label: progress.item(audio_id, current_step, current, total, label))
                if progress is not None
                else None
            ),
        )
        sections.append(
            section(
                "vad",
                "PASS",
                [
                    kv("backend", vad_config.get("backend", "silero")),
                    kv("threshold", vad_config.get("threshold", 0.5)),
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
        diarization_chunks = build_diarization_chunks(
            waveform,
            sample_rate,
            vad_segments,
            output_dir,
            max_chunk_seconds,
            progress_callback=(
                (lambda current, total, label: progress.item(audio_id, current_step, current, total, label))
                if progress is not None
                else None
            ),
        )
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
            backend = str(diarization_config.get("backend", "sortformer"))
            if backend in {"pixit", "pyannote", "pyannote_pixit", "diarizen"}:
                progress.item(audio_id, current_step, 0, 1, f"loading {backend} model")
        diarizer = load_diarizer(diarization_config, dry_run=dry_run)
        if isinstance(diarizer, (PyannotePixitDiarizer, DiariZenDiarizer)):
            running_label = f"running {standardized_path.name} device={diarizer.resolved_device}"
            if progress is not None:
                progress.item(audio_id, current_step, 0, 1, running_label)
            heartbeat_seconds = float(config.get("logging", {}).get("heartbeat_seconds", 15.0))
            segments = run_with_progress_heartbeat(
                lambda: diarizer.diarize(standardized_path, vad_segments),
                progress,
                audio_id,
                current_step,
                running_label,
                heartbeat_seconds,
            )
            if progress is not None:
                progress.item(audio_id, current_step, 1, 1, f"{diarization_config.get('backend', 'diarization')} done")
        else:
            segments = diarizer.diarize_chunks(
                diarization_chunks,
                progress_callback=(
                    (lambda current, total, label: progress.item(audio_id, current_step, current, total, label))
                    if progress is not None
                    else None
                ),
            )
        segments = annotate_overlaps(segments, threshold=float(config.get("overlap", {}).get("threshold_seconds", 0.05)))
        speaker_linking_payload = build_speaker_linking_artifact(
            backend=str(diarization_config.get("backend", "sortformer")),
            model=str(diarization_config.get("model", "nvidia/diar_sortformer_4spk-v1")),
            segments=segments,
            chunks=diarization_chunks,
            linking=getattr(diarizer, "speaker_linking", {}),
        )
        speaker_linking_summary = write_speaker_linking_artifact(output_dir, speaker_linking_payload)
        sections.append(
            section(
                "diarization",
                "PASS",
                [
                    kv("backend", diarization_config.get("backend", "sortformer")),
                    kv("model", diarization_config.get("model", "nvidia/diar_sortformer_4spk-v1")),
                    kv("device", getattr(diarizer, "resolved_device", diarization_config.get("device", ""))),
                    kv("segments", len(segments)),
                    kv("speaker_linking", speaker_linking_summary.get("audio", "")),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "music_separation"
        if progress is not None:
            progress.start(audio_id, current_step)
        music_config = config.get("music_separation", {})
        music_warnings = []
        music_separator = load_music_separator(music_config, dry_run=dry_run, warnings=music_warnings)
        music_result = apply_music_separation(waveform, sample_rate, output_dir, music_separator)
        speaker_waveform = music_result["waveform"]
        music_summary = {
            "enabled": bool(music_config.get("enabled", False)),
            "applied": bool(music_result.get("applied", False)),
            "backend": music_config.get("backend", "demucs") if bool(music_config.get("enabled", False)) else "",
            "model": music_config.get("model", "htdemucs") if bool(music_config.get("enabled", False)) else "",
            "audio": music_result.get("audio", ""),
        }
        music_attrs = [
            kv("enabled", music_summary["enabled"]),
            kv("applied", music_summary["applied"]),
            kv("backend", music_summary["backend"]),
            kv("model", music_summary["model"]),
            kv("device", getattr(music_separator, "resolved_device", getattr(music_separator, "device", music_config.get("device", "")))),
        ]
        if music_summary["audio"]:
            music_attrs.append(kv("audio", music_summary["audio"]))
        if music_warnings:
            music_attrs.append(kv("warning", music_warnings[0]))
        sections.append(section("music_separation", "PASS" if music_separator is not None else "SKIP", music_attrs))
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "overlap_separation"
        if progress is not None:
            progress.start(audio_id, current_step)
        overlap_config = config.get("overlap_separation", {})
        overlap_warnings = []
        separator = load_overlap_separator(overlap_config, dry_run=dry_run, warnings=overlap_warnings)
        overlap_result = apply_overlap_separation(
            speaker_waveform,
            sample_rate,
            segments,
            separator,
            overlap_threshold=float(overlap_config.get("overlap_threshold_seconds", config.get("overlap", {}).get("threshold_seconds", 0.05))),
            output_dir=output_dir,
            progress_callback=(
                (lambda current, total, label: progress.item(audio_id, current_step, current, total, label))
                if progress is not None
                else None
            ),
        )
        overlap_attrs = [
            kv("enabled", bool(overlap_config.get("enabled", False))),
            kv("backend", overlap_config.get("backend", "")),
            kv("model", overlap_config.get("model_name", overlap_config.get("model", ""))),
            kv("device", getattr(separator, "resolved_device", getattr(separator, "device", overlap_config.get("device", "")))),
            kv("regions", len(overlap_result["overlap_regions"])),
            kv("enhanced_segments", len(overlap_result["segment_audio"])),
        ]
        if overlap_result["overlap_regions"]:
            overlap_attrs.append(kv("audio_dir", "overlap"))
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
            speaker_waveform,
            sample_rate,
            segments,
            output_dir,
            write_segment_wavs=write_segment_wavs,
            segment_audio_overrides=overlap_result["segment_audio"],
            progress_callback=(
                (lambda current, total, label: progress.item(audio_id, "tracks_segments", current, total, label))
                if progress is not None
                else None
            ),
        )
        if progress is not None:
            progress._close_item_bar(audio_id, "tracks_segments")
        audacity_labels = export_audacity_labels(output_dir, segments, vad_segments)
        if bool(config.get("export", {}).get("write_visualizations", True)):
            write_visualizations(speaker_waveform, sample_rate, segments, tracks, output_dir)
        asr_segments = export_speaker_asr_audio(
            output_dir,
            tracks,
            vad_runner,
            progress_callback=(
                (lambda current, total, label: progress.item(audio_id, "tracks_asr_audio", current, total, label))
                if progress is not None
                else None
            ),
        )
        if progress is not None:
            progress._close_item_bar(audio_id, "tracks_asr_audio")
        sections.append(
            section(
                "tracks",
                "PASS",
                [
                    kv("speakers", len(tracks)),
                    kv("write_segment_wavs", write_segment_wavs),
                    kv("asr_segments", len(asr_segments)),
                    kv("labels", "labels"),
                ],
            )
        )
        if progress is not None:
            progress.complete(audio_id, current_step)

        if until == "pre_asr":
            current_step = "manifest"
            if progress is not None:
                progress.start(audio_id, current_step)
            resolved_config = write_resolved_config(
                output_dir,
                config,
                input_path=audio_path,
                output_root=output_root,
                dry_run=dry_run,
                sample_rate=sample_rate,
                components={
                    "vad": _component_summary(vad_config, True, vad_runner),
                    "diarization": _component_summary(diarization_config, True, diarizer),
                    "music_separation": _component_summary(music_config, bool(music_config.get("enabled", False)), music_separator),
                    "overlap_separation": _component_summary(
                        overlap_config,
                        bool(overlap_config.get("enabled", False)),
                        separator,
                        model_key="model_name",
                    ),
                    "asr": _component_summary(config.get("asr", {}), bool(config.get("asr", {}).get("enabled", False))),
                    "state_labeling": _component_summary(
                        config.get("state_labeling", {}),
                        bool(config.get("state_labeling", {}).get("enabled", False)),
                    ),
                },
            )
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
                asr_segments=asr_segments,
                diarization_chunks=diarization_chunks,
                speaker_linking=speaker_linking_summary,
                run_config=resolved_config,
                music_separation=music_summary,
                overlap_separation={
                    "enabled": bool(separator is not None),
                    "overlap_regions": overlap_result["overlap_regions"],
                    "enhanced_segment_count": len(overlap_result["segment_audio"]),
                },
                transcript=[],
                state_labeling={"enabled": False},
            )
            sections.append(
                section(
                    "manifest",
                    "PASS",
                    [
                        kv("output", manifest_path.name),
                        kv("transcript", ""),
                        kv("phase", "pre_asr"),
                    ],
                )
            )
            if progress is not None:
                progress.complete(audio_id, current_step)
            sections.append(section("done", "PASS", [kv("elapsed_sec", time.perf_counter() - started), kv("phase", "pre_asr")]))
            return manifest_path, sections

        current_step = "asr"
        if progress is not None:
            progress.start(audio_id, current_step)
        transcript = []
        asr_config = config.get("asr", {})
        asr_runner = load_asr_runner(config, dry_run=dry_run)
        if asr_runner is not None:
            transcript = transcribe_asr_segments(
                output_dir,
                asr_segments,
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
                    kv("device", getattr(asr_runner, "resolved_device", asr_config.get("device", ""))),
                    kv("language", asr_config.get("language", "")),
                    kv("unit", "asr_audio"),
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
            state_summary = write_state_outputs(audio_id, output_dir, audio_state_dir, labeled_records)
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
        state_attrs.append(kv("state_dir", state_summary.get("state_dir", relative_path(audio_state_dir, Path.cwd()))))
        sections.append(section("state_labeling", "PASS" if labeling_runner is not None else "SKIP", state_attrs))
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "manifest"
        if progress is not None:
            progress.start(audio_id, current_step)
        resolved_config = write_resolved_config(
            output_dir,
            config,
            input_path=audio_path,
            output_root=output_root,
            dry_run=dry_run,
            sample_rate=sample_rate,
            components={
                "vad": _component_summary(vad_config, True, vad_runner),
                "diarization": _component_summary(diarization_config, True, diarizer),
                "music_separation": _component_summary(music_config, bool(music_config.get("enabled", False)), music_separator),
                "overlap_separation": _component_summary(
                    overlap_config,
                    bool(overlap_config.get("enabled", False)),
                    separator,
                    model_key="model_name",
                ),
                "asr": _component_summary(asr_config, bool(asr_config.get("enabled", False)), asr_runner),
                "state_labeling": _component_summary(
                    state_labeling_config,
                    bool(state_labeling_config.get("enabled", False)),
                    labeling_runner,
                ),
            },
        )
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
            asr_segments=asr_segments,
            diarization_chunks=diarization_chunks,
            speaker_linking=speaker_linking_summary,
            run_config=resolved_config,
            music_separation=music_summary,
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


def process_post_asr(
    audio_path: Path,
    config: dict,
    output_root: Path,
    state_dir: Path,
    dry_run: bool,
    progress: ProgressBar | None = None,
) -> tuple[Path, list]:
    audio_id = audio_path.stem
    output_dir = output_root / audio_id
    audio_state_dir = state_dir_for_audio(output_dir, state_dir)
    started = time.perf_counter()
    sections = [
        section("input", "PASS", [kv("path", audio_path), kv("output_dir", output_dir), kv("phase", "post_asr")]),
    ]
    current_step = "post_asr"
    try:
        manifest_path = output_dir / "manifest.timeline.json"
        transcript_path = output_dir / "transcript.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest for post-ASR phase: {manifest_path}")
        if not transcript_path.exists():
            raise FileNotFoundError(f"Missing transcript for post-ASR phase: {transcript_path}")

        if progress is not None:
            progress.start(audio_id, "transcript")
        manifest = load_config(manifest_path)
        transcript = load_config(transcript_path)
        if progress is not None:
            progress.complete(audio_id, "transcript")
        sections.append(section("asr", "PASS", [kv("source", "transcript.json"), kv("transcripts", len(transcript))]))

        current_step = "state_labeling"
        if progress is not None:
            progress.start(audio_id, current_step)
        state_labeling_config = config.get("state_labeling", {})
        state_summary = {"enabled": False}
        labeling_runner = load_labeling_runner(config, dry_run=dry_run)
        if labeling_runner is not None:
            labeled_records = label_transcripts(transcript, labeling_runner)
            state_summary = write_state_outputs(audio_id, output_dir, audio_state_dir, labeled_records)
            state_summary["labels"] = list(labeling_runner.labels)
            state_summary["model"] = labeling_runner.model_name
            transcript = labeled_records
            write_transcript_json(transcript_path, transcript)
        state_attrs = [
            kv("enabled", bool(state_labeling_config.get("enabled", False))),
            kv("backend", state_labeling_config.get("backend", "")),
            kv("model", state_labeling_config.get("model", "")),
            kv("labels", ",".join(state_labeling_config.get("labels", []))),
            kv("labeled", state_summary.get("labeled", 0)),
        ]
        for label, count in sorted(state_summary.get("counts", {}).items()):
            state_attrs.append(kv(label, count))
        state_attrs.append(kv("state_dir", state_summary.get("state_dir", relative_path(audio_state_dir, Path.cwd()))))
        sections.append(section("state_labeling", "PASS" if labeling_runner is not None else "SKIP", state_attrs))
        if progress is not None:
            progress.complete(audio_id, current_step)

        current_step = "manifest"
        if progress is not None:
            progress.start(audio_id, current_step)
        manifest["transcript"] = transcript
        manifest["state_labeling"] = state_summary
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        sections.append(section("manifest", "PASS", [kv("output", manifest_path.name), kv("transcript", "transcript.json")]))
        if progress is not None:
            progress.complete(audio_id, current_step)
        sections.append(section("done", "PASS", [kv("elapsed_sec", time.perf_counter() - started), kv("phase", "post_asr")]))
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
    parser.add_argument("--until", choices=["", "pre_asr"], default="")
    parser.add_argument("--from", dest="from_phase", choices=["", "post_asr"], default="")
    parser.add_argument("--dry-run", action="store_true")
    add_model_option_arguments(parser)
    args = parser.parse_args()

    config = apply_model_overrides(load_config(Path(args.config)), args)
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
                kv("state_dir", state_dir_for_batch_log(output_root, state_dir, files)),
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
        progress_total = 8 if args.until == "pre_asr" else 3 if args.from_phase == "post_asr" else 10
        progress = ProgressBar(total=progress_total, enabled=progress_enabled)
        try:
            _, sections = process_one(
                audio_path,
                config,
                output_root,
                state_dir,
                dry_run=dry_run,
                until=args.until,
                from_phase=args.from_phase,
                progress=progress,
            )
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
