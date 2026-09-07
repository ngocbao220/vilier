from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from pipeline.benchmark import benchmark_run_dir, run_benchmark
from pipeline.cli import ProgressBar, load_config, process_one, resolve_state_dir


def _format_table(rows: list[tuple[str, object]]) -> str:
    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
    body = [border]
    body.extend(f"| {key.ljust(key_width)} | {str(value).ljust(value_width)} |" for key, value in rows)
    body.append(border)
    return "\n".join(body)


def format_pipeline_modules(config: dict) -> str:
    diarization = config.get("diarization", {})
    overlap = config.get("overlap_separation", {})
    modules = [
        ("Preparing input", "load_mono, write_wav"),
        ("VAD", f"SileroVadRunner ({config.get('vad', {}).get('model', 'silero_vad')})"),
        ("Speaker Diarization", f"build_diarization_chunks, {diarization.get('backend', 'diarization')} ({diarization.get('model', '')})"),
        ("Overlap Separation", f"{overlap.get('backend', '')} ({overlap.get('model_name', overlap.get('model', ''))})"),
        ("Concatenating", "export_segments_and_tracks"),
        ("Benchmark", "pipeline.benchmark"),
    ]
    return "\n".join(["Pipeline: Vilier", "[Modules participating]"] + [f"-> {phase}: {module}" for phase, module in modules])


def format_benchmark_report(row: dict, prediction_paths: list[Path], *, include_heading: bool = True) -> str:
    predictions = ", ".join(str(path) for path in prediction_paths)
    meanings = [
        ("reconstruction_snr_db", "higher: summed tracks resemble input"),
        ("residual_rms", "lower: less input left unexplained"),
        ("active_track_ratio", "share of samples active in either track"),
        ("multi_active_ratio", "share of samples active in both tracks"),
        ("clipped_sample_ratio", "lower: fewer clipped samples"),
        ("mean_track_leakage_ratio", "lower: energy outside diarized label"),
    ]
    results = [(name, row.get(name)) for name, _ in meanings]
    return "\n".join(
        [
            *(["========= 3. Running benchmark ========="] if include_heading else []),
            f"Predict: {predictions}",
            "Ground Truth: null",
            "Vì không có ground truth nên không tính PIT-SI-SDR, SI-SDR, STOI, PESQ hay các metric tham chiếu.",
            "Các metric dưới đây là proxy nội bộ để kiểm tra tái tạo, clipping và timeline; chúng không chứng minh chất lượng separation tuyệt đối.",
            "Metric meanings:",
            _format_table(meanings),
            "Benchmark result:",
            _format_table(results),
        ]
    )


def native_metric_summary(row: dict) -> dict:
    return {
        key: row.get(key)
        for key in (
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vilier on one audio file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input audio file does not exist: {args.input}")
    output = args.output_dir or Path("outputs") / args.input.stem
    config = load_config(Path("config.json"))
    config["entrypoint"]["input_path"] = str(args.input)
    print(format_pipeline_modules(config), flush=True)
    with tempfile.TemporaryDirectory(prefix="vilier_single_") as temp:
        root = Path(temp)
        visible_steps = {"preprocess", "vad", "diarization_chunks", "diarization", "overlap_separation", "tracks"}
        manifest_path, sections = process_one(
            args.input,
            config,
            root,
            resolve_state_dir(config, ""),
            dry_run=False,
            progress=ProgressBar(total=6, visible_steps=visible_steps),
        )
        manifest = json.loads(manifest_path.read_text())
        tracks = manifest["speakers"]
        if len(tracks) != 2:
            raise RuntimeError(f"Vilier expected two speaker tracks, got {len(tracks)}")
        output.mkdir(parents=True, exist_ok=True)
        for target, record in zip((output / "speakerA.wav", output / "speakerB.wav"), tracks):
            shutil.copy2(manifest_path.parent / record["track_wav"], target)
        print("========= 3. Running benchmark =========", flush=True)
        benchmark_progress = ProgressBar(total=1, visible_steps={"benchmark"})
        benchmark_progress.start(args.input.stem, "benchmark")
        benchmark_summary = run_benchmark(root, output / "benchmark")
        benchmark_progress.complete(args.input.stem, "benchmark")
        benchmark_path = output / "benchmark.json"
        reference_metrics = ("pit_si_sdr", "delta_si_sdr", "sir", "sar", "stoi", "pesq",
                             "crosstalk_rate", "leakage_p50_db", "leakage_p95_db", "vad_f1",
                             "onset_mae", "offset_mae", "overlap_f1", "overlap_iou")
        native_row = benchmark_run_dir(manifest_path.parent)
        benchmark_row = {"metric_status": "native_proxy", "reference_status": "unavailable",
            **{key: None for key in reference_metrics}, "native_report": str(benchmark_summary),
            "native_metrics": native_metric_summary(native_row)}
        benchmark_path.write_text(json.dumps(benchmark_row, indent=2) + "\n")
        print(format_benchmark_report(native_row, [output / "speakerA.wav", output / "speakerB.wav"], include_heading=False), flush=True)
        if args.debug:
            shutil.copytree(manifest_path.parent, output / "debug", dirs_exist_ok=True)
        (output / "run.json").write_text(json.dumps({"input": str(args.input), "speakerA": str(output / "speakerA.wav"),
            "speakerB": str(output / "speakerB.wav"), "debug": args.debug,
            "debug_dir": str(output / "debug") if args.debug else None, "benchmark": {"status": "native_proxy",
            "reference_status": "unavailable", "report": str(benchmark_path), "native_report": str(benchmark_summary)},
            "phases": sections}, default=str, indent=2) + "\n")
    print("Complete!", flush=True)
    print(f"output={output.resolve()}", flush=True)
    print(f"speakerA={output / 'speakerA.wav'}", flush=True)
    print(f"speakerB={output / 'speakerB.wav'}", flush=True)
    print(f"run_json={output / 'run.json'}", flush=True)
    print(f"benchmark={output / 'benchmark.json'}", flush=True)
    if args.debug:
        print(f"debug={output / 'debug'}", flush=True)


if __name__ == "__main__":
    main()
