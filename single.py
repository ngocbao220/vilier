from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from pipeline.benchmark import run_benchmark
from pipeline.cli import ProgressBar, load_config, process_one, resolve_state_dir


def _print_benchmark_table(row: dict) -> None:
    import pandas as pd
    rows = [{"chỉ_số": key, "kết_quả": value} for key, value in row.items()
            if isinstance(value, (str, int, float, bool)) or value is None]
    print(pd.DataFrame(rows, columns=["chỉ_số", "kết_quả"]).to_string(index=False), flush=True)


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
    print("========= Phase 1: Running Pipeline: Vilier =========", flush=True)
    with tempfile.TemporaryDirectory(prefix="vilier_single_") as temp:
        root = Path(temp)
        manifest_path, sections = process_one(args.input, config, root, resolve_state_dir(config, ""),
                                              dry_run=False, progress=ProgressBar(total=10))
        manifest = json.loads(manifest_path.read_text())
        tracks = manifest["speakers"]
        if len(tracks) != 2:
            raise RuntimeError(f"Vilier expected two speaker tracks, got {len(tracks)}")
        output.mkdir(parents=True, exist_ok=True)
        for target, record in zip((output / "speakerA.wav", output / "speakerB.wav"), tracks):
            shutil.copy2(manifest_path.parent / record["track_wav"], target)
        print("========= Phase 2: Running Benchmark: Vilier =========", flush=True)
        benchmark_progress = ProgressBar(total=1)
        benchmark_progress.start(args.input.stem, "benchmark")
        benchmark_summary = run_benchmark(root, output / "benchmark")
        benchmark_progress.complete(args.input.stem, "benchmark")
        benchmark_path = output / "benchmark.json"
        reference_metrics = ("pit_si_sdr", "delta_si_sdr", "sir", "sar", "stoi", "pesq",
                             "crosstalk_rate", "leakage_p50_db", "leakage_p95_db", "vad_f1",
                             "onset_mae", "offset_mae", "overlap_f1", "overlap_iou")
        benchmark_row = {"metric_status": "unavailable", "reference_status": "unavailable",
            **{key: None for key in reference_metrics}, "native_report": str(benchmark_summary)}
        benchmark_path.write_text(json.dumps(benchmark_row, indent=2) + "\n")
        _print_benchmark_table(benchmark_row)
        if args.debug:
            shutil.copytree(manifest_path.parent, output / "debug", dirs_exist_ok=True)
        (output / "run.json").write_text(json.dumps({"input": str(args.input), "speakerA": str(output / "speakerA.wav"),
            "speakerB": str(output / "speakerB.wav"), "debug": args.debug,
            "debug_dir": str(output / "debug") if args.debug else None, "benchmark": {"status": "unavailable",
            "reference_status": "unavailable", "report": str(benchmark_path), "native_report": str(benchmark_summary)},
            "phases": sections}, default=str, indent=2) + "\n")
    print("========= Done: Vilier =========", flush=True)
    print(f"output={output.resolve()}", flush=True)
    print(f"speakerA={output / 'speakerA.wav'}", flush=True)
    print(f"speakerB={output / 'speakerB.wav'}", flush=True)
    print(f"run_json={output / 'run.json'}", flush=True)
    print(f"benchmark={output / 'benchmark.json'}", flush=True)
    if args.debug:
        print(f"debug={output / 'debug'}", flush=True)


if __name__ == "__main__":
    main()
