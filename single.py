from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from pipeline.cli import ProgressBar, load_config, process_one, resolve_state_dir


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
        if args.debug:
            shutil.copytree(manifest_path.parent, output / "debug", dirs_exist_ok=True)
        (output / "run.json").write_text(json.dumps({"input": str(args.input), "speakerA": str(output / "speakerA.wav"),
            "speakerB": str(output / "speakerB.wav"), "debug": args.debug,
            "debug_dir": str(output / "debug") if args.debug else None, "phases": sections}, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
