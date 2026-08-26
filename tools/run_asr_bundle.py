import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asr import load_asr_runner, transcribe_asr_segments, transcribe_vad_audio, write_transcript_json
from pipeline.schema import SpeakerSegment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASR only on a prepared Vilier output bundle")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--bundle-dir", required=True, help="Directory containing asr_audio/ or vad_audio/, plus manifest.timeline.json")
    parser.add_argument("--output-dir", default="", help="Where to write transcript.json; defaults to bundle-dir")
    parser.add_argument("--device", default="", help="Override config asr.device, e.g. 0 on Kaggle GPU")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else bundle_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    if args.device:
        config.setdefault("asr", {})["device"] = args.device
    runner = load_asr_runner(config, dry_run=args.dry_run)
    if runner is None:
        raise ValueError("ASR is disabled in config")

    manifest = _read_json(bundle_dir / "manifest.timeline.json")
    asr_segments = manifest.get("asr_segments", [])
    if asr_segments:
        transcript = transcribe_asr_segments(bundle_dir, asr_segments, runner)
    else:
        vad_segments = manifest.get("vad_segments") or _read_json(bundle_dir / "vad.json")
        vad_audio = [Path(path) for path in manifest.get("vad_audio", [])]
        if not vad_audio:
            vad_audio = sorted(Path("vad_audio") / path.name for path in (bundle_dir / "vad_audio").glob("*.wav"))
        speaker_segments = [_speaker_segment(item) for item in manifest.get("segments", [])]
        transcript = transcribe_vad_audio(bundle_dir, vad_segments, vad_audio, speaker_segments, runner)
    transcript_path = output_dir / "transcript.json"
    write_transcript_json(transcript_path, transcript)
    (output_dir / "asr_result.json").write_text(
        json.dumps(
            {
                "audio_id": manifest.get("audio_id", bundle_dir.name),
                "bundle_dir": str(bundle_dir),
                "transcript": transcript_path.name,
                "transcripts": len(transcript),
                "model": runner.model_name,
                "language": runner.language,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(transcript)} transcripts to {transcript_path}")
    return 0


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _speaker_segment(item: dict) -> SpeakerSegment:
    return SpeakerSegment(
        id=str(item.get("id", "")),
        speaker=str(item.get("speaker", "SPEAKER_UNKNOWN")),
        start=float(item.get("start", 0.0)),
        end=float(item.get("end", 0.0)),
        segment_wav=str(item.get("segment_wav", "")),
        is_overlap=bool(item.get("is_overlap", False)),
        overlap_group_id=item.get("overlap_group_id"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
