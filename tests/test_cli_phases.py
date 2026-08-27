import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from pipeline.cli import format_elapsed, run_with_progress_heartbeat


class CliPhaseTest(unittest.TestCase):
    def test_pre_asr_and_post_asr_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            config_path = Path(tmp) / "config.json"
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
            config.setdefault("state_labeling", {})["enabled"] = True
            config_path.write_text(json.dumps(config), encoding="utf-8")
            input_path = Path("inputs/podcast_single_30s.wav")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipeline.cli",
                    "--config",
                    str(config_path),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_root),
                    "--until",
                    "pre_asr",
                    "--dry-run",
                ],
                check=True,
            )

            output_dir = output_root / input_path.stem
            manifest_path = output_dir / "manifest.timeline.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "vad.json").exists())
            self.assertTrue((output_dir / "vad_audio").exists())
            self.assertTrue((output_dir / "asr_audio").exists())
            self.assertGreater(len(manifest["asr_segments"]), 0)
            self.assertEqual(manifest["transcript"], [])
            self.assertFalse((output_dir / "transcript.json").exists())

            transcript = [
                {
                    "id": "asr_00000",
                    "asr_segment_id": manifest["asr_segments"][0]["id"],
                    "audio": manifest["asr_segments"][0]["audio"],
                    "start": manifest["asr_segments"][0]["start"],
                    "end": manifest["asr_segments"][0]["end"],
                    "duration": manifest["asr_segments"][0]["duration"],
                    "speaker": manifest["asr_segments"][0]["speaker"],
                    "text": "xin chao",
                    "model": "vinai/PhoWhisper-large",
                    "language": "vi",
                }
            ]
            (output_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipeline.cli",
                    "--config",
                    str(config_path),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_root),
                    "--from",
                    "post_asr",
                    "--dry-run",
                ],
                check=True,
            )

            final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            final_transcript = json.loads((output_dir / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(final_manifest["transcript"][0]["text"], "xin chao")
            self.assertEqual(final_manifest["state_labeling"]["labeled"], 1)
            self.assertEqual(final_transcript[0]["state_label"], "complete")
            self.assertTrue((output_dir / "state" / "index.json").exists())

    def test_pixit_backend_runs_in_dry_run_without_loading_pyannote(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            config_path = Path(tmp) / "config.json"
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
            config["entrypoint"]["input_path"] = "inputs/podcast_single_30s.wav"
            config["entrypoint"]["output_path"] = str(output_root)
            config["runtime"]["dry_run"] = True
            config.setdefault("diarization", {})["backend"] = "pixit"
            config["diarization"]["model"] = "pyannote/speech-separation-ami-1.0"
            config.setdefault("logging", {})["progress_bar"] = False
            config_path.write_text(json.dumps(config), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipeline.cli",
                    "--config",
                    str(config_path),
                    "--until",
                    "pre_asr",
                ],
                check=True,
            )

            manifest = json.loads((output_root / "podcast_single_30s" / "manifest.timeline.json").read_text(encoding="utf-8"))
            self.assertGreater(len(manifest["segments"]), 0)
            self.assertEqual(manifest["speaker_linking"]["audio"], "speaker_linking.json")
            self.assertEqual(manifest["speaker_linking"]["strategy"], "native_global")
            self.assertEqual(manifest["run_config"]["audio"], "config.resolved.json")
            self.assertTrue((output_root / "podcast_single_30s" / "speaker_linking.json").exists())
            self.assertTrue((output_root / "podcast_single_30s" / "config.resolved.json").exists())
            resolved_config = json.loads((output_root / "podcast_single_30s" / "config.resolved.json").read_text(encoding="utf-8"))
            self.assertEqual(resolved_config["components"]["diarization"]["model"], "pyannote/speech-separation-ami-1.0")

    def test_diarizen_backend_runs_in_dry_run_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            config_path = Path(tmp) / "config.json"
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
            config["entrypoint"]["input_path"] = "inputs/podcast_single_30s.wav"
            config["entrypoint"]["output_path"] = str(output_root)
            config["runtime"]["dry_run"] = True
            config.setdefault("diarization", {})["backend"] = "diarizen"
            config["diarization"]["model"] = "BUT-FIT/diarizen-wavlm-large-s80-md"
            config.setdefault("logging", {})["progress_bar"] = False
            config_path.write_text(json.dumps(config), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipeline.cli",
                    "--config",
                    str(config_path),
                    "--until",
                    "pre_asr",
                ],
                check=True,
            )

            manifest = json.loads((output_root / "podcast_single_30s" / "manifest.timeline.json").read_text(encoding="utf-8"))
            self.assertGreater(len(manifest["segments"]), 0)

    def test_progress_heartbeat_reports_elapsed_time_during_long_step(self):
        class FakeProgress:
            def __init__(self):
                self.items = []

            def item(self, audio_id, step, current, total, label):
                self.items.append((audio_id, step, current, total, label))

        progress = FakeProgress()

        result = run_with_progress_heartbeat(
            lambda: (time.sleep(0.03), "done")[1],
            progress,
            "real",
            "diarization",
            "running audio.standardized.wav",
            0.01,
        )

        self.assertEqual(result, "done")
        self.assertGreaterEqual(len(progress.items), 1)
        self.assertIn("elapsed=", progress.items[-1][-1])

    def test_format_elapsed_is_compact(self):
        self.assertEqual(format_elapsed(13), "13s")
        self.assertEqual(format_elapsed(73), "1m13s")
        self.assertEqual(format_elapsed(3673), "1h01m13s")


if __name__ == "__main__":
    unittest.main()
