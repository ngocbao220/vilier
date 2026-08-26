import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RunShTest(unittest.TestCase):
    def test_run_sh_has_no_upload_or_fixed_python_commands(self):
        script = Path("run.sh").read_text(encoding="utf-8")
        self.assertNotIn("/opt/anaconda3/envs/sommelier/bin/python", script)
        self.assertNotIn("tools/kaggle_asr.py", script)
        self.assertNotIn("tools/kaggle_pipeline.py", script)
        self.assertNotIn("KAGGLE_", script)
        self.assertNotIn("ASR_BACKEND", script)
        self.assertNotIn("RUNTIME_BACKEND", script)

    def test_run_sh_runs_current_environment_pipeline_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
            config["entrypoint"]["input_path"] = "inputs/podcast_single_30s.wav"
            config["entrypoint"]["output_path"] = str(root / "outputs")
            config.setdefault("runtime", {})["dry_run"] = True
            config.setdefault("asr", {})["enabled"] = False
            config.setdefault("state_labeling", {})["enabled"] = False
            config.setdefault("logging", {})["progress_bar"] = False
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            env = os.environ.copy()
            env["CONFIG_PATH"] = str(config_path)
            env["PYTHON_BIN"] = sys.executable
            result = subprocess.run(["bash", "run.sh"], check=True, text=True, capture_output=True, env=env)

            output = result.stdout + result.stderr
            self.assertIn("[INFO] Pipeline component usage", output)
            self.assertIn("asr SKIP", output)
            self.assertNotIn("Bundle đã tạo", output)
            self.assertTrue((root / "outputs" / "podcast_single_30s" / "manifest.timeline.json").exists())

    def test_run_sh_forwards_model_options_to_config_log_and_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
            config["entrypoint"]["input_path"] = "inputs/podcast_single_30s.wav"
            config["entrypoint"]["output_path"] = str(root / "outputs")
            config.setdefault("runtime", {})["dry_run"] = True
            config.setdefault("logging", {})["progress_bar"] = False
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            env = os.environ.copy()
            env["CONFIG_PATH"] = str(config_path)
            env["PYTHON_BIN"] = sys.executable
            result = subprocess.run(
                [
                    "bash",
                    "run.sh",
                    "--diarization-backend",
                    "sortformer",
                    "--diarization-model",
                    "nvidia/diar_sortformer_4spk-v1",
                    "--diarization-device",
                    "auto",
                    "--enable-asr",
                    "--asr-device",
                    "auto",
                    "--until",
                    "pre_asr",
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            output = result.stdout + result.stderr
            self.assertIn("| Diarization        | V       | sortformer", output)
            self.assertIn("| ASR                | V       | phowhisper_local", output)
            self.assertIn("device=dry-run", output)


if __name__ == "__main__":
    unittest.main()
