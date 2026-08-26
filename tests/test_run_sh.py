import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RunShTest(unittest.TestCase):
    def test_kaggle_backend_is_skipped_when_asr_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
            config["entrypoint"]["input_path"] = "inputs/podcast_single_30s.wav"
            config["entrypoint"]["output_path"] = str(root / "outputs")
            config.setdefault("runtime", {})["backend"] = "local"
            config["runtime"]["dry_run"] = True
            config.setdefault("asr", {})["enabled"] = False
            config["asr"]["asr_backend"] = "kaggle"
            config.setdefault("logging", {})["progress_bar"] = False
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            env = os.environ.copy()
            env["CONFIG_PATH"] = str(config_path)
            env["PYTHON_BIN"] = sys.executable
            result = subprocess.run(["bash", "run.sh"], check=True, text=True, capture_output=True, env=env)

            output = result.stdout + result.stderr
            self.assertIn("[INFO] Pipeline component usage", output)
            self.assertIn("| ASR                | X", output)
            self.assertIn("| State labeling     | X", output)
            self.assertIn("ASR disabled in config, skipping ASR backend kaggle", output)
            self.assertIn("asr SKIP", output)
            self.assertNotIn("Bundle đã tạo", output)
            self.assertFalse((root / "outputs" / "podcast_single_30s" / "transcript.json").exists())

    def test_kaggle_backend_can_be_configured_without_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
            config.setdefault("runtime", {})["backend"] = "local"
            config["entrypoint"]["input_path"] = "inputs/podcast_single_30s.wav"
            config["entrypoint"]["output_path"] = str(root / "outputs")
            config["runtime"]["dry_run"] = True
            config.setdefault("asr", {})["enabled"] = True
            config["asr"]["asr_backend"] = "kaggle"
            config.setdefault("state_labeling", {})["enabled"] = False
            config.setdefault("logging", {})["progress_bar"] = False
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            env = os.environ.copy()
            env["CONFIG_PATH"] = str(config_path)
            env["PYTHON_BIN"] = sys.executable
            env.pop("ASR_BACKEND", None)
            result = subprocess.run(["bash", "run.sh"], check=True, text=True, capture_output=True, env=env)

            output = result.stdout + result.stderr
            self.assertIn("Bundle đã tạo, đang upload lên kaggle để lấy kết quả", output)
            self.assertIn("Đã có kết quả, đang tải transcript xuống", output)
            self.assertTrue((root / "outputs" / "podcast_single_30s" / "transcript.json").exists())

    def test_runtime_kaggle_runs_full_pipeline_dry_run_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads(Path("config.json").read_text(encoding="utf-8"))
            config["entrypoint"]["input_path"] = "inputs/podcast_single_30s.wav"
            config["entrypoint"]["output_path"] = str(root / "outputs")
            config.setdefault("runtime", {})["backend"] = "kaggle"
            config["runtime"]["dry_run"] = True
            config.setdefault("asr", {})["enabled"] = True
            config["asr"]["asr_backend"] = "local"
            config.setdefault("state_labeling", {})["enabled"] = False
            config.setdefault("logging", {})["progress_bar"] = False
            config["kaggle"] = {
                "dataset_slug": "ngocbaotrinhtuan/vilier-pipeline-bundle",
                "kernel_slug": "ngocbaotrinhtuan/vilier-gpu-pipeline",
                "accelerator": "NvidiaTeslaT4",
                "work_dir": str(root / "kaggle_work"),
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            env = os.environ.copy()
            env["CONFIG_PATH"] = str(config_path)
            env["PYTHON_BIN"] = sys.executable
            result = subprocess.run(["bash", "run.sh"], check=True, text=True, capture_output=True, env=env)

            output = result.stdout + result.stderr
            self.assertIn("Bundle đã tạo, đang upload lên kaggle để chạy pipeline", output)
            self.assertIn("Kaggle đang chạy pipeline trên GPU", output)
            self.assertIn("Đã có kết quả, đang tải outputs xuống", output)
            self.assertIn("Đã tải outputs, đang chạy các phase còn lại", output)
            self.assertTrue((root / "outputs" / "podcast_single_30s" / "manifest.timeline.json").exists())
            self.assertTrue((root / "outputs" / "podcast_single_30s" / "transcript.json").exists())


if __name__ == "__main__":
    unittest.main()
