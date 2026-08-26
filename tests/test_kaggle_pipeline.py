import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from tools.kaggle_pipeline import KagglePipelineRunner, validate_pipeline_result, write_source_archive


class KagglePipelineTest(unittest.TestCase):
    def test_dry_run_writes_downloaded_output_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
            config = {
                "entrypoint": {"input_path": str(audio), "output_path": str(root / "outputs"), "sample_rate": 16000},
                "runtime": {"dry_run": True, "backend": "kaggle"},
                "asr": {"enabled": True, "model": "vinai/PhoWhisper-large", "language": "vi"},
                "state_labeling": {"enabled": False},
            }
            runner = KagglePipelineRunner(
                audio_id="sample",
                audio_path=audio,
                config=config,
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-pipeline-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-gpu-pipeline",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
                dry_run=True,
            )

            output_dir = runner.run()

            validate_pipeline_result(output_dir, asr_enabled=True)
            transcript = json.loads((output_dir / "transcript.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "manifest.timeline.json").read_text(encoding="utf-8"))
            self.assertEqual(transcript[0]["text"], "dry-run kaggle pipeline transcript 1")
            self.assertEqual(manifest["audio_id"], "sample")
            self.assertTrue((output_dir / "audio.standardized.wav").exists())

    def test_prepare_dataset_includes_raw_audio_config_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
            config = {
                "entrypoint": {"input_path": str(audio), "output_path": "outputs", "sample_rate": 16000},
                "runtime": {"dry_run": False, "backend": "kaggle"},
                "overlap_separation": {"enabled": False},
                "asr": {"enabled": True},
            }
            runner = KagglePipelineRunner(
                audio_id="sample",
                audio_path=audio,
                config=config,
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-pipeline-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-gpu-pipeline",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
            )

            dataset_dir = runner.prepare_dataset_dir()
            kernel_dir = runner.prepare_kernel_dir()

            metadata = json.loads((dataset_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["id"], "ngocbaotrinhtuan/vilier-pipeline-bundle")
            self.assertTrue((dataset_dir / "sample.wav").exists())
            self.assertTrue((dataset_dir / "config.json").exists())
            self.assertTrue((dataset_dir / "vilier_source.zip").exists())
            with zipfile.ZipFile(dataset_dir / "vilier_source.zip") as zf:
                self.assertIn("pipeline/cli.py", zf.namelist())
                self.assertIn("pipeline/diarization.py", zf.namelist())
                self.assertIn("tools/run_asr_bundle.py", zf.namelist())

            kernel_meta = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(kernel_meta["dataset_sources"], ["ngocbaotrinhtuan/vilier-pipeline-bundle"])
            self.assertEqual(kernel_meta["machine_shape"], "NvidiaTeslaT4")
            kernel_code = (kernel_dir / "kernel.py").read_text(encoding="utf-8")
            self.assertIn("HUGGINGFACE_TOKEN", kernel_code)
            self.assertIn("pipeline.cli", kernel_code)
            self.assertIn("pipeline_result.zip", kernel_code)

    def test_source_archive_contains_pipeline_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "source.zip"
            write_source_archive(archive, include_sepreformer=False, sepreformer_model_name="")

            with zipfile.ZipFile(archive) as zf:
                names = set(zf.namelist())

        self.assertIn("pipeline/music.py", names)
        self.assertIn("pipeline/overlap_separation.py", names)
        self.assertIn("run_pipeline.sh", names)

    def test_wait_for_dataset_files_follows_next_page_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
            runner = KagglePipelineRunner(
                audio_id="sample",
                audio_path=audio,
                config={"entrypoint": {"sample_rate": 16000}, "asr": {"enabled": False}},
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-pipeline-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-gpu-pipeline",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
            )
            first_page = mock.Mock(returncode=0, stdout="Next Page Token = token-2\nname size\n", stderr="")
            second_page = mock.Mock(returncode=0, stdout="sample.wav\nconfig.json\nvilier_source.zip\n", stderr="")

            with mock.patch("tools.kaggle_pipeline.run_command", side_effect=[first_page, second_page]) as run_command:
                runner.wait_for_dataset_files()

            self.assertEqual(
                run_command.call_args_list[1].args[0],
                ["kaggle", "datasets", "files", "ngocbaotrinhtuan/vilier-pipeline-bundle", "--page-size", "200", "--page-token", "token-2"],
            )

    def test_wait_for_dataset_files_continues_after_successful_paginated_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
            runner = KagglePipelineRunner(
                audio_id="sample",
                audio_path=audio,
                config={"entrypoint": {"sample_rate": 16000}, "asr": {"enabled": False}},
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-pipeline-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-gpu-pipeline",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
                poll_seconds=0.01,
                dataset_ready_seconds=0,
            )
            page = mock.Mock(returncode=0, stdout="Next Page Token = token-2\nname size\n", stderr="")

            with mock.patch("tools.kaggle_pipeline.run_command", return_value=page):
                runner.wait_for_dataset_files()


if __name__ == "__main__":
    unittest.main()
