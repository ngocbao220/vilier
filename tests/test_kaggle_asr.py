import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from tools.kaggle_asr import KaggleAsrRunner, resolve_command, validate_transcript


class KaggleAsrTest(unittest.TestCase):
    def test_resolve_command_uses_configured_kaggle_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            kaggle_bin = Path(tmp) / "kaggle"
            kaggle_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {"KAGGLE_BIN": str(kaggle_bin)}):
                self.assertEqual(resolve_command(["kaggle", "kernels", "status", "owner/slug"])[0], str(kaggle_bin))

    def test_prepare_metadata_uses_fixed_ngocbao_slugs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_bundle(root, vad_count=1)
            runner = KaggleAsrRunner(
                audio_id="sample",
                bundle=bundle,
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-asr-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-phowhisper-asr",
                repo_url="https://github.com/ngocbao220/vilier.git",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
            )

            dataset_dir = runner.prepare_dataset_dir()
            kernel_dir = runner.prepare_kernel_dir()

            dataset_meta = json.loads((dataset_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
            kernel_meta = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
            kernel_code = (kernel_dir / "kernel.py").read_text(encoding="utf-8")

            self.assertEqual(dataset_meta["id"], "ngocbaotrinhtuan/vilier-asr-bundle")
            self.assertEqual(kernel_meta["id"], "ngocbaotrinhtuan/vilier-phowhisper-asr")
            self.assertTrue(kernel_meta["enable_gpu"])
            self.assertEqual(kernel_meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(kernel_meta["dataset_sources"], ["ngocbaotrinhtuan/vilier-asr-bundle"])
            with zipfile.ZipFile(dataset_dir / "vilier_source.zip") as zf:
                self.assertIn("tools/run_asr_bundle.py", zf.namelist())
                self.assertIn("pipeline/asr.py", zf.namelist())
                self.assertIn("config.json", zf.namelist())
            self.assertIn("prepare_source", kernel_code)
            self.assertIn("vilier_source.zip", kernel_code)
            self.assertIn("EMBEDDED_SOURCE_ZIP", kernel_code)
            self.assertIn("base64.b64decode", kernel_code)
            self.assertIn("find_bundle_source", kernel_code)
            self.assertIn("manifest.timeline.json", kernel_code)
            self.assertIn("shutil.copytree", kernel_code)
            self.assertIn("tools", kernel_code)
            self.assertIn("run_asr_bundle.py", kernel_code)

    def test_dry_run_writes_valid_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_bundle(root, vad_count=2)
            output_dir = root / "outputs" / "sample"
            output_dir.mkdir(parents=True)
            with zipfile.ZipFile(bundle) as zf:
                (output_dir / "manifest.timeline.json").write_text(zf.read("manifest.timeline.json").decode("utf-8"), encoding="utf-8")
            runner = KaggleAsrRunner(
                audio_id="sample",
                bundle=bundle,
                output_dir=output_dir,
                dataset_slug="ngocbaotrinhtuan/vilier-asr-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-phowhisper-asr",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
                dry_run=True,
            )

            transcript_path = runner.run()

            validate_transcript(transcript_path, output_dir / "manifest.timeline.json")
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            self.assertEqual(len(transcript), 2)
            self.assertEqual(transcript[0]["text"], "dry-run kaggle transcript 1")
            self.assertEqual(transcript[0]["asr_segment_id"], "asrseg_00000")
            self.assertEqual(transcript[0]["audio"], "asr_audio/SPEAKER_00/audio_00001.wav")

    def test_push_kernel_requests_t4_accelerator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_bundle(root, vad_count=1)
            runner = KaggleAsrRunner(
                audio_id="sample",
                bundle=bundle,
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-asr-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-phowhisper-asr",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
            )
            kernel_dir = root / "kernel"
            kernel_dir.mkdir()

            with mock.patch("tools.kaggle_asr.run_command") as run_command:
                run_command.return_value.returncode = 0
                runner.push_kernel(kernel_dir)

            command = run_command.call_args.args[0]
            self.assertEqual(command, ["kaggle", "kernels", "push", "-p", str(kernel_dir), "--accelerator", "NvidiaTeslaT4"])

    def test_wait_for_dataset_files_accepts_extracted_bundle_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_bundle(root, vad_count=1)
            runner = KaggleAsrRunner(
                audio_id="sample",
                bundle=bundle,
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-asr-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-phowhisper-asr",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
            )

            with mock.patch("tools.kaggle_asr.run_command") as run_command:
                run_command.return_value.returncode = 0
                run_command.return_value.stdout = f"{bundle.stem}/asr_audio/SPEAKER_00/audio_00001.wav\n"
                run_command.return_value.stderr = ""
                runner.wait_for_dataset_files()

            self.assertEqual(
                run_command.call_args.args[0],
                ["kaggle", "datasets", "files", "ngocbaotrinhtuan/vilier-asr-bundle", "--page-size", "200"],
            )

    def test_wait_for_dataset_files_follows_next_page_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_bundle(root, vad_count=1)
            runner = KaggleAsrRunner(
                audio_id="sample",
                bundle=bundle,
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-asr-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-phowhisper-asr",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
            )
            first_page = mock.Mock(returncode=0, stdout="Next Page Token = token-2\nname size\n", stderr="")
            second_page = mock.Mock(returncode=0, stdout=f"{bundle.stem}/asr_audio/SPEAKER_00/audio_00001.wav\n", stderr="")

            with mock.patch("tools.kaggle_asr.run_command", side_effect=[first_page, second_page]) as run_command:
                runner.wait_for_dataset_files()

            self.assertEqual(
                run_command.call_args_list[1].args[0],
                ["kaggle", "datasets", "files", "ngocbaotrinhtuan/vilier-asr-bundle", "--page-size", "200", "--page-token", "token-2"],
            )

    def test_wait_for_dataset_files_continues_after_successful_paginated_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = make_bundle(root, vad_count=1)
            runner = KaggleAsrRunner(
                audio_id="sample",
                bundle=bundle,
                output_dir=root / "outputs" / "sample",
                dataset_slug="ngocbaotrinhtuan/vilier-asr-bundle",
                kernel_slug="ngocbaotrinhtuan/vilier-phowhisper-asr",
                repo_url="",
                repo_ref="main",
                accelerator="NvidiaTeslaT4",
                work_dir=root / "work",
                poll_seconds=0.01,
                dataset_ready_seconds=0,
            )
            page = mock.Mock(returncode=0, stdout="Next Page Token = token-2\nname size\n", stderr="")

            with mock.patch("tools.kaggle_asr.run_command", return_value=page):
                runner.wait_for_dataset_files()


def make_bundle(root: Path, vad_count: int) -> Path:
    bundle_dir = root / "bundle_src"
    asr_dir = bundle_dir / "asr_audio" / "SPEAKER_00"
    asr_dir.mkdir(parents=True)
    vad_segments = []
    asr_segments = []
    for idx in range(1, vad_count + 1):
        sf.write(asr_dir / f"audio_{idx:05d}.wav", np.zeros(16000, dtype=np.float32), 16000)
        vad_segments.append({"id": f"vad_{idx - 1:05d}", "start": float(idx - 1), "end": float(idx), "duration": 1.0})
        asr_segments.append(
            {
                "id": f"asrseg_{idx - 1:05d}",
                "speaker": "SPEAKER_00",
                "start": float(idx - 1),
                "end": float(idx),
                "duration": 1.0,
                "audio": f"asr_audio/SPEAKER_00/audio_{idx:05d}.wav",
            }
        )
    manifest = {
        "audio_id": "sample",
        "vad_segments": vad_segments,
        "asr_segments": asr_segments,
        "segments": [],
    }
    (bundle_dir / "manifest.timeline.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (bundle_dir / "vad.json").write_text(json.dumps(vad_segments, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle = root / "sample_asr_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        for item in bundle_dir.rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(bundle_dir).as_posix())
    return bundle


if __name__ == "__main__":
    unittest.main()
