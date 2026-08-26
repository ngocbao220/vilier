import json
import builtins
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

from pipeline.diarization import (
    DiariZenDiarizer,
    _allow_torch_checkpoint_globals,
    _import_diarizen_pipeline,
    _hf_hub_download_use_auth_token_compat,
    _load_pyannote_pipeline,
    _patch_torchaudio_audio_metadata,
    _speechbrain_device,
    _speechbrain_use_auth_token_compat,
    load_diarizer,
    PyannotePixitDiarizer,
    build_diarization_chunks,
    pyannote_annotation_to_segments,
    remap_chunk_segments_to_original,
    resolve_torch_device,
)
from pipeline.music import _suppress_accompaniment, apply_music_separation, load_music_separator
from pipeline.overlap_separation import _find_checkpoint_dir, _import_sepreformer_model_class, _resolve_sepreformer_path, apply_overlap_separation
from pipeline.schema import SpeakerSegment
from pipeline.timeline import annotate_overlaps, export_audacity_labels, export_segments_and_tracks, write_manifest
from pipeline.vad import SileroVadRunner, _TorchHubSileroVad, cleanup_intervals, export_vad_audio, write_vad_txt
from pipeline.asr import (
    DryRunAsrRunner,
    PhoWhisperLocalRunner,
    export_speaker_asr_audio,
    normalize_pipeline_device,
    transcribe_asr_segments,
    transcribe_vad_audio,
)
from pipeline.audio import load_mono


class TimelineTest(unittest.TestCase):
    def test_normalize_pipeline_device_converts_numeric_strings(self):
        self.assertEqual(normalize_pipeline_device("0"), 0)
        self.assertEqual(normalize_pipeline_device("-1"), -1)
        self.assertEqual(normalize_pipeline_device("cpu"), "cpu")

    def test_resolve_torch_device_prefers_cuda_then_mps_for_auto(self):
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        self.assertEqual(resolve_torch_device(torch_module, "auto"), "mps")

        torch_module.cuda.is_available = lambda: True
        self.assertEqual(resolve_torch_device(torch_module, "auto"), "cuda")

    def test_resolve_torch_device_falls_back_when_mps_unavailable(self):
        torch_module = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(resolve_torch_device(torch_module, "mps"), "cpu")

    def test_pixit_dry_run_reports_dry_run_device(self):
        diarizer = PyannotePixitDiarizer({"backend": "pixit", "device": "auto"}, dry_run=True)
        self.assertEqual(diarizer.resolved_device, "dry-run")

    def test_load_mono_falls_back_to_ffmpeg_when_soundfile_cannot_decode(self):
        with mock.patch("pipeline.audio.sf.read", side_effect=RuntimeError("bad codec")), mock.patch(
            "pipeline.audio.subprocess.check_output",
            return_value=np.array([0.1, -0.1], dtype=np.float32).tobytes(),
        ) as check_output:
            waveform, sample_rate = load_mono(Path("inputs/example.mp3"), 16000)

        self.assertEqual(sample_rate, 16000)
        np.testing.assert_allclose(waveform, np.array([0.1, -0.1], dtype=np.float32))
        self.assertIn("ffmpeg", check_output.call_args.args[0][0])

    def test_cleanup_intervals_merges_and_filters(self):
        cleaned = cleanup_intervals(
            [
                {"start": 0.0, "end": 0.1},
                {"start": 1.0, "end": 1.5},
                {"start": 1.6, "end": 2.0},
            ],
            min_duration=0.25,
            merge_gap=0.2,
        )
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["start"], 1.0)
        self.assertEqual(cleaned[0]["end"], 2.0)

    def test_silero_detect_passes_configured_threshold(self):
        calls = []

        class FakeModel:
            vad_model = object()

            def get_speech_timestamps(self, audio, model, **kwargs):
                calls.append(kwargs)
                return [{"start": 0, "end": 1600}]

        runner = SileroVadRunner.__new__(SileroVadRunner)
        runner.config = {"threshold": 0.35, "min_duration_seconds": 0.01, "merge_gap_seconds": 0.2}
        runner.sample_rate = 16000
        runner.model = _TorchHubSileroVad(FakeModel().vad_model, FakeModel().get_speech_timestamps, sampling_rate=16000)

        with mock.patch.dict(
            sys.modules,
            {
                "librosa": SimpleNamespace(resample=lambda audio, orig_sr, target_sr: audio),
            },
        ):
            segments = runner._silero_detect(np.zeros(16000, dtype=np.float32))

        self.assertEqual(calls[0]["threshold"], 0.35)
        self.assertEqual(segments, [{"id": "vad_00000", "start": 0.0, "end": 0.1}])

    def test_silero_load_model_uses_packaged_silero_vad(self):
        fake_package = SimpleNamespace(
            load_silero_vad=lambda: object(),
            get_speech_timestamps=lambda audio, model, **kwargs: [],
        )
        runner = SileroVadRunner.__new__(SileroVadRunner)
        runner.config = {"model": "silero_vad"}
        runner.sample_rate = 16000
        runner.dry_run = False

        with mock.patch.dict(sys.modules, {"silero_vad": fake_package}):
            model = runner._load_model()

        self.assertEqual(model.sampling_rate, 16000)

    def test_silero_load_model_falls_back_to_torchhub_without_models_or_package(self):
        class FakeHub:
            def load(self, **kwargs):
                self.kwargs = kwargs
                return object(), (lambda audio, model, **call_kwargs: [],)

        fake_hub = FakeHub()
        fake_torch = SimpleNamespace(hub=fake_hub)
        runner = SileroVadRunner.__new__(SileroVadRunner)
        runner.config = {"model": "silero_vad"}
        runner.sample_rate = 16000
        runner.dry_run = False

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in {"models", "silero_vad"}:
                raise ModuleNotFoundError(f"No module named '{name}'", name=name)
            return original_import(name, *args, **kwargs)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch("builtins.__import__", side_effect=fake_import):
            model = runner._load_model()

        self.assertEqual(model.sampling_rate, 16000)
        self.assertEqual(fake_hub.kwargs["repo_or_dir"], "snakers4/silero-vad")
        self.assertEqual(fake_hub.kwargs["model"], "silero_vad")

    def test_overlap_grouping(self):
        segments = [
            SpeakerSegment("a", "SPEAKER_00", 0.0, 2.0),
            SpeakerSegment("b", "SPEAKER_01", 1.5, 2.5),
            SpeakerSegment("c", "SPEAKER_00", 3.0, 4.0),
        ]
        annotate_overlaps(segments, threshold=0.1)
        self.assertTrue(segments[0].is_overlap)
        self.assertTrue(segments[1].is_overlap)
        self.assertEqual(segments[0].overlap_group_id, segments[1].overlap_group_id)
        self.assertFalse(segments[2].is_overlap)

    def test_track_export_preserves_full_duration(self):
        sample_rate = 16000
        waveform = np.zeros(sample_rate * 3, dtype=np.float32)
        waveform[0:sample_rate] = 0.1
        waveform[sample_rate * 2 : sample_rate * 3] = 0.2
        segments = [
            SpeakerSegment("00000_SPEAKER_00", "SPEAKER_00", 0.0, 1.0),
            SpeakerSegment("00001_SPEAKER_01", "SPEAKER_01", 2.0, 3.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tracks = export_segments_and_tracks(waveform, sample_rate, segments, out)
            self.assertEqual(len(tracks), 2)
            for track in tracks:
                info = sf.info(out / track.track_wav)
                self.assertEqual(info.frames, len(waveform))
            vad_segments = [{"id": "vad_00000", "start": 0.0, "end": 1.5}]
            manifest = write_manifest(
                out,
                "sample",
                Path.cwd() / "inputs" / "source.wav",
                out / "audio.standardized.wav",
                3.0,
                sample_rate,
                segments,
                tracks,
                vad_segments,
            )
            data = json.loads(manifest.read_text())
            self.assertEqual(data["source_audio"], "inputs/source.wav")
            self.assertEqual(data["transcript"], [])
            self.assertEqual(len(data["segments"]), 2)
            self.assertEqual(data["vad_segments"], [{"id": "vad_00000", "start": 0.0, "end": 1.5, "duration": 1.5}])

    def test_apply_music_separation_writes_cleaned_audio(self):
        class FakeMusicSeparator:
            def separate_music(self, waveform, sample_rate):
                return waveform * 0.5

        sample_rate = 10
        waveform = np.ones(sample_rate, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = apply_music_separation(waveform, sample_rate, out, FakeMusicSeparator())

            self.assertTrue(result["applied"])
            self.assertEqual(result["audio"], "music_cleaned.wav")
            np.testing.assert_allclose(result["waveform"], 0.5, atol=1e-4)
            audio, sr = sf.read(out / "music_cleaned.wav", dtype="float32")
            self.assertEqual(sr, sample_rate)
            np.testing.assert_allclose(audio, 0.5, atol=1e-4)

    def test_load_music_separator_dry_run_uses_noop_separator(self):
        sample_rate = 10
        waveform = np.linspace(-0.5, 0.5, sample_rate, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            separator = load_music_separator({"enabled": True, "backend": "demucs"}, dry_run=True)
            result = apply_music_separation(waveform, sample_rate, out, separator)

            self.assertTrue(result["applied"])
            np.testing.assert_allclose(result["waveform"], waveform)
            self.assertTrue((out / "music_cleaned.wav").exists())

    def test_suppress_accompaniment_subtracts_residual_music(self):
        vocals = np.array([0.8, 0.2, -0.4], dtype=np.float32)
        accompaniment = np.array([0.2, -0.2, -0.2], dtype=np.float32)

        cleaned = _suppress_accompaniment(vocals, accompaniment, strength=0.5)

        np.testing.assert_allclose(cleaned, np.array([0.7, 0.3, -0.3], dtype=np.float32), atol=1e-6)

    def test_manifest_preserves_transcript_records(self):
        sample_rate = 16000
        waveform = np.zeros(sample_rate, dtype=np.float32)
        segments = [SpeakerSegment("00000_SPEAKER_00", "SPEAKER_00", 0.0, 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tracks = export_segments_and_tracks(waveform, sample_rate, segments, out)
            transcript = [
                {
                    "id": "asr_00000",
                    "vad_id": "vad_00000",
                    "audio": "vad_audio/audio_1.wav",
                    "start": 0.0,
                    "end": 1.0,
                    "duration": 1.0,
                    "speaker": "SPEAKER_00",
                    "text": "xin chao",
                    "model": "vinai/PhoWhisper-large",
                    "language": "vi",
                }
            ]
            manifest = write_manifest(
                out,
                "sample",
                Path("source.wav"),
                out / "audio.standardized.wav",
                1.0,
                sample_rate,
                segments,
                tracks,
                [{"id": "vad_00000", "start": 0.0, "end": 1.0}],
                music_separation={"enabled": True, "applied": True, "audio": "music_cleaned.wav"},
                transcript=transcript,
            )
            data = json.loads(manifest.read_text())
            self.assertEqual(data["transcript"], transcript)
            self.assertEqual(data["music_separation"]["audio"], "music_cleaned.wav")

    def test_transcribe_vad_audio_assigns_largest_overlap_speaker(self):
        sample_rate = 16000
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            vad_dir = out / "vad_audio"
            vad_dir.mkdir()
            sf.write(vad_dir / "audio_1.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)
            vad_segments = [{"id": "vad_00000", "start": 1.0, "end": 3.0}]
            speaker_segments = [
                SpeakerSegment("short", "SPEAKER_00", 1.0, 1.3),
                SpeakerSegment("long", "SPEAKER_01", 1.5, 3.0),
            ]
            transcript = transcribe_vad_audio(
                output_dir=out,
                vad_segments=vad_segments,
                vad_audio=[Path("vad_audio/audio_1.wav")],
                speaker_segments=speaker_segments,
                runner=DryRunAsrRunner(model_name="vinai/PhoWhisper-large", language="vi"),
            )
            self.assertEqual(len(transcript), 1)
            self.assertEqual(transcript[0]["speaker"], "SPEAKER_01")
            self.assertEqual(transcript[0]["start"], 1.0)
            self.assertEqual(transcript[0]["end"], 3.0)
            self.assertEqual(transcript[0]["text"], "dry-run transcript 1")

    def test_export_speaker_asr_audio_segments_speaker_tracks(self):
        class FakeVadRunner:
            def detect(self, waveform):
                if np.max(np.abs(waveform)) < 0.05:
                    return []
                return [{"id": "vad_00000", "start": 0.5, "end": 1.0}]

        sample_rate = 10
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tracks_dir = out / "tracks"
            tracks_dir.mkdir()
            speaker_00 = np.zeros(sample_rate * 2, dtype=np.float32)
            speaker_00[5:10] = 0.4
            sf.write(tracks_dir / "SPEAKER_00.wav", speaker_00, sample_rate)
            sf.write(tracks_dir / "SPEAKER_01.wav", np.zeros(sample_rate * 2, dtype=np.float32), sample_rate)

            asr_segments = export_speaker_asr_audio(
                output_dir=out,
                speaker_tracks=[
                    {"id": "SPEAKER_00", "track_wav": "tracks/SPEAKER_00.wav"},
                    {"id": "SPEAKER_01", "track_wav": "tracks/SPEAKER_01.wav"},
                ],
                vad_runner=FakeVadRunner(),
            )

            self.assertEqual(
                asr_segments,
                [
                    {
                        "id": "asrseg_00000",
                        "speaker": "SPEAKER_00",
                        "start": 0.5,
                        "end": 1.0,
                        "duration": 0.5,
                        "audio": "asr_audio/SPEAKER_00/audio_00001.wav",
                    }
                ],
            )
            audio, sr = sf.read(out / "asr_audio" / "SPEAKER_00" / "audio_00001.wav", dtype="float32")
            self.assertEqual(sr, sample_rate)
            self.assertEqual(len(audio), 5)

    def test_transcribe_asr_segments_uses_speaker_audio(self):
        sample_rate = 16000
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            asr_dir = out / "asr_audio" / "SPEAKER_00"
            asr_dir.mkdir(parents=True)
            sf.write(asr_dir / "audio_00001.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)
            transcript = transcribe_asr_segments(
                output_dir=out,
                asr_segments=[
                    {
                        "id": "asrseg_00000",
                        "speaker": "SPEAKER_00",
                        "start": 0.0,
                        "end": 1.0,
                        "audio": "asr_audio/SPEAKER_00/audio_00001.wav",
                    }
                ],
                runner=DryRunAsrRunner(model_name="vinai/PhoWhisper-large", language="vi"),
            )

            self.assertEqual(transcript[0]["asr_segment_id"], "asrseg_00000")
            self.assertEqual(transcript[0]["audio"], "asr_audio/SPEAKER_00/audio_00001.wav")
            self.assertEqual(transcript[0]["speaker"], "SPEAKER_00")
            self.assertNotIn("vad_id", transcript[0])

    def test_phowhisper_runner_passes_loaded_audio_array_to_pipeline(self):
        class FakePipeline:
            def __init__(self):
                self.calls = []

            def __call__(self, audio, **kwargs):
                self.calls.append((audio, kwargs))
                return {"text": "xin chao"}

        sample_rate = 16000
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            sf.write(path, np.zeros(sample_rate, dtype=np.float32), sample_rate)
            runner = PhoWhisperLocalRunner({"model": "vinai/PhoWhisper-large", "language": "vi", "device": "cpu", "chunk_length_seconds": 30.0})
            fake = FakePipeline()
            runner._pipeline = fake

            self.assertEqual(runner.transcribe(path, index=1), "xin chao")

            audio_arg, kwargs = fake.calls[0]
            self.assertIn("array", audio_arg)
            self.assertEqual(audio_arg["sampling_rate"], sample_rate)
            self.assertEqual(kwargs["chunk_length_s"], 30.0)
            self.assertTrue(kwargs["ignore_warning"])
            self.assertEqual(kwargs["generate_kwargs"], {"language": "vi", "task": "transcribe"})

    def test_transcribe_vad_audio_reports_progress_per_vad_audio(self):
        progress = []
        sample_rate = 16000
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            vad_dir = out / "vad_audio"
            vad_dir.mkdir()
            sf.write(vad_dir / "audio_1.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)
            sf.write(vad_dir / "audio_2.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)

            transcribe_vad_audio(
                output_dir=out,
                vad_segments=[
                    {"id": "vad_00000", "start": 0.0, "end": 1.0},
                    {"id": "vad_00001", "start": 1.0, "end": 2.0},
                ],
                vad_audio=[Path("vad_audio/audio_1.wav"), Path("vad_audio/audio_2.wav")],
                speaker_segments=[],
                runner=DryRunAsrRunner(model_name="vinai/PhoWhisper-large", language="vi"),
                progress_callback=lambda current, total, label: progress.append((current, total, label)),
            )

        self.assertEqual(
            progress,
            [
                (1, 2, "vad_audio/audio_1.wav"),
                (2, 2, "vad_audio/audio_2.wav"),
            ],
        )

    def test_transcribe_vad_audio_adds_audio_context_to_asr_errors(self):
        class BrokenRunner:
            model_name = "broken"
            language = "vi"

            def transcribe(self, audio_path, index):
                raise ValueError("decode failed")

        sample_rate = 16000
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            vad_dir = out / "vad_audio"
            vad_dir.mkdir()
            sf.write(vad_dir / "audio_1.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)

            with self.assertRaisesRegex(ValueError, "ASR failed for vad_00000 at vad_audio/audio_1.wav"):
                transcribe_vad_audio(
                    output_dir=out,
                    vad_segments=[{"id": "vad_00000", "start": 0.0, "end": 1.0}],
                    vad_audio=[Path("vad_audio/audio_1.wav")],
                    speaker_segments=[],
                    runner=BrokenRunner(),
                )

    def test_track_export_can_skip_segment_wavs(self):
        sample_rate = 16000
        waveform = np.zeros(sample_rate * 3, dtype=np.float32)
        waveform[0:sample_rate] = 0.1
        waveform[sample_rate * 2 : sample_rate * 3] = 0.2
        segments = [
            SpeakerSegment("00000_SPEAKER_00", "SPEAKER_00", 0.0, 1.0),
            SpeakerSegment("00001_SPEAKER_01", "SPEAKER_01", 2.0, 3.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tracks = export_segments_and_tracks(waveform, sample_rate, segments, out, write_segment_wavs=False)
            self.assertFalse((out / "segments").exists())
            self.assertEqual([segment.segment_wav for segment in segments], ["", ""])
            self.assertEqual(len(tracks), 2)
            for track in tracks:
                info = sf.info(out / track.track_wav)
                self.assertEqual(info.frames, len(waveform))

    def test_write_vad_txt_uses_tab_separated_times_and_label(self):
        vad_segments = [
            {"id": "vad_00000", "start": 0.0, "end": 1.5},
            {"id": "vad_00001", "start": 2.25, "end": 3.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vad.txt"
            write_vad_txt(path, vad_segments)
            self.assertEqual(path.read_text(encoding="utf-8"), "0.000\t1.500\tspeech\n2.250\t3.000\tspeech\n")

    def test_export_vad_audio_writes_one_indexed_utterance_files(self):
        sample_rate = 16000
        waveform = np.zeros(sample_rate * 3, dtype=np.float32)
        waveform[0:sample_rate] = 0.1
        waveform[sample_rate * 2 : sample_rate * 3] = 0.2
        vad_segments = [
            {"id": "vad_00000", "start": 0.0, "end": 1.0},
            {"id": "vad_00001", "start": 2.0, "end": 3.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            files = export_vad_audio(waveform, sample_rate, vad_segments, out)
            self.assertEqual(files, [Path("vad_audio/audio_1.wav"), Path("vad_audio/audio_2.wav")])
            self.assertEqual(sf.info(out / "vad_audio" / "audio_1.wav").frames, sample_rate)
            self.assertEqual(sf.info(out / "vad_audio" / "audio_2.wav").frames, sample_rate)

    def test_export_audacity_labels_writes_combined_and_per_speaker_files(self):
        segments = [
            SpeakerSegment("00000_SPEAKER_00", "SPEAKER_00", 0.0, 1.5),
            SpeakerSegment("00001_SPEAKER_01", "SPEAKER_01", 2.25, 3.0),
            SpeakerSegment("00002_SPEAKER_00", "SPEAKER_00", 4.0, 5.0),
        ]
        vad_segments = [{"id": "vad_00000", "start": 0.0, "end": 3.0}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            files = export_audacity_labels(out, segments, vad_segments)
            self.assertEqual(files["vad"], Path("labels/vad.txt"))
            self.assertEqual(files["speakers"], Path("labels/speakers.txt"))
            self.assertEqual(files["speaker_labels"]["SPEAKER_00"], Path("labels/SPEAKER_00.txt"))
            self.assertEqual((out / "labels" / "vad.txt").read_text(encoding="utf-8"), "0.000\t3.000\tspeech\n")
            self.assertEqual(
                (out / "labels" / "speakers.txt").read_text(encoding="utf-8"),
                "0.000\t1.500\tSPEAKER_00\n2.250\t3.000\tSPEAKER_01\n4.000\t5.000\tSPEAKER_00\n",
            )
            self.assertEqual(
                (out / "labels" / "SPEAKER_00.txt").read_text(encoding="utf-8"),
                "0.000\t1.500\tSPEAKER_00\n4.000\t5.000\tSPEAKER_00\n",
            )

    def test_build_diarization_chunks_concatenates_vad_audio_under_limit(self):
        sample_rate = 10
        waveform = np.arange(sample_rate * 10, dtype=np.float32) / 100.0
        vad_segments = [
            {"id": "vad_00000", "start": 1.0, "end": 2.0},
            {"id": "vad_00001", "start": 3.0, "end": 5.0},
            {"id": "vad_00002", "start": 7.0, "end": 8.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            chunks = build_diarization_chunks(waveform, sample_rate, vad_segments, out, max_chunk_seconds=3.0)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["audio"], "diarization_chunks/chunk_1.wav")
            self.assertLess(chunks[0]["duration"], 3.0)
            self.assertLess(chunks[1]["duration"], 3.0)
            self.assertEqual(chunks[0]["mapping"][0]["source_start"], 1.0)
            self.assertEqual(chunks[0]["mapping"][0]["chunk_start"], 0.0)
            self.assertEqual(chunks[0]["mapping"][1]["source_start"], 3.0)
            self.assertEqual(chunks[0]["mapping"][1]["chunk_start"], 1.0)
            self.assertTrue((out / "diarization_chunks" / "chunk_1.wav").exists())
            self.assertTrue((out / "diarization_chunks" / "chunk_2.wav").exists())

    def test_pyannote_annotation_to_segments_normalizes_speakers(self):
        class Turn:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        class Annotation:
            def itertracks(self, yield_label=False):
                self.assertTrue(yield_label)
                yield Turn(1.0, 2.0), None, "speaker_b"
                yield Turn(0.0, 0.1), None, "speaker_short"
                yield Turn(0.2, 0.8), None, "speaker_a"

        annotation = Annotation()
        annotation.assertTrue = self.assertTrue

        segments = pyannote_annotation_to_segments(annotation, min_duration=0.25)

        self.assertEqual([(segment.start, segment.end, segment.speaker) for segment in segments], [(0.2, 0.8, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")])

    def test_load_diarizer_accepts_diarizen_backend_in_dry_run(self):
        diarizer = load_diarizer({"backend": "diarizen"}, dry_run=True)

        self.assertIsInstance(diarizer, DiariZenDiarizer)
        self.assertEqual(diarizer.model_name, "BUT-FIT/diarizen-wavlm-large-s80-md")

    def test_diarizen_diarizer_loads_pipeline_and_normalizes_annotation(self):
        test_case = self

        class Turn:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        class Annotation:
            uri = None

            def itertracks(self, yield_label=False):
                self.assertTrue(yield_label)
                yield Turn(0.0, 1.0), None, "speaker_b"
                yield Turn(1.2, 2.0), None, "speaker_a"

        class Pipeline:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return cls()

            def __call__(self, audio_path, sess_name=None):
                self.audio_path = audio_path
                self.sess_name = sess_name
                annotation = Annotation()
                annotation.assertTrue = test_case.assertTrue
                return annotation

        inference = types.ModuleType("diarizen.pipelines.inference")
        inference.DiariZenPipeline = Pipeline
        pipelines = types.ModuleType("diarizen.pipelines")
        diarizen = types.ModuleType("diarizen")

        with mock.patch.dict(
            sys.modules,
            {
                "diarizen": diarizen,
                "diarizen.pipelines": pipelines,
                "diarizen.pipelines.inference": inference,
            },
        ):
            diarizer = DiariZenDiarizer(
                {
                    "model": "BUT-FIT/test-model",
                    "cache_dir": "/tmp/diarizen-cache",
                    "rttm_out_dir": "/tmp/rttm",
                    "min_duration_seconds": 0.25,
                }
            )
            segments = diarizer.diarize(Path("meeting.wav"), [])

        self.assertEqual(
            Pipeline.calls,
            [
                (
                    ("BUT-FIT/test-model",),
                    {"cache_dir": "/tmp/diarizen-cache", "rttm_out_dir": "/tmp/rttm"},
                )
            ],
        )
        self.assertEqual(diarizer.pipeline.audio_path, "meeting.wav")
        self.assertEqual(diarizer.pipeline.sess_name, "meeting")
        self.assertEqual([(segment.start, segment.end, segment.speaker) for segment in segments], [(0.0, 1.0, "SPEAKER_00"), (1.2, 2.0, "SPEAKER_01")])

    def test_diarizen_diarizer_default_load_matches_upstream_example(self):
        class Pipeline:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return cls()

        inference = types.ModuleType("diarizen.pipelines.inference")
        inference.DiariZenPipeline = Pipeline
        pipelines = types.ModuleType("diarizen.pipelines")
        diarizen = types.ModuleType("diarizen")

        with mock.patch.dict(
            sys.modules,
            {
                "diarizen": diarizen,
                "diarizen.pipelines": pipelines,
                "diarizen.pipelines.inference": inference,
            },
        ):
            DiariZenDiarizer({"model": "BUT-FIT/diarizen-wavlm-large-s80-md"})

        self.assertEqual(Pipeline.calls, [(("BUT-FIT/diarizen-wavlm-large-s80-md",), {})])

    def test_import_diarizen_pipeline_retries_after_pyannote_audio_key_error(self):
        class Pipeline:
            pass

        inference = types.ModuleType("diarizen.pipelines.inference")
        inference.DiariZenPipeline = Pipeline
        calls = []
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "diarizen.pipelines.inference":
                calls.append(name)
                if len(calls) == 1:
                    sys.modules["pyannote.audio"] = types.ModuleType("pyannote.audio")
                    raise KeyError("pyannote.audio")
                return inference
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch.object(builtins, "__import__", side_effect=fake_import):
            result = _import_diarizen_pipeline()

        self.assertIs(result, Pipeline)
        self.assertEqual(calls, ["diarizen.pipelines.inference", "diarizen.pipelines.inference"])
        self.assertNotIn("pyannote.audio", sys.modules)

    def test_allow_torch_checkpoint_globals_registers_required_classes(self):
        class TorchVersion:
            pass

        class Specifications:
            pass

        class Serialization:
            calls = []

            @classmethod
            def add_safe_globals(cls, globals_to_add):
                cls.calls.append(globals_to_add)

        TorchModule = SimpleNamespace(
            torch_version=SimpleNamespace(TorchVersion=TorchVersion),
            serialization=Serialization,
        )

        _allow_torch_checkpoint_globals(TorchModule, extra_globals=[Specifications])

        self.assertIn(TorchVersion, Serialization.calls[0])
        self.assertIn(Specifications, Serialization.calls[0])

    def test_torchaudio_audio_metadata_compat_patches_top_level_attribute(self):
        class AudioMetaData:
            pass

        torchaudio = types.ModuleType("torchaudio")
        backend = types.ModuleType("torchaudio._backend")
        common = types.ModuleType("torchaudio._backend.common")
        common.AudioMetaData = AudioMetaData

        with mock.patch.dict(
            sys.modules,
            {
                "torchaudio": torchaudio,
                "torchaudio._backend": backend,
                "torchaudio._backend.common": common,
            },
        ):
            _patch_torchaudio_audio_metadata()

        self.assertIs(torchaudio.AudioMetaData, AudioMetaData)

    def test_torchaudio_audio_metadata_compat_creates_fallback_class(self):
        torchaudio = types.ModuleType("torchaudio")
        torchaudio.info = lambda path: None

        with mock.patch.dict(sys.modules, {"torchaudio": torchaudio}):
            _patch_torchaudio_audio_metadata()

        metadata = torchaudio.AudioMetaData(
            sample_rate=16000,
            num_frames=32000,
            num_channels=1,
            bits_per_sample=16,
            encoding="PCM_S",
        )
        self.assertEqual(metadata.sample_rate, 16000)
        self.assertEqual(metadata.num_frames, 32000)
        self.assertEqual(metadata.num_channels, 1)
        self.assertEqual(metadata.bits_per_sample, 16)
        self.assertEqual(metadata.encoding, "PCM_S")
        self.assertIn("ffmpeg", torchaudio.list_audio_backends())
        self.assertIsNone(torchaudio.get_audio_backend())
        self.assertIsNone(torchaudio.set_audio_backend("soundfile"))

    def test_torchaudio_backend_compat_patches_when_metadata_already_exists(self):
        class AudioMetaData:
            pass

        torchaudio = types.ModuleType("torchaudio")
        torchaudio.AudioMetaData = AudioMetaData
        torchaudio.load = lambda path: None

        with mock.patch.dict(sys.modules, {"torchaudio": torchaudio}):
            _patch_torchaudio_audio_metadata()

        self.assertIs(torchaudio.AudioMetaData, AudioMetaData)
        self.assertIn("ffmpeg", torchaudio.list_audio_backends())
        self.assertIsNone(torchaudio.get_audio_backend())

    def test_load_pyannote_pipeline_omits_auth_when_token_is_empty(self):
        class Pipeline:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return "pipeline"

        self.assertEqual(_load_pyannote_pipeline(Pipeline, "pyannote/model", None), "pipeline")
        self.assertEqual(Pipeline.calls, [(("pyannote/model",), {})])

    def test_load_pyannote_pipeline_retries_without_unsupported_auth_kwarg(self):
        class Pipeline:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                if "use_auth_token" in kwargs:
                    raise TypeError("Pretrained.__init__() got an unexpected keyword argument 'use_auth_token'")
                return "pipeline"

        self.assertEqual(_load_pyannote_pipeline(Pipeline, "pyannote/model", "hf_token"), "pipeline")
        self.assertEqual(
            Pipeline.calls,
            [
                (("pyannote/model",), {"use_auth_token": "hf_token"}),
                (("pyannote/model",), {}),
            ],
        )

    def test_hf_hub_download_compat_maps_use_auth_token_to_token(self):
        huggingface_hub = types.ModuleType("huggingface_hub")
        calls = []

        def hf_hub_download(*args, **kwargs):
            calls.append((args, kwargs))
            if "use_auth_token" in kwargs:
                raise TypeError("hf_hub_download() got an unexpected keyword argument 'use_auth_token'")
            return "model.bin"

        huggingface_hub.hf_hub_download = hf_hub_download

        with mock.patch.dict(sys.modules, {"huggingface_hub": huggingface_hub}):
            with _hf_hub_download_use_auth_token_compat():
                result = huggingface_hub.hf_hub_download("repo", "file", use_auth_token="hf_token")

        self.assertEqual(result, "model.bin")
        self.assertEqual(calls, [(("repo", "file"), {"token": "hf_token"})])
        self.assertIs(huggingface_hub.hf_hub_download, hf_hub_download)

    def test_hf_hub_download_compat_patches_imported_module_references(self):
        huggingface_hub = types.ModuleType("huggingface_hub")
        fetching = types.ModuleType("speechbrain.utils.fetching")
        calls = []

        def hf_hub_download(*args, **kwargs):
            calls.append((args, kwargs))
            if "use_auth_token" in kwargs:
                raise TypeError("hf_hub_download() got an unexpected keyword argument 'use_auth_token'")
            return "model.bin"

        huggingface_hub.hf_hub_download = hf_hub_download
        fetching.hf_hub_download = hf_hub_download

        with mock.patch.dict(
            sys.modules,
            {
                "huggingface_hub": huggingface_hub,
                "speechbrain.utils.fetching": fetching,
            },
        ):
            with _hf_hub_download_use_auth_token_compat():
                result = fetching.hf_hub_download("repo", "file", use_auth_token=True)

        self.assertEqual(result, "model.bin")
        self.assertEqual(calls, [(("repo", "file"), {"token": True})])
        self.assertIs(fetching.hf_hub_download, hf_hub_download)

    def test_speechbrain_compat_removes_unsupported_kwargs(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            self.skipTest("speechbrain is not installed")

        original = EncoderClassifier.from_hparams
        calls = []

        class FakeDevice:
            def __str__(self):
                return "cpu"

        def fake_from_hparams(*args, **kwargs):
            calls.append((args, kwargs))
            if "use_auth_token" in kwargs:
                raise TypeError("Pretrained.__init__() got an unexpected keyword argument 'use_auth_token'")
            if "revision" in kwargs:
                raise TypeError("Pretrained.__init__() got an unexpected keyword argument 'revision'")
            self.assertEqual(kwargs["run_opts"]["device"], "cpu")
            return "classifier"

        EncoderClassifier.from_hparams = fake_from_hparams
        try:
            with _speechbrain_use_auth_token_compat():
                result = EncoderClassifier.from_hparams(
                    source="speechbrain/model",
                    run_opts={"device": FakeDevice()},
                    use_auth_token="hf_token",
                    revision="main",
                )
        finally:
            EncoderClassifier.from_hparams = original

        self.assertEqual(result, "classifier")
        self.assertEqual(calls[0][1]["run_opts"], {"device": "cpu"})
        self.assertIn("use_auth_token", calls[0][1])
        self.assertIn("revision", calls[0][1])
        self.assertEqual(calls[1][1]["run_opts"], {"device": "cpu"})
        self.assertNotIn("use_auth_token", calls[1][1])
        self.assertIn("revision", calls[1][1])
        self.assertEqual(calls[2][1]["run_opts"], {"device": "cpu"})
        self.assertNotIn("use_auth_token", calls[2][1])
        self.assertNotIn("revision", calls[2][1])

    def test_speechbrain_compat_maps_mps_device_to_cpu(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            self.skipTest("speechbrain is not installed")

        original = EncoderClassifier.from_hparams
        calls = []

        def fake_from_hparams(*args, **kwargs):
            calls.append((args, kwargs))
            self.assertEqual(kwargs["run_opts"]["device"], "cpu")
            return "classifier"

        EncoderClassifier.from_hparams = fake_from_hparams
        try:
            with _speechbrain_use_auth_token_compat():
                result = EncoderClassifier.from_hparams(source="speechbrain/model", run_opts={"device": "mps"})
        finally:
            EncoderClassifier.from_hparams = original

        self.assertEqual(result, "classifier")
        self.assertEqual(calls[0][1]["run_opts"], {"device": "cpu"})

    def test_speechbrain_device_maps_mps_to_cpu(self):
        self.assertEqual(_speechbrain_device("mps"), "cpu")
        self.assertEqual(_speechbrain_device("mps:0"), "cpu")
        self.assertEqual(_speechbrain_device("cuda:0"), "cuda:0")

    def test_remap_chunk_segments_to_original_splits_across_concatenated_vad_boundaries(self):
        chunk_segments = [SpeakerSegment("local", "SPEAKER_00", 0.5, 2.5)]
        mapping = [
            {"chunk_start": 0.0, "chunk_end": 1.0, "source_start": 10.0, "source_end": 11.0},
            {"chunk_start": 1.0, "chunk_end": 3.0, "source_start": 20.0, "source_end": 22.0},
        ]
        remapped = remap_chunk_segments_to_original(chunk_segments, mapping, chunk_idx=1, min_duration=0.0)
        self.assertEqual([(segment.start, segment.end, segment.speaker) for segment in remapped], [(10.5, 11.0, "SPEAKER_00"), (20.0, 21.5, "SPEAKER_00")])

    def test_overlap_separation_reconstructs_segment_audio_with_separated_overlap(self):
        class FakeSeparator:
            def separate(self, audio_segment, sample_rate):
                return np.full_like(audio_segment, 0.8), np.full_like(audio_segment, 0.2)

        sample_rate = 10
        waveform = np.full(sample_rate * 4, 0.1, dtype=np.float32)
        waveform[:10] = 0.3
        waveform[20:30] = 0.3
        segments = [
            SpeakerSegment("a", "SPEAKER_00", 0.0, 3.0),
            SpeakerSegment("b", "SPEAKER_01", 1.0, 2.0),
        ]
        result = apply_overlap_separation(waveform, sample_rate, segments, FakeSeparator(), overlap_threshold=0.1)

        self.assertEqual(len(result["overlap_regions"]), 1)
        self.assertEqual(result["overlap_regions"][0]["start"], 1.0)
        self.assertEqual(result["overlap_regions"][0]["end"], 2.0)
        self.assertIn("a", result["segment_audio"])
        self.assertIn("b", result["segment_audio"])
        np.testing.assert_allclose(result["segment_audio"]["a"][:10], 0.3)
        np.testing.assert_allclose(result["segment_audio"]["a"][10:20], 0.3, atol=1e-6)
        np.testing.assert_allclose(result["segment_audio"]["a"][20:30], 0.3)
        np.testing.assert_allclose(result["segment_audio"]["b"], 0.07, atol=1e-6)

    def test_resolve_sepreformer_path_uses_project_local_checkout(self):
        resolved = _resolve_sepreformer_path("SepReformer")

        self.assertEqual(resolved, (Path.cwd() / "SepReformer").resolve())

    def test_find_checkpoint_dir_accepts_scratch_weight_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "models" / "SepReformer_Base_WSJ0" / "log" / "scratch_weight"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "model.pth").write_bytes(b"placeholder")

            self.assertEqual(_find_checkpoint_dir(root), checkpoint_dir)

    def test_find_checkpoint_dir_uses_configured_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "models" / "SepReformer_Large_DM_WSJ0" / "log" / "scratch_weights"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "model.pth").write_bytes(b"placeholder")

            self.assertEqual(_find_checkpoint_dir(root, "SepReformer_Large_DM_WSJ0"), checkpoint_dir)

    def test_sepreformer_import_ignores_existing_models_and_utils_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            (fake_root / "models").mkdir()
            (fake_root / "models" / "__init__.py").write_text("", encoding="utf-8")
            (fake_root / "utils").mkdir()
            (fake_root / "utils" / "__init__.py").write_text("", encoding="utf-8")

            saved_modules = {name: sys.modules.get(name) for name in ("models", "utils")}
            with mock.patch.object(sys, "path", [str(fake_root), *sys.path]):
                try:
                    for name in ("models", "utils"):
                        sys.modules.pop(name, None)
                    import models
                    import utils

                    self.assertEqual(Path(models.__file__).parent, fake_root / "models")
                    self.assertEqual(Path(utils.__file__).parent, fake_root / "utils")

                    Model = _import_sepreformer_model_class(Path("SepReformer").resolve(), "SepReformer_Base_WSJ0", {})

                    self.assertEqual(Model.__name__, "Model")
                finally:
                    for name in ("models", "utils"):
                        sys.modules.pop(name, None)
                        if saved_modules[name] is not None:
                            sys.modules[name] = saved_modules[name]

    def test_track_export_uses_overlap_separated_segment_audio(self):
        sample_rate = 10
        waveform = np.full(sample_rate * 3, 0.1, dtype=np.float32)
        segments = [
            SpeakerSegment("a", "SPEAKER_00", 0.0, 2.0),
            SpeakerSegment("b", "SPEAKER_01", 1.0, 3.0),
        ]
        overrides = {
            "a": np.full(sample_rate * 2, 0.7, dtype=np.float32),
            "b": np.full(sample_rate * 2, 0.3, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_segments_and_tracks(waveform, sample_rate, segments, out, write_segment_wavs=False, segment_audio_overrides=overrides)
            speaker_00, _ = sf.read(out / "tracks" / "SPEAKER_00.wav", dtype="float32")
            speaker_01, _ = sf.read(out / "tracks" / "SPEAKER_01.wav", dtype="float32")
            np.testing.assert_allclose(speaker_00[:20], 0.7, atol=1e-4)
            np.testing.assert_allclose(speaker_00[20:], 0.0, atol=1e-4)
            np.testing.assert_allclose(speaker_01[:10], 0.0, atol=1e-4)
            np.testing.assert_allclose(speaker_01[10:], 0.3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
