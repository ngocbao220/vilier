import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from pipeline.diarization import build_diarization_chunks, remap_chunk_segments_to_original
from pipeline.overlap_separation import _find_checkpoint_dir, _resolve_sepreformer_path, apply_overlap_separation
from pipeline.schema import SpeakerSegment
from pipeline.timeline import annotate_overlaps, export_audacity_labels, export_segments_and_tracks, write_manifest
from pipeline.vad import cleanup_intervals, export_vad_audio, write_vad_txt
from pipeline.asr import DryRunAsrRunner, PhoWhisperLocalRunner, transcribe_vad_audio
from pipeline.audio import load_mono


class TimelineTest(unittest.TestCase):
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
                transcript=transcript,
            )
            data = json.loads(manifest.read_text())
            self.assertEqual(data["transcript"], transcript)

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
