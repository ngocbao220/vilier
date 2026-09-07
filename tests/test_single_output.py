import unittest
from io import StringIO
from pathlib import Path

from pipeline.cli import ProgressBar
from single import format_benchmark_report, format_pipeline_modules, native_metric_summary


class SingleOutputTest(unittest.TestCase):
    def test_pipeline_modules_name_each_visible_phase(self):
        output = format_pipeline_modules(
            {
                "vad": {"model": "silero_vad"},
                "diarization": {"backend": "pyannote", "model": "community-1"},
                "overlap_separation": {"backend": "clearvoice", "model_name": "MossFormer2"},
            }
        )

        self.assertIn("Pipeline: Vilier", output)
        self.assertIn("Speaker Diarization", output)
        self.assertIn("Overlap Separation", output)

    def test_benchmark_report_marks_reference_metrics_unavailable_without_ground_truth(self):
        output = format_benchmark_report(
            {
                "reconstruction_snr_db": 12.5,
                "residual_rms": 0.01,
                "active_track_ratio": 0.4,
                "multi_active_ratio": 0.1,
                "clipped_sample_ratio": 0.0,
                "mean_track_leakage_ratio": 0.2,
            },
            [Path("speakerA.wav"), Path("speakerB.wav")],
        )

        self.assertIn("Ground Truth: null", output)
        self.assertIn("========= 3. Running benchmark =========", output)
        self.assertIn("không tính PIT-SI-SDR", output)
        self.assertIn("reconstruction_snr_db", output)
        self.assertIn("12.5", output)

    def test_progress_uses_requested_phase_headings_and_hides_internal_steps(self):
        stream = StringIO()
        progress = ProgressBar(total=2, stream=stream, visible_steps={"vad", "overlap_separation"})
        progress.start("sample", "vad")
        progress.complete("sample", "vad")
        progress.start("sample", "diarization")
        progress.start("sample", "overlap_separation")
        progress.complete("sample", "overlap_separation")

        output = stream.getvalue()
        self.assertIn("========= 2. Running Pipeline =========", output)
        self.assertIn("=== 2.1 VAD ===", output)
        self.assertIn("=== 2.3 Overlap Separation ===", output)
        self.assertNotIn("Diarization", output)
        self.assertNotIn("RUN sample", output)
        self.assertNotIn("DONE sample", output)

    def test_native_metric_summary_excludes_temporary_artifact_paths(self):
        summary = native_metric_summary(
            {"run_dir": "/tmp/expired", "reference_audio": "/tmp/expired/audio.wav", "track_count": 2, "residual_rms": 0.1}
        )

        self.assertNotIn("run_dir", summary)
        self.assertNotIn("reference_audio", summary)
        self.assertEqual(summary["track_count"], 2)
        self.assertEqual(summary["residual_rms"], 0.1)

    def test_progress_details_shows_only_the_first_three_intervals(self):
        stream = StringIO()
        progress = ProgressBar(total=1, stream=stream)
        progress.details("chunks", [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)])

        output = stream.getvalue()
        self.assertIn("Done, found 4 chunks", output)
        self.assertIn("-> Chunk 3: [2.000, 3.000]", output)
        self.assertNotIn("-> Chunk 4", output)
        self.assertTrue(output.rstrip().endswith("..."))


if __name__ == "__main__":
    unittest.main()
