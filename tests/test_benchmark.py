import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.benchmark import benchmark_run_dir, run_benchmark


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_run_dir_scores_speaker_tracks(self):
        sample_rate = 8000
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "outputs" / "sample"
            tracks_dir = run_dir / "tracks"
            tracks_dir.mkdir(parents=True)
            reference = np.zeros(sample_rate * 2, dtype=np.float32)
            reference[:sample_rate] = 0.1
            reference[sample_rate:] = 0.2
            speaker_00 = np.zeros_like(reference)
            speaker_00[:sample_rate] = 0.1
            speaker_01 = np.zeros_like(reference)
            speaker_01[sample_rate:] = 0.2
            sf.write(run_dir / "audio.standardized.wav", reference, sample_rate)
            sf.write(tracks_dir / "SPEAKER_00.wav", speaker_00, sample_rate)
            sf.write(tracks_dir / "SPEAKER_01.wav", speaker_01, sample_rate)
            (run_dir / "manifest.timeline.json").write_text(
                json.dumps(
                    {
                        "audio_id": "sample",
                        "sample_rate": sample_rate,
                        "standardized_audio": "audio.standardized.wav",
                        "speakers": [
                            {"id": "SPEAKER_00", "track_wav": "tracks/SPEAKER_00.wav"},
                            {"id": "SPEAKER_01", "track_wav": "tracks/SPEAKER_01.wav"},
                        ],
                        "segments": [
                            {"id": "s0", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
                            {"id": "s1", "speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
                        ],
                        "overlap_separation": {"backend": "dry", "overlap_regions": [], "enhanced_segment_count": 0},
                    }
                ),
                encoding="utf-8",
            )

            row = benchmark_run_dir(run_dir)

        self.assertEqual(row["metric_status"], "computed")
        self.assertEqual(row["track_count"], 2)
        self.assertEqual(row["duration_sec"], 2.0)
        self.assertEqual(row["multi_active_ratio"], 0.0)
        self.assertGreater(row["reconstruction_snr_db"], 80.0)
        self.assertEqual(len(row["tracks"]), 2)
        self.assertEqual(row["tracks"][0]["labeled_duration_sec"], 1.0)
        self.assertEqual(row["tracks"][0]["leakage_energy_ratio"], 0.0)

    def test_run_benchmark_writes_summary_and_jsonl(self):
        sample_rate = 8000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "sample"
            (run_dir / "tracks").mkdir(parents=True)
            audio = np.zeros(sample_rate, dtype=np.float32)
            sf.write(run_dir / "audio.standardized.wav", audio, sample_rate)
            sf.write(run_dir / "tracks" / "SPEAKER_00.wav", audio, sample_rate)
            (run_dir / "manifest.timeline.json").write_text(
                json.dumps(
                    {
                        "audio_id": "sample",
                        "sample_rate": sample_rate,
                        "standardized_audio": "audio.standardized.wav",
                        "speakers": [{"id": "SPEAKER_00", "track_wav": "tracks/SPEAKER_00.wav"}],
                    }
                ),
                encoding="utf-8",
            )

            summary_path = run_benchmark(root / "outputs", root / "benchmark")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics_lines = (root / "benchmark" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["metric_status"], "computed")
        self.assertEqual(len(metrics_lines), 1)


if __name__ == "__main__":
    unittest.main()
