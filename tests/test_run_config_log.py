import unittest

from pipeline.run_config_log import enabled_mark, format_component_usage_table


class RunConfigLogTest(unittest.TestCase):
    def test_enabled_mark_uses_v_and_x(self):
        self.assertEqual(enabled_mark(True), "V")
        self.assertEqual(enabled_mark(False), "X")

    def test_format_component_usage_table_marks_enabled_components(self):
        output = format_component_usage_table(
            {
                "vad": {"enabled": True, "backend": "silero", "model": "silero_vad"},
                "diarization": {"backend": "pixit", "model": "pyannote"},
                "music_separation": {"enabled": False, "backend": "demucs", "model": "htdemucs"},
                "overlap_separation": {"enabled": True, "backend": "sepreformer", "model_name": "SepReformer_Base_WSJ0"},
                "asr": {"enabled": False, "asr_backend": "local", "model": "vinai/PhoWhisper-large"},
                "state_labeling": {"enabled": False, "backend": "qwen", "model": "qwen3.8-max"},
            },
            runtime_backend="local",
            asr_backend="kaggle",
        )

        self.assertIn("[INFO] Pipeline component usage", output)
        self.assertIn("| VAD                | V", output)
        self.assertIn("| Diarization        | V", output)
        self.assertIn("| Music separation   | X", output)
        self.assertIn("| Overlap separation | V", output)
        self.assertIn("| ASR                | X       | kaggle", output)
        self.assertIn("| State labeling     | X", output)


if __name__ == "__main__":
    unittest.main()
