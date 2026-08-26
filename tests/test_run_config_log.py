import unittest
from types import SimpleNamespace

from pipeline.run_config_log import enabled_mark, format_component_usage_table, resolve_device_label


class RunConfigLogTest(unittest.TestCase):
    def test_enabled_mark_uses_v_and_x(self):
        self.assertEqual(enabled_mark(True), "V")
        self.assertEqual(enabled_mark(False), "X")

    def test_format_component_usage_table_marks_enabled_components(self):
        output = format_component_usage_table(
            {
                "vad": {"enabled": True, "backend": "silero", "model": "silero_vad"},
                "diarization": {"backend": "pixit", "model": "pyannote", "device": "auto"},
                "music_separation": {"enabled": False, "backend": "demucs", "model": "htdemucs"},
                "overlap_separation": {"enabled": True, "backend": "sepreformer", "model_name": "SepReformer_Base_WSJ0", "device": "cuda"},
                "asr": {"enabled": False, "backend": "phowhisper_local", "model": "vinai/PhoWhisper-large", "device": "cpu"},
                "state_labeling": {"enabled": False, "backend": "qwen", "model": "qwen3.8-max"},
            }
        )

        self.assertIn("[INFO] Pipeline component usage", output)
        self.assertIn("| Component          | Enabled | Backend          | Model", output)
        self.assertIn("Device", output)
        self.assertIn("| VAD                | V", output)
        self.assertIn("| Diarization        | V", output)
        self.assertIn("| Diarization        | V       | pixit", output)
        self.assertIn("(auto)", output)
        self.assertIn("| Music separation   | X", output)
        self.assertIn("| Overlap separation | V", output)
        self.assertIn("| ASR                | X       | phowhisper_local", output)
        self.assertIn("| State labeling     | X", output)

    def test_resolve_device_label_shows_resolved_auto_device(self):
        torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        self.assertEqual(resolve_device_label({"device": "auto"}, torch_module=torch_module), "cuda (auto)")

        torch_module.cuda.is_available = lambda: False
        self.assertEqual(resolve_device_label({"device": "auto"}, torch_module=torch_module), "cpu (auto)")


if __name__ == "__main__":
    unittest.main()
