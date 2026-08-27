import argparse
import unittest
from types import SimpleNamespace

from pipeline.devices import resolve_auto_device
from pipeline.model_options import add_model_option_arguments, apply_model_overrides


class ModelOptionsTest(unittest.TestCase):
    def test_apply_model_overrides_updates_backend_model_device_and_enabled(self):
        parser = argparse.ArgumentParser()
        add_model_option_arguments(parser)
        args = parser.parse_args(
            [
                "--diarization-backend",
                "sortformer",
                "--diarization-model",
                "nvidia/diar_sortformer_4spk-v1",
                "--diarization-device",
                "auto",
                "--enable-asr",
                "--asr-model",
                "vinai/PhoWhisper-large",
                "--asr-device",
                "cuda",
                "--overlap-separation-model",
                "alibabasglab/MossFormer2_SS_16K",
                "--overlap-separation-backend",
                "clearvoice",
            ]
        )
        config = {
            "diarization": {"backend": "pyannote", "model": "pyannote/speaker-diarization-community-1", "device": "cpu"},
            "asr": {"enabled": False, "backend": "phowhisper_local", "model": "old", "device": "cpu"},
            "overlap_separation": {"enabled": True, "backend": "sepreformer", "model_name": "old", "device": "cpu"},
        }

        updated = apply_model_overrides(config, args)

        self.assertEqual(updated["diarization"]["backend"], "sortformer")
        self.assertEqual(updated["diarization"]["model"], "nvidia/diar_sortformer_4spk-v1")
        self.assertEqual(updated["diarization"]["device"], "auto")
        self.assertTrue(updated["asr"]["enabled"])
        self.assertEqual(updated["asr"]["model"], "vinai/PhoWhisper-large")
        self.assertEqual(updated["asr"]["device"], "cuda")
        self.assertEqual(updated["overlap_separation"]["backend"], "clearvoice")
        self.assertEqual(updated["overlap_separation"]["model_name"], "alibabasglab/MossFormer2_SS_16K")
        self.assertEqual(config["diarization"]["backend"], "pyannote")

    def test_disable_optional_phase(self):
        parser = argparse.ArgumentParser()
        add_model_option_arguments(parser)
        args = parser.parse_args(["--disable-music-separation"])

        updated = apply_model_overrides({"music_separation": {"enabled": True}}, args)

        self.assertFalse(updated["music_separation"]["enabled"])

    def test_auto_device_prefers_cuda_then_cpu(self):
        torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        self.assertEqual(resolve_auto_device(torch_module, "auto"), "cuda")

        torch_module.cuda.is_available = lambda: False
        self.assertEqual(resolve_auto_device(torch_module, "auto"), "cpu")


if __name__ == "__main__":
    unittest.main()
