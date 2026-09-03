import json
import unittest
from pathlib import Path


class ConfigCommentsTest(unittest.TestCase):
    def test_pyannote_pixit_is_documented_under_separation_examples(self):
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))

        overlap_examples = config["overlap_separation"]["_model_replacement_examples"]
        pixit_examples = [
            example
            for example in overlap_examples
            if example.get("backend") == "pyannote_pixit"
        ]

        self.assertEqual(len(pixit_examples), 1)
        self.assertEqual(pixit_examples[0]["model_name"], "pyannote/speech-separation-ami-1.0")
        self.assertEqual(pixit_examples[0]["requirements"], "requirements/pyannote-pixit.txt")
        self.assertIn("diarization.backend=pixit", pixit_examples[0]["runtime_note"])

    def test_diarization_examples_do_not_advertise_pixit_as_primary_backend(self):
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))

        diarization_examples = config["diarization"]["_model_replacement_examples"]
        diarization_backends = {example.get("backend") for example in diarization_examples}

        self.assertNotIn("pixit", diarization_backends)
        self.assertNotIn("pyannote_pixit", diarization_backends)


if __name__ == "__main__":
    unittest.main()
