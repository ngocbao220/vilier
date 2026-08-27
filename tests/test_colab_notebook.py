import json
import unittest
from pathlib import Path


class ColabNotebookTest(unittest.TestCase):
    def test_colab_notebook_uses_drive_input_and_run_sh(self):
        notebook = json.loads(Path("notebooks/colab.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("from google.colab import drive", source)
        self.assertIn("drive.mount('/content/drive')", source)
        self.assertIn("!rm -rf /content/vilier", source)
        self.assertIn('DRIVE_AUDIO_PATH = "/content/drive/MyDrive/VDT-TurnTaking/inputs/real.wav"', source)
        self.assertIn('DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/VDT-TurnTaking/outputs"', source)
        self.assertIn('os.environ["INPUT_PATH"] = str(audio_path)', source)
        self.assertIn('os.environ["OUTPUT_PATH"] = str(output_dir)', source)
        self.assertIn('"pyannote/speaker-diarization-3.1"', source)
        self.assertIn('PYANNOTE_PROFILE = "pyannote-3.1"', source)
        self.assertIn("requirements/clearvoice-separation.txt", source)
        self.assertIn('OVERLAP_SEPARATION_BACKEND = "clearvoice"', source)
        self.assertIn('OVERLAP_SEPARATION_MODEL = "alibabasglab/MossFormer2_SS_16K"', source)
        self.assertIn('config["overlap_separation"]["backend"] = OVERLAP_SEPARATION_BACKEND', source)
        self.assertIn(
            "!bash run.sh --enable-overlap-separation --overlap-separation-backend clearvoice --overlap-separation-model alibabasglab/MossFormer2_SS_16K",
            source,
        )
        self.assertNotIn("/kaggle/input", source)

    def test_colab_notebook_has_no_saved_outputs(self):
        notebook = json.loads(Path("notebooks/colab.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
