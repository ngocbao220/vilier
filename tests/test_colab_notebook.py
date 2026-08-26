import json
import unittest
from pathlib import Path


class ColabNotebookTest(unittest.TestCase):
    def test_colab_notebook_uses_drive_input_and_run_sh(self):
        notebook = json.loads(Path("notebooks/colab.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("from google.colab import drive", source)
        self.assertIn("drive.mount('/content/drive')", source)
        self.assertIn('DRIVE_AUDIO_PATH = "/content/drive/MyDrive/vilier/input/real.wav"', source)
        self.assertIn('DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/vilier/outputs"', source)
        self.assertIn('os.environ["INPUT_PATH"] = str(audio_path)', source)
        self.assertIn('os.environ["OUTPUT_PATH"] = str(output_dir)', source)
        self.assertIn("!bash run.sh", source)
        self.assertNotIn("/kaggle/input", source)

    def test_colab_notebook_has_no_saved_outputs(self):
        notebook = json.loads(Path("notebooks/colab.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
