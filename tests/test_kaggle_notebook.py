import json
import unittest
from pathlib import Path


class KaggleNotebookTest(unittest.TestCase):
    def test_kaggle_notebook_uses_simple_five_step_cli_flow(self):
        notebook = json.loads(Path("notebooks/kaggle.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertEqual(len(notebook["cells"]), 5)
        self.assertIn("!git clone https://github.com/ngocbao220/vilier.git", source)
        self.assertIn("requirements/pyannote-community.txt", source)
        self.assertNotIn("pip install -r /kaggle/working/vilier/requirements.txt", source)
        self.assertIn("%cd /kaggle/working/vilier", source)
        self.assertNotIn("/kaggle/working/vilier/requirements/", source)
        self.assertNotIn("/kaggle/working/vilier/config.json", source)
        self.assertNotIn("/kaggle/working/vilier/run.sh", source)
        self.assertIn("HUGGINGFACE_TOKEN = input(", source)
        self.assertIn("INPUT_PATH = input(", source)
        self.assertIn("OUTPUT_PATH = input(", source)
        self.assertIn("!HUGGINGFACE_TOKEN=\"$HUGGINGFACE_TOKEN\" INPUT_PATH=\"$INPUT_PATH\" OUTPUT_PATH=\"$OUTPUT_PATH\" bash run.sh", source)
        self.assertNotIn("find ", source)
        self.assertNotIn("zip -r", source)

    def test_kaggle_notebook_has_no_saved_outputs(self):
        notebook = json.loads(Path("notebooks/kaggle.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
