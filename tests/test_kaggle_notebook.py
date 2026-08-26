import json
import unittest
from pathlib import Path


class KaggleNotebookTest(unittest.TestCase):
    def test_kaggle_notebook_uses_split_requirements_profile(self):
        notebook = json.loads(Path("notebooks/kaggle.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("requirements/pyannote-community.txt", source)
        self.assertNotIn("pip install -r /kaggle/working/vilier/requirements.txt", source)
        self.assertIn("DRY_RUN=1", source)
        self.assertIn("HUGGINGFACE_TOKEN=YOUR_HUGGINGFACE_TOKEN", source)

    def test_kaggle_notebook_has_no_saved_outputs(self):
        notebook = json.loads(Path("notebooks/kaggle.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
