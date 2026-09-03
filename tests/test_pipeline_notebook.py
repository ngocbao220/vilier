import json
import unittest
from pathlib import Path


class PipelineNotebookTest(unittest.TestCase):
    def test_pipeline_notebook_has_component_cells_with_inputs(self):
        notebook = json.loads(Path("notebooks/pipeline.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertGreaterEqual(len(notebook["cells"]), 10)
        for name in (
            "preprocess_input",
            "vad_input",
            "diarization_chunks_input",
            "diarization_input",
            "music_input",
            "separation_input",
            "tracks_input",
            "asr_input",
            "state_labeling_input",
            "manifest_input",
        ):
            self.assertIn(f"{name} = input(", source)

        self.assertIn("SileroVadRunner", source)
        self.assertIn("load_diarizer", source)
        self.assertIn("load_overlap_separator", source)
        self.assertIn("transcribe_asr_segments", source)
        self.assertIn("write_manifest", source)

    def test_pipeline_notebook_has_no_saved_outputs(self):
        notebook = json.loads(Path("notebooks/pipeline.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
