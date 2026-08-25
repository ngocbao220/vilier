import unittest
import tempfile
from pathlib import Path

from pipeline.tree_log import format_tree, kv, section, write_tree_log
from pipeline.cli import batch_log_path, format_progress_bar, log_path_for_audio, resolve_log_dir


class TreeLogTest(unittest.TestCase):
    def test_format_tree_renders_nested_sections_and_key_values(self):
        output = format_tree(
            "audio=vi_one",
            [
                section("preprocess", "PASS", [kv("sample_rate", 16000), kv("duration_sec", 1097.5)]),
                section("asr", "SKIP", [kv("enabled", False)]),
            ],
        )

        self.assertEqual(
            output,
            "\n".join(
                [
                    "[INFO] audio=vi_one",
                    "├── preprocess PASS",
                    "│   ├── sample_rate=16000",
                    "│   └── duration_sec=1097.50",
                    "└── asr SKIP",
                    "    └── enabled=0",
                ]
            ),
        )

    def test_format_tree_escapes_newlines_in_values(self):
        output = format_tree("audio=vi_one", [section("asr", "FAIL", [kv("error", "line1\nline2")])], level="ERROR")

        self.assertIn("[ERROR] audio=vi_one", output)
        self.assertIn("error=line1\\nline2", output)

    def test_format_tree_renders_paths_relative_to_cwd(self):
        output = format_tree("batch", [kv("input_path", Path.cwd() / "inputs" / "vi_one.wav")])

        self.assertIn("input_path=inputs/vi_one.wav", output)

    def test_write_tree_log_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "logs" / "2026-08-24" / "vi_one.wav" / "pipeline.log"
            write_tree_log(log_path, "audio=vi_one", [section("done", "PASS")])

            self.assertIn("[INFO] audio=vi_one", log_path.read_text(encoding="utf-8"))

    def test_log_dir_is_not_nested_under_output_root(self):
        log_dir = resolve_log_dir({"logging": {"log_dir": "logs"}}, "2026-08-24")

        self.assertEqual(log_dir, (Path.cwd() / "logs" / "2026-08-24").resolve())
        self.assertEqual(batch_log_path(log_dir), (Path.cwd() / "logs" / "2026-08-24" / "batch.log").resolve())
        self.assertEqual(
            log_path_for_audio(log_dir, Path("inputs/vi_one.wav")),
            (Path.cwd() / "logs" / "2026-08-24" / "vi_one.wav" / "pipeline.log").resolve(),
        )

    def test_log_dir_arg_can_be_exact_daily_dir(self):
        log_dir = resolve_log_dir({}, "2026-08-24", "logs/2026-08-24")

        self.assertEqual(log_dir, (Path.cwd() / "logs" / "2026-08-24").resolve())

    def test_format_progress_bar_shows_step_status_and_ratio(self):
        output = format_progress_bar(3, 9, "vi_one/asr", "RUN", width=9)

        self.assertEqual(output, "[###------] 3/9 RUN vi_one/asr")


if __name__ == "__main__":
    unittest.main()
