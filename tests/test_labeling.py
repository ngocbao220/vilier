import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.labeling import (
    DryRunLabelingRunner,
    QwenLabelingRunner,
    label_transcripts,
    write_state_outputs,
)


class FakeQwenClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def classify(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.content


class LabelingTest(unittest.TestCase):
    def test_qwen_runner_accepts_valid_configured_label(self):
        runner = QwenLabelingRunner(
            {
                "model": "qwen-plus",
                "labels": ["complete", "incomplete"],
                "api_key_env": "DASHSCOPE_API_KEY",
            },
            client=FakeQwenClient('{"label":"complete","confidence":0.91,"reason":"finished thought"}'),
        )

        result = runner.label({"text": "toi dong y", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0})

        self.assertEqual(result["label"], "complete")
        self.assertEqual(result["confidence"], 0.91)
        self.assertEqual(result["model"], "qwen-plus")

    def test_qwen_runner_rejects_unknown_label(self):
        runner = QwenLabelingRunner(
            {
                "model": "qwen-plus",
                "labels": ["complete", "incomplete"],
                "api_key_env": "DASHSCOPE_API_KEY",
            },
            client=FakeQwenClient('{"label":"other","confidence":0.5,"reason":"bad"}'),
        )

        with self.assertRaisesRegex(ValueError, "not in configured labels"):
            runner.label({"text": "noi tiep", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0})

    def test_empty_transcript_defaults_to_incomplete_without_qwen_call(self):
        client = FakeQwenClient('{"label":"complete","confidence":1.0,"reason":"unused"}')
        runner = QwenLabelingRunner(
            {
                "model": "qwen-plus",
                "labels": ["complete", "incomplete"],
                "api_key_env": "DASHSCOPE_API_KEY",
            },
            client=client,
        )

        result = runner.label({"text": "  ", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0})

        self.assertEqual(result["label"], "incomplete")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(client.calls, [])

    def test_dry_run_runner_is_deterministic(self):
        runner = DryRunLabelingRunner(labels=["complete", "incomplete"], model_name="qwen-plus")

        self.assertEqual(runner.label({"id": "asr_00000", "text": "a"})["label"], "complete")
        self.assertEqual(runner.label({"id": "asr_00001", "text": "b"})["label"], "incomplete")

    def test_label_transcripts_and_write_state_outputs(self):
        sample_rate = 16000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs" / "vi_one"
            vad_dir = output_dir / "vad_audio"
            vad_dir.mkdir(parents=True)
            sf.write(vad_dir / "audio_1.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)
            sf.write(vad_dir / "audio_2.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)
            transcripts = [
                {
                    "id": "asr_00000",
                    "vad_id": "vad_00000",
                    "audio": "vad_audio/audio_1.wav",
                    "text": "xin chao",
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 1.0,
                },
                {
                    "id": "asr_00001",
                    "vad_id": "vad_00001",
                    "audio": "vad_audio/audio_2.wav",
                    "text": "dang noi do",
                    "speaker": "SPEAKER_01",
                    "start": 1.0,
                    "end": 2.0,
                },
            ]
            labels = label_transcripts(transcripts, DryRunLabelingRunner(["complete", "incomplete"], "qwen-plus"))
            summary = write_state_outputs("vi_one", output_dir, root / "state", labels)

            self.assertEqual(summary["counts"], {"complete": 1, "incomplete": 1})
            self.assertTrue((root / "state" / "complete" / "complete_01.wav").exists())
            self.assertTrue((root / "state" / "incomplete" / "incomplete_01.wav").exists())
            self.assertFalse((output_dir / "state").exists())
            index = json.loads((root / "state" / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index), 2)
            self.assertEqual(index[0]["audio_id"], "vi_one")
            self.assertEqual(index[0]["state_audio"], "state/complete/complete_01.wav")
            sidecar = json.loads((root / "state" / "complete" / "complete_01.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["transcript"], "xin chao")


if __name__ == "__main__":
    unittest.main()
