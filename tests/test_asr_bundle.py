import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


class AsrBundleTest(unittest.TestCase):
    def test_run_asr_bundle_dry_run_writes_transcript(self):
        sample_rate = 16000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "outputs" / "sample"
            asr_dir = bundle / "asr_audio" / "SPEAKER_00"
            asr_dir.mkdir(parents=True)
            sf.write(asr_dir / "audio_00001.wav", np.zeros(sample_rate, dtype=np.float32), sample_rate)
            (bundle / "manifest.timeline.json").write_text(
                json.dumps(
                    {
                        "audio_id": "sample",
                        "vad_segments": [{"id": "vad_00000", "start": 0.0, "end": 1.0, "duration": 1.0}],
                        "asr_segments": [
                            {
                                "id": "asrseg_00000",
                                "speaker": "SPEAKER_00",
                                "start": 0.0,
                                "end": 1.0,
                                "duration": 1.0,
                                "audio": "asr_audio/SPEAKER_00/audio_00001.wav",
                            }
                        ],
                        "segments": [{"id": "seg_00000", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}],
                    }
                ),
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "asr": {
                            "enabled": True,
                            "backend": "phowhisper_local",
                            "model": "vinai/PhoWhisper-large",
                            "language": "vi",
                        }
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "tools/run_asr_bundle.py",
                    "--config",
                    str(config),
                    "--bundle-dir",
                    str(bundle),
                    "--dry-run",
                ],
                check=True,
            )

            transcript = json.loads((bundle / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(transcript[0]["text"], "dry-run transcript 1")
            self.assertEqual(transcript[0]["speaker"], "SPEAKER_00")
            self.assertEqual(transcript[0]["asr_segment_id"], "asrseg_00000")
            self.assertEqual(transcript[0]["audio"], "asr_audio/SPEAKER_00/audio_00001.wav")


if __name__ == "__main__":
    unittest.main()
