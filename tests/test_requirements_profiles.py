import unittest
from pathlib import Path


class RequirementsProfilesTest(unittest.TestCase):
    def test_root_requirements_uses_default_profile_only(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8")

        self.assertIn("-r requirements/pyannote-community.txt", requirements)
        self.assertNotIn("pyannote.audio[separation]==3.3.2", requirements)
        self.assertNotIn("nemo_toolkit[asr]", requirements)
        self.assertNotIn("git+https://github.com/BUTSpeechFIT/DiariZen.git", requirements)

    def test_backend_profiles_are_split_to_avoid_pyannote_conflicts(self):
        community = Path("requirements/pyannote-community.txt").read_text(encoding="utf-8")
        pyannote31 = Path("requirements/pyannote-3.1.txt").read_text(encoding="utf-8")
        pixit = Path("requirements/pyannote-pixit.txt").read_text(encoding="utf-8")
        diarizen = Path("requirements/diarizen.txt").read_text(encoding="utf-8")
        sortformer = Path("requirements/sortformer.txt").read_text(encoding="utf-8")
        clearvoice_separation = Path("requirements/clearvoice-separation.txt").read_text(encoding="utf-8")
        speechbrain_separation = Path("requirements/speechbrain-separation.txt").read_text(encoding="utf-8")

        self.assertIn("pyannote.audio>=4.0", community)
        self.assertNotIn("pyannote.audio[separation]==3.3.2", community)
        self.assertIn("pyannote.audio==3.3.2", pyannote31)
        self.assertNotIn("pyannote.audio>=4.0", pyannote31)
        self.assertNotIn("pyannote.audio[separation]", pyannote31)
        self.assertIn("pyannote.audio[separation]==3.3.2", pixit)
        self.assertNotIn("pyannote.audio>=4.0", pixit)
        self.assertIn("#subdirectory=pyannote-audio", diarizen)
        self.assertIn("torch==2.1.1", diarizen)
        self.assertIn("torchaudio==2.1.1", diarizen)
        self.assertIn("torchvision==0.16.1", diarizen)
        self.assertIn("numpy==1.26.4", diarizen)
        self.assertNotIn("pyannote.audio>=4.0", diarizen)
        self.assertNotIn("pyannote.audio[separation]", diarizen)
        self.assertIn("numba<0.66", sortformer)
        self.assertIn("clearvoice", clearvoice_separation)
        self.assertNotIn("pyannote.audio", clearvoice_separation)
        self.assertIn("speechbrain", speechbrain_separation)
        self.assertNotIn("pyannote.audio", speechbrain_separation)

    def test_no_all_profile_mixes_incompatible_backends(self):
        self.assertFalse(Path("requirements/all.txt").exists())


if __name__ == "__main__":
    unittest.main()
