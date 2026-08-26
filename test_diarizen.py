import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))

from pipeline.diarization import DiariZenDiarizer

config = {"backend": "diarizen"}
try:
    diarizer = DiariZenDiarizer(config)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
