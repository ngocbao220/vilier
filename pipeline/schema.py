from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any


@dataclass
class SpeakerSegment:
    id: str
    speaker: str
    start: float
    end: float
    segment_wav: str = ""
    is_overlap: bool = False
    overlap_group_id: str | None = None

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("segment id is required")
        if not self.speaker:
            raise ValueError(f"segment {self.id} has empty speaker")
        if self.end <= self.start:
            raise ValueError(f"segment {self.id} has end <= start")

    def to_manifest(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["duration"] = self.duration
        return payload


@dataclass
class SpeakerTrack:
    id: str
    track_wav: str


def relative_path(path: Path, root: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return os.path.relpath(path, root)
