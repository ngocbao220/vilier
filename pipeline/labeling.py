import json
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from .schema import relative_path


class LabelingRunner(Protocol):
    labels: list[str]
    model_name: str

    def label(self, transcript: dict) -> dict:
        ...


class QwenChatClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout_seconds: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def classify(self, messages: list[dict]) -> str:
        payload = json.dumps({"model": self.model, "messages": messages}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen labeling request failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen labeling request failed: {exc}") from exc
        return str(data["choices"][0]["message"]["content"])


class QwenLabelingRunner:
    def __init__(self, config: dict, client: QwenChatClient | None = None):
        self.labels = _labels_from_config(config)
        self.model_name = str(config.get("model", "qwen-plus"))
        self.base_url = str(config.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"))
        self.api_key_env = str(config.get("api_key_env", "DASHSCOPE_API_KEY"))
        self.timeout_seconds = float(config.get("timeout_seconds", 60.0))
        self.default_empty_label = _default_empty_label(self.labels)
        self.client = client

    def label(self, transcript: dict) -> dict:
        text = str(transcript.get("text", "")).strip()
        if not text:
            return self._result(
                label=self.default_empty_label,
                confidence=1.0,
                reason="empty transcript",
            )

        content = self._client().classify(self._messages(transcript))
        payload = _parse_json_object(content)
        label = str(payload.get("label", "")).strip()
        if label not in self.labels:
            raise ValueError(f"Qwen label {label!r} is not in configured labels: {', '.join(self.labels)}")
        return self._result(
            label=label,
            confidence=_coerce_confidence(payload.get("confidence", 0.0)),
            reason=str(payload.get("reason", "")).strip(),
        )

    def _client(self) -> QwenChatClient:
        if self.client is not None:
            return self.client
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing Qwen API key environment variable: {self.api_key_env}")
        self.client = QwenChatClient(api_key, self.base_url, self.model_name, self.timeout_seconds)
        return self.client

    def _messages(self, transcript: dict) -> list[dict]:
        labels = ", ".join(self.labels)
        return [
            {
                "role": "system",
                "content": (
                    "You label Vietnamese ASR transcript snippets. "
                    "Return only one JSON object. "
                    f"Allowed labels: {labels}. "
                    'Schema: {"label":"<allowed label>","confidence":0.0,"reason":"short reason"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"speaker={transcript.get('speaker', '')}\n"
                    f"start={transcript.get('start', '')}\n"
                    f"end={transcript.get('end', '')}\n"
                    f"transcript={transcript.get('text', '')}"
                ),
            },
        ]

    def _result(self, label: str, confidence: float, reason: str) -> dict:
        return {
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "model": self.model_name,
        }


class DryRunLabelingRunner:
    def __init__(self, labels: list[str] | None = None, model_name: str = "qwen-plus"):
        self.labels = labels or ["complete", "incomplete"]
        self.model_name = model_name

    def label(self, transcript: dict) -> dict:
        idx = _index_from_transcript_id(str(transcript.get("id", "")))
        label = self.labels[idx % len(self.labels)]
        return {
            "label": label,
            "confidence": 1.0,
            "reason": "dry-run deterministic label",
            "model": self.model_name,
        }


def load_labeling_runner(config: dict, dry_run: bool = False) -> LabelingRunner | None:
    labeling_config = config.get("state_labeling", {})
    if not bool(labeling_config.get("enabled", False)):
        return None
    labels = _labels_from_config(labeling_config)
    if dry_run:
        return DryRunLabelingRunner(labels=labels, model_name=str(labeling_config.get("model", "qwen-plus")))
    backend = str(labeling_config.get("backend", "qwen"))
    if backend != "qwen":
        raise ValueError(f"Unsupported state labeling backend: {backend}")
    return QwenLabelingRunner(labeling_config)


def resolve_state_dir(config: dict, state_dir_arg: str = "") -> Path:
    raw_override = state_dir_arg or os.environ.get("STATE_DIR")
    if raw_override:
        override = Path(raw_override).expanduser()
        return override.resolve() if override.is_absolute() else override
    raw_state_dir = config.get("state_labeling", {}).get("state_dir", "state")
    return Path(raw_state_dir).expanduser()


def label_transcripts(transcripts: list[dict], runner: LabelingRunner) -> list[dict]:
    labeled = []
    for transcript in transcripts:
        state = runner.label(transcript)
        record = dict(transcript)
        record["state_label"] = state["label"]
        record["state_confidence"] = state["confidence"]
        record["state_reason"] = state["reason"]
        record["state_model"] = state["model"]
        labeled.append(record)
    return labeled


def write_state_outputs(audio_id: str, output_dir: Path, state_dir: Path, records: list[dict]) -> dict:
    state_dir.mkdir(parents=True, exist_ok=True)
    existing_index = _read_index(state_dir / "index.json")
    counters = _next_counters(existing_index)
    written = []

    for record in records:
        label = str(record["state_label"])
        counters[label] = counters.get(label, 0) + 1
        stem = f"{label}_{counters[label]:02d}"
        label_dir = state_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        source_audio = output_dir / str(record["audio"])
        state_audio = label_dir / f"{stem}.wav"
        shutil.copy2(source_audio, state_audio)
        metadata = _metadata(audio_id, output_dir, state_dir, state_audio, record)
        metadata_path = label_dir / f"{stem}.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(metadata)

    index = existing_index + written
    (state_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {}
    for record in written:
        counts[record["label"]] = counts.get(record["label"], 0) + 1
    return {
        "enabled": True,
        "state_dir": relative_path(state_dir, Path.cwd()),
        "index": relative_path(state_dir / "index.json", Path.cwd()),
        "labeled": len(written),
        "counts": counts,
        "records": written,
    }


def _metadata(audio_id: str, output_dir: Path, state_dir: Path, state_audio: Path, record: dict) -> dict:
    source_audio = output_dir / str(record["audio"])
    return {
        "audio_id": audio_id,
        "asr_id": str(record.get("id", "")),
        "vad_id": str(record.get("vad_id", "")),
        "source_audio": relative_path(source_audio, Path.cwd()),
        "state_audio": relative_path(state_audio, state_dir.parent),
        "transcript": str(record.get("text", "")),
        "speaker": str(record.get("speaker", "")),
        "start": float(record.get("start", 0.0)),
        "end": float(record.get("end", 0.0)),
        "label": str(record["state_label"]),
        "confidence": float(record["state_confidence"]),
        "reason": str(record.get("state_reason", "")),
        "model": str(record.get("state_model", "")),
    }


def _read_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _next_counters(index: list[dict]) -> dict[str, int]:
    counters = {}
    pattern = re.compile(r"(.+)_([0-9]+)\.wav$")
    for item in index:
        name = Path(str(item.get("state_audio", ""))).name
        match = pattern.match(name)
        if match:
            label = match.group(1)
            counters[label] = max(counters.get(label, 0), int(match.group(2)))
    return counters


def _labels_from_config(config: dict) -> list[str]:
    labels = [str(label).strip() for label in config.get("labels", ["complete", "incomplete"]) if str(label).strip()]
    if not labels:
        raise ValueError("state_labeling.labels must contain at least one label")
    return labels


def _default_empty_label(labels: list[str]) -> str:
    return "incomplete" if "incomplete" in labels else labels[-1]


def _parse_json_object(content: str) -> dict:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Qwen response is not JSON: {content}")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Qwen response JSON must be an object")
    return payload


def _coerce_confidence(value) -> float:
    confidence = float(value)
    return max(0.0, min(1.0, confidence))


def _index_from_transcript_id(transcript_id: str) -> int:
    match = re.search(r"(\d+)$", transcript_id)
    if not match:
        return 0
    return int(match.group(1))
