import argparse
import json
import warnings
from pathlib import Path

from .devices import resolve_auto_device
from .model_options import add_model_option_arguments, apply_model_overrides


COMPONENTS = [
    ("vad", "VAD"),
    ("diarization", "Diarization"),
    ("music_separation", "Music separation"),
    ("overlap_separation", "Overlap separation"),
    ("asr", "ASR"),
    ("state_labeling", "State labeling"),
]


def enabled_mark(value) -> str:
    return "V" if bool(value) else "X"


def format_component_usage_table(config: dict) -> str:
    rows = []
    for key, label in COMPONENTS:
        section = config.get(key, {})
        enabled = bool(section.get("enabled", key in {"vad", "diarization"}))
        backend = _configured_backend(key, section)
        model = _configured_model(key, section)
        device = resolve_device_label(section)
        rows.append([label, enabled_mark(enabled), backend, model, device])

    headers = ["Component", "Enabled", "Backend", "Model", "Device"]
    widths = [
        max(len(str(row[idx])) for row in [headers] + rows)
        for idx in range(len(headers))
    ]
    lines = ["[INFO] Pipeline component usage"]
    lines.append(_format_row(headers, widths))
    lines.append(_format_row(["-" * width for width in widths], widths))
    lines.extend(_format_row(row, widths) for row in rows)
    return "\n".join(lines)


def _configured_backend(key: str, section: dict) -> str:
    if key == "asr":
        return str(section.get("backend") or "")
    if key in {"vad", "diarization", "music_separation", "overlap_separation", "state_labeling"}:
        return str(section.get("backend", ""))
    return ""


def _configured_model(key: str, section: dict) -> str:
    if key == "overlap_separation":
        return str(section.get("model_name") or section.get("model") or "")
    return str(section.get("model", ""))


def resolve_device_label(section: dict, torch_module=None) -> str:
    requested = str(section.get("device", "")).strip()
    if not requested:
        return ""
    normalized = requested.lower()
    if normalized not in {"auto", "gpu", "cuda"} and not normalized.startswith("cuda:"):
        return requested
    if torch_module is None:
        try:
            import torch as torch_module
        except Exception:
            return f"{requested} (torch unavailable)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resolved = resolve_auto_device(torch_module, normalized, warn_label="device")
    return f"{resolved} ({requested})" if resolved != requested else resolved


def _format_row(values: list[str], widths: list[int]) -> str:
    padded = [str(value).ljust(widths[idx]) for idx, value in enumerate(values)]
    return "| " + " | ".join(padded) + " |"


def main() -> int:
    parser = argparse.ArgumentParser(description="Print enabled/disabled pipeline component usage")
    parser.add_argument("--config", default="config.json")
    add_model_option_arguments(parser)
    args, _ = parser.parse_known_args()

    config = apply_model_overrides(json.loads(Path(args.config).read_text(encoding="utf-8")), args)
    print(format_component_usage_table(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
