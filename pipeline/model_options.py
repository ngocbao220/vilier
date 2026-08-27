import argparse
import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseOption:
    key: str
    cli: str
    backends: tuple[str, ...]
    model_key: str = "model"
    has_device: bool = True
    optional: bool = False


PHASE_OPTIONS = (
    PhaseOption("vad", "vad", ("silero",), has_device=False),
    PhaseOption("diarization", "diarization", ("pyannote", "pixit", "pyannote_pixit", "sortformer", "diarizen")),
    PhaseOption("music_separation", "music-separation", ("demucs",), optional=True),
    PhaseOption(
        "overlap_separation",
        "overlap-separation",
        ("sepreformer", "speechbrain", "clearvoice", "mossformer2"),
        model_key="model_name",
        optional=True,
    ),
    PhaseOption("asr", "asr", ("phowhisper_local",), optional=True),
    PhaseOption("state_labeling", "state-labeling", ("qwen",), has_device=False, optional=True),
)


def add_model_option_arguments(parser: argparse.ArgumentParser) -> None:
    for phase in PHASE_OPTIONS:
        prefix = phase.cli
        parser.add_argument(f"--{prefix}-backend", choices=phase.backends, default=None)
        parser.add_argument(f"--{prefix}-model", default=None)
        if phase.has_device:
            parser.add_argument(f"--{prefix}-device", default=None)
        if phase.optional:
            enable_attr = f"{phase.key}_enabled"
            enable = parser.add_mutually_exclusive_group()
            enable.add_argument(f"--enable-{prefix}", dest=enable_attr, action="store_true", default=None)
            enable.add_argument(f"--disable-{prefix}", dest=enable_attr, action="store_false", default=None)


def apply_model_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = copy.deepcopy(config)
    for phase in PHASE_OPTIONS:
        section = updated.setdefault(phase.key, {})
        backend = getattr(args, f"{phase.key}_backend", None)
        model = getattr(args, f"{phase.key}_model", None)
        device = getattr(args, f"{phase.key}_device", None) if phase.has_device else None
        enabled = getattr(args, f"{phase.key}_enabled", None) if phase.optional else None

        if backend:
            section["backend"] = backend
        if model:
            section[phase.model_key] = model
        if device:
            section["device"] = device
        if enabled is not None:
            section["enabled"] = bool(enabled)
    return updated
