import datetime
import logging
import os
import re
import sys
import warnings
from contextlib import contextmanager
import importlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .audio import slice_waveform, write_wav
from .schema import SpeakerSegment, relative_path


def load_diarizer(config: dict, dry_run: bool = False):
    backend = str(config.get("backend", "sortformer"))
    if backend == "sortformer":
        return SortformerDiarizer(config, dry_run=dry_run)
    if backend in {"pixit", "pyannote_pixit"}:
        return PyannotePixitDiarizer(config, dry_run=dry_run)
    if backend == "diarizen":
        return DiariZenDiarizer(config, dry_run=dry_run)
    raise ValueError(f"Unsupported diarization backend: {backend}")


class SortformerDiarizer:
    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.nemo_log_level = str(config.get("nemo_log_level", "WARNING")).upper()
        self.model = None if dry_run else self._load_model()

    def diarize(self, audio_path: Path, vad_segments: list[dict]) -> list[SpeakerSegment]:
        if self.model is None:
            return self._dry_run_segments(vad_segments)

        return self._diarize_audio(audio_path)

    def diarize_chunks(self, chunks: list[dict], progress_callback: Callable[[int, int, str], None] | None = None) -> list[SpeakerSegment]:
        if self.model is None:
            if progress_callback is not None:
                total = len(chunks)
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    progress_callback(chunk_idx, total, str(chunk.get("id", f"chunk_{chunk_idx}")))
            return self._dry_run_chunk_segments(chunks)

        segments = []
        min_duration = float(self.config.get("min_duration_seconds", 0.25))
        total = len(chunks)
        chunk_results = []

        for chunk_idx, chunk in enumerate(chunks, start=1):
            if progress_callback is not None:
                progress_callback(chunk_idx, total, f"diarize {chunk.get('id', f'chunk_{chunk_idx}')}")
            chunk_segments = self._diarize_audio(chunk["path"])
            chunk_results.append((chunk, chunk_segments))

        if total > 1:
            if progress_callback is not None:
                progress_callback(total, total, "clustering global speakers")
            global_mapping = self._cluster_local_speakers(chunk_results)
        else:
            global_mapping = None

        for chunk_idx, (chunk, chunk_segments) in enumerate(chunk_results, start=1):
            if global_mapping:
                for seg in chunk_segments:
                    seg.speaker = global_mapping.get((chunk_idx, seg.speaker), seg.speaker)
            segments.extend(remap_chunk_segments_to_original(chunk_segments, chunk["mapping"], chunk_idx, min_duration))
        return segments

    def _cluster_local_speakers(self, chunk_results: list[tuple[dict, list[SpeakerSegment]]]) -> dict:
        import torch
        import torchaudio
        from speechbrain.inference.speaker import EncoderClassifier
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        import numpy as np

        device = self.config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        device = resolve_torch_device(torch, device) or "cpu"

        try:
            with _speechbrain_use_auth_token_compat():
                classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": device},
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
        except Exception as e:
            import logging
            logging.warning(f"Could not load ECAPA-TDNN for global clustering: {e}")
            return {}

        embeddings_list = []
        labels = []

        for chunk_idx, (chunk, chunk_segments) in enumerate(chunk_results, start=1):
            try:
                signal, fs = torchaudio.load(chunk["path"])
                if fs != 16000:
                    signal = torchaudio.functional.resample(signal, fs, 16000)
            except Exception:
                continue
            
            speaker_to_segments = {}
            for seg in chunk_segments:
                speaker_to_segments.setdefault(seg.speaker, []).append(seg)
                
            for speaker, segs in speaker_to_segments.items():
                speaker_wavs = []
                for seg in segs:
                    start_sample = int(seg.start * 16000)
                    end_sample = int(seg.end * 16000)
                    if end_sample > start_sample:
                        speaker_wavs.append(signal[:, start_sample:end_sample])
                
                if not speaker_wavs:
                    continue
                speaker_signal = torch.cat(speaker_wavs, dim=1)
                
                with torch.no_grad():
                    embeddings = classifier.encode_batch(speaker_signal.to(device))
                    emb = embeddings.squeeze().cpu().numpy()
                
                if emb.ndim == 1:
                    embeddings_list.append(emb)
                    labels.append((chunk_idx, speaker))
                elif emb.ndim == 0:
                    embeddings_list.append(np.expand_dims(emb, 0))
                    labels.append((chunk_idx, speaker))

        if not embeddings_list:
            return {}
            
        X = np.stack(embeddings_list)
        if len(X) < 2:
            return {}

        distances = pdist(X, metric='cosine')
        Z = linkage(distances, method='average')
        
        # 0.6 is a standard threshold for ECAPA-TDNN cosine distance
        cluster_labels = fcluster(Z, t=0.6, criterion='distance')
        
        mapping = {}
        for (chunk_idx, old_spk), cluster_id in zip(labels, cluster_labels):
            mapping[(chunk_idx, old_spk)] = f"SPEAKER_{cluster_id - 1:02d}"
            
        return mapping

    def _diarize_audio(self, audio_path: Path) -> list[SpeakerSegment]:
        _configure_nemo_logging(self.nemo_log_level)
        predicted_segments, _ = self.model.diarize(
            audio=str(audio_path),
            batch_size=1,
            include_tensor_outputs=True,
        )
        frame = sortformer_to_dataframe(predicted_segments)
        return dataframe_to_segments(frame, min_duration=float(self.config.get("min_duration_seconds", 0.25)))

    def _load_model(self):
        _configure_nemo_logging(self.nemo_log_level)
        from nemo.collections.asr.models import SortformerEncLabelModel

        model = SortformerEncLabelModel.from_pretrained(self.config.get("model", "nvidia/diar_sortformer_4spk-v1"))
        model.eval()
        return model

    def _dry_run_segments(self, vad_segments: list[dict]) -> list[SpeakerSegment]:
        segments = []
        for idx, vad in enumerate(vad_segments):
            speaker = f"SPEAKER_{idx % 2:02d}"
            segments.append(
                SpeakerSegment(
                    id=f"{idx:05d}_{speaker}",
                    speaker=speaker,
                    start=float(vad["start"]),
                    end=float(vad["end"]),
                )
            )
        return segments

    def _dry_run_chunk_segments(self, chunks: list[dict]) -> list[SpeakerSegment]:
        segments = []
        idx = 0
        for chunk in chunks:
            for item in chunk["mapping"]:
                speaker = f"SPEAKER_{idx % 2:02d}"
                segments.append(
                    SpeakerSegment(
                        id=f"{idx:05d}_{speaker}",
                        speaker=speaker,
                        start=float(item["source_start"]),
                        end=float(item["source_end"]),
                    )
                )
                idx += 1
        return segments


class PyannotePixitDiarizer:
    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.model_name = str(config.get("model", "pyannote/speech-separation-ami-1.0"))
        self.device = str(config.get("device", "auto"))
        self.resolved_device = "dry-run" if dry_run else ""
        self.token_env = str(config.get("token_env", "HUGGINGFACE_TOKEN"))
        self.min_duration = float(config.get("min_duration_seconds", 0.25))
        self.pipeline = None if dry_run else self._load_pipeline()

    def diarize(self, audio_path: Path, vad_segments: list[dict]) -> list[SpeakerSegment]:
        if self.pipeline is None:
            return self._dry_run_segments(vad_segments)
        output = self.pipeline(str(audio_path))
        diarization = output[0] if isinstance(output, tuple) else getattr(output, "speaker_diarization", output)
        return pyannote_annotation_to_segments(diarization, self.min_duration)

    def diarize_chunks(self, chunks: list[dict], progress_callback: Callable[[int, int, str], None] | None = None) -> list[SpeakerSegment]:
        if self.pipeline is None:
            if progress_callback is not None:
                total = len(chunks)
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    progress_callback(chunk_idx, total, str(chunk.get("id", f"chunk_{chunk_idx}")))
            return SortformerDiarizer(self.config, dry_run=True)._dry_run_chunk_segments(chunks)
        raise RuntimeError("PixIT diarization must run on the full standardized audio file, not diarization chunks")

    def _load_pipeline(self):
        try:
            import pkg_resources  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyannote.audio 3.3.2 requires pkg_resources. Install a compatible setuptools with "
                "`python -m pip install 'setuptools<81'` in the active environment."
            ) from exc

        if self.device in {"auto", "mps"}:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        import torch
        _patch_torchaudio_audio_metadata()
        from pyannote.audio import Pipeline

        _allow_torch_checkpoint_globals(torch)

        token = os.environ.get(self.token_env) or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        with _hf_hub_download_use_auth_token_compat(), _speechbrain_use_auth_token_compat():
            pipeline = _load_pyannote_pipeline(Pipeline, self.model_name, token)
            resolved_device = resolve_torch_device(torch, self.device)
            self.resolved_device = resolved_device
            if resolved_device:
                pipeline.to(torch.device(resolved_device))
        return pipeline

    def _dry_run_segments(self, vad_segments: list[dict]) -> list[SpeakerSegment]:
        return SortformerDiarizer(self.config, dry_run=True)._dry_run_segments(vad_segments)


class DiariZenDiarizer:
    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.model_name = str(config.get("model", "BUT-FIT/diarizen-wavlm-large-s80-md"))
        self.cache_dir = str(config.get("cache_dir", "") or "")
        self.rttm_out_dir = str(config.get("rttm_out_dir", "") or "")
        self.min_duration = float(config.get("min_duration_seconds", 0.25))
        self.resolved_device = "dry-run" if dry_run else str(config.get("device", "auto"))
        self.pipeline = None if dry_run else self._load_pipeline()

    def diarize(self, audio_path: Path, vad_segments: list[dict]) -> list[SpeakerSegment]:
        if self.pipeline is None:
            return self._dry_run_segments(vad_segments)
        output = self.pipeline(str(audio_path), sess_name=audio_path.stem)
        return pyannote_annotation_to_segments(output, self.min_duration)

    def diarize_chunks(self, chunks: list[dict], progress_callback: Callable[[int, int, str], None] | None = None) -> list[SpeakerSegment]:
        if self.pipeline is None:
            if progress_callback is not None:
                total = len(chunks)
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    progress_callback(chunk_idx, total, str(chunk.get("id", f"chunk_{chunk_idx}")))
            return SortformerDiarizer(self.config, dry_run=True)._dry_run_chunk_segments(chunks)
        raise RuntimeError("DiariZen diarization must run on the full standardized audio file, not diarization chunks")

    def _load_pipeline(self):
        _patch_torchaudio_audio_metadata()
        try:
            from diarizen.pipelines.inference import DiariZenPipeline
        except ModuleNotFoundError as exc:
            if exc.name and not exc.name.startswith("diarizen"):
                raise
            raise ModuleNotFoundError(
                "DiariZen is not installed in the active environment. Install the upstream "
                "BUTSpeechFIT/DiariZen package before using diarization.backend=diarizen."
            ) from exc

        kwargs = {
            "cache_dir": self.cache_dir or None,
            "rttm_out_dir": self.rttm_out_dir or None,
        }
        return DiariZenPipeline.from_pretrained(self.model_name, **kwargs)

    def _dry_run_segments(self, vad_segments: list[dict]) -> list[SpeakerSegment]:
        return SortformerDiarizer(self.config, dry_run=True)._dry_run_segments(vad_segments)


def _allow_torch_checkpoint_globals(torch_module, extra_globals: list | None = None) -> None:
    add_safe_globals = getattr(getattr(torch_module, "serialization", None), "add_safe_globals", None)
    if not callable(add_safe_globals):
        return

    safe_globals = []
    torch_version = getattr(getattr(torch_module, "torch_version", None), "TorchVersion", None)
    if torch_version is not None:
        safe_globals.append(torch_version)

    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        safe_globals.extend([Specifications, Problem, Resolution])
    except Exception:
        pass

    if extra_globals:
        safe_globals.extend(extra_globals)
    if safe_globals:
        add_safe_globals(safe_globals)


def _patch_torchaudio_audio_metadata() -> None:
    try:
        import torchaudio
    except Exception:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        for module_name in ("torchaudio._backend.common", "torchaudio.backend.common", "torchaudio._backend.utils"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            audio_metadata = getattr(module, "AudioMetaData", None)
            if audio_metadata is not None:
                torchaudio.AudioMetaData = audio_metadata
                break

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(
                self,
                sample_rate: int,
                num_frames: int,
                num_channels: int,
                bits_per_sample: int,
                encoding: str,
            ):
                self.sample_rate = sample_rate
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.bits_per_sample = bits_per_sample
                self.encoding = encoding

            def __repr__(self) -> str:
                return (
                    "AudioMetaData("
                    f"sample_rate={self.sample_rate}, "
                    f"num_frames={self.num_frames}, "
                    f"num_channels={self.num_channels}, "
                    f"bits_per_sample={self.bits_per_sample}, "
                    f"encoding={self.encoding!r})"
                )

        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends():
            backends = []
            if hasattr(torchaudio, "load") or hasattr(torchaudio, "info"):
                backends.append("ffmpeg")
            try:
                import soundfile  # noqa: F401
            except Exception:
                pass
            else:
                backends.append("soundfile")
            return backends or ["ffmpeg"]

        torchaudio.list_audio_backends = list_audio_backends

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: None

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend=None: None

    if not hasattr(torchaudio, "info"):
        def info(filepath, **kwargs):
            import soundfile
            info_sf = soundfile.info(filepath)
            return torchaudio.AudioMetaData(
                sample_rate=info_sf.samplerate,
                num_frames=info_sf.frames,
                num_channels=info_sf.channels,
                bits_per_sample=16,
                encoding="PCM_S"
            )
        torchaudio.info = info


def resolve_torch_device(torch_module, requested: str) -> str:
    requested = str(requested or "cpu").strip().lower()
    if requested == "":
        return ""
    if requested == "auto":
        if getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)():
            return "cuda"
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
            return "mps"
        return "cpu"
    if requested == "mps":
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
            return "mps"
        warnings.warn("Requested diarization.device=mps but MPS is not available; falling back to CPU.", RuntimeWarning)
        return "cpu"
    if requested == "cuda":
        if getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)():
            return "cuda"
        warnings.warn("Requested diarization.device=cuda but CUDA is not available; falling back to CPU.", RuntimeWarning)
        return "cpu"
    return requested


def _load_pyannote_pipeline(pipeline_class, model_name: str, token: str | None):
    if not token:
        return pipeline_class.from_pretrained(model_name)

    try:
        return pipeline_class.from_pretrained(model_name, use_auth_token=token)
    except TypeError as exc:
        if "use_auth_token" not in str(exc):
            raise
        return pipeline_class.from_pretrained(model_name)


@contextmanager
def _hf_hub_download_use_auth_token_compat():
    try:
        import huggingface_hub
    except Exception:
        yield
        return

    original = getattr(huggingface_hub, "hf_hub_download", None)
    if not callable(original):
        yield
        return

    def compatible_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)
        return original(*args, **kwargs)

    patched = []

    def patch_module(module):
        if module is None or getattr(module, "hf_hub_download", None) is not original:
            return
        module.hf_hub_download = compatible_hf_hub_download
        patched.append(module)

    patch_module(huggingface_hub)
    for module_name in (
        "huggingface_hub.file_download",
        "huggingface_hub.hf_api",
        "pyannote.audio.core.io",
        "pyannote.audio.core.model",
        "pyannote.audio.core.pipeline",
        "speechbrain.pretrained.fetching",
        "speechbrain.utils.fetching",
    ):
        patch_module(sys.modules.get(module_name))

    try:
        yield
    finally:
        for module in patched:
            module.hf_hub_download = original


@contextmanager
def _speechbrain_use_auth_token_compat():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        yield
        return

    original_from_hparams = EncoderClassifier.from_hparams

    def compatible_from_hparams(*args, **kwargs):
        retry_kwargs = dict(kwargs)
        run_opts = retry_kwargs.get("run_opts")
        if isinstance(run_opts, dict) and "device" in run_opts:
            retry_kwargs["run_opts"] = {**run_opts, "device": _speechbrain_device(str(run_opts["device"]))}
        while True:
            try:
                return original_from_hparams(*args, **retry_kwargs)
            except AttributeError as exc:
                if "device_type" not in str(exc):
                    raise
                run_opts = retry_kwargs.get("run_opts")
                if not isinstance(run_opts, dict) or run_opts.get("device") == "cpu":
                    raise
                retry_kwargs["run_opts"] = {**run_opts, "device": "cpu"}
            except TypeError as exc:
                match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
                if match is None or match.group(1) not in retry_kwargs:
                    raise
                retry_kwargs.pop(match.group(1), None)

    EncoderClassifier.from_hparams = compatible_from_hparams
    try:
        yield
    finally:
        EncoderClassifier.from_hparams = original_from_hparams


def _speechbrain_device(device: str) -> str:
    normalized = str(device or "cpu").strip().lower()
    if normalized.startswith("mps"):
        return "cpu"
    return str(device)


def build_diarization_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    vad_segments: list[dict],
    output_dir: Path,
    max_chunk_seconds: float,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    chunks_dir = output_dir / "diarization_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    max_chunk_samples = max(1, int(round(max_chunk_seconds * sample_rate)) - 1)
    chunks = []
    current_audio = []
    current_mapping = []
    current_samples = 0

    def flush() -> None:
        nonlocal current_audio, current_mapping, current_samples
        if not current_audio:
            return
        chunk_idx = len(chunks) + 1
        chunk_path = chunks_dir / f"chunk_{chunk_idx}.wav"
        chunk_audio = np.concatenate(current_audio).astype(np.float32, copy=False)
        write_wav(chunk_path, chunk_audio, sample_rate)
        chunks.append(
            {
                "id": f"chunk_{chunk_idx}",
                "path": chunk_path,
                "audio": relative_path(chunk_path, output_dir),
                "duration": round(len(chunk_audio) / sample_rate, 6),
                "mapping": current_mapping,
            }
        )
        current_audio = []
        current_mapping = []
        current_samples = 0

    total = len(vad_segments)
    for vad_idx, vad in enumerate(vad_segments, start=1):
        if progress_callback is not None:
            progress_callback(vad_idx, total, str(vad.get("id", f"vad_{vad_idx - 1:05d}")))
        source_start = float(vad["start"])
        source_end = float(vad["end"])
        utterance = slice_waveform(waveform, sample_rate, source_start, source_end)
        offset = 0
        while offset < len(utterance):
            if current_samples >= max_chunk_samples:
                flush()
            available = max_chunk_samples - current_samples
            if available <= 0:
                flush()
                available = max_chunk_samples
            take = min(len(utterance) - offset, available)
            piece = utterance[offset : offset + take]
            chunk_start = current_samples / sample_rate
            chunk_end = (current_samples + take) / sample_rate
            piece_source_start = source_start + offset / sample_rate
            piece_source_end = source_start + (offset + take) / sample_rate
            current_audio.append(piece)
            current_mapping.append(
                {
                    "vad_id": str(vad.get("id", "")),
                    "chunk_start": round(chunk_start, 6),
                    "chunk_end": round(chunk_end, 6),
                    "source_start": round(piece_source_start, 6),
                    "source_end": round(piece_source_end, 6),
                }
            )
            current_samples += take
            offset += take
            if current_samples >= max_chunk_samples:
                flush()

    flush()
    return chunks


def _configure_nemo_logging(level_name: str) -> None:
    level = getattr(logging, level_name, logging.WARNING)
    for logger_name in ("nemo", "nemo_logger", "nemo.utils", "nemo.collections.asr.parts.utils.diarization_utils"):
        logging.getLogger(logger_name).setLevel(level)
    try:
        from nemo.utils import logging as nemo_logging
    except Exception:
        return
    if hasattr(nemo_logging, "setLevel"):
        nemo_logging.setLevel(level)


def remap_chunk_segments_to_original(
    segments: list[SpeakerSegment],
    mapping: list[dict],
    chunk_idx: int,
    min_duration: float = 0.25,
) -> list[SpeakerSegment]:
    remapped = []
    for segment in segments:
        for item in mapping:
            overlap_start = max(segment.start, float(item["chunk_start"]))
            overlap_end = min(segment.end, float(item["chunk_end"]))
            if overlap_end <= overlap_start:
                continue
            source_start = float(item["source_start"]) + (overlap_start - float(item["chunk_start"]))
            source_end = float(item["source_start"]) + (overlap_end - float(item["chunk_start"]))
            source_start = round(source_start, 3)
            source_end = round(source_end, 3)
            if source_end - source_start < min_duration:
                continue
            idx = len(remapped)
            remapped.append(
                SpeakerSegment(
                    id=f"chunk_{chunk_idx:03d}_{idx:05d}_{segment.speaker}",
                    speaker=segment.speaker,
                    start=source_start,
                    end=source_end,
                )
            )
    return remapped


def sortformer_to_dataframe(predicted_segments) -> pd.DataFrame:
    lists = [x for x in predicted_segments if isinstance(x, (list, tuple))]
    if not lists:
        lists = predicted_segments
    rows = []
    for sub in lists:
        for raw_segment in sub:
            start_s, end_s, speaker_raw = raw_segment.split()
            speaker_num = int(speaker_raw.split("_")[1])
            start = float(start_s)
            end = float(end_s)
            rows.append(
                {
                    "segment": _format_segment(start, end),
                    "label": chr(ord("A") + len(rows) % 26),
                    "speaker": f"SPEAKER_{speaker_num:02d}",
                    "start": start,
                    "end": end,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
    return pd.DataFrame(rows).sort_values(["start", "end", "speaker"]).reset_index(drop=True)


def dataframe_to_segments(frame: pd.DataFrame, min_duration: float) -> list[SpeakerSegment]:
    segments = []
    for idx, row in frame.iterrows():
        start = round(float(row["start"]), 3)
        end = round(float(row["end"]), 3)
        if end - start < min_duration:
            continue
        speaker = str(row["speaker"])
        segments.append(SpeakerSegment(id=f"{idx:05d}_{speaker}", speaker=speaker, start=start, end=end))
    return segments


def pyannote_annotation_to_segments(annotation, min_duration: float = 0.25) -> list[SpeakerSegment]:
    rows = []
    if hasattr(annotation, "itertracks"):
        iterator = annotation.itertracks(yield_label=True)
        for turn, _, speaker in iterator:
            rows.append((float(turn.start), float(turn.end), str(speaker)))
    elif hasattr(annotation, "speaker_diarization"):
        return pyannote_annotation_to_segments(annotation.speaker_diarization, min_duration=min_duration)
    else:
        raise TypeError(f"Unsupported pyannote diarization output: {type(annotation).__name__}")

    speaker_map = {}
    segments = []
    for idx, (start_raw, end_raw, speaker_raw) in enumerate(sorted(rows, key=lambda item: (item[0], item[1], item[2]))):
        start = round(start_raw, 3)
        end = round(end_raw, 3)
        if end - start < min_duration:
            continue
        if speaker_raw not in speaker_map:
            speaker_map[speaker_raw] = f"SPEAKER_{len(speaker_map):02d}"
        speaker = speaker_map[speaker_raw]
        segments.append(SpeakerSegment(id=f"{idx:05d}_{speaker}", speaker=speaker, start=start, end=end))
    return segments


def _format_segment(start: float, end: float) -> str:
    def fmt(sec: float) -> str:
        td = datetime.timedelta(seconds=sec)
        hrs = td.seconds // 3600 + td.days * 24
        mins = (td.seconds // 60) % 60
        secs = td.seconds % 60
        ms = int(td.microseconds / 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

    return f"[ {fmt(start)} --> {fmt(end)}]"
