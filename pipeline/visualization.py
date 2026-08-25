from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .audio import slice_waveform
from .schema import SpeakerSegment, SpeakerTrack


def write_visualizations(
    waveform: np.ndarray,
    sample_rate: int,
    segments: list[SpeakerSegment],
    tracks: list[SpeakerTrack],
    output_dir: Path,
) -> None:
    visualization_dir = output_dir / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    _write_source_spectrogram(waveform, sample_rate, visualization_dir / "spectrogram_source.png")
    _write_track_spectrograms(waveform, sample_rate, tracks, output_dir, visualization_dir / "spectrogram_tracks.png")
    _write_timeline(segments, visualization_dir / "timeline.png")


def _write_source_spectrogram(waveform: np.ndarray, sample_rate: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10", category=RuntimeWarning)
        ax.specgram(waveform, Fs=sample_rate, NFFT=512, noverlap=384, cmap="magma")
    ax.set_title("Source audio spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_track_spectrograms(
    waveform: np.ndarray,
    sample_rate: int,
    tracks: list[SpeakerTrack],
    output_dir: Path,
    path: Path,
) -> None:
    if not tracks:
        _write_source_spectrogram(waveform, sample_rate, path)
        return
    import soundfile as sf

    fig, axes = plt.subplots(len(tracks), 1, figsize=(12, max(2.5, 2.2 * len(tracks))), sharex=True, constrained_layout=True)
    if len(tracks) == 1:
        axes = [axes]
    for ax, track in zip(axes, tracks):
        track_audio, _ = sf.read(output_dir / track.track_wav, dtype="float32")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in log10", category=RuntimeWarning)
            ax.specgram(track_audio, Fs=sample_rate, NFFT=512, noverlap=384, cmap="viridis")
        ax.set_title(track.id)
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_timeline(segments: list[SpeakerSegment], path: Path) -> None:
    speakers = sorted({segment.speaker for segment in segments})
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.8 * len(speakers) + 1)), constrained_layout=True)
    speaker_y = {speaker: idx for idx, speaker in enumerate(speakers)}
    for segment in segments:
        y = speaker_y[segment.speaker]
        color = "#d97706" if segment.is_overlap else "#2563eb"
        ax.broken_barh([(segment.start, segment.duration)], (y - 0.35, 0.7), facecolors=color)
    ax.set_yticks(list(speaker_y.values()), list(speaker_y.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_title("Speaker timeline")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)
