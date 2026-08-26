# Vilier

Vilier creates speaker-aligned audio artifacts from raw Vietnamese conversation audio.

Version 1 focuses only on speaker splitting and timeline reconstruction. Model choices are configured in `config.json`; the code is not tied to one diarization or separation model:

- Silero VAD for local speech activity detection.
- NVIDIA Sortformer or pyannote.audio PixIT for speaker diarization.
- Speech-only diarization chunks built from consecutive VAD utterances when Sortformer is selected.
- Optional Demucs vocal extraction before SepReformer overlap separation.
- PhoWhisper Large ASR for Vietnamese transcripts.
- Qwen transcript-based state labeling for speaker-channel ASR segments.
- Full-duration per-speaker tracks that can be played in parallel.
- Optional per-speaker segment WAV files.
- Audacity-compatible label files for VAD and speaker segments.

Transcript runs after VAD and diarization. Optional Demucs music separation can clean the waveform before SepReformer, speaker-track export, and ASR. After speaker tracks are exported, a second VAD pass segments each speaker channel for `vinai/PhoWhisper-large` ASR and Qwen labeling.
State labeling runs after ASR. Qwen receives transcript text only and assigns one configured label, initially `complete` or `incomplete`.

## Layout

```text
vilier/
  config.json
  run_pipeline.sh
  pipeline/
  tests/
```

## Run A Smoke Test Without Heavy Models

Dry-run mode uses deterministic adapters and does not load Sortformer, PixIT, SepReformer, PhoWhisper, or Qwen.
It also writes placeholder ASR text, so use it only to test file flow and timeline contracts.

```bash
DRY_RUN=1 \
INPUT_PATH=inputs/vi_conv_sample_5min.mp3 \
OUTPUT_PATH=output/ \
bash run.sh
```

`run.sh` uses the active environment's `python3`/`python` by default. Set `PYTHON_BIN=/path/to/python` only when you intentionally want to override it.

## Run With Configured Models

Install dependencies in the active environment:

```bash
python -m pip install -r requirements.txt
```

`run.sh` reads `entrypoint.input_path` from `config.json`. Set it to one audio file to process only that file:

```json
{
  "entrypoint": {
    "input_path": "inputs/haveasip_khanhvi_5m_2.wav",
    "output_path": "outputs"
  }
}
```

```bash
bash run.sh
```

For a one-off run without editing config, `INPUT_PATH=/path/to/audio.wav bash run.sh` still overrides the config value. If `entrypoint.input_path` points to a folder, the pipeline processes every supported audio file in that folder.

On cloud notebooks or GPU VMs such as Kaggle, Colab, RunPod, Paperspace, or a remote server where the repo and audio are already present, run the same script directly:

```bash
INPUT_PATH=/path/to/mounted/audio.wav bash run.sh
```

In notebook cells, prefix the shell command with `!`:

```bash
!INPUT_PATH=/path/to/mounted/audio.wav bash run.sh
```

The current `config.json` selects one set of models for a run, but each stage can be changed independently.
VAD and diarization are core pipeline stages, so they always run; configure their backend/model/device in `vad` and `diarization`. Optional stages are controlled by `*.enabled`: `music_separation`, `overlap_separation`, `asr`, and `state_labeling`.

## Model Choices

### Diarization

| Backend | Config values | Notes |
|---------|---------------|-------|
| NVIDIA Sortformer | `diarization.backend=sortformer`, `diarization.model=nvidia/diar_sortformer_4spk-v1` | Uses `nemo.collections.asr.models.SortformerEncLabelModel`. VAD utterances are concatenated into speech-only files shorter than `diarization.max_chunk_seconds`, then diarization timestamps are mapped back to the original timeline. |
| pyannote PixIT | `diarization.backend=pixit` or `pyannote`, `diarization.model=pyannote/speech-separation-ami-1.0` | Runs on `audio.standardized.wav` directly. Install `pyannote.audio[separation]==3.3.2`, accept the Hugging Face conditions for the pyannote model, and set the token env configured by `diarization.token_env`, usually `HUGGINGFACE_TOKEN`. `diarization.device=auto` uses CUDA if available, then Apple MPS, then CPU. Set `diarization.device=mps` to force Apple GPU on macOS; unsupported MPS ops can still fall back to CPU through PyTorch. |
| DiariZen | `diarization.backend=diarizen`, `diarization.model=BUT-FIT/diarizen-wavlm-large-s80-md` | Runs on `audio.standardized.wav` directly through `diarizen.pipelines.inference.DiariZenPipeline`. Install DiariZen from `https://github.com/BUTSpeechFIT/DiariZen` in the active environment before using this backend. Upstream releases the model weights under CC BY-NC 4.0, so treat them as research/non-commercial weights. |

Example Sortformer config:

```json
{
  "diarization": {
    "backend": "sortformer",
    "model": "nvidia/diar_sortformer_4spk-v1",
    "device": "cpu",
    "nemo_log_level": "ERROR",
    "min_duration_seconds": 0.25,
    "max_chunk_seconds": 180.0
  }
}
```

Example PixIT config:

```json
{
  "diarization": {
    "backend": "pixit",
    "model": "pyannote/speech-separation-ami-1.0",
    "token_env": "HUGGINGFACE_TOKEN",
    "device": "auto",
    "min_duration_seconds": 0.25
  }
}
```

Example DiariZen config:

```json
{
  "diarization": {
    "backend": "diarizen",
    "model": "BUT-FIT/diarizen-wavlm-large-s80-md",
    "cache_dir": "",
    "rttm_out_dir": "",
    "min_duration_seconds": 0.25
  }
}
```

### Music Separation

| Backend | Config values | Notes |
|---------|---------------|-------|
| Disabled | `music_separation.enabled=false` | Keeps the standardized waveform unchanged. |
| Demucs | `music_separation.enabled=true`, `music_separation.backend=demucs`, `music_separation.model=htdemucs` | Extracts vocals before SepReformer overlap separation. Increase `music_separation.residual_subtract` gradually if accompaniment still leaks into `music_cleaned.wav`; higher values can distort speech. |

### Overlap Separation

SepReformer is optional. If `overlap_separation.enabled=false`, overlapping regions remain unchanged and the pipeline still exports diarization labels and speaker tracks.

| Backend | Config values | Notes |
|---------|---------------|-------|
| SepReformer | `overlap_separation.backend=sepreformer`, `overlap_separation.model_name=<model_dir>` | `model_name` is the directory under `SepReFormer/models`. The corresponding checkpoint must exist under that model's `log/pretrain_weights`, `log/pretrained_weights`, `log/scratch_weights`, or `log/scratch_weight`. |

Model directories present in this checkout:

| Size | `overlap_separation.model_name` | Training set |
|------|----------------------------------|--------------|
| Base | `SepReformer_Base_WSJ0` | WSJ0 |
| Large | `SepReformer_Large_DM_WSJ0` | WSJ0 |
| Large | `SepReformer_Large_DM_WHAM` | WHAM |
| Large | `SepReformer_Large_DM_WHAMR` | WHAMR |

The three Large variants are selected only by changing `overlap_separation.model_name`; no code change is needed as long as the matching config and checkpoint exist under that model directory.

Example SepReformer config:

```json
{
  "overlap_separation": {
    "enabled": true,
    "backend": "sepreformer",
    "sepreformer_path": "SepReFormer",
    "model_name": "SepReformer_Large_DM_WSJ0",
    "device": "cpu",
    "overlap_threshold_seconds": 0.2
  }
}
```

### ASR And Labeling

| Stage | Config values | Notes |
|-------|---------------|-------|
| PhoWhisper ASR | `asr.enabled=true`, `asr.backend=phowhisper_local`, `asr.model=vinai/PhoWhisper-large` | Loads PhoWhisper through Hugging Face Transformers in the active environment. On GPU machines, set `asr.device=0`; on CPU set `asr.device=cpu`. |
| Skip ASR | `asr.enabled=false` | Skips ASR. |
| Qwen state labeling | `state_labeling.enabled=true`, `state_labeling.model=qwen3.8-max` | Uses DashScope/OpenAI-compatible chat completions. Set `DASHSCOPE_API_KEY` before non-dry runs. |

## Run PhoWhisper ASR Only

```bash
PYTHONPATH=. python -m pipeline.asr /path/to/audio.wav
```

## Log Format

Pipeline logs use relative paths. The same tree logs printed to the terminal are also saved under the project-level log directory, separate from `OUTPUT_PATH`:

```text
logs/<YYYY-MM-DD>/batch.log
logs/<YYYY-MM-DD>/<input_file>/pipeline.log
```

Set `LOG_DIR=logs/2026-08-24` or pass `--log-dir logs/2026-08-24` to pin a specific daily log directory.

Each run prints one tree block for the batch and one tree block per audio file:

```text
[INFO] Vilier pipeline
└── batch
    ├── input_path=inputs/vi_one.wav
    ├── output_path=outputs
    ├── log_dir=logs/2026-08-24
    ├── state_dir=outputs/vi_one/state
    ├── dry_run=1
    └── files=1

[INFO] audio=vi_one
├── preprocess PASS
│   ├── sample_rate=16000
│   └── duration_sec=1097.50
├── vad PASS
│   └── segments=551
├── asr PASS
│   ├── model=vinai/PhoWhisper-large
│   └── transcripts=551
├── state_labeling PASS
│   ├── labels=complete,incomplete
│   ├── complete=276
│   └── incomplete=275
└── done PASS
    └── elapsed_sec=123.45
```

Statuses are `PASS`, `SKIP`, or `FAIL`. Batch logs do not print transcript text or secrets.

## Output

For each input audio:

```text
outputs/<audio_id>/
  audio.standardized.wav
  vad.json
  vad.txt
  transcript.json
  manifest.timeline.json
  labels/vad.txt
  labels/speakers.txt
  labels/SPEAKER_00.txt
  vad_audio/audio_1.wav
  vad_audio/audio_2.wav
  music_cleaned.wav
  asr_audio/SPEAKER_00/audio_00001.wav
  asr_audio/SPEAKER_01/audio_00001.wav
  diarization_chunks/chunk_1.wav
  diarization_chunks/chunk_2.wav
  tracks/SPEAKER_00.wav
  state/
    complete/complete_01.wav
    complete/complete_01.json
    incomplete/incomplete_01.wav
    incomplete/incomplete_01.json
    index.json
```

`manifest.timeline.json` includes `vad_segments`, VAD utterance audio paths, `music_separation`, `asr_segments`, diarization chunk paths with source-time mapping, speaker segments, track paths, and Audacity label file paths.
It also includes `transcript`, with one speaker-tagged transcript record per speaker-channel ASR segment when `asr.enabled` is `true`, plus state label fields when `state_labeling.enabled` is `true`.
`state_labeling` in the manifest summarizes the configured labels, Qwen model, counts, and `outputs/<audio_id>/state/index.json`.
`transcript.json` contains the same transcript records as a standalone inspectable file.
`vad.txt` is a tab-separated view of the same VAD intervals: `start_time<TAB>end_time<TAB>label`.
`vad_audio/audio_*.wav` contains one WAV file per source-audio VAD utterance, numbered from `audio_1.wav` in VAD order. These files are for VAD review and diarization support, not the default ASR input.
`music_cleaned.wav` is written only when `music_separation.enabled` is active. SepReformer, speaker tracks, and downstream ASR use this cleaned waveform.
`asr_audio/SPEAKER_*/*.wav` contains speech segments detected on each exported speaker track. These files are the default ASR and Qwen labeling input.
`diarization_chunks/chunk_*.wav` contains concatenated VAD utterances for Sortformer compatibility. PixIT diarization runs on `audio.standardized.wav` directly.
When `music_separation.enabled` is `true`, Demucs runs before SepReformer so overlap separation receives the vocal-cleaned waveform, following the Sommelier ordering.
When `overlap_separation.enabled` is `true`, overlapping speaker regions are separated before speaker tracks are exported. This follows the Sommelier SepReformer flow: detect overlapping diarization pairs, separate only the mixed overlap region, match separated source volume to each speaker's non-overlap RMS, then reconstruct enhanced per-speaker audio for track export.

## Audacity Import

Import audio files with `File > Import > Audio`:

- `audio.standardized.wav` for the source audio.
- `tracks/SPEAKER_*.wav` for full-duration speaker-separated tracks.

Import label tracks with `File > Import > Labels`:

- `labels/vad.txt` for speech activity labels.
- `labels/speakers.txt` for all diarized speaker segments in one label track.
- `labels/SPEAKER_*.txt` for one label track per speaker.

All label files use Audacity's tab-separated format: `start_time<TAB>end_time<TAB>label`.

Segment WAV export is disabled by default for faster Audacity-oriented generation. Set `export.write_segment_wavs` to `true` in `config.json` when you need individual files under `segments/SPEAKER_*`.
Matplotlib visualization export is also disabled by default. Set `export.write_visualizations` to `true` when you need PNG files under `visualization/`.

Set `overlap_separation.enabled` to `false` to skip SepReFormer. When it is enabled, point `overlap_separation.sepreformer_path` to the local `SepReFormer` checkout and choose a `model_name` whose checkpoint exists.

`tracks/SPEAKER_*.wav` are full-duration files. Playing them in parallel reconstructs the speaker timing from the original conversation.
