# Vilier

Vilier creates speaker-aligned audio artifacts from raw Vietnamese conversation audio.

Version 1 focuses only on speaker splitting and timeline reconstruction:

- Silero VAD for local speech activity detection.
- Speech-only diarization chunks built from consecutive VAD utterances.
- NVIDIA Sortformer for speaker diarization.
- Local PhoWhisper Large ASR for Vietnamese transcripts.
- Qwen transcript-based state labeling for VAD utterances.
- Full-duration per-speaker tracks that can be played in parallel.
- Optional per-speaker segment WAV files.
- Audacity-compatible label files for VAD and speaker segments.

Transcript runs after VAD and diarization. Each VAD utterance is transcribed with `vinai/PhoWhisper-large`, then assigned to the diarized speaker with the largest time overlap.
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

Dry-run mode uses deterministic adapters and does not load Sortformer.
It also writes placeholder ASR text, so use it only to test file flow and timeline contracts.

```bash
PYTHON_BIN=/opt/anaconda3/envs/sommelier/bin/python \
DRY_RUN=1 \
INPUT_PATH=inputs/vi_conv_sample_5min.mp3 \
OUTPUT_PATH=output/ \
STATE_DIR=/tmp/vilier_state_smoke \
bash run_pipeline.sh
```

## Run With Silero VAD + NVIDIA Sortformer

```bash
PYTHON_BIN=/opt/anaconda3/envs/sommelier/bin/python \
INPUT_PATH=/path/to/audio_or_folder \
OUTPUT_PATH=outputs \
STATE_DIR=state \
bash run_pipeline.sh
```

The default Sortformer model is `nvidia/diar_sortformer_4spk-v1`.
The default ASR model is `vinai/PhoWhisper-large`, loaded locally through Hugging Face Transformers. The first non-dry run downloads the model into the local Hugging Face cache; later runs reuse the cached model.
The default state labeling backend is Qwen through DashScope/OpenAI-compatible chat completions. Set `DASHSCOPE_API_KEY` before non-dry runs.
By default, VAD utterances are concatenated into speech-only files shorter than `diarization.max_chunk_seconds` before diarization. The default is `180.0` seconds.

## Run PhoWhisper ASR Only

```bash
PYTHONPATH=. /opt/anaconda3/envs/sommelier/bin/python -m pipeline.asr /path/to/audio.wav
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
    ├── state_dir=state
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

`manifest.timeline.json` includes `vad_segments`, VAD utterance audio paths, diarization chunk paths with source-time mapping, speaker segments, track paths, and Audacity label file paths.
It also includes `transcript`, with one speaker-tagged transcript record per VAD utterance when `asr.enabled` is `true`, plus state label fields when `state_labeling.enabled` is `true`.
`state_labeling` in the manifest summarizes the configured labels, Qwen model, counts, and `state/index.json`.
`transcript.json` contains the same transcript records as a standalone inspectable file.
`vad.txt` is a tab-separated view of the same VAD intervals: `start_time<TAB>end_time<TAB>label`.
`vad_audio/audio_*.wav` contains one WAV file per VAD utterance, numbered from `audio_1.wav` in VAD order.
`diarization_chunks/chunk_*.wav` contains concatenated VAD utterances for Sortformer. Diarization timestamps from these speech-only chunks are mapped back to the original audio timeline before labels and speaker tracks are written.
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

Overlap separation is disabled by default because SepReformer weights are not bundled. Set `overlap_separation.enabled` to `true` and point `overlap_separation.sepreformer_path` to a local SepReformer checkout when those weights are available.

`tracks/SPEAKER_*.wav` are full-duration files. Playing them in parallel reconstructs the speaker timing from the original conversation.
