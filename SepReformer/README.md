# SepReformer vendored runtime

This directory is a trimmed SepReformer checkout used by the Vilier
`overlap_separation` pipeline.

Kept:

- `models/SepReformer_*/configs.yaml`
- `models/SepReformer_*/model.py`
- `models/SepReformer_*/modules/`
- checkpoint folders such as `models/SepReformer_Base_WSJ0/log/scratch_weights/`
- `utils/decorators.py`
- `LICENSE`

Removed:

- upstream git metadata
- dataset preparation assets
- sample wav files
- train/test entrypoints and engines
- metrics, optimizer, scheduler, and dataset utilities

The pipeline defaults to `SepReformer_Base_WSJ0`. To use another retained model
version, set `overlap_separation.model_name` in the project `config.json` and
place a compatible `.pt` or `.pth` checkpoint under that model's `log/` weight
directory.
