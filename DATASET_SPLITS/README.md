# DATASET_SPLITS

This directory stores the split definitions used for the font-held-out experiments.

## Intended meaning

- `train/`: training split metadata;
- `val/`: validation split metadata;
- `test/`: held-out test metadata.

The held-out test split is intended to contain font families that are disjoint from the training split.

## Expected contents

Each split directory should contain at least a `meta.jsonl` describing the samples in that split.

Two usage patterns are reasonable:

1. **Split-as-dataset-root**: each split directory contains or mirrors the actual images/masks/SVG assets referenced by its `meta.jsonl`.
2. **Split-as-selector**: each split directory contains only `meta.jsonl`, and external tooling treats it as a selector over a larger dataset root.

Whichever pattern you use, document it clearly and keep it consistent.

## Public release rule

If a run in `runs_split/` points to `DATASET_SPLITS/test/meta.jsonl`, make sure the corresponding data layout is documented well enough that another user can understand how that file was evaluated.
