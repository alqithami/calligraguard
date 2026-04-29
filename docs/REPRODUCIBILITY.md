# Reproducibility notes

This repository should make it possible for another researcher to do three things:

1. rebuild a CFDefect-style dataset from documented source fonts;
2. rerun the public baselines and CalligraGuard-Lite scripts;
3. regenerate the exact paper-facing numbers from released artifacts.

## Minimum public artifact standard

A paper-facing repository should include:

- a root `README.md` that matches the actual tree and actual public scripts;
- `CITATION.cff`;
- a real `LICENSE`;
- exact commands for dataset building, training, prediction, evaluation, and table generation;
- a manifest of source fonts, versions, licenses, and checksums;
- a manifest of released run artifacts;
- the final paper PDF and source package, or a clear statement that they are not yet included.

## Minimum run artifact standard

For every experiment row that appears in the paper, the public artifact should include:

- `metrics.json`;
- `pred.jsonl`;
- `train_info.json` for learned methods;
- `model.pt` or a link to a Release/DOI artifact containing the checkpoint.

If the checkpoint is too large for git, attach it to a GitHub Release and record the download URL in `docs/RUNS_MANIFEST.csv`.

## Dataset provenance

The dataset generator is only half of reproducibility. The other half is provenance.

Before final release, record for every source font:

- source filename;
- human-readable font name;
- source URL;
- version / release date;
- license;
- license URL;
- checksum;
- split assignment;
- notes on whether the font can be redistributed.

Use `docs/FONTS_MANIFEST_TEMPLATE.csv` as the starting point.

## Manuscript synchronization

The repository and manuscript must agree on:

- dataset names;
- sample counts;
- run folder names;
- table values;
- figure-source values.

If they do not, the repository should temporarily state which side is the source of truth. `docs/RESULTS_AND_ARTIFACTS.md` is the place to record that.

## Recommended release pattern

- keep source code and small configs in git;
- keep very large checkpoints, predictions, and generated datasets in Releases or an external archive;
- keep exact paths, checksums, and download links in the repository docs.
