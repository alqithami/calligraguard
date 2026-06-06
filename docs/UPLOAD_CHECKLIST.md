# Upload checklist for a complete public repository

This file lists the remaining items that still need manual upload or manual completion before the repository can be called complete.

## 1) Font provenance

Replace the placeholder row in `docs/FONTS_MANIFEST.csv` with the real source-font manifest.

Required columns:
- `source_filename`
- `font_name`
- `source_url`
- `version_or_release_date`
- `license`
- `license_url`
- `sha256`
- `split_assignment`
- `redistributable`
- `notes`

## 2) Missing run artifacts

### `runs_2m/calligra_ref_nosvgv/`
Upload:
- `pred.jsonl`
- `train_info.json`
- `model.pt` **or** a Release / DOI link recorded in the docs

### `runs/unet_scoremax/`
Upload these files **if this row is cited in the manuscript**:
- `pred.jsonl`
- `train_info.json`
- `model.pt` **or** a Release / DOI link recorded in the docs

## 3) Paper snapshot

Either commit these files under `paper/`:
- `paper/manuscript.pdf`
- `paper/source.zip`

or attach them to a GitHub Release and update `paper/README.md` with the public location.

## 4) Figure assets

Add either:
- figure-generation scripts; or
- figure-source JSON / CSV / NPZ files with a short README explaining which files reproduce which figures.

A simple public layout would be:

```text
figures/
  make_roc_large.py
  make_score_hist.py
  data/
    roc_large.json
    score_hist_calligra_ft.csv
    score_hist_templatediff.csv
```

## 5) Non-file GitHub settings

These cannot be completed by a normal file commit:
- update the GitHub **About** description so it matches the README;
- create a GitHub **Release** for large checkpoints / predictions that should not live in git.

## 6) Final public consistency check

Before calling the repository complete, verify that:
- `README.md` matches the actual code tree;
- `docs/RESULTS_AND_ARTIFACTS.md` matches the currently released runs;
- `docs/RUNS_MANIFEST.csv` matches the visible artifact folders;
- public `metrics.json` and generated `tables/*.tex` agree with the manuscript, or any mismatch is explicitly explained.
