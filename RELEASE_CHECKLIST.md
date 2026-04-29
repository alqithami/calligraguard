# Release checklist

Use this before calling the public repository complete.

## Naming and metadata

- [ ] Pick one canonical paper title and use it consistently in the README, `CITATION.cff`, manuscript, and any dataset names.
- [ ] Pick one canonical dataset name for the large build (for example, `CFDefect-Large`) and stop mixing it with `CFDefect-2M` unless there is actually a 2M-sample release.
- [ ] Add a real `LICENSE` file.
- [ ] Create a tagged GitHub Release and attach large artifacts that do not belong in git history.

## Reproducibility files

- [ ] Fill `docs/FONTS_MANIFEST_TEMPLATE.csv` with exact font provenance, version, license, URL, and checksum.
- [ ] Keep `docs/RUNS_MANIFEST.csv` synchronized with the run folders that are publicly uploaded.
- [ ] Add figure-generation scripts or at least figure-source CSV/JSON files.
- [ ] Add the final manuscript PDF and source archive under `paper/` or remove any claim that the repo can compile the paper.

## Artifacts

- [ ] For every table/figure row cited in the manuscript, upload `metrics.json`.
- [ ] For every learned method cited in the manuscript, upload `train_info.json`.
- [ ] For every run cited in the manuscript, upload `pred.jsonl`.
- [ ] Upload `model.pt` or provide an external checkpoint link in the Release notes.

## Numeric consistency

- [ ] Reconcile the large-run numbers in the manuscript with `runs_2m/calligra_ref_nosvgv/metrics.json` and `tables/detection_fpr_calligra_2m.tex`.
- [ ] Reconcile the U-Net AUROC discrepancy between the main manuscript table and the strict-FPR table.
- [ ] Reconcile any sample-count mismatch between the manuscript text, figure captions, and `DATASET_2M_stats.json`.

## Clean-up

- [ ] Remove committed `__pycache__` directories.
- [ ] Remove stale `.bak` files.
- [ ] Confirm that `requirements.txt` and/or `environment.yml` match the environment used for the released runs.
