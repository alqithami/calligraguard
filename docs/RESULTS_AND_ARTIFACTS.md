# Results and artifacts manifest

This document maps the currently visible public artifact tree to the manuscript tables/figures and records the places that still need synchronization.

## Public artifact inventory visible in the repository tree

| Public path | Intended use | Files currently visible | Status |
|---|---|---|---|
| `runs_split/calligra_ref_svgv/` | held-out referenced CalligraGuard + SVG-V | `metrics.json`, `model.pt`, `pred.jsonl`, `train_info.json` | complete for a public run folder |
| `runs_split/template_diff/` | held-out TemplateDiff baseline | `metrics.json`, `pred.jsonl` | complete for a non-learned baseline |
| `runs_split/unet/` | held-out supervised U-Net baseline | `metrics.json`, `model.pt`, `pred.jsonl`, `train_info.json` | complete for a public run folder |
| `runs/unet_scoremax/` | alternate U-Net scoring run | `metrics.json` | partial |
| `runs_2m/calligra_ref_nosvgv/` | large referenced CalligraGuard without SVG-V | `metrics.json` | partial |
| `tables/` | generated paper-facing LaTeX tables | multiple `.tex` files | present |

## Exact numbers currently exposed by the public repository

### Dataset stats file

`DATASET_2M_stats.json` currently reports:

- `n = 247996`
- `num_fonts = 31`
- `clean = 35428`
- `defective = 212568`

### Held-out referenced run

`runs_split/calligra_ref_svgv/metrics.json` currently reports:

- detection AUROC = `0.7671967455621301`
- Dice = `0.4179791795179665`
- mIoU = `0.3991775250385872`
- Top-1 = `0.7958579881656804`
- Top-3 = `0.9467455621301775`

`runs_split/calligra_ref_svgv/train_info.json` currently reports:

- `mode = referenced`
- `size = 256`
- `base = 32`
- `use_svgv = true`
- `epochs = 10`
- `batch_size = 16`
- `lr = 0.001`

### Large referenced run

`runs_2m/calligra_ref_nosvgv/metrics.json` currently reports:

- detection AUROC = `0.7951435014657792`
- Dice = `0.3712257743133997`
- mIoU = `0.3482554360270337`
- Top-1 = `0.4744411200180648`
- Top-3 = `0.8523013812050731`

`tables/detection_fpr_calligra_2m.tex` currently reports:

- AUROC = `0.795`
- Recall@1%FPR = `0.611`
- Recall@5%FPR = `0.649`

### U-Net discrepancy that should be explained publicly

`runs/unet_scoremax/metrics.json` currently reports AUROC `0.4355593869446079`, while `tables/detection_fpr.tex` contains both:

- `unet = 0.565 / 0.001 / 0.001`
- `unet_scoremax = 0.564 / 0.000 / 0.000`

This is not necessarily wrong, but it must be explained. The public docs should say whether `unet` and `unet_scoremax` are two different score definitions derived from the same segmentation model, or two distinct runs.

## What still needs to be uploaded for a truly complete repository

- missing `pred.jsonl` and `train_info.json` for `runs_2m/calligra_ref_nosvgv/`;
- missing `pred.jsonl`, `train_info.json`, and checkpoint for `runs/unet_scoremax/` if that run is cited in the manuscript;
- fonts manifest with provenance and licenses;
- figure-generation scripts or figure-source data;
- paper PDF and source package under `paper/`.

## Recommended public policy until everything is synchronized

Until the manuscript and repository fully agree, state explicitly in the README that the released `metrics.json` and generated `.tex` files in the repository are the source of truth for the public artifact.
