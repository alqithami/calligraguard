# CalligraGuard / CFDefect

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.XXXXX-b31b1b.svg)](https://arxiv.org/)

Code, dataset-generation scripts, evaluation utilities, and released artifacts for Arabic glyph defect inspection and the CFDefect benchmark.

## Repository status

This public repository is close to a reviewer-facing reproducibility package, but it is **not fully complete yet**.

The public tree already includes:

- dataset generation code under `cfdefect/`;
- held-out split definitions under `DATASET_SPLITS/`;
- selected released run artifacts under `runs_split/`, `runs/`, and `runs_2m/`;
- generated paper-facing LaTeX tables under `tables/`;
- repository documentation under `docs/`.

The main remaining gaps are:

- fill `docs/FONTS_MANIFEST.csv` with the **actual** source-font provenance;
- upload missing run artifacts for `runs_2m/calligra_ref_nosvgv/`;
- upload missing run artifacts for `runs/unet_scoremax/` if that run is cited in the manuscript;
- upload figure-generation scripts or figure-source data;
- create a GitHub Release for large checkpoints / predictions that should not live in git;
- optionally upload repository-facing manuscript assets under `paper/`.

See `docs/UPLOAD_CHECKLIST.md` for the exact filenames and paths that still need to be uploaded or completed manually.

Until the manuscript and repository are fully synchronized, treat the released `metrics.json` files and generated `tables/*.tex` in this repository as the **public source of truth** for released artifact values.

## What is in the code tree

The main package is `cfdefect/`:

- `cfdefect/export_glyphs.py` — export per-glyph SVG files from TTF/OTF/TTC fonts.
- `cfdefect/build_dataset.py` — build paired clean/defective raster samples, masks, and `meta.jsonl`.
- `cfdefect/precompute_svgv.py` — pre-render SVG-V inputs.
- `cfdefect/evaluate.py` — compute detection, localization, classification, and attribution metrics.
- `cfdefect/make_detection_fpr_table.py` — generate a strict-FPR LaTeX table for one prediction file.
- `cfdefect/make_latex_tables.py` — generate summary LaTeX tables from one or more `metrics.json` files.
- `cfdefect/baselines/template_diff.py` — classical referenced differencing baseline.
- `cfdefect/baselines/calligraguard_train.py` — CalligraGuard-Lite training entry point.
- `cfdefect/baselines/calligraguard_predict.py` — CalligraGuard-Lite inference entry point.

The repository also exposes selected artifacts under `runs_split/`, `runs/`, and `runs_2m/`, plus generated LaTeX tables under `tables/`.

## Installation

### Option A: pip + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate calligraguard
```

## Dataset format

A built dataset root is expected to look like this:

```text
DATASET_ROOT/
├── images/
├── masks/
├── svg/
├── clean_svg/
├── clean_images/
├── svgv/          # added after precompute_svgv
├── clean_svgv/    # added after precompute_svgv
└── meta.jsonl
```

Each line in `meta.jsonl` is a single sample record. Common fields include:

- `id`
- `font_id`
- `unicode`
- `form`
- `render`
- `is_defective`
- `defects`
- `image_path`, `mask_path`, `svg_path`
- `clean_image_path`, `clean_svg_path`

## Quick start

### 1) Export glyphs from fonts

```bash
python -m cfdefect.export_glyphs \
  --fonts_dir /path/to/fonts \
  --out_dir /path/to/out_svg \
  --chars_file chars_arabic.txt \
  --font_glob "*.ttf"
```

Repeat with `*.otf` or `*.ttc` if needed.

### 2) Build the paired clean/defective dataset

```bash
python -m cfdefect.build_dataset \
  --svg_dir /path/to/out_svg \
  --out_dir /path/to/DATASET_ROOT \
  --renders "64,96" \
  --variants_per_glyph 4 \
  --seed 123 \
  --workers 8
```

### 3) Precompute SVG-V

```bash
python -m cfdefect.precompute_svgv \
  --dataset_root /path/to/DATASET_ROOT \
  --size 64
```

## Run the released baselines / model

### TemplateDiff

```bash
python -m cfdefect.baselines.template_diff \
  --dataset /path/to/DATASET_ROOT \
  --out_pred /path/to/runs/template_diff/pred.jsonl

python -m cfdefect.evaluate \
  --gt /path/to/DATASET_ROOT/meta.jsonl \
  --pred /path/to/runs/template_diff/pred.jsonl \
  --out /path/to/runs/template_diff/metrics.json \
  --dataset_root /path/to/DATASET_ROOT
```

### CalligraGuard-Lite (template-referenced, with SVG-V)

```bash
python -m cfdefect.baselines.calligraguard_train \
  --dataset /path/to/DATASET_ROOT \
  --out_dir /path/to/runs/calligra_ref_svgv \
  --mode referenced \
  --use_svgv \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-3

python -m cfdefect.baselines.calligraguard_predict \
  --dataset /path/to/DATASET_ROOT \
  --ckpt /path/to/runs/calligra_ref_svgv/model.pt \
  --out_pred /path/to/runs/calligra_ref_svgv/pred.jsonl \
  --score_mode maskmax

python -m cfdefect.evaluate \
  --gt /path/to/DATASET_ROOT/meta.jsonl \
  --pred /path/to/runs/calligra_ref_svgv/pred.jsonl \
  --out /path/to/runs/calligra_ref_svgv/metrics.json \
  --dataset_root /path/to/DATASET_ROOT
```

### CalligraGuard-Lite (template-referenced, no SVG-V)

```bash
python -m cfdefect.baselines.calligraguard_train \
  --dataset /path/to/DATASET_ROOT \
  --out_dir /path/to/runs/calligra_ref_nosvgv \
  --mode referenced \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-3

python -m cfdefect.baselines.calligraguard_predict \
  --dataset /path/to/DATASET_ROOT \
  --ckpt /path/to/runs/calligra_ref_nosvgv/model.pt \
  --out_pred /path/to/runs/calligra_ref_nosvgv/pred.jsonl \
  --score_mode maskmax
```

### CalligraGuard-Lite (universal)

```bash
python -m cfdefect.baselines.calligraguard_train \
  --dataset /path/to/DATASET_ROOT \
  --out_dir /path/to/runs/calligra_uni \
  --mode universal \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-3

python -m cfdefect.baselines.calligraguard_predict \
  --dataset /path/to/DATASET_ROOT \
  --ckpt /path/to/runs/calligra_uni/model.pt \
  --out_pred /path/to/runs/calligra_uni/pred.jsonl \
  --score_mode maskmax
```

## Generate paper-facing tables

### Strict-FPR detection table for one run

```bash
python -m cfdefect.make_detection_fpr_table \
  --gt /path/to/DATASET_ROOT/meta.jsonl \
  --pred /path/to/runs/calligra_ref_svgv/pred.jsonl \
  --out_tex /path/to/tables/detection_fpr_calligra.tex \
  --method_name calligra_ref_svgv
```

### Summary LaTeX table across multiple runs

```bash
python -m cfdefect.make_latex_tables \
  --metrics_glob "/path/to/runs/*/metrics.json" \
  --out_dir /path/to/tables
```

## Public artifact policy

If a run is cited in the manuscript, its public artifact should include:

- `metrics.json`;
- `pred.jsonl`;
- `train_info.json` for learned methods;
- `model.pt` or a Release / DOI link to the checkpoint.

See:

- `docs/RESULTS_AND_ARTIFACTS.md`
- `docs/RUNS_MANIFEST.csv`
- `docs/UPLOAD_CHECKLIST.md`
- `docs/ADMIN_STEPS.md`

## Dataset provenance

Before calling the repository complete, replace the placeholder `docs/FONTS_MANIFEST.csv` with the **actual** font manifest: source URL, font version, license, checksum, split assignment, and redistribution status for every source font.

## Paper assets

A placeholder `paper/` directory is included for repository-facing manuscript assets.

If you want the repository itself to store the manuscript snapshot, upload:

- `paper/manuscript.pdf`
- `paper/source.zip`

If those files are not committed to the repository, keep them in a GitHub Release or external archive and update `paper/README.md` accordingly.

## Citation

Please cite the repository and the accompanying manuscript as appropriate. A `CITATION.cff` file is included.

## License

This repository is released under the MIT License.
