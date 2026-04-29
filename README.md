# CalligraGuard / CFDefect

Code, data-generation scripts, evaluation utilities, and released artifacts for Arabic glyph defect inspection and the CFDefect benchmark.

## What this public repository is meant to contain

This repository is the reproducibility companion for the CalligraGuard / CFDefect work. The public tree is meant to cover four pieces:

1. dataset generation from real font files;
2. released baseline and reference-model code;
3. exact run artifacts behind the reported tables and figures;
4. paper-facing assets such as generated tables and the final manuscript source/PDF.

## Current scope of the public codebase

The code tree is organized around the `cfdefect/` package:

- `cfdefect/export_glyphs.py`: export per-glyph SVG files from TTF/OTF/TTC fonts.
- `cfdefect/build_dataset.py`: generate paired clean/defective raster samples, masks, and `meta.jsonl`.
- `cfdefect/precompute_svgv.py`: pre-render SVG-V inputs.
- `cfdefect/evaluate.py`: compute detection, localization, classification, and attribution metrics.
- `cfdefect/make_detection_fpr_table.py`: generate a strict-FPR LaTeX table for one prediction file.
- `cfdefect/make_latex_tables.py`: generate summary LaTeX tables from one or more `metrics.json` files.
- `cfdefect/baselines/template_diff.py`: classical referenced differencing baseline.
- `cfdefect/baselines/calligraguard_train.py`: CalligraGuard-Lite training entry point.
- `cfdefect/baselines/calligraguard_predict.py`: CalligraGuard-Lite inference entry point.

The public repository currently also exposes selected artifacts under `runs_split/`, `runs/`, `runs_2m/`, and generated paper tables under `tables/`.

## What is intentionally not redistributed by default

The repository may not redistribute everything needed for exact regeneration unless you add it explicitly. In particular, you should assume that the following still need to be documented and/or uploaded:

- source font files and their licenses;
- full generated datasets, if they are too large for Git;
- large checkpoints and prediction dumps that are better attached to Releases or an external archive;
- any private or unpublished baseline implementations not present under `cfdefect/`;
- the exact paper source and PDF, unless they are added under `paper/`.

## Recommended repository layout

```text
calligraguard/
├── README.md
├── CITATION.cff
├── LICENSE                  # choose and add a real license before final release
├── requirements.txt
├── environment.yml
├── DATASET_SPLITS/
│   ├── README.md
│   ├── train/
│   ├── val/
│   └── test/
├── cfdefect/
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── RESULTS_AND_ARTIFACTS.md
│   ├── RUNS_MANIFEST.csv
│   ├── FONTS_MANIFEST_TEMPLATE.csv
│   └── PAPER_RELEASE_NOTES.md
├── paper/
│   ├── README.md
│   ├── manuscript.pdf
│   ├── source.zip
│   └── figures/             # optional, if you want paper assets inside the repo
├── runs/
├── runs_split/
├── runs_2m/
└── tables/
```

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

Each line in `meta.jsonl` is a single sample record. The key fields used throughout the codebase are:

- `id`: stable sample identifier;
- `font_id`: source font identifier;
- `unicode`: codepoint label such as `U+0628`;
- `form`: contextual-form field (currently often `unknown` in the released generator);
- `render`: render configuration dictionary;
- `is_defective`: image-level binary label;
- `defects`: list of defect annotations;
- `image_path`, `mask_path`, `svg_path`;
- `clean_image_path`, `clean_svg_path`.

## Quick start: build a local dataset

### 1) Export glyphs from fonts

```bash
python -m cfdefect.export_glyphs \
  --fonts_dir /path/to/fonts \
  --out_dir /path/to/out_svg \
  --chars_file chars_arabic.txt \
  --font_glob "*.ttf"
```

If you have OTF or TTC fonts, rerun with `--font_glob "*.otf"` or `--font_glob "*.ttc"`.

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

## Quick start: run the released baselines / model

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

## Public artifact inventory to keep in sync

At minimum, if a run is cited in the paper, its folder should include:

- `metrics.json`;
- `pred.jsonl`;
- `train_info.json` for learned methods;
- `model.pt` or a Release/DOI link to the checkpoint.

See `docs/RESULTS_AND_ARTIFACTS.md` and `docs/RUNS_MANIFEST.csv` for the current artifact matrix and the places where the repository still needs synchronization with the manuscript.

## Dataset provenance

Populate `docs/FONTS_MANIFEST_TEMPLATE.csv` before calling the repository complete. If source fonts cannot be redistributed, the manifest should still record source URL, version, license, checksum, and split assignment.

## Paper assets

If the README claims that the repository can regenerate or compile the paper, the `paper/` directory should contain the actual manuscript PDF and source package. See `paper/README.md`.

## Citation

Please cite the repository and the accompanying manuscript. A ready-to-edit `CITATION.cff` is included.

## License

Choose a real license before final release. A `LICENSE-MIT.txt` option is included in this bundle for convenience, but you should only rename/upload it if that is the license you want to apply.
