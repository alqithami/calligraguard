# CFDefect Toolkit (Reference Pipeline)
This toolkit is a **results-producing pipeline** intended to accompany the CalligraGuard / CFDefect-2M paper.

It provides:
1) **Font → SVG export** (via `fontTools`)
2) **SVG → raster rendering** (via `cairosvg`)
3) **Synthetic defect injection at SVG level** + automatic mask generation
4) **Dataset packaging** in a simple JSONL + files layout
5) **Baseline runners** (template-diff, PatchCore-like anomaly baseline, supervised U-Net)
6) **Evaluation** (AUROC/F1, mIoU/Dice, attribution F1) from prediction JSONL
7) **Paper table generation** (LaTeX) from metrics JSON

> Goal: you can run experiments and automatically populate `tables/*.tex` in the paper.

---

## 0) Installation
This repository is pure Python. Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you already have `torch`, `fonttools`, `cairosvg`, `numpy`, `scikit-learn`, `opencv-python`, `scikit-image`, you can skip or selectively install.

---

## 1) Dataset format
After building, you will get:

```
DATASET_ROOT/
  images/                # raster PNGs
  masks/                 # binary mask PNGs (defect-only; empty for clean)
  svg/                   # SVG files (defective)
  clean_svg/             # SVG files (clean reference for same glyph)
  clean_images/          # raster PNGs (clean reference)
  meta.jsonl             # one JSON per sample
```

`meta.jsonl` schema (per line):
```json
{
  "id": "font=.../unicode=U+0628/form=initial/render=64aa/variant=spur_001",
  "font_id": "MyFont-Regular",
  "unicode": "U+0628",
  "form": "initial",
  "render": {"size": 64, "antialias": true},
  "is_defective": true,
  "defects": [
    {"type": "spur", "severity": 0.4, "path_ids": [7], "bbox": [x1,y1,x2,y2]}
  ],
  "image_path": "images/....png",
  "mask_path": "masks/....png",
  "svg_path": "svg/....svg",
  "clean_image_path": "clean_images/....png",
  "clean_svg_path": "clean_svg/....svg"
}
```

---

## 2) Build a dataset (synthetic + clean)
### 2.1 Export glyphs from fonts to SVG
```bash
python -m cfdefect.export_glyphs \
  --fonts_dir /path/to/fonts \
  --out_dir /path/to/out_svg \
  --chars_file chars_arabic.txt \
  --font_glob "*.ttf"
```

`chars_arabic.txt` can contain either Unicode codepoints (`U+0628`) or actual characters.

### 2.2 Build dataset with synthetic defects
```bash
python -m cfdefect.build_dataset \
  --workers 8 \
  --svg_dir /path/to/out_svg \
  --out_dir /path/to/DATASET_ROOT \
  --renders "64,96" \
  --variants_per_glyph 4 \
  --seed 123
```

This creates clean reference renders and defective variants plus masks.

---

## 3) Run baselines
### 3.1 Referenced template-diff baseline
Uses (clean image) as template.

```bash
python -m cfdefect.baselines.template_diff \
  --dataset /path/to/DATASET_ROOT \
  --out_pred /path/to/runs/template_diff/pred.jsonl
```

### 3.2 Universal anomaly baseline (PatchCore-like)
```bash
python -m cfdefect.baselines.patchcore_like \
  --dataset /path/to/DATASET_ROOT \
  --out_dir /path/to/runs/patchcore_like \
  --max_train 50000
```

### 3.3 Supervised U-Net baseline
```bash
python -m cfdefect.baselines.unet_train \
  --dataset /path/to/DATASET_ROOT \
  --out_dir /path/to/runs/unet \
  --epochs 20
python -m cfdefect.baselines.unet_predict \
  --ckpt /path/to/runs/unet/model.pt \
  --dataset /path/to/DATASET_ROOT \
  --out_pred /path/to/runs/unet/pred.jsonl
```

---

## 4) Evaluate predictions
```bash
python -m cfdefect.evaluate \
  --gt /path/to/DATASET_ROOT/meta.jsonl \
  --pred /path/to/runs/unet/pred.jsonl \
  --out /path/to/runs/unet/metrics.json
```

---

## 5) Generate LaTeX tables for the paper
Collect multiple `metrics.json` files in a folder and generate tables:

```bash
python -m cfdefect.make_latex_tables \
  --metrics_glob "/path/to/runs/*/metrics.json" \
  --out_dir /path/to/paper/tables
```

This will create:
- `paper/tables/main_results.tex`
- `paper/tables/ablations.tex` (if your metrics include "variant" keys)

You can then compile the paper and it will automatically `\input{...}` these files.

---

## 6) Demo (no fonts needed)
To sanity check the pipeline end-to-end without font files:
```bash
python -m cfdefect.demo --out_dir /tmp/cfdefect_demo
python -m cfdefect.baselines.template_diff --dataset /tmp/cfdefect_demo --out_pred /tmp/cfdefect_demo/pred.jsonl
python -m cfdefect.evaluate --gt /tmp/cfdefect_demo/meta.jsonl --pred /tmp/cfdefect_demo/pred.jsonl --out /tmp/cfdefect_demo/metrics.json
```

---

## Notes
- The defect injection is intentionally **simple but extensible**: it manipulates SVG `path d=` commands via a minimal parser.
- For a production-quality CFDefect-2M build, you may want:
  - more sophisticated dot/diacritic identification,
  - mark-anchor simulation at shaped-text level,
  - additional vector validity checks (winding, self-intersection tests).
