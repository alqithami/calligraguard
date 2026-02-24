from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

from ..utils.io import read_jsonl
from ..utils.img import read_image
from ..render.rasterize import compute_diff_mask, refine_mask
from ..utils.rle import rle_encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Dataset root containing meta.jsonl")
    ap.add_argument("--out_pred", type=str, required=True)
    ap.add_argument("--thresh", type=int, default=16)
    ap.add_argument("--min_area_frac", type=float, default=0.002, help="Min connected component area as fraction of image")
    args = ap.parse_args()

    root = Path(args.dataset)
    meta_path = root / "meta.jsonl"
    out_path = Path(args.out_pred)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in tqdm(read_jsonl(meta_path), desc="template_diff"):
            img = read_image(root / r["image_path"], mode="L")
            tmpl = read_image(root / r["clean_image_path"], mode="L")

            mask = compute_diff_mask(tmpl, img, thresh=args.thresh)
            min_area = max(10, int(args.min_area_frac * img.shape[0] * img.shape[1]))
            mask = refine_mask(mask, min_area=min_area)

            score = float(mask.mean())  # fraction of changed pixels

            pred: Dict[str, Any] = {
                "id": r["id"],
                "score": score,
                "mask_rle": rle_encode(mask),
                "types": ["unknown"],
                "path_ids": []
            }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            n += 1

    print(f"[template_diff] wrote {n} preds to {out_path}")


if __name__ == "__main__":
    main()
