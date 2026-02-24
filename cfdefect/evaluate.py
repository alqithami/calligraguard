from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .utils.io import read_jsonl, write_json
from .utils.img import read_image
from .eval.metrics import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True, help="Path to ground-truth meta.jsonl")
    ap.add_argument("--pred", type=str, required=True, help="Path to predictions jsonl")
    ap.add_argument("--out", type=str, required=True, help="Output metrics.json")
    ap.add_argument("--dataset_root", type=str, default="", help="Optional dataset root; defaults to gt parent")
    args = ap.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    out_path = Path(args.out)

    dataset_root = Path(args.dataset_root) if args.dataset_root else gt_path.parent

    gt_records = list(read_jsonl(gt_path))
    pred_records = list(read_jsonl(pred_path))

    def load_mask(rel_path: str) -> np.ndarray:
        p = dataset_root / rel_path
        arr = read_image(p, mode="L")
        return (arr > 0).astype(np.uint8)

    metrics = evaluate(gt_records, pred_records, load_mask_fn=load_mask)
    metrics["gt"] = str(gt_path)
    metrics["pred"] = str(pred_path)

    write_json(out_path, metrics, indent=2)
    print(f"[evaluate] wrote metrics to {out_path}")

if __name__ == "__main__":
    main()
