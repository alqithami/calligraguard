from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .utils.io import read_jsonl
from .utils.rle import rle_decode

def _safe_auc(y_true: List[int], y_score: List[float]) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def threshold_for_fpr(clean_scores: np.ndarray, target_fpr: float) -> float:
    clean = np.sort(clean_scores.astype(np.float32))
    n = clean.size
    # allow at most floor(target_fpr*n) false positives among n clean samples
    max_fp = int(np.floor(target_fpr * n))
    k = max(0, n - max_fp - 1)
    k = min(k, n - 1)
    return float(clean[k])

def recall_at_fpr(scores: np.ndarray, y: np.ndarray, target_fpr: float):
    clean = scores[y == 0]
    thr = threshold_for_fpr(clean, target_fpr)
    # STRICT '>' to avoid tie explosions when many clean scores are identical
    yp = (scores > thr).astype(np.int32)
    tp = int(((yp==1)&(y==1)).sum()); fn = int(((yp==0)&(y==1)).sum())
    fp = int(((yp==1)&(y==0)).sum()); tn = int(((yp==0)&(y==0)).sum())
    rec = tp/(tp+fn+1e-9)
    fpr = fp/(fp+tn+1e-9)
    return float(rec), float(fpr), float(thr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True, help="Path to meta.jsonl")
    ap.add_argument("--pred", type=str, required=True, help="Path to pred.jsonl")
    ap.add_argument("--out_tex", type=str, required=True, help="Output LaTeX table (complete table)")
    ap.add_argument("--method_name", type=str, default="", help="Method name for the table row")
    ap.add_argument("--invert_score", action="store_true", help="Use 1-score (for normality scores)")
    args = ap.parse_args()

    gt = list(read_jsonl(Path(args.gt)))
    pred = {p["id"]: p for p in read_jsonl(Path(args.pred))}

    y = []
    s = []
    for r in gt:
        y.append(1 if r.get("is_defective", False) else 0)
        p = pred.get(r["id"])
        s.append(float(p.get("score", 0.0)) if p else 0.0)

    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float32)
    if args.invert_score:
        s = 1.0 - s

    auc = _safe_auc(y.tolist(), s.tolist())
    rec1, fpr1, thr1 = recall_at_fpr(s, y, 0.01)
    rec5, fpr5, thr5 = recall_at_fpr(s, y, 0.05)

    method = args.method_name or Path(args.pred).parent.name
    method_tex = method.replace("_", r"\_")

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"Method & AUROC$\uparrow$ & Recall@1\%FPR$\uparrow$ & Recall@5\%FPR$\uparrow$ \\")
    tex.append(r"\midrule")
    tex.append(f"{method_tex} & {auc:.3f} & {rec1:.3f} & {rec5:.3f} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\caption{Detection recall at fixed false-positive rates on clean samples (strict thresholding).}")
    tex.append(r"\label{tab:detection_fpr_single}")
    tex.append(r"\end{table}")
    Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_tex).write_text("\n".join(tex) + "\n", encoding="utf-8")

    print("[make_detection_fpr_table] wrote", args.out_tex)
    print("AUROC:", auc, "Recall@1%FPR:", rec1, "Recall@5%FPR:", rec5)

if __name__ == "__main__":
    main()
