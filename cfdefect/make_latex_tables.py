from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import glob
import math

from .utils.io import read_json

def fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{x:.{digits}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_glob", type=str, required=True, help="Glob for metrics.json files")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir for LaTeX tables")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.metrics_glob))
    if not paths:
        raise SystemExit("No metrics files matched the glob")

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for p in paths:
        m = read_json(p)
        method = Path(p).parent.name  # run folder name
        rows.append((method, m))

    cols = [
        ("AUROC", lambda m: m["detection"]["auroc"]),
        ("F1", lambda m: m["detection"]["f1@0.5"]),
        ("mIoU", lambda m: m["localization"]["miou"]),
        ("Dice", lambda m: m["localization"]["dice"]),
        ("Top-1", lambda m: m["classification"]["top1"]),
        ("Top-3", lambda m: m["classification"]["top3"]),
        ("Path-F1", lambda m: m["attribution"]["path_f1"]),
    ]

    best: Dict[str, float] = {}
    for name, getter in cols:
        vals = []
        for _, m in rows:
            v = getter(m)
            if v is None:
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            vals.append(float(v))
        best[name] = max(vals) if vals else float("nan")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for method, m in rows:
        parts = [method.replace("_", r"\_")]
        for name, getter in cols:
            v = getter(m)
            s = fmt(float(v) if v is not None else float("nan"))
            if s != "--" and not math.isnan(best[name]) and abs(float(v) - best[name]) < 1e-12:
                s = rf"\textbf{{{s}}}"
            parts.append(s)
        lines.append(" & ".join(parts) + r" \\")
    body = "\n".join(lines) + "\n"

    (out_dir / "main_results_rows.tex").write_text(body, encoding="utf-8")

    table = r"""
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}lccccccc@{}}
\toprule
Method & AUROC$\uparrow$ & F1$\uparrow$ & mIoU$\uparrow$ & Dice$\uparrow$ & Top-1$\uparrow$ & Top-3$\uparrow$ & Path-F1$\uparrow$ \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\caption{Main results (auto-generated).}
\label{tab:main_results_auto}
\end{table*}
"""
    (out_dir / "main_results.tex").write_text(table, encoding="utf-8")
    print(f"[make_latex_tables] wrote {out_dir/'main_results.tex'}")

if __name__ == "__main__":
    main()
