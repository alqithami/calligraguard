from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any
import random

from .build_dataset import build_for_svg
from .utils.io import write_jsonl

def make_simple_svg(path_d: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path d="{path_d}" fill="black"/>
</svg>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parseArgs() if hasattr(ap, "parseArgs") else ap.parse_args()

    out_dir = Path(args.out_dir)
    svg_dir = out_dir / "svg_src" / "DemoFont"
    svg_dir.mkdir(parents=True, exist_ok=True)

    # Create a few simple shapes as "glyphs"
    glyphs = {
        "U+0001": "M 40 40 L 160 40 L 160 160 L 40 160 Z",  # square
        "U+0002": "M 100 30 C 150 30 170 60 170 100 C 170 150 130 170 100 170 C 60 170 30 140 30 100 C 30 60 50 30 100 30 Z",
        "U+0003": "M 40 120 Q 100 20 160 120 L 160 160 L 40 160 Z"
    }
    for name, d in glyphs.items():
        (svg_dir / f"{name}.svg").write_text(make_simple_svg(d), encoding="utf-8")

    # Build dataset
    records: List[Dict[str, Any]] = []
    for svg_file in sorted(svg_dir.glob("*.svg")):
        records.extend(build_for_svg(
            svg_file=svg_file,
            out_dir=out_dir,
            render_sizes=[128],
            variants_per_glyph=2,
            defect_types=["spur","gap","jitter"],
            seed=args.seed
        ))
    write_jsonl(out_dir / "meta.jsonl", records)
    print(f"[demo] wrote demo dataset with {len(records)} samples to {out_dir}")

if __name__ == "__main__":
    main()
