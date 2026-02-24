from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image

from .render.rasterize import render_svg_to_array

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True, help="Dataset root containing svg/ and clean_svg/")
    ap.add_argument("--size", type=int, default=64, help="SVG-V render size (square)")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    size = int(args.size)

    def render_tree(src_dirname: str, dst_dirname: str):
        src = root / src_dirname
        dst = root / dst_dirname
        svg_files = sorted(src.glob("**/*.svg"))
        if not svg_files:
            print(f"[precompute_svgv] No SVGs under {src}")
            return 0
        n = 0
        for svg in tqdm(svg_files, desc=f"render {src_dirname}->{dst_dirname}"):
            rel = svg.relative_to(src)
            out = (dst / rel).with_suffix(".png")
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            arr = render_svg_to_array(svg, size=size, background="white", mode="L")
            Image.fromarray(arr).save(out)
            n += 1
        return n

    n1 = render_tree("svg", "svgv")
    n2 = render_tree("clean_svg", "clean_svgv")
    print(f"[precompute_svgv] rendered {n1} corrupted SVGs and {n2} clean SVGs at {size}x{size}.")

if __name__ == "__main__":
    main()
