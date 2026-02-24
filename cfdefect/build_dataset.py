from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib
import json
from tqdm import tqdm
import multiprocessing as mp

import numpy as np

from .utils.img import write_image
from .utils.rle import bbox_from_mask
from .svg.xml import load_svg_paths, save_svg
from .svg.ops import inject_defect, InjectResult
from .render.rasterize import render_svg_to_array, compute_diff_mask, refine_mask

DEFAULT_DEFECTS = ["missing_diacritic", "misplaced_diacritic", "spur", "gap", "jitter"]

def _stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def build_for_svg(svg_file: Path,
                  out_dir: Path,
                  render_sizes: List[int],
                  variants_per_glyph: int,
                  defect_types: List[str],
                  seed: int,
                  add_clean: bool = True) -> List[Dict[str, Any]]:
    """
    Build clean + defective variants for a single SVG glyph file.
    Returns list of metadata dicts (one per sample = one render size x variant).
    This function writes images/SVGs to out_dir (safe for multiprocessing as filenames are unique).
    """
    font_id = svg_file.parent.name
    unicode = svg_file.stem  # e.g., U+0628
    form = "unknown"

    records: List[Dict[str, Any]] = []
    # Copy clean SVG
    clean_svg_rel = Path("clean_svg") / font_id / svg_file.name
    clean_svg_path = out_dir / clean_svg_rel
    clean_svg_path.parent.mkdir(parents=True, exist_ok=True)
    clean_svg_path.write_bytes(svg_file.read_bytes())

    # Render clean images
    clean_imgs: Dict[int, Path] = {}
    for sz in render_sizes:
        arr = render_svg_to_array(clean_svg_path, size=sz, background="white", mode="L")
        out_rel = Path("clean_images") / font_id / f"{unicode}__sz{sz}.png"
        out_path = out_dir / out_rel
        write_image(out_path, arr)
        clean_imgs[sz] = out_rel

    # Load SVG paths for injections
    tree, root, paths = load_svg_paths(clean_svg_path)
    paths_d = [p.d for p in paths]

    # Defective variants
    for v in range(variants_per_glyph):
        rnd = random.Random(seed + hash((str(svg_file), v)) % (10**9))
        defect_type = rnd.choice(defect_types)
        severity = float(rnd.uniform(0.2, 1.0))

        # reload clean each variant
        tree_v, _, paths_v = load_svg_paths(clean_svg_path)
        paths_d_v = [p.d for p in paths_v]
        inj: InjectResult = inject_defect(tree_v, paths_d_v, defect_type=defect_type, severity=severity)

        variant_tag = f"{inj.defect_type}_{v:03d}"
        sample_id_base = f"{font_id}/{unicode}/{form}/{variant_tag}"
        sid = _stable_id(sample_id_base)

        svg_rel = Path("svg") / font_id / f"{unicode}__{variant_tag}__{sid}.svg"
        svg_out = out_dir / svg_rel
        save_svg(tree_v, svg_out)

        for sz in render_sizes:
            corrupted = render_svg_to_array(svg_out, size=sz, background="white", mode="L")
            clean = render_svg_to_array(clean_svg_path, size=sz, background="white", mode="L")
            mask = compute_diff_mask(clean, corrupted, thresh=16)
            mask = refine_mask(mask, min_area=max(10, int(0.002 * sz * sz)))

            img_rel = Path("images") / font_id / f"{unicode}__{variant_tag}__sz{sz}__{sid}.png"
            mask_rel = Path("masks") / font_id / f"{unicode}__{variant_tag}__sz{sz}__{sid}_mask.png"
            write_image(out_dir / img_rel, corrupted)
            write_image(out_dir / mask_rel, (mask * 255).astype(np.uint8))

            x1,y1,x2,y2 = bbox_from_mask(mask)
            rec: Dict[str, Any] = {
                "id": f"{sid}__sz{sz}",
                "font_id": font_id,
                "unicode": unicode,
                "form": form,
                "render": {"size": int(sz), "antialias": True},
                "is_defective": True,
                "defects": [{
                    "type": inj.defect_type,
                    "severity": float(inj.severity),
                    "path_ids": [int(i) for i in inj.path_indices],
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "note": inj.note
                }],
                "image_path": str(img_rel).replace("\\", "/"),
                "mask_path": str(mask_rel).replace("\\", "/"),
                "svg_path": str(svg_rel).replace("\\", "/"),
                "clean_image_path": str(clean_imgs[sz]).replace("\\", "/"),
                "clean_svg_path": str(clean_svg_rel).replace("\\", "/"),
            }
            records.append(rec)

    if add_clean:
        for sz in render_sizes:
            sid = _stable_id(f"{font_id}/{unicode}/{form}/clean")
            img_rel = Path("images") / font_id / f"{unicode}__clean__sz{sz}__{sid}.png"
            mask_rel = Path("masks") / font_id / f"{unicode}__clean__sz{sz}__{sid}_mask.png"
            arr = render_svg_to_array(clean_svg_path, size=sz, background="white", mode="L")
            write_image(out_dir / img_rel, arr)
            write_image(out_dir / mask_rel, np.zeros_like(arr, dtype=np.uint8))
            rec = {
                "id": f"{sid}__sz{sz}",
                "font_id": font_id,
                "unicode": unicode,
                "form": form,
                "render": {"size": int(sz), "antialias": True},
                "is_defective": False,
                "defects": [],
                "image_path": str(img_rel).replace("\\", "/"),
                "mask_path": str(mask_rel).replace("\\", "/"),
                "svg_path": str(clean_svg_rel).replace("\\", "/"),
                "clean_image_path": str(clean_imgs[sz]).replace("\\", "/"),
                "clean_svg_path": str(clean_svg_rel).replace("\\", "/"),
            }
            records.append(rec)

    return records

def _worker(args_tuple):
    svg_file, out_dir, render_sizes, variants_per_glyph, defect_types, seed, add_clean = args_tuple
    try:
        return build_for_svg(Path(svg_file), Path(out_dir), render_sizes, variants_per_glyph, defect_types, seed, add_clean)
    except Exception as e:
        return [{"_error": str(e), "_svg_file": str(svg_file)}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg_dir", type=str, required=True, help="Directory containing exported SVGs (e.g., out_svg/font_id/U+0628.svg)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--renders", type=str, default="64", help="Comma-separated render sizes, e.g. '64,96'")
    ap.add_argument("--variants_per_glyph", type=int, default=2)
    ap.add_argument("--defect_types", type=str, default=",".join(DEFAULT_DEFECTS))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of SVG files for quick tests")
    ap.add_argument("--workers", type=int, default=0, help="Use multiprocessing with this many workers (0=off)")
    ap.add_argument("--no_clean", action="store_true", help="Do not add clean samples")
    args = ap.parse_args()

    svg_dir = Path(args.svg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_sizes = [int(x) for x in args.renders.split(",") if x.strip()]
    defect_types = [x.strip() for x in args.defect_types.split(",") if x.strip()]
    seed = int(args.seed)
    add_clean = not args.no_clean

    svg_files = sorted(svg_dir.glob("**/*.svg"))
    if args.limit and args.limit > 0:
        svg_files = svg_files[: args.limit]

    meta_path = out_dir / "meta.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream write for scalability
    n_written = 0
    with meta_path.open("w", encoding="utf-8") as f:
        if args.workers and args.workers > 0:
            work = [(str(p), str(out_dir), render_sizes, args.variants_per_glyph, defect_types, seed, add_clean) for p in svg_files]
            with mp.Pool(processes=args.workers) as pool:
                for recs in tqdm(pool.imap_unordered(_worker, work), total=len(work), desc="Building dataset"):
                    for r in recs:
                        if "_error" in r:
                            print(f"[WARN] {r['_svg_file']}: {r['_error']}")
                            continue
                        f.write(_json_dumps(r) + "\n")
                        n_written += 1
        else:
            for svg_file in tqdm(svg_files, desc="Building dataset"):
                recs = build_for_svg(svg_file, out_dir, render_sizes, args.variants_per_glyph, defect_types, seed, add_clean)
                for r in recs:
                    f.write(_json_dumps(r) + "\n")
                    n_written += 1

    print(f"[build_dataset] wrote {n_written} samples to {meta_path}")

if __name__ == "__main__":
    main()
