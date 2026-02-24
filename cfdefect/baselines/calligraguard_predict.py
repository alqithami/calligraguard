from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from .calligraguard_model import CalligraGuardLite
from .calligraguard_data import scan_meta
from ..utils.io import read_jsonl, write_jsonl
from ..utils.rle import rle_encode

def _resize(arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    im = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    im = im.resize((size, size), resample=resample)
    return np.array(im)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Dataset root with meta.jsonl")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model.pt produced by calligraguard_train")
    ap.add_argument("--out_pred", type=str, required=True, help="Output pred.jsonl")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--path_k", type=int, default=3, help="Top-K path ids to output")
    ap.add_argument("--type_k", type=int, default=3, help="Top-K defect types to output (excluding 'clean')")
    ap.add_argument("--score_mode", type=str, default="maskmax", choices=["maskmax","type_notclean"])
    args = ap.parse_args()

    root = Path(args.dataset)
    meta_path = root / "meta.jsonl"
    recs = list(read_jsonl(meta_path))

    ckpt = torch.load(args.ckpt, map_location="cpu")
    defect_vocab: List[str] = ckpt.get("defect_vocab", ["clean"])
    max_path_id: int = int(ckpt.get("max_path_id", 1))
    mode: str = ckpt.get("mode", "referenced")
    size: int = int(ckpt.get("size", 256))
    base: int = int(ckpt.get("base", 32))
    use_svgv: bool = bool(ckpt.get("use_svgv", False))

    in_ch = 1 if mode == "universal" else 4
    model = CalligraGuardLite(in_ch=in_ch, num_types=len(defect_vocab), max_path_id=max_path_id,
                              base=base, use_svgv=use_svgv)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    def _load_gray(rel: str) -> np.ndarray:
        return np.array(Image.open(root / rel).convert("L"))

    def _svgv_path(rel_svg: str, clean: bool) -> Path:
        rel = Path(rel_svg)
        base = rel.with_suffix(".png").name
        subdir = "clean_svgv" if clean else "svgv"
        return root / subdir / rel.parent.name / base

    out_rows: List[Dict[str, Any]] = []
    for r in tqdm(recs, desc="predict"):
        img0 = _load_gray(r["image_path"])
        h0, w0 = img0.shape[:2]
        img = _resize(img0, size, is_mask=False).astype(np.float32)/255.0

        # build channels
        chs: List[np.ndarray] = []
        if mode == "universal":
            chs.append(img)
        else:
            ref0 = _load_gray(r["clean_image_path"])
            ref = _resize(ref0, size, is_mask=False).astype(np.float32)/255.0
            chs.append(img); chs.append(ref); chs.append(np.abs(img-ref)); chs.append(img-ref)
        x = np.stack(chs, axis=0)
        x_t = torch.from_numpy(x).unsqueeze(0).to(args.device)

        svgv_t: Optional[torch.Tensor] = None
        if use_svgv:
            p = _svgv_path(r["svg_path"], clean=False)
            if p.exists():
                s0 = np.array(Image.open(p).convert("L"))
                s = s0.astype(np.float32)/255.0
            else:
                # fallback: downsample input image
                s = np.array(Image.fromarray(img0).resize((64,64), resample=Image.BILINEAR)).astype(np.float32)/255.0
            svgv_t = torch.from_numpy(s).unsqueeze(0).unsqueeze(0).to(args.device)  # (1,1,S,S)

        with torch.no_grad():
            out = model(x_t, svgv=svgv_t)
            mask_prob = torch.sigmoid(out.mask_logits)[0,0].detach().cpu().numpy()
            # resize mask prob back to original
            mask_prob0 = _resize((mask_prob*255).astype(np.uint8), h0, is_mask=False).astype(np.float32)/255.0
            mask_bin0 = (mask_prob0 >= 0.5).astype(np.uint8)
            mask_rle = rle_encode(mask_bin0)

            # score
            if args.score_mode == "maskmax":
                score = float(mask_prob.max())
            else:
                type_prob = torch.softmax(out.type_logits, dim=-1)[0].detach().cpu().numpy()
                score = float(1.0 - type_prob[0])  # 1 - P(clean)

            # types: top-k excluding clean
            type_prob = torch.softmax(out.type_logits, dim=-1)[0].detach().cpu().numpy()
            # indices 1.. end
            idxs = np.argsort(-type_prob[1:])[:args.type_k]
            pred_types = [defect_vocab[1+i] for i in idxs.tolist() if (1+i) < len(defect_vocab)]

            # paths: top-k
            path_prob = torch.sigmoid(out.path_logits)[0].detach().cpu().numpy()
            pid_idxs = np.argsort(-path_prob)[:args.path_k]
            pred_paths = [int(i) for i in pid_idxs.tolist()]

        out_rows.append({
            "id": r["id"],
            "score": score,
            "mask_rle": mask_rle,
            "types": pred_types,
            "path_ids": pred_paths,
        })

    out_path = Path(args.out_pred)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, out_rows)
    print("[calligraguard_predict] wrote", out_path)

if __name__ == "__main__":
    main()
