from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from .calligraguard_model import CalligraGuardLite, dice_loss_from_logits
from .calligraguard_data import CFDefectCalligraDataset, scan_meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Dataset root containing meta.jsonl")
    ap.add_argument("--out_dir", type=str, required=True, help="Output run directory")
    ap.add_argument("--mode", type=str, default="referenced", choices=["universal","referenced"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--use_svgv", action="store_true", help="Enable SVG-V FiLM conditioning (requires precomputed svgv/)")
    ap.add_argument("--max_path_cap", type=int, default=512, help="Maximum number of path ids to model (clip higher ids).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = root / "meta.jsonl"
    index = scan_meta(meta_path, max_path_cap=args.max_path_cap)

    in_ch = 1 if args.mode == "universal" else 4
    model = CalligraGuardLite(
        in_ch=in_ch,
        num_types=len(index.defect_vocab),
        max_path_id=index.max_path_id,
        base=args.base,
        use_svgv=bool(args.use_svgv),
    ).to(args.device)

    ds = CFDefectCalligraDataset(root=root, index=index, mode=args.mode, size=args.size, use_svgv=bool(args.use_svgv))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Class weights: encourage not predicting "clean" for defective samples
    # (optional; keep simple here)
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}")
        running = 0.0
        for x, svgv, y_mask, y_type, y_paths, _, _, _ in pbar:
            x = x.to(args.device)
            y_mask = y_mask.to(args.device)
            y_type = y_type.to(args.device)
            y_paths = y_paths.to(args.device)
            svgv = svgv.to(args.device) if (svgv is not None and args.use_svgv) else None

            out = model(x, svgv=svgv)

            # Segmentation losses
            bce = F.binary_cross_entropy_with_logits(out.mask_logits, y_mask)
            dloss = dice_loss_from_logits(out.mask_logits, y_mask)
            seg_loss = bce + dloss

            # Type classification
            type_loss = F.cross_entropy(out.type_logits, y_type)

            # Path attribution (multi-label)
            path_loss = F.binary_cross_entropy_with_logits(out.path_logits, y_paths)

            loss = 1.0*seg_loss + 0.5*type_loss + 0.2*path_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            running = 0.9*running + 0.1*float(loss.item()) if running > 0 else float(loss.item())
            pbar.set_postfix(loss=f"{running:.4f}")

    # Save
    ckpt = {
        "state_dict": model.state_dict(),
        "mode": args.mode,
        "size": int(args.size),
        "base": int(args.base),
        "use_svgv": bool(args.use_svgv),
        "defect_vocab": index.defect_vocab,
        "max_path_id": int(index.max_path_id),
    }
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "train_info.json").write_text(json.dumps({
        "mode": args.mode,
        "size": int(args.size),
        "base": int(args.base),
        "use_svgv": bool(args.use_svgv),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "defect_vocab": index.defect_vocab,
        "max_path_id": int(index.max_path_id),
    }, indent=2), encoding="utf-8")

    print("[calligraguard_train] wrote", out_dir / "model.pt")
    print("[calligraguard_train] wrote", out_dir / "train_info.json")

if __name__ == "__main__":
    main()
