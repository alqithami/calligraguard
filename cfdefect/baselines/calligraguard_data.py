from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from ..utils.io import read_jsonl

def _resize_u8(arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    im = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    im = im.resize((size, size), resample=resample)
    return np.array(im)

def _resize_gray(arr_u8: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(arr_u8)
    im = im.resize((size, size), resample=Image.BILINEAR)
    return np.array(im)

@dataclass
class MetaIndex:
    defect_vocab: List[str]          # includes "clean" at index 0
    defect_to_id: Dict[str, int]
    max_path_id: int

def scan_meta(meta_path: Path, max_path_cap: int = 1024) -> MetaIndex:
    """Scan meta.jsonl to build a defect vocabulary and max path id."""
    defect_types: set[str] = set()
    max_pid = 0
    for r in read_jsonl(meta_path):
        for d in r.get("defects", []) or []:
            t = d.get("type")
            if t:
                defect_types.add(str(t))
            for pid in (d.get("path_ids", []) or []):
                try:
                    max_pid = max(max_pid, int(pid))
                except Exception:
                    continue
    vocab = ["clean"] + sorted(defect_types)
    defect_to_id = {t:i for i,t in enumerate(vocab)}
    max_path_id = min(max_pid + 1, int(max_path_cap))
    return MetaIndex(defect_vocab=vocab, defect_to_id=defect_to_id, max_path_id=max_path_id)

class CFDefectCalligraDataset(Dataset):
    """
    Batch-safe dataset for CalligraGuard-Lite.

    Returns (8 items, all batch-collatable):
      x: (C,H,W) float32 in [0,1]
      svgv: (1,S,S) float32 in [0,1]  (ALWAYS a tensor; dummy zeros if use_svgv=False)
      y_mask: (1,H,W) float32 {0,1}
      y_type: int64 (0=clean)
      y_paths: (P,) float32 multi-hot (P=max_path_id)
      sample_id: str
      orig_hw: (H0,W0) tuple[int,int]
      dummy: int  (placeholder to avoid returning raw dicts)
    """
    def __init__(self,
                 root: Path,
                 index: MetaIndex,
                 mode: str = "universal",
                 size: int = 256,
                 use_svgv: bool = True,
                 svgv_size: int = 64,
                 svgv_dirname: str = "svgv",
                 clean_svgv_dirname: str = "clean_svgv"):
        self.root = Path(root)
        self.index = index
        self.mode = str(mode)
        assert self.mode in ("universal", "referenced"), "mode must be universal or referenced"
        self.size = int(size)
        self.use_svgv = bool(use_svgv)
        self.svgv_size = int(svgv_size)
        self.svgv_dirname = str(svgv_dirname)
        self.clean_svgv_dirname = str(clean_svgv_dirname)

        self.recs: List[Dict[str, Any]] = list(read_jsonl(self.root / "meta.jsonl"))

    def __len__(self) -> int:
        return len(self.recs)

    def _load_gray(self, rel_path: str) -> np.ndarray:
        p = Path(rel_path)
        # if rel_path is absolute, Path join preserves it; if relative, it is under root
        full = p if p.is_absolute() else (self.root / p)
        return np.array(Image.open(full).convert("L"))

    def _svgv_path(self, rel_svg: str, clean: bool) -> Path:
        rel = Path(rel_svg)
        base = rel.with_suffix(".png").name
        subdir = self.clean_svgv_dirname if clean else self.svgv_dirname
        # Use parent folder name as "font id" directory (works for svg/... and clean_svg/...)
        return self.root / subdir / rel.parent.name / base

    def _load_svgv_tensor(self, r: Dict[str, Any], img0_u8: np.ndarray) -> torch.Tensor:
        # Always return a tensor so DataLoader can collate batches, even when conditioning is disabled.
        if not self.use_svgv:
            return torch.zeros((1, self.svgv_size, self.svgv_size), dtype=torch.float32)

        # Candidate PNGs to try (robust to whether svg_path points to svg/ or clean_svg/)
        candidates: List[Path] = []
        for key, clean_flag in [
            ("svg_path", False),
            ("svg_path", True),
            ("clean_svg_path", True),
            ("clean_svg_path", False),
        ]:
            v = r.get(key)
            if v:
                candidates.append(self._svgv_path(str(v), clean=clean_flag))

        arr_u8: Optional[np.ndarray] = None
        for p in candidates:
            if p.exists():
                arr_u8 = np.array(Image.open(p).convert("L"))
                break

        if arr_u8 is None:
            # fallback: downsample the raster image
            arr_u8 = _resize_gray(img0_u8, self.svgv_size)

        # Ensure correct size
        if arr_u8.shape[0] != self.svgv_size or arr_u8.shape[1] != self.svgv_size:
            arr_u8 = _resize_gray(arr_u8, self.svgv_size)

        svgv = arr_u8.astype(np.float32) / 255.0
        return torch.from_numpy(svgv).unsqueeze(0)  # (1,S,S)

    def __getitem__(self, idx: int):
        r = self.recs[idx]
        img0 = self._load_gray(r["image_path"])
        mask0 = self._load_gray(r["mask_path"])

        h0, w0 = img0.shape[:2]

        # Resize to model size
        img = _resize_u8(img0, self.size, is_mask=False).astype(np.float32) / 255.0
        mask = _resize_u8(mask0, self.size, is_mask=True)
        mask = (mask > 0).astype(np.float32)

        # Build input channels
        chs: List[np.ndarray] = []
        if self.mode == "universal":
            chs.append(img)
        else:
            ref0 = self._load_gray(r["clean_image_path"])
            ref = _resize_u8(ref0, self.size, is_mask=False).astype(np.float32) / 255.0
            chs.append(img)
            chs.append(ref)
            chs.append(np.abs(img - ref))
            chs.append(img - ref)

        x = np.stack(chs, axis=0)  # (C,H,W)
        x_t = torch.from_numpy(x)

        # SVG-V conditioning tensor (always collatable)
        svgv_t = self._load_svgv_tensor(r, img0)

        # Targets
        is_def = bool(r.get("is_defective", False))
        if not is_def:
            y_type = 0
            y_paths = np.zeros((self.index.max_path_id,), dtype=np.float32)
        else:
            defects = r.get("defects", []) or []
            t = (defects[0].get("type") if defects else None) or "clean"
            y_type = self.index.defect_to_id.get(str(t), 0)
            y_paths = np.zeros((self.index.max_path_id,), dtype=np.float32)
            for d in defects:
                for pid in (d.get("path_ids", []) or []):
                    try:
                        pid = int(pid)
                    except Exception:
                        continue
                    if 0 <= pid < self.index.max_path_id:
                        y_paths[pid] = 1.0

        y_mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)
        y_type_t = torch.tensor(y_type, dtype=torch.long)
        y_paths_t = torch.from_numpy(y_paths)

        # IMPORTANT: last element must NOT be the raw dict (breaks default_collate)
        return x_t, svgv_t, y_mask_t, y_type_t, y_paths_t, r["id"], (h0, w0), 0
