from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple, List

def rle_encode(mask: np.ndarray) -> Dict[str, Any]:
    """
    Simple COCO-style RLE for 2D binary mask.
    Returns dict with 'size' and 'counts' (list of run lengths).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Flatten in column-major (Fortran) order like COCO
    pixels = mask.T.flatten()
    # Ensure starts with 0 run
    counts: List[int] = []
    prev = 0
    run_len = 0
    for p in pixels:
        if p == prev:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            prev = p
    counts.append(run_len)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}

def rle_decode(rle: Dict[str, Any]) -> np.ndarray:
    h, w = rle["size"]
    counts = rle["counts"]
    # Reconstruct in column-major order
    arr = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for run in counts:
        if run > 0:
            arr[idx: idx + run] = val
            idx += run
            val = 1 - val
    mask = arr.reshape((w, h)).T  # back to (h,w)
    return mask

def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)
