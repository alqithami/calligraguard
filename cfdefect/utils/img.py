from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image

PathLike = Union[str, Path]

def read_image(path: PathLike, mode: str = "L") -> np.ndarray:
    """Read image as numpy array. mode='L' (grayscale) by default."""
    img = Image.open(path).convert(mode)
    return np.array(img)

def write_image(path: PathLike, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask
