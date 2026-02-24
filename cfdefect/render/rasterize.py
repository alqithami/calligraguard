from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple, Optional
import io
import numpy as np
from PIL import Image
import cairosvg

PathLike = Union[str, Path]

def svg_file_to_png_bytes(svg_path: PathLike, out_w: int, out_h: int, background: str = "white") -> bytes:
    svg_path = str(svg_path)
    # cairosvg supports background_color
    return cairosvg.svg2png(url=svg_path, output_width=out_w, output_height=out_h, background_color=background)

def png_bytes_to_array(png_bytes: bytes, mode: str = "L") -> np.ndarray:
    img = Image.open(io.BytesIO(png_bytes)).convert(mode)
    return np.array(img)

def render_svg_to_array(svg_path: PathLike, size: int, background: str = "white", mode: str = "L") -> np.ndarray:
    """
    Render SVG to a square numpy array of shape (size,size) in uint8.
    """
    png = svg_file_to_png_bytes(svg_path, out_w=size, out_h=size, background=background)
    arr = png_bytes_to_array(png, mode=mode)
    return arr

def compute_diff_mask(clean: np.ndarray, corrupted: np.ndarray, thresh: int = 16) -> np.ndarray:
    """
    Compute a binary mask of changed pixels between clean and corrupted raster images.
    Assumes uint8 grayscale. Returns uint8 mask with values {0,1}.
    """
    if clean.shape != corrupted.shape:
        raise ValueError("clean and corrupted must have same shape")
    diff = np.abs(clean.astype(np.int16) - corrupted.astype(np.int16)).astype(np.int16)
    mask = (diff >= thresh).astype(np.uint8)
    return mask

def refine_mask(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Morphological cleanup (remove small components).
    """
    import cv2
    m = (mask > 0).astype(np.uint8)
    # close then open
    kernel = np.ones((3,3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    # connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out.astype(np.uint8)
