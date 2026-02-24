from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

from fontTools.ttLib import TTFont, TTCollection
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen


def parse_chars_file(path: Path) -> List[int]:
    """
    Accepts:
      - lines like "U+0628"
      - literal characters (single char per line)
      - hex codepoint like "0628"
    Returns list of codepoints (ints).
    """
    cps: List[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.upper().startswith("U+"):
            cps.append(int(s[2:], 16))
        elif len(s) == 1:
            cps.append(ord(s))
        else:
            cps.append(int(s, 16))
    return cps


def _sanitize(name: str, max_len: int = 80) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)
    name = name.strip("_")
    if not name:
        name = "font"
    return name[:max_len]


def _best_font_name(font: TTFont) -> str:
    # Try common nameIDs: 4=Full name, 1=Family, 6=PostScript name
    try:
        n = font["name"]
        for nid in (4, 1, 6):
            s = n.getDebugName(nid)
            if s:
                return s
    except Exception:
        pass
    return "UnknownFont"


def glyph_to_svg(font: TTFont, glyph_name: str, fill: str = "black") -> str:
    glyph_set = font.getGlyphSet()
    glyph = glyph_set[glyph_name]

    # Bounds in font units
    bpen = BoundsPen(glyph_set)
    glyph.draw(bpen)
    bounds = bpen.bounds  # (xMin, yMin, xMax, yMax) or None
    if bounds is None:
        bounds = (0, 0, 0, 0)
    xMin, yMin, xMax, yMax = bounds

    # Path commands
    pen = SVGPathPen(glyph_set)
    glyph.draw(pen)
    d = pen.getCommands() or ""

    # Padding
    pad = 20
    vb_x = xMin - pad
    vb_y = yMin - pad
    vb_w = (xMax - xMin) + 2 * pad
    vb_h = (yMax - yMin) + 2 * pad

    # Flip Y from font coords (y up) to SVG coords (y down)
    transform = f"translate(0,{yMax}) scale(1,-1)"

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb_x} {vb_y} {vb_w} {vb_h}">
  <path d="{d}" fill="{fill}" transform="{transform}"/>
</svg>
"""
    return svg


def export_ttfont(font: TTFont, font_id: str, out_dir: Path, codepoints: List[int]) -> int:
    cmap = font.getBestCmap() or {}
    exported = 0
    for cp in codepoints:
        gname = cmap.get(cp)
        if not gname:
            continue
        svg = glyph_to_svg(font, gname)
        out_path = out_dir / font_id / f"U+{cp:04X}.svg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(svg, encoding="utf-8")
        exported += 1
    return exported


def export_font_file(font_path: Path, out_dir: Path, codepoints: List[int]) -> int:
    ext = font_path.suffix.lower()

    if ext == ".ttc":
        # TrueType Collection: multiple fonts inside
        ttc = TTCollection(str(font_path))
        total = 0
        for idx, f in enumerate(ttc.fonts):
            subname = _sanitize(_best_font_name(f))
            font_id = _sanitize(f"{font_path.stem}__{idx}__{subname}")
            try:
                n = export_ttfont(f, font_id, out_dir, codepoints)
                total += n
                print(f"[export] {font_path.name} (#{idx} {subname}): {n} glyphs")
            except Exception as e:
                print(f"[export][WARN] {font_path} (#{idx}): {e}")
        return total

    # Regular TTF/OTF
    font = TTFont(str(font_path))
    try:
        font_id = _sanitize(font_path.stem)
        n = export_ttfont(font, font_id, out_dir, codepoints)
        print(f"[export] {font_path.name}: {n} glyphs")
        return n
    finally:
        try:
            font.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--chars_file", type=str, required=True)
    ap.add_argument("--font_glob", type=str, default="*.ttf")
    args = ap.parse_args()

    fonts_dir = Path(args.fonts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cps = parse_chars_file(Path(args.chars_file))
    font_paths = sorted(fonts_dir.glob(args.font_glob))
    if not font_paths:
        print(f"No fonts matched {args.font_glob} in {fonts_dir}")
        return

    total = 0
    for fp in font_paths:
        try:
            total += export_font_file(fp, out_dir, cps)
        except Exception as e:
            print(f"[export][WARN] {fp}: {e}")

    print(f"[export] total exported glyph SVGs: {total}")


if __name__ == "__main__":
    main()
