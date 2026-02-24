from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

SVG_NS = "{http://www.w3.org/2000/svg}"

@dataclass
class SvgPath:
    idx: int
    elem_id: str
    d: str
    style: Dict[str, str]

def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag

def load_svg_paths(svg_path: Path) -> Tuple[ET.ElementTree, ET.Element, List[SvgPath]]:
    """
    Load SVG XML and return (tree, root, list_of_paths).
    Each path gets an integer idx in document order and an elem_id if present.
    """
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    paths: List[SvgPath] = []
    # iterate all <path> regardless of namespace issues
    for i, elem in enumerate(root.iter()):
        if _strip_ns(elem.tag) != "path":
            continue
        d = elem.attrib.get("d", "")
        elem_id = elem.attrib.get("id", f"path_{len(paths)}")
        style_str = elem.attrib.get("style", "")
        style: Dict[str, str] = {}
        if style_str:
            for part in style_str.split(";"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    style[k.strip()] = v.strip()
        paths.append(SvgPath(idx=len(paths), elem_id=elem_id, d=d, style=style))
    return tree, root, paths

def set_svg_path_d(tree: ET.ElementTree, target_idx: int, new_d: str) -> None:
    root = tree.getroot()
    path_elems = [e for e in root.iter() if _strip_ns(e.tag) == "path"]
    if target_idx < 0 or target_idx >= len(path_elems):
        raise IndexError("target_idx out of range")
    path_elems[target_idx].set("d", new_d)

def translate_svg_path(tree: ET.ElementTree, target_idx: int, dx: float, dy: float) -> None:
    """
    Apply a translation by wrapping the path in a transform attribute.
    This is a conservative operation that does not re-write the path commands.
    """
    root = tree.getroot()
    path_elems = [e for e in root.iter() if _strip_ns(e.tag) == "path"]
    if target_idx < 0 or target_idx >= len(path_elems):
        raise IndexError("target_idx out of range")
    e = path_elems[target_idx]
    existing = e.attrib.get("transform", "")
    t = f"translate({dx:.3f},{dy:.3f})"
    if existing:
        e.set("transform", existing + " " + t)
    else:
        e.set("transform", t)

def save_svg(tree: ET.ElementTree, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
