from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .parse import parse_path_d, segments_to_d, bbox_of_segments, Segment
from .xml import load_svg_paths, set_svg_path_d, translate_svg_path

@dataclass
class InjectResult:
    defect_type: str
    severity: float
    path_indices: List[int]
    note: str = ""

def _bbox_area(b: Tuple[float,float,float,float]) -> float:
    x1,y1,x2,y2 = b
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def find_diacritic_like_paths(paths_d: List[str]) -> List[int]:
    """
    Heuristic: identify small paths likely to be dots/diacritics by bbox size and position.
    Works best when SVG uses separate paths for dots/marks.
    """
    bbs = []
    for d in paths_d:
        segs = parse_path_d(d) if d else []
        bbs.append(bbox_of_segments(segs))
    # global bbox
    gx1 = min(b[0] for b in bbs) if bbs else 0.0
    gy1 = min(b[1] for b in bbs) if bbs else 0.0
    gx2 = max(b[2] for b in bbs) if bbs else 0.0
    gy2 = max(b[3] for b in bbs) if bbs else 0.0
    garea = _bbox_area((gx1,gy1,gx2,gy2)) + 1e-9
    gh = max(1e-9, gy2-gy1)
    gyc = (gy1+gy2)/2.0

    candidates = []
    for i,b in enumerate(bbs):
        area = _bbox_area(b)
        bw = b[2]-b[0]
        bh = b[3]-b[1]
        yc = (b[1]+b[3])/2.0
        # small area + small height + located away from center
        if area < 0.08 * garea and bh < 0.35 * gh and abs(yc - gyc) > 0.12 * gh:
            candidates.append(i)
    return candidates

def _choose_nonempty_path(paths_d: List[str], exclude: Optional[List[int]] = None) -> int:
    exclude = exclude or []
    indices = [i for i,d in enumerate(paths_d) if d and i not in exclude]
    if not indices:
        # fallback
        return random.randrange(len(paths_d))
    return random.choice(indices)

def inject_missing_diacritic(tree, paths_d: List[str], severity: float) -> InjectResult:
    cand = find_diacritic_like_paths(paths_d)
    if not cand:
        # fallback: remove any small-ish path
        idx = _choose_nonempty_path(paths_d)
    else:
        idx = random.choice(cand)
    set_svg_path_d(tree, idx, "")  # remove geometry
    return InjectResult(defect_type="missing_diacritic", severity=severity, path_indices=[idx],
                        note="Removed one diacritic-like path")

def inject_misplaced_diacritic(tree, paths_d: List[str], severity: float) -> InjectResult:
    cand = find_diacritic_like_paths(paths_d)
    if not cand:
        idx = _choose_nonempty_path(paths_d)
    else:
        idx = random.choice(cand)
    # translate proportional to severity; direction random
    dx = float(np.sign(random.uniform(-1,1)) * severity * random.uniform(5.0, 25.0))
    dy = float(np.sign(random.uniform(-1,1)) * severity * random.uniform(5.0, 25.0))
    translate_svg_path(tree, idx, dx, dy)
    return InjectResult(defect_type="misplaced_diacritic", severity=severity, path_indices=[idx],
                        note=f"Translated path by ({dx:.1f},{dy:.1f})")

def inject_spur(tree, paths_d: List[str], severity: float) -> InjectResult:
    # choose a non-diacritic path
    cand_diac = set(find_diacritic_like_paths(paths_d))
    idx = _choose_nonempty_path(paths_d, exclude=list(cand_diac))
    d = paths_d[idx]
    segs = parse_path_d(d)
    # choose an L segment point to attach spur; fallback to last point
    l_indices = [i for i,s in enumerate(segs) if s.cmd == "L"]
    if l_indices:
        si = random.choice(l_indices)
    else:
        si = max(0, len(segs)-1)
    # get current point at that segment end
    # We insert after segment si
    # Determine point p = end of segs[si] if it has endpoint
    p = None
    if segs[si].cmd in ("M","L"):
        x,y = segs[si].params
        p = (x,y)
    elif segs[si].cmd == "C":
        x1,y1,x2,y2,x,y = segs[si].params
        p = (x,y)
    elif segs[si].cmd == "Q":
        x1,y1,x,y = segs[si].params
        p = (x,y)
    else:
        # find next point
        for j in range(si, -1, -1):
            if segs[j].cmd in ("M","L"):
                p = segs[j].params
                break
    if p is None:
        return InjectResult(defect_type="spur", severity=severity, path_indices=[idx], note="No point to spur")

    x,y = p
    # Spur parameters
    length = severity * random.uniform(8.0, 40.0)
    angle = random.uniform(0, 2*np.pi)
    sx = x + length * float(np.cos(angle))
    sy = y + length * float(np.sin(angle))
    spur1 = Segment("L", (sx, sy))
    spur2 = Segment("L", (x, y))
    # Insert after si
    segs2 = segs[:si+1] + [spur1, spur2] + segs[si+1:]
    new_d = segments_to_d(segs2, precision=3)
    set_svg_path_d(tree, idx, new_d)
    return InjectResult(defect_type="spur", severity=severity, path_indices=[idx],
                        note=f"Inserted spur length~{length:.1f}")

def inject_gap(tree, paths_d: List[str], severity: float) -> InjectResult:
    cand_diac = set(find_diacritic_like_paths(paths_d))
    idx = _choose_nonempty_path(paths_d, exclude=list(cand_diac))
    d = paths_d[idx]
    segs = parse_path_d(d)
    # Choose an L segment to break
    l_indices = [i for i,s in enumerate(segs) if s.cmd == "L"]
    if not l_indices:
        # fallback: do nothing
        return InjectResult(defect_type="gap", severity=severity, path_indices=[idx], note="No L segment found")
    si = random.choice(l_indices)
    # Replace that L with M at the same endpoint (break contour)
    x,y = segs[si].params
    segs[si] = Segment("M", (x,y))
    new_d = segments_to_d(segs, precision=3)
    set_svg_path_d(tree, idx, new_d)
    return InjectResult(defect_type="gap", severity=severity, path_indices=[idx], note="Broke contour by L->M")

def inject_jitter(tree, paths_d: List[str], severity: float) -> InjectResult:
    idx = _choose_nonempty_path(paths_d)
    d = paths_d[idx]
    segs = parse_path_d(d)
    # jitter all control points slightly; endpoints less
    amp = severity * random.uniform(0.5, 3.0)  # in SVG units
    segs2: List[Segment] = []
    for s in segs:
        if s.cmd in ("M","L"):
            x,y = s.params
            jx = x + random.uniform(-amp, amp) * 0.2
            jy = y + random.uniform(-amp, amp) * 0.2
            segs2.append(Segment(s.cmd, (jx,jy)))
        elif s.cmd == "C":
            x1,y1,x2,y2,x,y = s.params
            x1 += random.uniform(-amp, amp)
            y1 += random.uniform(-amp, amp)
            x2 += random.uniform(-amp, amp)
            y2 += random.uniform(-amp, amp)
            x  += random.uniform(-amp, amp) * 0.2
            y  += random.uniform(-amp, amp) * 0.2
            segs2.append(Segment("C", (x1,y1,x2,y2,x,y)))
        elif s.cmd == "Q":
            x1,y1,x,y = s.params
            x1 += random.uniform(-amp, amp)
            y1 += random.uniform(-amp, amp)
            x  += random.uniform(-amp, amp) * 0.2
            y  += random.uniform(-amp, amp) * 0.2
            segs2.append(Segment("Q", (x1,y1,x,y)))
        else:
            segs2.append(s)
    new_d = segments_to_d(segs2, precision=3)
    set_svg_path_d(tree, idx, new_d)
    return InjectResult(defect_type="jitter", severity=severity, path_indices=[idx], note=f"Jitter amp~{amp:.2f}")

def inject_defect(tree, paths_d: List[str], defect_type: str, severity: float) -> InjectResult:
    """
    Apply a defect to the SVG tree and return metadata.
    defect_type in {'missing_diacritic','misplaced_diacritic','spur','gap','jitter'}.
    """
    if defect_type == "missing_diacritic":
        return inject_missing_diacritic(tree, paths_d, severity)
    if defect_type == "misplaced_diacritic":
        return inject_misplaced_diacritic(tree, paths_d, severity)
    if defect_type == "spur":
        return inject_spur(tree, paths_d, severity)
    if defect_type == "gap":
        return inject_gap(tree, paths_d, severity)
    if defect_type == "jitter":
        return inject_jitter(tree, paths_d, severity)
    raise ValueError(f"Unknown defect_type: {defect_type}")
