from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

_CMD_RE = re.compile(r"[MmLlHhVvCcQqZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Number of parameters per primitive segment for each command
_PARAM_COUNTS = {
    "M": 2, "m": 2,
    "L": 2, "l": 2,
    "H": 1, "h": 1,
    "V": 1, "v": 1,
    "C": 6, "c": 6,
    "Q": 4, "q": 4,
    "Z": 0, "z": 0,
}

@dataclass
class Segment:
    cmd: str  # 'M','L','C','Q','Z'
    params: Tuple[float, ...]  # absolute coordinates

def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        # fallback: strip commas
        return float(x.replace(",", ""))

def parse_path_d(d: str) -> List[Segment]:
    """
    Parse an SVG path 'd' string into a list of absolute segments.
    Supports M/L/H/V/C/Q/Z and their relative variants.
    Converts H/V into L. Converts repeated coordinate pairs appropriately.
    """
    tokens = _CMD_RE.findall(d.replace(",", " "))
    if not tokens:
        return []

    segs: List[Segment] = []
    i = 0
    cmd = None
    cx, cy = 0.0, 0.0
    sx, sy = 0.0, 0.0  # current subpath start

    def next_numbers(n: int) -> List[float]:
        nonlocal i
        nums = []
        for _ in range(n):
            if i >= len(tokens):
                raise ValueError("Unexpected end of path data")
            t = tokens[i]
            if re.match(r"[A-Za-z]", t):
                raise ValueError(f"Expected number, got command {t}")
            nums.append(_to_float(t))
            i += 1
        return nums

    while i < len(tokens):
        t = tokens[i]
        if re.match(r"[A-Za-z]", t):
            cmd = t
            i += 1
        elif cmd is None:
            raise ValueError("Path data must start with a command")
        # Now parse segments for this cmd until next command
        assert cmd is not None
        pc = _PARAM_COUNTS[cmd]
        if cmd in "Zz":
            segs.append(Segment("Z", tuple()))
            cx, cy = sx, sy
            cmd = None
            continue

        # For commands with params, consume in chunks of pc until a new command appears
        # Special case: M/m first pair is moveto, subsequent pairs are treated as L/l
        is_move = cmd in "Mm"
        first_move = True

        while True:
            # Stop if next token is a command (but allow for implicit repeats)
            if i >= len(tokens):
                break
            if re.match(r"[A-Za-z]", tokens[i]):
                break
            nums = next_numbers(pc)

            if cmd in "Hh":
                x = nums[0]
                if cmd == "h":
                    x = cx + x
                segs.append(Segment("L", (x, cy)))
                cx, cy = x, cy
            elif cmd in "Vv":
                y = nums[0]
                if cmd == "v":
                    y = cy + y
                segs.append(Segment("L", (cx, y)))
                cx, cy = cx, y
            elif cmd in "Mm":
                x, y = nums
                if cmd == "m":
                    x, y = cx + x, cy + y
                if first_move:
                    segs.append(Segment("M", (x, y)))
                    sx, sy = x, y
                    first_move = False
                else:
                    # Subsequent moveto pairs are lineto per SVG spec
                    segs.append(Segment("L", (x, y)))
                cx, cy = x, y
            elif cmd in "Ll":
                x, y = nums
                if cmd == "l":
                    x, y = cx + x, cy + y
                segs.append(Segment("L", (x, y)))
                cx, cy = x, y
            elif cmd in "Cc":
                x1, y1, x2, y2, x, y = nums
                if cmd == "c":
                    x1, y1, x2, y2, x, y = x1+cx, y1+cy, x2+cx, y2+cy, x+cx, y+cy
                segs.append(Segment("C", (x1, y1, x2, y2, x, y)))
                cx, cy = x, y
            elif cmd in "Qq":
                x1, y1, x, y = nums
                if cmd == "q":
                    x1, y1, x, y = x1+cx, y1+cy, x+cx, y+cy
                segs.append(Segment("Q", (x1, y1, x, y)))
                cx, cy = x, y
            else:
                raise ValueError(f"Unsupported command {cmd}")

        cmd = None  # require explicit command next, per spec; implicit repeats handled above by while loop

    return segs

def segments_to_d(segs: List[Segment], precision: int = 3) -> str:
    """
    Serialize segments back to a compact SVG path string (absolute).
    """
    fmt = f"{{:.{precision}f}}"
    parts: List[str] = []
    for s in segs:
        parts.append(s.cmd)
        if s.params:
            parts.extend(fmt.format(v) for v in s.params)
    return " ".join(parts)

def bbox_of_segments(segs: List[Segment]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    cx, cy = 0.0, 0.0
    sx, sy = 0.0, 0.0
    for s in segs:
        if s.cmd == "M":
            cx, cy = s.params
            sx, sy = cx, cy
            xs.append(cx); ys.append(cy)
        elif s.cmd == "L":
            cx, cy = s.params
            xs.append(cx); ys.append(cy)
        elif s.cmd == "C":
            x1,y1,x2,y2,x,y = s.params
            xs.extend([x1,x2,x]); ys.extend([y1,y2,y])
            cx, cy = x, y
        elif s.cmd == "Q":
            x1,y1,x,y = s.params
            xs.extend([x1,x]); ys.extend([y1,y])
            cx, cy = x, y
        elif s.cmd == "Z":
            cx, cy = sx, sy
            xs.append(cx); ys.append(cy)
    if not xs:
        return (0.0,0.0,0.0,0.0)
    return (min(xs), min(ys), max(xs), max(ys))
