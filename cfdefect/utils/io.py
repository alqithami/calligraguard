from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

JsonDict = Dict[str, Any]

def read_jsonl(path: Union[str, Path]) -> Iterator[JsonDict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: Union[str, Path], records: Iterable[JsonDict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_json(path: Union[str, Path]) -> JsonDict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Union[str, Path], obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
