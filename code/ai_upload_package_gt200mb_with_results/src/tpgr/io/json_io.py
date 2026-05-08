from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w", encoding="utf-8")

    def write(self, item: Dict[str, Any]) -> None:
        self.fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.fp.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
