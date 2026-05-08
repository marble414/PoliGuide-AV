from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    records: list[dict] = []
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fp:
        for item in records:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def filter_by_split(records: Iterable[dict], split: str | None = None) -> list[dict]:
    if split is None or str(split).lower() in {"", "all", "*"}:
        return list(records)
    wanted = str(split).lower()
    accepted = {"train", "val"} if wanted in {"trainval", "train_val", "train+val"} else {wanted}
    return [item for item in records if str(item.get("split", "train")).lower() in accepted]


def resolve_path(path: str | Path, manifest_path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (Path(manifest_path).resolve().parent / p).resolve()

