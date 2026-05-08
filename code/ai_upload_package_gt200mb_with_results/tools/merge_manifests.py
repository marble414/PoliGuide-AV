#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from tpgr.data.manifests import read_jsonl, resolve_path, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="合并多个 manifest，并为 sample_id 增加来源前缀")
    parser.add_argument("--inputs", nargs="+", required=True, help="多个 jsonl manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prefixes",
        nargs="*",
        default=None,
        help="可选，与 inputs 一一对应；若不提供则使用 m0/m1/... 前缀",
    )
    args = parser.parse_args()
    output_path = Path(args.output).resolve()
    output_dir = output_path.parent

    prefixes = args.prefixes or [f"m{i}" for i in range(len(args.inputs))]
    if len(prefixes) != len(args.inputs):
        raise ValueError("prefixes 数量必须与 inputs 数量一致")

    merged = []
    for prefix, path in zip(prefixes, args.inputs):
        source_manifest = Path(path).resolve()
        for item in read_jsonl(path):
            record = dict(item)
            sample_id = str(record.get("sample_id", "sample"))
            record["sample_id"] = f"{prefix}_{sample_id}"
            for key in ["video_path", "keypoints_path", "feature_path"]:
                if key in record:
                    resolved = resolve_path(record[key], source_manifest)
                    record[key] = os.path.relpath(resolved, output_dir)
            attrs = dict(record.get("attributes", {}))
            attrs.setdefault("source_manifest", path)
            attrs.setdefault("source_prefix", prefix)
            record["attributes"] = attrs
            merged.append(record)

    write_jsonl(merged, args.output)
    print({"samples": len(merged), "output": args.output})


if __name__ == "__main__":
    main()
