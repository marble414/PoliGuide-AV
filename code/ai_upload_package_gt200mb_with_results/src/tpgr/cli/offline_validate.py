from __future__ import annotations

import argparse
from pathlib import Path

from tpgr.cli.common import configure_runtime
from tpgr.config import load_config
from tpgr.metrics.sequence import command_switch_delay, sequence_edit_similarity, stable_output_latency
from tpgr.pipeline.system import TrafficPoliceGestureSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="离线验证脚本（基于预计算检测/姿态序列）")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-jsonl", required=True, help="由 infer_video.py 输出的 jsonl")
    args = parser.parse_args()

    import json

    preds = []
    gts = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            preds.append(item["command"])
            if "gt_command" in item:
                gts.append(item["gt_command"])

    if not gts:
        print({"message": "输入 jsonl 中无 gt_command，已完成结构校验但无法计算时序指标。", "frames": len(preds)})
        return

    summary = {
        "frames": len(preds),
        "sequence_edit_similarity": sequence_edit_similarity(preds, gts),
        "command_switch_delay_frames": command_switch_delay(gts, preds),
        "stable_output_latency_frames": stable_output_latency(gts, preds),
    }
    print(summary)


if __name__ == "__main__":
    main()
