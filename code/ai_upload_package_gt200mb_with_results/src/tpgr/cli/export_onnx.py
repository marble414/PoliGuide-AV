from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tpgr.cli.common import configure_runtime, resolve_device
from tpgr.config import load_config
from tpgr.models.builder import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 ONNX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise RuntimeError("未安装 onnx，请先安装 requirements-optional.txt") from e

    configure_runtime(1)
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "cpu"))
    model = build_model(cfg["model"]).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    if cfg["dataset"].get("representation", "features") == "features":
        dummy = torch.randn(1, cfg["dataset"]["clip_len"], cfg["model"]["input_dim"], device=device)
        dynamic_axes = {"input": {0: "batch", 1: "time"}, "logits": {0: "batch"}}
    else:
        dummy = torch.randn(1, cfg["model"].get("in_channels", 3), cfg["dataset"]["clip_len"], cfg["model"].get("num_nodes", 17), device=device)
        dynamic_axes = {"input": {0: "batch", 2: "time"}, "logits": {0: "batch"}}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    print({"onnx_path": args.output})


if __name__ == "__main__":
    main()
