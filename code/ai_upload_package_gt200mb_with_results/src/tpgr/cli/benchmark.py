from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tpgr.cli.common import configure_runtime, resolve_device
from tpgr.config import load_config
from tpgr.metrics.latency import LatencyProfiler
from tpgr.models.builder import build_model
from tpgr.pipeline.system import TrafficPoliceGestureSystem


def benchmark_model(cfg: dict, iterations: int = 200) -> dict:
    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(cfg["model"]).to(device).eval()
    if cfg["dataset"].get("representation", "features") == "features":
        x = torch.randn(1, cfg["dataset"]["clip_len"], cfg["model"]["input_dim"], device=device)
    else:
        x = torch.randn(
            1,
            cfg["model"].get("in_channels", 3),
            cfg["dataset"]["clip_len"],
            cfg["model"].get("num_nodes", 17),
            device=device,
        )
    prof = LatencyProfiler()
    with torch.no_grad():
        for _ in range(iterations):
            with prof.track():
                _ = model(x)
    return prof.summary()


def benchmark_system(cfg: dict, iterations: int = 200, width: int = 640, height: int = 480) -> dict:
    system = TrafficPoliceGestureSystem(cfg)
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    prof = LatencyProfiler()
    for i in range(iterations):
        with prof.track():
            _ = system.process_frame(dummy_frame, fps=25.0)
    return prof.summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="基准测试")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mode", choices=["model", "system"], default="model")
    args = parser.parse_args()

    configure_runtime(1)
    cfg = load_config(args.config)
    summary = benchmark_model(cfg, args.iterations) if args.mode == "model" else benchmark_system(cfg, args.iterations)
    print(summary)


if __name__ == "__main__":
    main()
