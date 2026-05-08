from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from tpgr.config import ensure_dir, load_config, save_config
from tpgr.logging_utils import setup_logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def save_json(data: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_runtime(num_threads: int = 1) -> None:
    try:
        import cv2
        cv2.setNumThreads(max(1, int(num_threads)))
    except Exception:
        pass
    torch.set_num_threads(max(1, int(num_threads)))
    try:
        torch.set_num_interop_threads(max(1, int(num_threads)))
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
