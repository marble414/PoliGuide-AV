from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


def iter_video(path: str | int) -> Generator[tuple[int, np.ndarray, float], None, None]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频源: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 1e-3 else 25.0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield index, frame, fps
        index += 1
    cap.release()


def get_video_writer(path: str | Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {path}")
    return writer
