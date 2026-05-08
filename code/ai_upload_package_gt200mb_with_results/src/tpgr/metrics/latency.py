from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Generator, List


class LatencyProfiler:
    def __init__(self) -> None:
        self.values_ms: List[float] = []

    @contextmanager
    def track(self) -> Generator[None, None, None]:
        tic = time.perf_counter()
        yield
        toc = time.perf_counter()
        self.values_ms.append((toc - tic) * 1000.0)

    def summary(self) -> dict:
        if not self.values_ms:
            return {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
        values = sorted(self.values_ms)
        p95_idx = int(0.95 * (len(values) - 1))
        mean_ms = statistics.mean(values)
        return {
            "count": len(values),
            "mean_ms": mean_ms,
            "p95_ms": values[p95_idx],
            "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        }
