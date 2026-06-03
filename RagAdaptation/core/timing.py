from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Any

import torch


def sync_cuda() -> None:
    """
    Synchronize CUDA before/after timing blocks.

    This is important because CUDA kernels are usually asynchronous from Python's
    point of view. Without synchronization, wall-clock timing can undercount GPU work.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimingRecorder:
    def __init__(self) -> None:
        self._sections: Dict[str, Dict[str, float | int]] = {}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        sync_cuda()
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            sync_cuda()
            elapsed_sec = (time.perf_counter_ns() - start_ns) / 1e9

            entry = self._sections.setdefault(
                name,
                {"seconds": 0.0, "count": 0},
            )
            entry["seconds"] = float(entry["seconds"]) + elapsed_sec
            entry["count"] = int(entry["count"]) + 1

    def to_dict(self) -> Dict[str, Any]:
        total = sum(float(v["seconds"]) for v in self._sections.values())
        return {
            "total_sec": total,
            "sections": {
                name: {
                    "seconds": float(v["seconds"]),
                    "count": int(v["count"]),
                    "avg_sec": float(v["seconds"]) / max(1, int(v["count"])),
                }
                for name, v in self._sections.items()
            },
        }