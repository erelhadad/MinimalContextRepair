from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create_p_true_function(masked_logps_att: Sequence[Any], seed: int | None = None, out_dir: str | Path = "", filename: str | None = None) -> str:
    if len(masked_logps_att) > 0 and isinstance(masked_logps_att[0], dict):
        y = np.array([d["p_true"] for d in masked_logps_att], dtype=float)
    else:
        y = np.array(masked_logps_att, dtype=float)

    out_dir = Path(out_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        s = "na" if seed is None else str(seed)
        filename = f"p_true_vs_iter_seed_{s}.png"

    file_path = out_dir / filename

    plt.figure(figsize=(8, 5))
    plt.plot(y)
    plt.ylim(0.0, 1.0)
    plt.ylabel("P(True)")
    plt.xlabel("Masking iteration")
    plt.title("P(True) as a function of masking iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()

    return str(file_path)
