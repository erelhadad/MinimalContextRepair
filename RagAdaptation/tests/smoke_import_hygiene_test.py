from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


def main():
    modules = [
        "RagAdaptation.core",
        "RagAdaptation.rag_expirement_init",
        "RagAdaptation.baseline.ContextCite_run",
        "RagAdaptation.baseline.mask_iter_recompute_attention",
    ]
    loaded = []
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        loaded.append(mod.__name__)
    print("[ok] imported modules:", ", ".join(loaded))


if __name__ == "__main__":
    main()
