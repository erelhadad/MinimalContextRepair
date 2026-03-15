from __future__ import annotations

import argparse
from pathlib import Path

from RagAdaptation.core.paths import RUNS_DIR
from RagAdaptation.pipeline.config import PipelineConfig
from RagAdaptation.pipeline.runner import run_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="report.json or examples.json")
    ap.add_argument("--out_dir", default=str(RUNS_DIR))
    ap.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3"],
    )
    ap.add_argument("--methods", nargs="+", default=["attention", "random", "context_cite","at2"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 10, 20, 40])
    ap.add_argument("--context_field", default="context")
    ap.add_argument("--recompute", nargs="+", default=["attention", "context_cite","at2"])
    ap.add_argument("--skip_recompute", type=int, default=1)
    ap.add_argument("--skip_examples", nargs="*", type=int, default=[])
    args = ap.parse_args()

    config = PipelineConfig(
        input_path=Path(args.input),
        output_root=Path(args.out_dir),
        models=list(args.models),
        methods=list(args.methods),
        seeds=list(args.seeds),
        recompute=list(args.recompute),
        context_field=args.context_field,
        skip_example_indices=list(args.skip_examples),
        skip_recompute=args.skip_recompute
    )

    run_root = run_dataset(config)
    print(f"[ok] wrote organized run to {run_root}")


if __name__ == "__main__":
    main()
