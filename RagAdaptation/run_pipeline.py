from __future__ import annotations

import argparse
from pathlib import Path

from RagAdaptation.core.paths import RUNS_DIR
from RagAdaptation.pipeline.config import PipelineConfig
from RagAdaptation.pipeline.runner import run_dataset
from typing import Tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="report.json that contains field of relevant example")
    ap.add_argument("--out_dir", default=str(RUNS_DIR))
    ap.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3",
                 "Qwen/Qwen3-4B-Instruct-2507"],
    )
    ap.add_argument("--methods", nargs="+", default=["attention", "random", "context_cite","at2"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 10, 20, 40])
    ap.add_argument("--context_field", default="context")
    ap.add_argument("--recompute", nargs="+", default=["attention", "context_cite","at2"])
    ap.add_argument("--skip_recompute",nargs="*", type=int, default=[5])
    ap.add_argument("--skip_examples", nargs="*", type=int, default=[])
    ap.add_argument("--save_logs",  action="store_true")
    ap.add_argument("--stop_at_flip", action="store_true")
    ap.add_argument("--examples_range", nargs=2, type=int,help="Range of examples to run")
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
        skip_recompute=args.skip_recompute,
        save_logs=args.save_logs,
        stop_at_flip=args.stop_at_flip,
        examples_range=args.examples_range,
    )

    run_root = run_dataset(config)
    print(f"[ok] wrote organized run to {run_root}")


if __name__ == "__main__":
    main()
