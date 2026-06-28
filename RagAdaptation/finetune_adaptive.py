from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple


_THIS_FILE = Path(__file__).resolve()

project_root = None
for p in [_THIS_FILE.parent] + list(_THIS_FILE.parents):
    if (p / "RagAdaptation").is_dir():
        project_root = p
        break

if project_root is None:
    project_root = _THIS_FILE.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "__")


def add_flag(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def add_bool_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def run_method_pipeline(
    *,
    examples_json: Path,
    out_root: Path,
    model: str,
    method: str,
    tau: float,
    epsilon: float,
    k: int,
    examples_range: Optional[Tuple[int, int]],
    save_logs: bool,
    stop_at_flip: bool,
    keep_going: bool,
    intervention_mode: str,
    running_env: str,
    prefer_at2_word_scorer: bool,
    use_yes_no_variants: bool,
    replacement_cache: Optional[str],
    neutral_model: str,
    conceptnet_min_weight: float,
    disable_replacement_semex_filter: bool,
    replacement_semex_spacy_model: str,
    max_scoring_prompt_batch_size: Optional[int],
    scoring_row_batch_size: int,
    streaming_chunk_size: int,
    cleanup_every_batches: int,
    aggressive_cleanup: bool,
    unload_models_between_runs: bool,
    attention_native_seq_len_limit: int,
) -> None:
    model_slug = sanitize_model_name(model)

    out_dir = (
        out_root
        / model_slug
        / method
        / f"tau_{tau:g}__eps_{epsilon:g}__k_{k}"
    )

    cmd = [
        sys.executable,
        "-m",
        "RagAdaptation.run_pipeline",
        "--input",
        str(examples_json),
        "--out_dir",
        str(out_dir),
        "--models",
        model,
        "--methods",
        method,
        "--tau",
        str(tau),
        "--epsilon",
        str(epsilon),
        "--k",
        str(k),
        "--intervention_mode",
        intervention_mode,
        "--running_env",
        running_env,
        "--neutral_model",
        neutral_model,
        "--conceptnet_min_weight",
        str(conceptnet_min_weight),
        "--replacement_semex_spacy_model",
        replacement_semex_spacy_model,
        "--scoring_row_batch_size",
        str(scoring_row_batch_size),
        "--streaming_chunk_size",
        str(streaming_chunk_size),
        "--cleanup_every_batches",
        str(cleanup_every_batches),
        "--attention_native_seq_len_limit",
        str(attention_native_seq_len_limit),
    ]

    if examples_range is not None:
        start, end = examples_range
        cmd += ["--examples_range", str(start), str(end)]

    add_flag(cmd, "--replacement_cache", replacement_cache)
    add_flag(cmd, "--max_scoring_prompt_batch_size", max_scoring_prompt_batch_size)

    add_bool_flag(cmd, "--save_logs", save_logs)
    add_bool_flag(cmd, "--stop_at_flip", stop_at_flip)
    add_bool_flag(cmd, "--prefer_at2_word_scorer", prefer_at2_word_scorer)
    add_bool_flag(cmd, "--use_yes_no_variants", use_yes_no_variants)
    add_bool_flag(cmd, "--disable_replacement_semex_filter", disable_replacement_semex_filter)
    add_bool_flag(cmd, "--aggressive_cleanup", aggressive_cleanup)
    add_bool_flag(cmd, "--unload_models_between_runs", unload_models_between_runs)

    print("\n[RUN]", " ".join(cmd), flush=True)

    try:
        subprocess.run(cmd, check=True, cwd=str(project_root))
    except subprocess.CalledProcessError:
        if keep_going:
            print("[warn] command failed, continuing because --keep_going was set", flush=True)
            return
        raise


def main() -> None:
    ap = argparse.ArgumentParser()

    # Accept both names, because run_pipeline uses --input but the old wrapper used --examples_json.
    ap.add_argument("--examples_json", "--input", dest="examples_json", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, default=None)

    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", type=str, required=True)

    # Keep this as a sweep interface.
    ap.add_argument("--tau", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    ap.add_argument("--epsilon", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    ap.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])

    ap.add_argument(
        "--examples_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Inclusive example range, e.g. --examples_range 0 499",
    )

    ap.add_argument("--save_logs", action="store_true")
    ap.add_argument("--stop_at_flip", action="store_true")
    ap.add_argument("--keep_going", action="store_true")

    # New/current run_pipeline options.
    ap.add_argument("--intervention_mode", type=str, default="mask_word")
    ap.add_argument("--running_env", type=str, default="newton")
    ap.add_argument("--prefer_at2_word_scorer", action="store_true")
    ap.add_argument("--use_yes_no_variants", action="store_true")

    ap.add_argument("--replacement_cache", type=str, default=None)
    ap.add_argument("--neutral_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--conceptnet_min_weight", type=float, default=1.0)
    ap.add_argument("--disable_replacement_semex_filter", action="store_true")
    ap.add_argument("--replacement_semex_spacy_model", type=str, default="en_core_web_sm")

    ap.add_argument("--max_scoring_prompt_batch_size", type=int, default=None)
    ap.add_argument("--scoring_row_batch_size", type=int, default=8)
    ap.add_argument("--streaming_chunk_size", type=int, default=32)
    ap.add_argument("--cleanup_every_batches", type=int, default=0)
    ap.add_argument("--aggressive_cleanup", action="store_true")
    ap.add_argument("--unload_models_between_runs", action="store_true")
    ap.add_argument("--attention_native_seq_len_limit", type=int, default=512)

    args = ap.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "nojid")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

    if args.out_root is None:
        args.out_root = (
            project_root
            / "outputs"
            / "adaptive_tuning"
            / f"runs_{job_id}_{task_id}"
        )

    examples_range = (
        tuple(args.examples_range) if args.examples_range is not None else None
    )

    for method in args.methods:
        for model, tau, epsilon, k in itertools.product(
            args.models,
            args.tau,
            args.epsilon,
            args.k,
        ):
            run_method_pipeline(
                examples_json=args.examples_json,
                out_root=args.out_root,
                model=model,
                method=method,
                tau=tau,
                epsilon=epsilon,
                k=k,
                examples_range=examples_range,
                save_logs=args.save_logs,
                stop_at_flip=args.stop_at_flip,
                keep_going=args.keep_going,
                intervention_mode=args.intervention_mode,
                running_env=args.running_env,
                prefer_at2_word_scorer=args.prefer_at2_word_scorer,
                use_yes_no_variants=args.use_yes_no_variants,
                replacement_cache=args.replacement_cache,
                neutral_model=args.neutral_model,
                conceptnet_min_weight=args.conceptnet_min_weight,
                disable_replacement_semex_filter=args.disable_replacement_semex_filter,
                replacement_semex_spacy_model=args.replacement_semex_spacy_model,
                max_scoring_prompt_batch_size=args.max_scoring_prompt_batch_size,
                scoring_row_batch_size=args.scoring_row_batch_size,
                streaming_chunk_size=args.streaming_chunk_size,
                cleanup_every_batches=args.cleanup_every_batches,
                aggressive_cleanup=args.aggressive_cleanup,
                unload_models_between_runs=args.unload_models_between_runs,
                attention_native_seq_len_limit=args.attention_native_seq_len_limit,
            )


if __name__ == "__main__":
    main()