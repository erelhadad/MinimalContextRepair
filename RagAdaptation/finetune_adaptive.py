from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


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
    return model_id.replace("/", "__").replace(":", "_")


def run_method_pipeline(
    *,
    examples_json: Path,
    model: str,
    method: str,
    tau: float,
    epsilon: float,
    k: int,
    examples_range: Optional[Tuple[int, int]],
    save_logs: bool,
) -> None:
    model_slug = sanitize_model_name(model)

    out_dir = (
        project_root
        / "outputs"
        / "adaptive_tuning"
        / model_slug
        / method
        / f"tau_{tau:g}__eps_{epsilon:g}__k_{k}"
    )

    cmd = [
        sys.executable,
        "-m","RagAdaptation.run_pipeline",
        "--input",str(examples_json),
        "--out_dir",str(out_dir),
        "--stop_at_flip",
        "--models",model,
        "--methods",method,
        "--tau",str(tau),"--epsilon",str(epsilon),"--k",str(k),
    ]

    if examples_range is not None:
        start, end = examples_range
        cmd += ["--examples_range", str(start), str(end)]

    if save_logs:
        cmd.append("--save_logs")

    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(project_root))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--examples_json", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--method", type=str, required=True)

    ap.add_argument("--tau", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--epsilon", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    ap.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])

    ap.add_argument(
        "--examples_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Inclusive example range, e.g. --examples_range 0 99",
    )

    ap.add_argument("--save_logs", action="store_true")
    ap.add_argument("--keep_going", action="store_true")

    args = ap.parse_args()

    examples_range = (
        tuple(args.examples_range) if args.examples_range is not None else None
    )

    for model, tau, epsilon, k in itertools.product(
        args.models,
        args.tau,
        args.epsilon,
        args.k,
    ):
        try:
            run_method_pipeline(
                examples_json=args.examples_json,
                model=model,
                method=args.method,
                tau=tau,
                epsilon=epsilon,
                k=k,
                examples_range=examples_range,
                save_logs=args.save_logs,
            )
        except subprocess.CalledProcessError as e:
            print(
                f"[FAILED] model={model} method={args.method} "
                f"tau={tau} epsilon={epsilon} k={k} returncode={e.returncode}",
                flush=True,
            )
            if not args.keep_going:
                raise


if __name__ == "__main__":
    main()