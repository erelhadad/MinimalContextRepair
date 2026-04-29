from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from RagAdaptation.pipeline.config import PipelineConfig
from RagAdaptation.pipeline.runner import run_dataset


'''
python -m RagAdaptation.debug.rerun_failed_qwen_at2 \
  --input ./outputs/reports/dataset_creation/hotpot_yesno__validation__all__full/report_flip_only__Qwen__Qwen3-4B-Instruct-2507.json \
  --previous_run_roots ./qwen_hotpot_pipe/qwen_hotpot_pipe_37  ./qwen_hotpot_pipe \
  --out_dir ./qwen_hotpot_rerun_fixed \
  --stop_at_flip \
  --save_logs

'''


QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_ts(path: Path) -> datetime:
    import re
    m = re.search(r"_(\d{8}_\d{6})\.json$", path.name)
    if not m:
        return datetime.min
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")


def get_latest_pipeline_result(model_dir: Path) -> Optional[Path]:
    candidates = list(model_dir.glob("pipeline_result_methods_*.json"))
    if not candidates:
        return None
    return max(candidates, key=extract_ts)


def at2_failed(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return True
    if payload.get("status") == "failed":
        return True
    if payload.get("error"):
        return True
    if payload.get("masked_stats") is None:
        return True
    return False


def find_failed_examples(run_root: Path) -> List[int]:
    failed = []
    examples_dir = run_root / "examples"
    for ex_dir in sorted(examples_dir.glob("ex*")):
        model_dir = ex_dir / "models" / QWEN_MODEL_ID.replace("/", "__")
        if not model_dir.exists():
            continue
        latest = get_latest_pipeline_result(model_dir)
        if latest is None:
            failed.append(int(ex_dir.name.replace("ex", "")))
            continue
        payload = load_json(latest)
        methods = payload.get("methods", {})
        at2_payload = methods.get("at2")
        rec_payload = methods.get("recompute_at2_SR5") or methods.get("recompute_at2")
        if at2_failed(at2_payload) or at2_failed(rec_payload):
            failed.append(int(ex_dir.name.replace("ex", "")))
    return failed


def filter_report_rows(report_path: Path, keep_indices: List[int]) -> Path:
    payload = load_json(report_path)
    if isinstance(payload, dict) and "results" in payload:
        rows = payload["results"]
        wrapper = dict(payload)
    elif isinstance(payload, list):
        rows = payload
        wrapper = None
    else:
        raise ValueError("Unsupported report format")

    keep = set(keep_indices)
    filtered = [row for i, row in enumerate(rows) if i in keep]
    out_path = report_path.parent / f"{report_path.stem}__failed_qwen_at2_only.json"
    if wrapper is None:
        out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        wrapper["results"] = filtered
        out_path.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Original report json used by run_pipeline")
    ap.add_argument("--previous_run_roots", nargs="+", required=True, help="One or more old run roots to scan for failed Qwen AT2 examples")
    ap.add_argument("--out_dir", required=True, help="Output root for the rerun")
    ap.add_argument("--save_logs", action="store_true")
    ap.add_argument("--stop_at_flip", action="store_true")
    ap.add_argument("--skip_recompute", nargs="*", type=int, default=[5])
    args = ap.parse_args()

    report_path = Path(args.input)
    previous_run_roots = [Path(x) for x in args.previous_run_roots]

    failed_examples = sorted({idx for run_root in previous_run_roots for idx in find_failed_examples(run_root)})
    if not failed_examples:
        print("[ok] no failed Qwen AT2 examples found")
        return

    print(f"[info] failed Qwen AT2 examples: {failed_examples}")
    filtered_report = filter_report_rows(report_path, failed_examples)
    print(f"[info] wrote filtered report to {filtered_report}")

    config = PipelineConfig(
        input_path=filtered_report,
        output_root=Path(args.out_dir),
        models=[QWEN_MODEL_ID],
        methods=["at2"],
        seeds=[0, 10, 20, 40],
        recompute=["at2"],
        skip_recompute=args.skip_recompute,
        save_logs=args.save_logs,
        stop_at_flip=args.stop_at_flip,
    )

    run_root = run_dataset(config)
    print(f"[done] rerun written to {run_root}")


if __name__ == "__main__":
    main()
