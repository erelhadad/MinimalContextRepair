from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ---------------- helpers ----------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def safe_std(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return statistics.pstdev(vals)


def fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "–"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "–"
    return f"{x:.{digits}f}"


def markdown_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def label_from_stats(stats: Dict[str, Any]) -> str:
    return "true" if float(stats.get("p_true", 0.0)) > 0.5 else "false"


def short_model_name(model_id: str) -> str:
    model_id = str(model_id)
    if "Mistral" in model_id:
        return "Mistral-7B"
    if "Phi-3" in model_id or "Phi3" in model_id:
        return "Phi-3"
    if "Qwen" in model_id:
        return "Qwen"
    return model_id.split("/")[-1]


def extract_ts(path: Path) -> datetime:
    m = re.search(r"_(\d{8}_\d{6})\.json$", path.name)
    if not m:
        return datetime.min
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")


def find_model_dir(example_dir: Path, model_id: str) -> Optional[Path]:
    model_short = short_model_name(model_id)
    model_dir = example_dir / "models" / model_short
    if model_dir.exists():
        return model_dir
    return None


def get_latest_pipeline_result(model_dir: Path) -> Optional[Path]:
    candidates = list(model_dir.glob("pipeline_result_methods_*.json"))
    if not candidates:
        return None
    return max(candidates, key=extract_ts)


def first_flip_step(masked_stats: List[Dict[str, Any]]) -> Optional[int]:
    """
    In the newer compute_probs_updated.py:
    - each row may contain: step_index, is_flipped
    - the FIRST flipped row also stores first_flip_index as 1-based
    """
    for st in masked_stats:
        if st.get("is_flipped"):
            if "first_flip_index" in st and st["first_flip_index"] is not None:
                return int(st["first_flip_index"])
            if "step_index" in st and st["step_index"] is not None:
                return int(st["step_index"])
    return None


def get_length_examples(query: str, report_with_lengths: Path, model_id: str) -> int:
    """
    Preferred path:
      ex["context_lengths"]["per_model"][model_id]["context_tokens"]

    Fallback path:
      ex["per_model"][model_id]["context_length_tokens"]
    """
    data = load_json(report_with_lengths)

    if isinstance(data, dict) and "results" in data:
        items = data["results"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(
            f"couldn't read data from json in get_length_examples: {report_with_lengths}"
        )

    for ex in items:
        if ex.get("query") != query:
            continue

        context_lengths = ex.get("context_lengths", {})
        per_model_ctx = context_lengths.get("per_model", {})
        if model_id in per_model_ctx:
            return int(per_model_ctx[model_id]["context_tokens"])

        per_model = ex.get("per_model", {})
        if model_id in per_model and "context_length_tokens" in per_model[model_id]:
            return int(per_model[model_id]["context_length_tokens"])

        raise ValueError(
            f"found query but no context length for model {model_id} in {report_with_lengths}"
        )

    raise ValueError(f"couldn't find query={query!r} in {report_with_lengths}")


def get_report_with_lengths_path(model_id: str) -> Path:
    """
    Update these paths if your files live elsewhere.
    """
    mapping = {
        "mistralai/Mistral-7B-Instruct-v0.3": Path(
            "/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Mistral_full_dataset_reports/report_flip_only__mistralai__Mistral-7B-Instruct-v0.3_with_context_lengths.json"
        ),
        "microsoft/Phi-3-mini-4k-instruct": Path(
            "/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Phi_full_dataset_reports/report_flip_only__microsoft__Phi-3-mini-4k-instruct_with_context_lengths.json"
        ),
        "Qwen/Qwen3-4B-Instruct-2507": Path(
            "/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Phi_full_dataset_reports/report_flip_only__Qwen__Qwen3-4B-Instruct-2507_with_context_lengths.json"
        ),
    }

    if model_id not in mapping:
        raise ValueError(f"couldn't find report-with-lengths path for model_id={model_id}")

    return mapping[model_id]


# ---------------- method-level processing ----------------

def build_method_result(payload: Dict[str, Any], context_tokens: int) -> Dict[str, Any]:
    if isinstance(payload, dict) and payload.get("status") == "failed":
        return {
            "success": False,
            "flip_step": None,
            "masked_percentage": None,
            "context_tokens": context_tokens,
            "error": payload.get("error"),
        }

    masked_stats = (payload or {}).get("masked_stats") or []
    flip_step = first_flip_step(masked_stats)

    return {
        "success": flip_step is not None,
        "flip_step": flip_step,
        "masked_percentage": (
            100.0 * flip_step / context_tokens
            if (flip_step is not None and context_tokens > 0)
            else None
        ),
        "context_tokens": context_tokens,
        "error": None,
    }


def build_random_method_result(payload: Dict[str, Any], context_tokens: int) -> Dict[str, Any]:
    """
    random payload structure:
      {
        "0": {"masked_stats": ..., "masked_logps": ...},
        "10": {...},
        ...
      }

    We summarize per example by averaging across successful seeds.
    """
    if not isinstance(payload, dict):
        return {
            "success": False,
            "flip_step": None,
            "masked_percentage": None,
            "context_tokens": context_tokens,
            "error": "random payload is not a dict",
        }

    seed_results = []
    for seed, seed_payload in payload.items():
        masked_stats = (seed_payload or {}).get("masked_stats") or []
        flip_step = first_flip_step(masked_stats)

        seed_results.append({
            "seed": seed,
            "success": flip_step is not None,
            "flip_step": flip_step,
            "masked_percentage": (
                100.0 * flip_step / context_tokens
                if (flip_step is not None and context_tokens > 0)
                else None
            ),
        })

    successes = [x for x in seed_results if x["success"]]
    flip_steps = [x["flip_step"] for x in successes if x["flip_step"] is not None]
    masked_percentages = [
        x["masked_percentage"]
        for x in successes
        if x["masked_percentage"] is not None
    ]

    return {
        "success": len(successes) > 0,
        "flip_step": safe_mean(flip_steps),
        "masked_percentage": safe_mean(masked_percentages),
        "context_tokens": context_tokens,
        "n_random_seeds": len(seed_results),
        "n_random_success": len(successes),
        "seed_details": seed_results,
        "error": None,
    }


# ---------------- collection ----------------

def iterate_for_model(
    model_id: str,
    outputs_dir: Path,
    report_with_lengths: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Passes on all the examples and documents the results for model_id
    on the examples this model actually saw.
    """
    if report_with_lengths is None:
        report_with_lengths = get_report_with_lengths_path(model_id)

    for example_dir in sorted(outputs_dir.iterdir()):
        if not example_dir.is_dir():
            continue

        model_dir = find_model_dir(example_dir, model_id)
        if model_dir is None:
            continue

        latest_result_file = get_latest_pipeline_result(model_dir)
        if latest_result_file is None:
            continue

        result = load_json(latest_result_file)

        query = str(result.get("query", ""))
        baseline_stats = ((result.get("baseline") or {}).get("stats") or {})
        baseline_label = label_from_stats(baseline_stats)
        context_tokens = get_length_examples(query, report_with_lengths, model_id)

        methods = result.get("methods") or {}

        row = {
            "example_dir": example_dir.name,
            "model_id": model_id,
            "model_short": short_model_name(model_id),
            "query": query,
            "baseline_label": baseline_label,
            "baseline_p_true": baseline_stats.get("p_true"),
            "context_tokens": context_tokens,
            "methods": {},
        }

        for method_name, payload in methods.items():
            if method_name == "random":
                row["methods"][method_name] = build_random_method_result(
                    payload, context_tokens
                )
            else:
                row["methods"][method_name] = build_method_result(
                    payload, context_tokens
                )

        yield row


def summarize_model(
    model_id: str,
    outputs_dir: Path,
    report_with_lengths: Optional[Path] = None,
) -> Dict[str, Any]:
    rows = list(iterate_for_model(model_id, outputs_dir, report_with_lengths))

    all_method_names = sorted({
        method_name
        for row in rows
        for method_name in row["methods"].keys()
    })

    summary = {
        "model_id": model_id,
        "model_short": short_model_name(model_id),
        "n_examples": len(rows),
        "methods": {},
        "rows": rows,
    }

    for method_name in all_method_names:
        method_rows = [
            row["methods"].get(method_name)
            for row in rows
            if method_name in row["methods"]
        ]

        successes = [m for m in method_rows if m and m.get("success")]
        flip_steps = [
            m.get("flip_step") for m in successes
            if m.get("flip_step") is not None
        ]
        masked_percentages = [
            m.get("masked_percentage") for m in successes
            if m.get("masked_percentage") is not None
        ]

        summary["methods"][method_name] = {
            "n_seen": len(method_rows),
            "n_success": len(successes),
            "success_rate": (
                len(successes) / len(method_rows) * 100.0
                if method_rows else None
            ),
            "avg_flip_step": safe_mean(flip_steps),
            "std_flip_step": safe_std(flip_steps),
            "avg_masked_percentage": safe_mean(masked_percentages),
            "std_masked_percentage": safe_std(masked_percentages),
        }

    return summary


# ---------------- rendering ----------------

def render_model_summary_table(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"## Model: {markdown_escape(summary['model_short'])}")
    lines.append("")
    lines.append(f"- Total examples seen: **{summary['n_examples']}**")
    lines.append("")
    lines.append("| Method | n | #flips | Success rate | Avg. masks | Avg. % masked |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for method_name, stats in summary["methods"].items():
        lines.append(
            f"| {markdown_escape(method_name)} "
            f"| {stats['n_seen']} "
            f"| {stats['n_success']} "
            f"| {fmt_num(stats['success_rate'])}% "
            f"| {fmt_num(stats['avg_flip_step'])} "
            f"| {fmt_num(stats['avg_masked_percentage'])}% |"
        )

    lines.append("")
    return "\n".join(lines)


def render_model_examples_table(summary: Dict[str, Any]) -> str:
    rows = summary["rows"]

    all_method_names = sorted({
        method_name
        for row in rows
        for method_name in row["methods"].keys()
    })

    lines: List[str] = []
    lines.append(f"### Per-example results for {markdown_escape(summary['model_short'])}")
    lines.append("")

    header = ["example_dir", "query", "baseline", "context_tokens"]
    for method_name in all_method_names:
        header.append(f"{method_name} flip")
        header.append(f"{method_name} %")

    lines.append("| " + " | ".join(markdown_escape(h) for h in header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in rows:
        table_row = [
            markdown_escape(row["example_dir"]),
            markdown_escape(row["query"]),
            markdown_escape(row["baseline_label"]),
            str(row["context_tokens"]),
        ]

        for method_name in all_method_names:
            method_data = row["methods"].get(method_name)

            if method_data is None:
                table_row.extend(["–", "–"])
                continue

            if method_data.get("error"):
                table_row.extend(["error", "–"])
                continue

            flip_val = method_data.get("flip_step")
            pct_val = method_data.get("masked_percentage")

            table_row.append(fmt_num(flip_val))
            table_row.append(f"{fmt_num(pct_val)}%")

        lines.append("| " + " | ".join(table_row) + " |")

    lines.append("")
    return "\n".join(lines)


def render_full_report(model_summaries: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    parts.append("# Run Pipeline Results by Model")
    parts.append("")
    parts.append("The report summarizes each model separately over all the examples for which that model has results.")
    parts.append("")

    for summary in model_summaries:
        parts.append(render_model_summary_table(summary))
        parts.append(render_model_examples_table(summary))

    return "\n".join(parts)


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outputs_dir",
        default="outputs/runs/examples",
        help="Directory that contains ex1/ex2/... example folders",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Full model ids, e.g. microsoft/Phi-3-mini-4k-instruct",
    )
    ap.add_argument(
        "--out_json",
        default=None,
        help="Optional path to save JSON summary",
    )
    ap.add_argument(
        "--out_md",
        default=None,
        help="Optional path to save Markdown report",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise ValueError(f"outputs_dir does not exist: {outputs_dir}")

    summaries: List[Dict[str, Any]] = []
    for model_id in args.models:
        summary = summarize_model(model_id=model_id, outputs_dir=outputs_dir)
        summaries.append(summary)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps({"models": summaries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[saved json] {out_json}")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md = render_full_report(summaries)
        out_md.write_text(md, encoding="utf-8")
        print(f"[saved md] {out_md}")

    if not args.out_json and not args.out_md:
        print(json.dumps({"models": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()