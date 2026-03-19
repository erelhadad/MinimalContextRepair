from __future__ import annotations

import argparse
import compileall
import contextlib
import io
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class StageResult:
    name: str
    status: str
    duration_sec: float
    error_type: Optional[str] = None
    error: Optional[str] = None
    traceback_file: Optional[str] = None
    preview: Optional[Dict[str, Any]] = None


def _find_project_root(explicit: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).resolve())
    candidates.extend([
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    ])
    for base in candidates:
        if (base / "RagAdaptation").is_dir():
            return base
    raise RuntimeError(
        "Could not find project root containing a 'RagAdaptation/' directory. "
        "Pass --project-root explicitly."
    )


def _ensure_on_path(project_root: Path) -> None:
    p = str(project_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def _load_raw_payload(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data, data["results"]
    if isinstance(data, list):
        return data, data
    raise ValueError("Unsupported JSON format. Expected list or dict with 'results'.")


def _merge_items_with_examples(items: List[Dict[str, Any]], examples: List[Dict[str, Any]]) -> int:
    merged_count = 0
    for i, item in enumerate(items):
        if item.get("context_path"):
            continue
        raw_idx = item.get("idx", i)
        if not isinstance(raw_idx, int) or raw_idx < 0 or raw_idx >= len(examples):
            raw_idx = i
        if raw_idx < 0 or raw_idx >= len(examples):
            continue
        raw = examples[raw_idx]
        for key in (
            "context_path",
            "source_id",
            "source_dataset",
            "source_split",
            "title",
            "expected_answer",
            "query",
        ):
            if key not in item and key in raw:
                item[key] = raw[key]
        merged_count += 1
    return merged_count


def _load_items(input_path: Path, companion_examples: Optional[Path]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    _, items = _load_raw_payload(input_path)
    note = None

    needs_merge = any(not item.get("context_path") for item in items)
    if not needs_merge:
        return items, note

    candidate_examples: List[Path] = []
    if companion_examples is not None:
        candidate_examples.append(companion_examples)
    sibling = input_path.with_name("examples.json")
    if sibling.exists():
        candidate_examples.append(sibling)

    for candidate in candidate_examples:
        try:
            _, raw_examples = _load_raw_payload(candidate)
        except Exception:
            continue
        merged = _merge_items_with_examples(items, raw_examples)
        if merged > 0 and all(item.get("context_path") for item in items):
            note = f"Merged missing context metadata from {candidate}"
            return items, note

    return items, note


def _pick_example(
    items: List[Dict[str, Any]],
    model_id: str,
    example_index: Optional[int],
    benchmark_split: Optional[str],
) -> int:
    def split_ok(ex: Dict[str, Any]) -> bool:
        if benchmark_split is None:
            return True
        return ex.get("benchmark_split") == benchmark_split

    if example_index is not None:
        if example_index < 0 or example_index >= len(items):
            raise IndexError(f"example_index={example_index} out of range for {len(items)} items")
        return example_index

    for i, ex in enumerate(items):
        if not split_ok(ex):
            continue
        per_model = ex.get("per_model")
        if isinstance(per_model, dict) and model_id in per_model:
            if per_model[model_id].get("relevant"):
                return i

    for i, ex in enumerate(items):
        if split_ok(ex) and ex.get("context_path"):
            return i

    raise ValueError("Could not auto-select an example with context_path.")


def _compute_detect_flip_to_true(ex: Dict[str, Any], model_id: str) -> bool:
    per_model = ex.get("per_model")
    if isinstance(per_model, dict) and model_id in per_model:
        return per_model[model_id].get("prob_label_with_context") == "false"

    expected = ex.get("expected_answer_norm", ex.get("expected_answer"))
    if isinstance(expected, bool):
        return not expected
    if expected is None:
        return False
    return str(expected).strip().lower() == "false"


def _build_preview(payload: Any) -> Dict[str, Any]:
    preview: Dict[str, Any] = {}
    if isinstance(payload, dict):
        if "masked_stats" in payload and isinstance(payload["masked_stats"], list):
            masked_stats = payload["masked_stats"]
            preview["num_masked_stats"] = len(masked_stats)
            if masked_stats:
                first = masked_stats[0]
                if isinstance(first, dict):
                    preview["first_masked_p_true"] = first.get("p_true")
                    preview["first_flip_index"] = first.get("first_flip_index")
        if "masked_logps" in payload and isinstance(payload["masked_logps"], list):
            preview["num_masked_logps"] = len(payload["masked_logps"])
        if "generation" in payload:
            preview["generation"] = payload["generation"]
        if "estimator" in payload:
            preview["estimator"] = payload["estimator"]
        if payload and all(isinstance(v, dict) for v in payload.values()):
            seed_preview = {}
            for seed, res in payload.items():
                if isinstance(res, dict) and "masked_stats" in res:
                    stats = res["masked_stats"]
                    first_flip = None
                    if isinstance(stats, list) and stats and isinstance(stats[0], dict):
                        first_flip = stats[0].get("first_flip_index")
                    seed_preview[str(seed)] = {
                        "num_masked_stats": len(stats) if isinstance(stats, list) else None,
                        "first_flip_index": first_flip,
                    }
            if seed_preview:
                preview["per_seed"] = seed_preview
    return _jsonable(preview)


def _run_stage(
    name: str,
    fn: Callable[[], Any],
    out_dir: Path,
) -> StageResult:
    stage_dir = out_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    trace_path = stage_dir / "traceback.txt"
    start = time.perf_counter()
    try:
        print(f"\n===== STAGE: {name} =====")
        payload = fn()
        dur = time.perf_counter() - start
        preview = _build_preview(payload)
        trace_path.write_text("", encoding="utf-8")
        print(f"[OK] {name} finished in {dur:.2f}s")
        return StageResult(
            name=name,
            status="ok",
            duration_sec=dur,
            traceback_file=str(trace_path),
            preview=preview,
        )
    except Exception as e:
        dur = time.perf_counter() - start
        tb = traceback.format_exc()
        trace_path.write_text(tb, encoding="utf-8")
        print(f"[FAIL] {name} finished in {dur:.2f}s with {type(e).__name__}: {e}")
        print(tb)
        return StageResult(
            name=name,
            status="failed",
            duration_sec=dur,
            error_type=type(e).__name__,
            error=str(e),
            traceback_file=str(trace_path),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test the RAG pipeline stage by stage and log full tracebacks.")
    ap.add_argument("--project-root", default=None, help="Directory that contains RagAdaptation/")
    ap.add_argument("--input", required=True, help="Path to report/examples JSON")
    ap.add_argument("--examples-json", default=None, help="Optional companion examples.json used to restore missing context metadata")
    ap.add_argument("--benchmark-split", choices=["train", "dev", "test"], default=None, help="Prefer an example from this internal split")
    ap.add_argument("--model", required=True, help="HF model id, e.g. microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--example-index", type=int, default=None, help="Specific example index to test")
    ap.add_argument("--out-dir", default="smoke_stage_test", help="Where to write logs/results")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 10, 20, 40])
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-at2", action="store_true")
    args = ap.parse_args()

    project_root = _find_project_root(args.project_root)
    _ensure_on_path(project_root)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    master_log_path = out_dir / "smoke_test.log"
    summary_path = out_dir / "smoke_summary.json"

    with master_log_path.open("w", encoding="utf-8") as log_f:
        tee_out = Tee(sys.stdout, log_f)
        tee_err = Tee(sys.stderr, log_f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"project_root={project_root}")
            print(f"input={args.input}")
            print(f"model={args.model}")
            print(f"out_dir={out_dir}")
            print(f"seeds={args.seeds}")

            summary: Dict[str, Any] = {
                "project_root": str(project_root),
                "input": str(Path(args.input).resolve()),
                "model": args.model,
                "out_dir": str(out_dir),
                "seeds": list(args.seeds),
                "benchmark_split": args.benchmark_split,
                "stages": [],
            }

            if not args.skip_compile:
                print("\n===== PREFLIGHT: compileall =====")
                try:
                    ok = compileall.compile_dir(str(project_root / "RagAdaptation"), quiet=1, force=False)
                    summary["compileall_ok"] = bool(ok)
                    print(f"compileall_ok={ok}")
                except Exception:
                    summary["compileall_ok"] = False
                    summary["compileall_traceback"] = traceback.format_exc()
                    print(summary["compileall_traceback"])

            from RagAdaptation.compute_probs_updated import compute_probs
            from RagAdaptation.core.documents import combine_document_text, load_documents_any
            from RagAdaptation.core.models import get_hf_scorer
            from RagAdaptation.core.prompting import ChatPromptTemplate
            from RagAdaptation.methods.attention import run_attention_method
            from RagAdaptation.methods.at2 import run_at2_method
            from RagAdaptation.methods.context_cite import run_context_cite_method
            from RagAdaptation.methods.random_mask import run_random_method
            from RagAdaptation.methods.recompute import run_recompute_method
            from RagAdaptation.prompts_format import TF_RAG_TEMPLATE

            items, merge_note = _load_items(
                Path(args.input).resolve(),
                Path(args.examples_json).resolve() if args.examples_json else None,
            )
            if merge_note:
                print(merge_note)
                summary["merge_note"] = merge_note

            ex_i = _pick_example(items, args.model, args.example_index, args.benchmark_split)
            ex = items[ex_i]
            query = ex.get("query") or ex.get("question")
            context_path = ex.get("context_path")
            if not query:
                raise ValueError(f"Example {ex_i} has no query/question")
            if not context_path:
                raise ValueError(
                    f"Example {ex_i} has no context_path. Supply --examples-json or use an enriched report."
                )

            docs = load_documents_any(context_path)
            full_context = combine_document_text(docs)
            detect_flip_to_true = _compute_detect_flip_to_true(ex, args.model)

            summary["example_index"] = ex_i
            summary["query"] = query
            summary["context_path"] = context_path
            summary["context_chars"] = len(full_context)
            summary["detect_flip_to_true"] = detect_flip_to_true
            summary["source_id"] = ex.get("source_id")
            summary["benchmark_split_selected"] = ex.get("benchmark_split")

            print(f"\nSelected example_index={ex_i}")
            print(f"query={query}")
            print(f"context_path={context_path}")
            print(f"context_chars={len(full_context)}")
            print(f"detect_flip_to_true={detect_flip_to_true}")
            if ex.get("benchmark_split"):
                print(f"benchmark_split={ex.get('benchmark_split')}")

            print("\n===== PREFLIGHT: load main scorer =====")
            hf_model, hf_tok, hf_device = get_hf_scorer(args.model)
            print(f"main_device={hf_device}")

            prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)
            baseline_prompt = prompt_template.format(context=full_context, question=query)
            baseline_stats_list, baseline_logps = compute_probs(
                hf_model,
                hf_tok,
                [baseline_prompt],
                hf_device,
                None,
                batch_size=1,
                return_full_logp=True,
                file_name=str(out_dir / "baseline_compute_probs.txt"),
                detect_flip_to_true=detect_flip_to_true,
            )
            baseline_stats = baseline_stats_list[0]
            summary["baseline"] = _jsonable({
                "stats": baseline_stats,
                "logps": baseline_logps,
            })
            print(f"baseline_stats={json.dumps(_jsonable(baseline_stats), indent=2)}")

            common_kwargs = dict(
                out_dir=str(out_dir),
                baseline_stats=baseline_stats,
                full_context=full_context,
                query=query,
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                p_true_flipping=detect_flip_to_true,
                dump_policy="flip",
                dump_window=1,
                true_variants=["true", "True", "TRUE"],
                false_variants=["false", "False", "FALSE"],
            )

            stages: List[Tuple[str, Callable[[], Any]]] = [
                (
                    "attention",
                    lambda: run_attention_method(
                        baseline_prompt=baseline_prompt,
                        **common_kwargs,
                    ),
                ),
                (
                    "random",
                    lambda: run_random_method(
                        seeds=list(args.seeds),
                        **common_kwargs,
                    ),
                ),
                (
                    "context_cite",
                    lambda: run_context_cite_method(**common_kwargs),
                ),
            ]

            if not args.skip_at2:
                stages.append(
                    (
                        "at2",
                        lambda: run_at2_method(
                            model_id=args.model,
                            **common_kwargs,
                        ),
                    )
                )

            stages.extend([
                (
                    "recompute_attention",
                    lambda: run_recompute_method(
                        out_dir=str(out_dir),
                        rec_method="attention",
                        model_id=args.model,
                        full_context=full_context,
                        query=query,
                        hf_model=hf_model,
                        hf_tok=hf_tok,
                        hf_device=hf_device,
                        p_true_flipping=detect_flip_to_true,
                        true_variants=["true", "True", "TRUE"],
                        false_variants=["false", "False", "FALSE"],
                    ),
                ),
                (
                    "recompute_context_cite",
                    lambda: run_recompute_method(
                        out_dir=str(out_dir),
                        rec_method="context_cite",
                        model_id=args.model,
                        full_context=full_context,
                        query=query,
                        hf_model=hf_model,
                        hf_tok=hf_tok,
                        hf_device=hf_device,
                        p_true_flipping=detect_flip_to_true,
                        true_variants=["true", "True", "TRUE"],
                        false_variants=["false", "False", "FALSE"],
                    ),
                ),
            ])

            if not args.skip_at2:
                stages.append(
                    (
                        "recompute_at2",
                        lambda: run_recompute_method(
                            out_dir=str(out_dir),
                            rec_method="at2",
                            model_id=args.model,
                            full_context=full_context,
                            query=query,
                            hf_model=hf_model,
                            hf_tok=hf_tok,
                            hf_device=hf_device,
                            p_true_flipping=detect_flip_to_true,
                            true_variants=["true", "True", "TRUE"],
                            false_variants=["false", "False", "FALSE"],
                        ),
                    )
                )

            for name, fn in stages:
                result = _run_stage(name, fn, out_dir)
                summary["stages"].append(asdict(result))
                summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

            ok_count = sum(1 for s in summary["stages"] if s["status"] == "ok")
            fail_count = sum(1 for s in summary["stages"] if s["status"] == "failed")
            summary["ok_count"] = ok_count
            summary["fail_count"] = fail_count
            summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
            print(f"\nDONE: ok_count={ok_count} fail_count={fail_count}")
            print(f"summary={summary_path}")
            print(f"log={master_log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
