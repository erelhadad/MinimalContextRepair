from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset


# --- robust project-root discovery (same spirit as the rest of the project) ---
_THIS_FILE = Path(__file__).resolve()
project_root = None
for p in [_THIS_FILE.parent] + list(_THIS_FILE.parents):
    if (p / "RagAdaptation").is_dir():
        project_root = p
        break

if project_root is None:
    # fallback so the file can still be inspected outside the repo
    project_root = _THIS_FILE.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from RagAdaptation.core.paths import DATA_DIR, REPORTS_DIR
except Exception:
    DATA_DIR = project_root / "data"
    REPORTS_DIR = project_root / "outputs" / "reports"


@dataclass
class NormalizedExample:
    query: str
    expected_answer: bool
    context_text: str
    source_dataset: str
    source_split: str
    source_id: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_examples_json_item(self, context_path: Path) -> Dict[str, Any]:
        out = {
            "query": self.query,
            "expected_answer": self.expected_answer,
            "context_path": str(context_path),
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "source_id": self.source_id,
        }
        if self.title:
            out["title"] = self.title
        if self.metadata:
            out.update(self.metadata)
        return out


# -----------------------------
# small helpers
# -----------------------------
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    text = text.strip("_")
    return text[:max_len] or "item"


def _ensure_bool_answer(answer: Any) -> bool:
    if isinstance(answer, bool):
        return answer
    s = str(answer).strip().lower()
    if s in {"yes", "true", "1"}:
        return True
    if s in {"no", "false", "0"}:
        return False
    raise ValueError(f"Unsupported binary answer value: {answer!r}")


def _load_payload_items(payload_or_path: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    if isinstance(payload_or_path, (str, Path)):
        payload = json.loads(Path(payload_or_path).read_text(encoding="utf-8"))
    else:
        payload = payload_or_path

    if isinstance(payload, dict) and "results" in payload:
        return payload, payload["results"]
    if isinstance(payload, list):
        return payload, payload
    raise ValueError("Unsupported JSON payload. Expected list or dict with 'results'.")


def _replace_items_in_payload(payload: Any, items: List[Dict[str, Any]]) -> Any:
    if isinstance(payload, dict) and "results" in payload:
        return {**payload, "results": items}
    return items


def _stable_item_key(item: Dict[str, Any], seed: int) -> str:
    key_bits = [
        str(item.get("source_id") or ""),
        str(item.get("query") or item.get("question") or ""),
        str(item.get("expected_answer") or item.get("expected_answer_norm") or ""),
    ]
    raw = f"{seed}|" + "|".join(key_bits)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalized_split_fracs(train_frac: float, dev_frac: float, test_frac: float) -> Dict[str, float]:
    total = float(train_frac + dev_frac + test_frac)
    if total <= 0:
        raise ValueError("train/dev/test fractions must sum to a positive value")
    return {
        "train": float(train_frac) / total,
        "dev": float(dev_frac) / total,
        "test": float(test_frac) / total,
    }


def build_internal_split_map(
    items: List[Dict[str, Any]],
    *,
    seed: int,
    train_frac: float,
    dev_frac: float,
    test_frac: float,
) -> Dict[str, str]:
    fracs = _normalized_split_fracs(train_frac, dev_frac, test_frac)
    ordered = sorted(items, key=lambda item: _stable_item_key(item, seed))
    n = len(ordered)

    if n == 0:
        return {}

    raw_counts = {name: n * frac for name, frac in fracs.items()}
    counts = {name: int(raw_counts[name]) for name in raw_counts}
    assigned = sum(counts.values())
    remainders = sorted(
        ((raw_counts[name] - counts[name], name) for name in counts),
        reverse=True,
    )
    for _, name in remainders[: max(0, n - assigned)]:
        counts[name] += 1

    # Prefer keeping at least one train example whenever possible.
    if n > 0 and counts["train"] == 0:
        for donor in ("test", "dev"):
            if counts[donor] > 1:
                counts[donor] -= 1
                counts["train"] += 1
                break
        if counts["train"] == 0:
            counts["train"] = 1
            for donor in ("test", "dev"):
                if counts[donor] > 0:
                    counts[donor] -= 1
                    break

    split_map: Dict[str, str] = {}
    cursor = 0
    for split_name in ("train", "dev", "test"):
        chunk = ordered[cursor : cursor + counts[split_name]]
        for item in chunk:
            split_map[_stable_item_key(item, seed)] = split_name
        cursor += counts[split_name]

    # Safety fallback in case a rounding corner left something unassigned.
    for item in ordered:
        key = _stable_item_key(item, seed)
        split_map.setdefault(key, "train")

    return split_map


def _annotate_items_with_split(
    items: List[Dict[str, Any]],
    *,
    split_map: Dict[str, str],
    seed: int,
    split_group: str,
) -> Dict[str, List[Dict[str, Any]]]:
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for item in items:
        split_name = split_map.get(_stable_item_key(item, seed), "train")
        annotated = dict(item)
        annotated["benchmark_split"] = split_name
        annotated["benchmark_group"] = split_group
        by_split[split_name].append(annotated)
    return by_split


def write_internal_split_files(
    *,
    payload_or_path: Any,
    split_map: Dict[str, str],
    seed: int,
    out_prefix: Path,
    split_group: str,
    split_config: Dict[str, Any],
) -> Dict[str, Path]:
    payload, items = _load_payload_items(payload_or_path)
    by_split = _annotate_items_with_split(items, split_map=split_map, seed=seed, split_group=split_group)

    out_paths: Dict[str, Path] = {}
    counts: Dict[str, int] = {}
    for split_name, split_items in by_split.items():
        out_payload = _replace_items_in_payload(payload, split_items)
        if isinstance(out_payload, dict):
            meta = dict(out_payload.get("benchmark_split_meta") or {})
            meta.update({
                "split_name": split_name,
                "split_group": split_group,
                **split_config,
            })
            out_payload["benchmark_split_meta"] = meta
        split_path = out_prefix.parent / f"{out_prefix.name}__{split_name}.json"
        _write_json(split_path, out_payload)
        out_paths[split_name] = split_path
        counts[split_name] = len(split_items)

    manifest_path = out_prefix.parent / f"{out_prefix.name}__split_manifest.json"
    _write_json(
        manifest_path,
        {
            "split_group": split_group,
            **split_config,
            "counts": counts,
        },
    )
    out_paths["manifest"] = manifest_path
    return out_paths


# -----------------------------
# dataset normalization
# -----------------------------
def _join_hotpot_context(context_field: Any, *, mode: str, supporting_facts: Optional[List[List[Any]]] = None) -> str:
    """
    context_field shape: [[title, [sent1, sent2, ...]], ...]
    supporting_facts shape: [[title, sent_id], ...]
    """
    if not isinstance(context_field, list):
        raise ValueError(f"Hotpot context has unexpected type: {type(context_field)}")

    sf_lookup = set()
    if supporting_facts:
        sf_lookup = {(str(title), int(sent_id)) for title, sent_id in supporting_facts}

    paragraphs: List[str] = []
    for item in context_field:
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, sentences = item
        title = str(title)
        if not isinstance(sentences, list):
            continue

        kept: List[str] = []
        if mode == "supporting_only":
            for i, sent in enumerate(sentences):
                if (title, i) in sf_lookup:
                    kept.append(str(sent).strip())
        else:
            kept = [str(sent).strip() for sent in sentences if str(sent).strip()]

        kept = [s for s in kept if s]
        if not kept:
            continue

        para = title + "\n" + " ".join(kept)
        paragraphs.append(para.strip())

    return "\n\n".join(paragraphs).strip()


def iter_boolq_examples(split: str, limit: Optional[int]) -> Iterable[NormalizedExample]:
    ds = load_dataset("google/boolq", split=split)
    count = 0
    for row in ds:
        query = str(row["question"]).strip()
        passage = str(row["passage"]).strip()
        title = str(row.get("title") or "").strip() or None
        answer = _ensure_bool_answer(row["answer"])

        context_text = passage if not title else f"{title}\n{passage}"
        if not query or not context_text:
            continue

        metadata = {
            "hf_dataset": "google/boolq",
            "hf_row_idx": count,
        }
        yield NormalizedExample(
            query=query,
            expected_answer=answer,
            context_text=context_text,
            source_dataset="boolq",
            source_split=split,
            source_id=f"boolq_{split}_{count}",
            title=title,
            metadata=metadata,
        )
        count += 1
        if limit is not None and count >= limit:
            break


def iter_hotpot_yesno_examples(split: str, limit: Optional[int], *, context_mode: str) -> Iterable[NormalizedExample]:
    ds = load_dataset("hotpotqa/hotpot_qa", split=split)
    seen = 0
    kept = 0
    for row in ds:
        seen += 1
        answer_raw = str(row.get("answer", "")).strip().lower()
        if answer_raw not in {"yes", "no", "true", "false"}:
            continue

        question = str(row["question"]).strip()
        supporting_facts = row.get("supporting_facts")
        context_text = _join_hotpot_context(
            row.get("context"),
            mode=context_mode,
            supporting_facts=supporting_facts,
        )
        if not question or not context_text:
            continue

        metadata = {
            "hf_dataset": "hotpotqa/hotpot_qa",
            "hf_original_id": row.get("id"),
            "hotpot_type": row.get("type"),
            "hotpot_level": row.get("level"),
            "supporting_facts": supporting_facts,
            "context_mode": context_mode,
            "hf_seen_rows": seen,
        }
        yield NormalizedExample(
            query=question,
            expected_answer=_ensure_bool_answer(answer_raw),
            context_text=context_text,
            source_dataset="hotpot_yesno",
            source_split=split,
            source_id=f"hotpot_yesno_{split}_{kept}",
            title=None,
            metadata=metadata,
        )
        kept += 1
        if limit is not None and kept >= limit:
            break


# -----------------------------
# raw examples build
# -----------------------------
def build_raw_examples(
    *,
    dataset_name: str,
    split: str,
    limit: Optional[int],
    out_name: str,
    hotpot_context_mode: str,
) -> Dict[str, Path]:
    base_dir = REPORTS_DIR / "dataset_creation" / out_name
    context_dir = DATA_DIR / "generated_contexts" / out_name
    base_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "boolq":
        iterator = iter_boolq_examples(split=split, limit=limit)
    elif dataset_name == "hotpot_yesno":
        iterator = iter_hotpot_yesno_examples(split=split, limit=limit, context_mode=hotpot_context_mode)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    raw_examples: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "dataset": dataset_name,
        "split": split,
        "limit": limit,
        "hotpot_context_mode": hotpot_context_mode,
        "examples_written": 0,
    }

    for idx, ex in enumerate(iterator):
        qslug = _slugify(ex.query, max_len=50)
        context_path = context_dir / f"{idx:05d}_{qslug}.txt"
        context_path.write_text(ex.context_text, encoding="utf-8")
        raw_examples.append(ex.to_examples_json_item(context_path))

    manifest["examples_written"] = len(raw_examples)

    examples_json_path = base_dir / "examples.json"
    manifest_path = base_dir / "manifest.json"
    _write_json(examples_json_path, raw_examples)
    _write_json(manifest_path, manifest)

    return {
        "base_dir": base_dir,
        "context_dir": context_dir,
        "examples_json": examples_json_path,
        "manifest_json": manifest_path,
    }


# -----------------------------
# evaluation + enrichment
# -----------------------------
def run_evaluate_questions(
    *,
    examples_json: Path,
    out_report: Path,
    models: List[str],
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    script_path = project_root / "RagAdaptation" / "evaluate_questions.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find evaluate_questions.py at {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--input", str(examples_json),
        "--out", str(out_report),
        "--device", device,
        "--batch_size", str(batch_size),
        "--max_new_tokens", str(max_new_tokens),
        "--models",
        *models,
    ]
    subprocess.run(cmd, check=True)

def enrich_eval_report_with_raw_examples(
    *,
    raw_examples_json: Path,
    raw_report_json: Path,
    out_report_json: Path,
) -> Dict[str, Any]:
    raw_examples = json.loads(raw_examples_json.read_text(encoding="utf-8"))
    payload = json.loads(raw_report_json.read_text(encoding="utf-8"))

    if not isinstance(raw_examples, list):
        raise ValueError("raw_examples_json must contain a list")
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("raw_report_json must contain a dict with a 'results' field")

    enriched_results: List[Dict[str, Any]] = []

    for item in payload.get("results", []):
        idx = item.get("idx")
        if not isinstance(idx, int):
            raise ValueError(f"Report item missing integer idx: {item}")
        if idx < 0 or idx >= len(raw_examples):
            raise IndexError(f"Report idx {idx} is out of range for raw examples length {len(raw_examples)}")

        ex = raw_examples[idx]

        merged = dict(ex)
        merged.update(item)

        # נשמור גם expected_answer_norm אם חסר
        if "expected_answer_norm" not in merged and "expected_answer" in merged:
            exp = merged["expected_answer"]
            if isinstance(exp, bool):
                merged["expected_answer_norm"] = "true" if exp else "false"

        enriched_results.append(merged)

    enriched_payload = dict(payload)
    enriched_payload["results"] = enriched_results
    enriched_payload.setdefault("meta", {})
    enriched_payload["meta"]["raw_examples_json"] = str(raw_examples_json)
    enriched_payload["meta"]["raw_report_json"] = str(raw_report_json)

    out_report_json.write_text(
        json.dumps(enriched_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return enriched_payload

def enrich_eval_report_with_raw_examples(
    *,
    raw_examples_path: Path,
    eval_report_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    raw_examples = json.loads(raw_examples_path.read_text(encoding="utf-8"))
    report_payload = json.loads(eval_report_path.read_text(encoding="utf-8"))
    results = report_payload.get("results", [])

    if not isinstance(raw_examples, list):
        raise ValueError("examples.json must be a list")
    if not isinstance(results, list):
        raise ValueError("eval report 'results' must be a list")
    if len(results) != len(raw_examples):
        raise ValueError(
            f"Mismatch between raw examples ({len(raw_examples)}) and eval report results ({len(results)})"
        )

    enriched_results: List[Dict[str, Any]] = []
    for i, row in enumerate(results):
        raw_idx = row.get("idx", i)
        if not isinstance(raw_idx, int) or raw_idx < 0 or raw_idx >= len(raw_examples):
            raw_idx = i
        raw_item = raw_examples[raw_idx]

        merged = dict(raw_item)
        merged.update(row)
        merged.setdefault("idx", i)
        merged.setdefault("context_path", raw_item.get("context_path"))
        merged.setdefault("source_dataset", raw_item.get("source_dataset"))
        merged.setdefault("source_split", raw_item.get("source_split"))
        merged.setdefault("source_id", raw_item.get("source_id"))
        if raw_item.get("title") and "title" not in merged:
            merged["title"] = raw_item["title"]
        enriched_results.append(merged)

    enriched_payload = {
        **report_payload,
        "results": enriched_results,
        "benchmark_build_meta": {
            "enriched_from_examples_json": str(raw_examples_path),
            "original_eval_report": str(eval_report_path),
        },
    }
    _write_json(output_path, enriched_payload)
    return enriched_payload


# -----------------------------
# filtered reports + summaries
# -----------------------------
def build_filtered_reports(
    *,
    report_payload_or_path: Any,
    target_models: List[str],
    out_dir: Path,
) -> Dict[str, Path]:
    payload, results = _load_payload_items(report_payload_or_path)

    any_flip: List[Dict[str, Any]] = []
    all_flip: List[Dict[str, Any]] = []
    per_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in target_models}
    flip_directions: Dict[str, Dict[str, int]] = {
        m: {"true_to_false": 0, "false_to_true": 0, "other": 0} for m in target_models
    }

    for item in results:
        flags = []
        per_item_model = item.get("per_model", {})
        for model_name in target_models:
            model_payload = per_item_model.get(model_name, {})
            relevant = bool(model_payload.get("relevant", False))
            flags.append(relevant)
            if relevant:
                per_model[model_name].append(item)
                no_ctx = (
                    model_payload.get("probs_without_context", {}) or {}
                ).get("label")
                with_ctx = model_payload.get("prob_label_with_context")
                if no_ctx == "true" and with_ctx == "false":
                    flip_directions[model_name]["true_to_false"] += 1
                elif no_ctx == "false" and with_ctx == "true":
                    flip_directions[model_name]["false_to_true"] += 1
                else:
                    flip_directions[model_name]["other"] += 1
        if any(flags):
            any_flip.append(item)
        if target_models and all(flags):
            all_flip.append(item)

    out_paths: Dict[str, Path] = {}

    any_path = out_dir / "report_any_flip.json"
    _write_json(any_path, {**payload, "results": any_flip})
    out_paths["any_flip"] = any_path

    all_path = out_dir / "report_all_models_flip.json"
    _write_json(all_path, {**payload, "results": all_flip})
    out_paths["all_models_flip"] = all_path

    for model_name, items in per_model.items():
        safe_name = model_name.replace("/", "__")
        p = out_dir / f"report_flip_only__{safe_name}.json"
        _write_json(p, {**payload, "results": items})
        out_paths[f"per_model::{model_name}"] = p

    summary = {
        "total_examples": len(results),
        "any_flip": len(any_flip),
        "all_models_flip": len(all_flip),
        "per_model_flip_counts": {m: len(v) for m, v in per_model.items()},
        "per_model_flip_directions": flip_directions,
    }
    summary_path = out_dir / "flip_summary.json"
    _write_json(summary_path, summary)
    out_paths["summary"] = summary_path

    return out_paths


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Create a Hugging Face benchmark, evaluate it with your models, "
            "and emit enriched report files that the current RagAdaptation pipeline can consume."
        )
    )
    ap.add_argument("--dataset", choices=["boolq", "hotpot_yesno"], required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_name", default=None, help="Folder name under outputs/reports/dataset_creation/")
    ap.add_argument("--hotpot_context_mode", choices=["full", "supporting_only"], default="full")
    ap.add_argument("--skip_eval", action="store_true", help="Only build examples.json and context files.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--no_internal_splits", action="store_true", help="Do not emit train/dev/test JSON files.")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3"],
    )
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    out_name = args.out_name
    if out_name is None:
        out_name = f"{args.dataset}__{args.split}__{('all' if args.limit is None else args.limit)}"
        if args.dataset == "hotpot_yesno":
            out_name += f"__{args.hotpot_context_mode}"

    paths = build_raw_examples(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        out_name=out_name,
        hotpot_context_mode=args.hotpot_context_mode,
    )

    print(f"[ok] wrote raw examples to: {paths['examples_json']}")
    print(f"[ok] wrote context files under: {paths['context_dir']}")

    # Optional: if your build_raw_examples already created internal splits, mention them.
    for key in ("examples_train_json", "examples_dev_json", "examples_test_json"):
        if key in paths:
            print(f"[ok] wrote raw split file: {paths[key]}")

    if args.skip_eval:
        return

    raw_report_path = paths["base_dir"] / "eval_report_raw.json"
    enriched_report_path = paths["base_dir"] / "eval_report.json"

    run_evaluate_questions(
        examples_json=paths["examples_json"],
        out_report=raw_report_path,
        models=list(args.models),
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[ok] wrote raw evaluation report to: {raw_report_path}")

    enriched_payload = enrich_eval_report_with_raw_examples(
        raw_examples_json=paths["examples_json"],
        raw_report_json=raw_report_path,
        out_report_json=enriched_report_path,
    )
    print(f"[ok] wrote enriched evaluation report to: {enriched_report_path}")

    filtered = build_filtered_reports(
        report_path=enriched_report_path,
        target_models=list(args.models),
    )

    print(f"[ok] wrote flip summary to: {filtered['summary']}")
    print(f"[ok] report for any-model flips: {filtered['any_flip']}")
    print(f"[ok] report for all-model flips: {filtered['all_models_flip']}")
    for model_name in args.models:
        key = f"per_model::{model_name}"
        if key in filtered:
            print(f"[ok] report for {model_name}: {filtered[key]}")

    # Optional: if you later add split-specific filtered reports, this won't break meanwhile.
    for key, value in filtered.items():
        if key.startswith("split::"):
            print(f"[ok] {key}: {value}")


if __name__ == "__main__":
    main()
