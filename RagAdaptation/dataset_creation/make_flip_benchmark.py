from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from RagAdaptation.core.model_config import ModelConfig
from datasets import load_dataset
import numpy as np

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
        out: Dict[str, Any] = {
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

def _normalize_hotpot_context(context_field: Any) -> List[Tuple[str, List[str]]]:
    # New HF-style format:
    # {"title": [...], "sentences": [[...], [...], ...]}
    if isinstance(context_field, dict):
        titles = context_field.get("title", [])
        sentences_lists = context_field.get("sentences", [])
        return [
            (str(title), sentences)
            for title, sentences in zip(titles, sentences_lists)
            if isinstance(sentences, list)
        ]

    # Older/expected format:
    # [[title, [sent1, sent2, ...]], ...]
    if isinstance(context_field, list):
        out = []
        for item in context_field:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sentences = item
                if isinstance(sentences, list):
                    out.append((str(title), sentences))
        return out

    raise ValueError(f"Hotpot context has unexpected type: {type(context_field)}")


def _normalize_supporting_facts(supporting_facts: Any) -> set[tuple[str, int]]:
    if not supporting_facts:
        return set()

    # New HF-style format:
    # {"title": [...], "sent_id": [...]}
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", [])
        return {
            (str(title), int(sent_id))
            for title, sent_id in zip(titles, sent_ids)
        }

    # Older/expected format:
    # [[title, sent_id], ...]
    if isinstance(supporting_facts, list):
        out = set()
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sent_id = item
                out.add((str(title), int(sent_id)))
        return out

    raise ValueError(f"Hotpot supporting_facts has unexpected type: {type(supporting_facts)}")


def _join_hotpot_context(
    context_field: Any,
    *,
    mode: str,
    supporting_facts: Optional[Any] = None,
) -> str:
    context_items = _normalize_hotpot_context(context_field)
    sf_lookup = _normalize_supporting_facts(supporting_facts)

    paragraphs: List[str] = []
    for title, sentences in context_items:
        kept: List[str] = []

        if mode == "supporting_only":
            for i, sent in enumerate(sentences):
                if (title, i) in sf_lookup:
                    sent = str(sent).strip()
                    if sent:
                        kept.append(sent)
        else:
            kept = [str(sent).strip() for sent in sentences if str(sent).strip()]

        if not kept:
            continue

        para = title + "\n" + " ".join(kept)
        paragraphs.append(para.strip())

    return "\n\n".join(paragraphs).strip()


def iter_boolq_examples(split: str, limit: Optional[int]) -> Iterable[NormalizedExample]:
    ds = load_dataset("google/boolq", "distractor",split=split)
    kept = 0
    for row_idx, row in enumerate(ds):
        query = str(row["question"]).strip()
        passage = str(row["passage"]).strip()
        title = str(row.get("title") or "").strip() or None
        answer = _ensure_bool_answer(row["answer"])

        context_text = passage if not title else f"{title}\n{passage}"
        if not query or not context_text:
            continue

        metadata = {
            "hf_dataset": "google/boolq",
            "hf_row_idx": row_idx,
        }
        yield NormalizedExample(
            query=query,
            expected_answer=answer,
            context_text=context_text,
            source_dataset="boolq",
            source_split=split,
            source_id=f"boolq_{split}_{row_idx}",
            title=title,
            metadata=metadata,
        )
        kept += 1
        if limit is not None and kept >= limit:
            break


def iter_hotpot_yesno_examples(
    split: str,
    limit: Optional[int],
    *,
    context_mode: str,
) -> Iterable[NormalizedExample]:
    ds = load_dataset("hotpotqa/hotpot_qa","distractor", split=split)
    kept = 0
    seen = 0
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
            metadata=metadata,
        )
        kept += 1
        if limit is not None and kept >= limit:
            break


def _stable_internal_split(
    examples: Sequence[Dict[str, Any]],
    *,
    train_ratio: float,
    dev_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, dev_ratio >= 0, train_ratio + dev_ratio < 1")

    tagged: List[Tuple[str, Dict[str, Any]]] = []
    for ex in examples:
        sid = str(ex.get("source_id") or ex.get("query") or "")
        digest = hashlib.sha1(sid.encode("utf-8")).hexdigest()
        tagged.append((digest, ex))

    tagged.sort(key=lambda t: t[0])
    ordered = [ex for _, ex in tagged]
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    if n >= 3:
        n_train = max(1, n_train)
        n_dev = max(1, n_dev) if dev_ratio > 0 else 0
        if n_train + n_dev >= n:
            n_dev = max(0, n - n_train - 1)
    n_test = n - n_train - n_dev

    return {
        "train": ordered[:n_train],
        "dev": ordered[n_train:n_train + n_dev],
        "test": ordered[n_train + n_dev:n_train + n_dev + n_test],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_raw_examples(
    *,
    dataset_name: str,
    split: str,
    limit: Optional[int],
    out_name: str,
    hotpot_context_mode: str,
    train_ratio: float,
    dev_ratio: float,
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
    for idx, ex in enumerate(iterator):
        qslug = _slugify(ex.query, max_len=50)
        context_path = context_dir / f"{idx:05d}_{qslug}.txt"
        context_path.write_text(ex.context_text, encoding="utf-8")
        raw_examples.append(ex.to_examples_json_item(context_path))

    manifest: Dict[str, Any] = {
        "dataset": dataset_name,
        "split": split,
        "limit": limit,
        "hotpot_context_mode": hotpot_context_mode,
        "examples_written": len(raw_examples),
        "internal_split_train_ratio": train_ratio,
        "internal_split_dev_ratio": dev_ratio,
        "internal_split_test_ratio": 1.0 - train_ratio - dev_ratio,
    }

    examples_json_path = base_dir / "examples.json"
    manifest_path = base_dir / "manifest.json"
    _write_json(examples_json_path, raw_examples)
    _write_json(manifest_path, manifest)

    split_map = _stable_internal_split(raw_examples, train_ratio=train_ratio, dev_ratio=dev_ratio)
    examples_train_json = base_dir / "examples_train.json"
    examples_dev_json = base_dir / "examples_dev.json"
    examples_test_json = base_dir / "examples_test.json"
    _write_json(examples_train_json, split_map["train"])
    _write_json(examples_dev_json, split_map["dev"])
    _write_json(examples_test_json, split_map["test"])

    return {
        "base_dir": base_dir,
        "context_dir": context_dir,
        "examples_json": examples_json_path,
        "manifest_json": manifest_path,
        "examples_train_json": examples_train_json,
        "examples_dev_json": examples_dev_json,
        "examples_test_json": examples_test_json,
    }

# add this import near the top
from transformers import AutoTokenizer


def _build_eval_prompt_for_length(tok, question: str, context: str) -> str:
    """
    Match the prompt structure used by evaluate_questions.py as closely as possible.
    """
    user_msg = (
        "Answer with exactly one word: true or false.\n"
        "Use ONLY the context.\n\n"
        f"Context:\n{context}\n\n"
        f"question: {question}\n"
        "Answer:"
    )

    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return user_msg


def run_evaluate_length(*,examples_json: Path,
    models: List[str],out_json: Optional[Path] = None,) -> Dict[str, Any]:
    """
    Update the JSON file with per-query token lengths for each model.
    If out_json is None, updates examples_json in place.
    """
    payload = json.loads(examples_json.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        rows = payload
        payload_kind = "list"
    elif isinstance(payload, dict) and "results" in payload:
        rows = payload["results"]
        payload_kind = "results_dict"
    else:
        raise ValueError("Unsupported JSON format: expected a list or a dict with 'results'")

    # load tokenizers once
    tokenizers: Dict[str, Any] = {}
    for model_id in models:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tokenizers[model_id] = tok

    for row in rows:
        query = row.get("query") or row.get("question")
        context_path = row.get("context_path")

        if not query:
            continue
        if not context_path:
            continue

        context_text = Path(context_path).read_text(encoding="utf-8")

        row.setdefault("detected_lengths_by_model", {})

        for model_id, tok in tokenizers.items():
            q_len = len(tok(query, add_special_tokens=False)["input_ids"])
            c_len = len(tok(context_text, add_special_tokens=False)["input_ids"])

            row["detected_lengths_by_model"][model_id] = {
                "question_tokens": int(q_len),
                "context_tokens": int(c_len),
            }

    out_path = out_json or examples_json

    if payload_kind == "list":
        final_payload = rows
    else:
        payload["results"] = rows
        final_payload = payload

    _write_json(out_path, final_payload)
    return final_payload


def run_evaluate_questions(
    *,
    examples_json: Path,
    out_report: Path,
    models: List[str],
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "RagAdaptation.evaluate_questions",
        "--input",
        str(examples_json),
        "--out",
        str(out_report),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--max_new_tokens",
        str(max_new_tokens),
        "--models",
        *models,
    ]
    subprocess.run(cmd, check=True, cwd=str(project_root))


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

    _write_json(out_report_json, enriched_payload)
    return enriched_payload


def _split_results_by_raw_internal_split(
    results: Sequence[Dict[str, Any]],
    *,
    train_ratio: float,
    dev_ratio: float,
) -> Dict[str, List[Dict[str, Any]]]:
    return _stable_internal_split(list(results), train_ratio=train_ratio, dev_ratio=dev_ratio)


def build_filtered_reports(
    *,
    report_path: Path,
    target_models: List[str],
    train_ratio: float,
    dev_ratio: float,
) -> Dict[str, Path]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results = payload.get("results", [])

    any_flip: List[Dict[str, Any]] = []
    all_flip: List[Dict[str, Any]] = []
    per_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in target_models}
    per_model_direction: Dict[str, Dict[str, int]] = {
        m: {"false_to_true": 0, "true_to_false": 0, "other": 0} for m in target_models
    }

    for item in results:
        flags: List[bool] = []
        per_item_model = item.get("per_model", {})
        for model_name in target_models:
            model_info = per_item_model.get(model_name, {})
            relevant = bool(model_info.get("relevant", False))
            flags.append(relevant)
            if relevant:
                per_model[model_name].append(item)
                before = model_info.get("probs_without_context", {}).get("label")
                after = model_info.get("prob_label_with_context")
                if before == "false" and after == "true":
                    per_model_direction[model_name]["false_to_true"] += 1
                elif before == "true" and after == "false":
                    per_model_direction[model_name]["true_to_false"] += 1
                else:
                    per_model_direction[model_name]["other"] += 1
        if any(flags):
            any_flip.append(item)
        if target_models and all(flags):
            all_flip.append(item)

    out_paths: Dict[str, Path] = {}
    base = report_path.parent

    def _write_report(name: str, rows: List[Dict[str, Any]]) -> Path:
        p = base / name
        report_payload = dict(payload)
        report_payload["results"] = rows
        _write_json(p, report_payload)
        return p

    out_paths["any_flip"] = _write_report("report_any_flip.json", any_flip)
    out_paths["all_models_flip"] = _write_report("report_all_models_flip.json", all_flip)

    for model_name, items in per_model.items():
        safe_name = model_name.replace("/", "__")
        out_paths[f"per_model::{model_name}"] = _write_report(f"report_flip_only__{safe_name}.json", items)

    summary = {
        "total_examples": len(results),
        "any_flip": len(any_flip),
        "all_models_flip": len(all_flip),
        "per_model_flip_counts": {m: len(v) for m, v in per_model.items()},
        "per_model_flip_direction_counts": per_model_direction,
    }
    summary_path = base / "flip_summary.json"
    _write_json(summary_path, summary)
    out_paths["summary"] = summary_path

    split_map = _split_results_by_raw_internal_split(results, train_ratio=train_ratio, dev_ratio=dev_ratio)
    for split_name, split_rows in split_map.items():
        split_report = dict(payload)
        split_report["results"] = split_rows
        split_path = base / f"eval_report__{split_name}.json"
        _write_json(split_path, split_report)
        out_paths[f"split::{split_name}"] = split_path

    return out_paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Create a Hugging Face benchmark, evaluate it with your models, "
            "and emit report files your current RagAdaptation pipeline can consume."
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
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--eval_length", action="store_true", help="Compute tokenizer lengths")

    ap.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3","Qwen/Qwen3-4B-Instruct-2507"],
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
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )

    print(f"[ok] wrote raw examples to: {paths['examples_json']}")
    print(f"[ok] wrote context files under: {paths['context_dir']}")
    print(f"[ok] wrote raw train/dev/test splits next to: {paths['examples_json']}")

    if args.skip_eval:
        return

    raw_report_path = paths["base_dir"] / "eval_report_raw.json"
    enriched_report_path = paths["base_dir"] / "eval_report.json"

    if args.eval_length:
        run_evaluate_length(
            examples_json=paths["examples_json"],
            models=list(args.models),
        )

    run_evaluate_questions(
        examples_json=paths["examples_json"],
        out_report=raw_report_path,
        models=list(args.models),
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[ok] wrote raw evaluation report to: {raw_report_path}")

    enrich_eval_report_with_raw_examples(
        raw_examples_json=paths["examples_json"],
        raw_report_json=raw_report_path,
        out_report_json=enriched_report_path,
    )
    print(f"[ok] wrote enriched evaluation report to: {enriched_report_path}")

    filtered = build_filtered_reports(
        report_path=enriched_report_path,
        target_models=list(args.models),
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )

    print(f"[ok] wrote flip summary to: {filtered['summary']}")
    print(f"[ok] report for any-model flips: {filtered['any_flip']}")
    print(f"[ok] report for all-model flips: {filtered['all_models_flip']}")
    for model_name in args.models:
        key = f"per_model::{model_name}"
        print(f"Current Model Evaluathion:{model_name}")
        if key in filtered:
            print(f"[ok] report for {model_name}: {filtered[key]}")
    for split_name in ("train", "dev", "test"):
        key = f"split::{split_name}"
        if key in filtered:
            print(f"[ok] eval split report {split_name}: {filtered[key]}")


if __name__ == "__main__":
    main()
