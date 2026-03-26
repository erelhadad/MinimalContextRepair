from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

# --- make project imports work even if you run this file directly ---
_THIS_FILE = Path(__file__).resolve()
project_root = None
for p in [_THIS_FILE.parent] + list(_THIS_FILE.parents):
    if (p / "RagAdaptation").is_dir():
        project_root = p
        break

if project_root is None:
    raise RuntimeError(
        "Could not find project root containing a 'RagAdaptation/' directory. "
        f"Started from {_THIS_FILE}."
    )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from RagAdaptation.core.documents import combine_document_text, load_documents_any


def load_report(path: str | Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Supports both:
      1. {"meta": ..., "results": [...]}
      2. [...]
    Returns:
      (full_json_object, results_list)

    If the input is a list, wraps it into {"results": ...} for uniform writing.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data, data["results"]

    if isinstance(data, list):
        wrapped = {"results": data}
        return wrapped, wrapped["results"]

    raise ValueError("Unsupported eval_report format. Expected dict with 'results' or a list.")


def resolve_models(report_obj: Dict[str, Any], results: List[Dict[str, Any]], cli_models: List[str] | None) -> List[str]:
    if cli_models:
        return cli_models

    meta_models = report_obj.get("meta", {}).get("models")
    if isinstance(meta_models, list) and meta_models:
        return [str(m) for m in meta_models]

    discovered = []
    seen = set()
    for row in results:
        per_model = row.get("per_model", {})
        if isinstance(per_model, dict):
            for model_id in per_model.keys():
                if model_id not in seen:
                    seen.add(model_id)
                    discovered.append(model_id)

    if discovered:
        return discovered

    raise ValueError(
        "Could not infer model ids. Pass them explicitly with --models."
    )


def load_tokenizer_safe(model_id: str):
    """
    Try normal tokenizer load first; if needed, retry with trust_remote_code=True.
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    return tok


def token_len_text(tok, text: str) -> int:
    enc = tok(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    return len(enc["input_ids"])


def build_chat_prompt(tok, question: str, context: str) -> str:
    user_msg = (
        "Answer with exactly one word: true or false.\n"
        "Use ONLY the context.\n\n"
        f"Context:\n{context}\n\n"
        f"question: {question}\n"
        "Answer:"
    )

    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        messages = [{"role": "user", "content": user_msg}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return user_msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to eval_report json")
    ap.add_argument("--out", default=None, help="Output path. If omitted and --inplace is not set, writes '<input>_with_context_lengths.json'")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input file")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model ids. If omitted, uses report.meta.models or per_model keys.",
    )
    ap.add_argument(
        "--also_prompt_lengths",
        action="store_true",
        help="Also compute full prompt length per model, not just raw context length",
    )
    args = ap.parse_args()

    report_obj, results = load_report(args.input)
    models = resolve_models(report_obj, results, args.models)

    print(f"[info] models = {models}")

    tokenizers: Dict[str, Any] = {}
    for model_id in models:
        print(f"[load] tokenizer for {model_id}")
        tokenizers[model_id] = load_tokenizer_safe(model_id)

    context_cache: Dict[str, str] = {}

    for row_idx, row in enumerate(results):
        context_path = row.get("context_path")
        if not context_path:
            raise ValueError(f"Row {row_idx} has no context_path")

        if context_path not in context_cache:
            docs = load_documents_any(context_path)
            context_cache[context_path] = combine_document_text(docs)

        full_context = context_cache[context_path]
        query = row.get("query") or row.get("question") or ""

        row.setdefault("per_model", {})
        row["context_lengths"] = {
            "chars": len(full_context),
            "words_split": len(full_context.split()),
            "per_model": {},
        }

        for model_id in models:
            tok = tokenizers[model_id]
            context_tokens = token_len_text(tok, full_context)

            row["context_lengths"]["per_model"][model_id] = {
                "tokenizer_name": getattr(tok, "name_or_path", model_id),
                "context_tokens": context_tokens,
            }

            # Also copy the per-model value into the existing per_model block
            row["per_model"].setdefault(model_id, {})
            row["per_model"][model_id]["context_length_tokens"] = context_tokens
            row["per_model"][model_id]["context_tokenizer_name"] = getattr(tok, "name_or_path", model_id)

            if args.also_prompt_lengths:
                prompt = build_chat_prompt(tok, question=query, context=full_context)
                prompt_tokens = token_len_text(tok, prompt)
                question_tokens = token_len_text(tok, query)

                row["context_lengths"]["per_model"][model_id]["question_tokens"] = question_tokens
                row["context_lengths"]["per_model"][model_id]["full_prompt_tokens"] = prompt_tokens

                row["per_model"][model_id]["question_length_tokens"] = question_tokens
                row["per_model"][model_id]["full_prompt_length_tokens"] = prompt_tokens

        if row_idx % 10 == 0:
            print(f"[done] processed row {row_idx + 1}/{len(results)}")

    report_obj.setdefault("meta", {})
    report_obj["meta"]["context_lengths_added"] = True
    report_obj["meta"]["context_length_models"] = models

    input_path = Path(args.input)
    if args.inplace:
        out_path = input_path
    elif args.out:
        out_path = Path(args.out)
    else:
        out_path = input_path.with_name(f"{input_path.stem}_with_context_lengths{input_path.suffix}")

    out_path.write_text(json.dumps(report_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()