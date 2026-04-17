from __future__ import annotations
from RagAdaptation.core.model_config import ModelConfig
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from RagAdaptation.core.artifacts import method_dir, write_json

from RagAdaptation.core.artifacts import create_run_root, example_dir, model_dir, write_example_inputs, write_manifest
from RagAdaptation.core.documents import combine_document_text, load_documents_any
from RagAdaptation.pipeline.config import PipelineConfig
from RagAdaptation.prompts_format import normalize_true_false
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- robustly add the project root (folder that contains RagAdaptation/) to sys.path ---
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()


def build_chat_prompt(tok, question: str, context: str = "") -> str:
    # Keep your instruction exactly: "Answer with exactly one word: true or false."
    user_msg = ("Answer with exactly one word: true or false.\n" "Use ONLY the context.\n\n" f"Context:\n{context}\n\n"
        f"question: {question}\n"  "Answer:")

    # If the tokenizer supports chat templates, use them:
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        messages = [{"role": "user", "content": user_msg}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback (works but less reliable for some models)
    return user_msg

# Walk up until we find a directory that contains "RagAdaptation"
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

from langchain_core.prompts import ChatPromptTemplate
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE, normalize_true_false
from RagAdaptation.compute_probs_updated import compute_probs
from RagAdaptation.document_handling.create_docuemnt import download_url
# -----------------------------
# Input loading (your examples.json schema)
# -----------------------------
def load_examples(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("examples.json must be a LIST of dicts")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{i} is not a dict: {type(item)}")

        q = item.get("query")
        exp = item.get("expected_answer")
        if exp is None:
            exp=item.get("expected_answer_raw")
        url = item.get("contradicting_url")
        context_path=item.get("context_path")
        if q is None or exp is None:
            raise ValueError(f"Missing query/expected_answer in item #{i}: {item}")
        out.append({"query": q, "expected_answer": exp, "contradicting_url": url,"context_path": context_path})
    return out


# -----------------------------
# Model load (once per model)
# -----------------------------


@torch.no_grad()
def generate_answer(model, tok, prompt: str, max_new_tokens: int = 20) -> str:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = out_ids[0, enc["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def norm_expected(x: Any) -> str:
    # expected_answer in your examples.json is usually boolean true/false
    if isinstance(x, bool):
        return "true" if x else "false"
    return normalize_true_false(str(x))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to examples.json", default="./data/examples.json")
    ap.add_argument("--out", default="report_compute_probs.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--download_url", action="store_true", default=False)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3"],
    )
    args = ap.parse_args()

    examples = load_examples(args.input)

    # טוענת את כל הקונטקסטים פעם אחת בלבד
    prepared_examples = []
    for i, ex in enumerate(examples):
        context_path = ex.get("context_path")
        if not context_path:
            raise ValueError(f"Example {i} has no context_path")

        docs = load_documents_any(context_path)
        full_context = combine_document_text(docs)

        prepared_examples.append({
            "idx": i,
            "query": ex["query"],
            "expected_answer_raw": ex["expected_answer"],
            "expected_answer_norm": norm_expected(ex["expected_answer"]),
            "contradicting_url": ex.get("contradicting_url"),
            "context_path": context_path,
            "full_context": full_context,
        })

    report: Dict[str, Any] = {
        "meta": {
            "device": str(args.device),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "models": args.models,
            "candidate_strings": {"true": " true", "false": " false"},
        },
        "results": [],
    }

    # מאתחלת מראש את התוצאות לפי דוגמאות
    for ex in prepared_examples:
        report["results"].append({
            "idx": ex["idx"],
            "query": ex["query"],
            "expected_answer_raw": ex["expected_answer_raw"],
            "expected_answer_norm": ex["expected_answer_norm"],
            "contradicting_url": ex["contradicting_url"],
            "prompt": "",
            "per_model": {},
        })

    # לולאה חיצונית על מודלים
    for mid in args.models:
        print("Evaluating model", mid)

        model_config = ModelConfig(mid)
        model, tok, _ = model_config.load()

        true_variants = model_config.get_true_variants()
        false_variants = model_config.get_false_variants()

        for ex in prepared_examples:
            i = ex["idx"]
            q = ex["query"]
            expected_norm = ex["expected_answer_norm"]
            full_context = ex["full_context"]

            row = report["results"][i]

            prompt = model_config.format_prompt(
                question=q,
                context="",
                context_cite_at2_formating=False,
                empty=True,
            )

            gen_text = generate_answer(model, tok, prompt, max_new_tokens=args.max_new_tokens)
            try:
                gen_norm = normalize_true_false(gen_text)
            except Exception:
                gen_norm = None

            stats_list, full_logps_without_context = compute_probs(
                model=model,
                tok=tok,
                prompts=[prompt],
                device=model.device,
                expected_result=None,
                batch_size=4,
                masked_context_list=None,
                true_variants=true_variants,
                false_variants=false_variants,
                return_full_logp=False,
                file_name=f"compute_probs_baseline_without_context_{mid.replace('/', '__')}_idx{i}.txt",
                detect_flip_to_true=(expected_norm == "false"),
                save_file=False,
                stop_on_flip=True,
            )
            tf_stats = stats_list[0]

            prob_label = "true" if tf_stats["logP_true"] > tf_stats["logP_false"] else "false"
            generated_eq_probs_without_context = (prob_label == gen_norm) if gen_norm is not None else None

            baseline_prompt = model_config.format_prompt(
                question=q,
                context=full_context,
                context_cite_at2_formating=False,
                empty=False,
            )

            baseline_stats_list, full_logps_with_context = compute_probs(
                model=model,
                tok=tok,
                prompts=[baseline_prompt],
                device=model.device,
                expected_result=None,
                batch_size=1,
                masked_context_list=None,
                return_full_logp=False,
                true_variants=true_variants,
                false_variants=false_variants,
                file_name=f"compute_probs_baseline_with_context_{mid.replace('/', '__')}_idx{i}.txt",
                detect_flip_to_true=(expected_norm == "false"),
                save_file=False,
                stop_on_flip=True,
            )

            gen_text_with_context = generate_answer(
                model, tok, baseline_prompt, max_new_tokens=args.max_new_tokens
            )
            try:
                gen_norm_with_context = normalize_true_false(gen_text_with_context)
            except Exception:
                gen_norm_with_context = None

            baseline_stats = baseline_stats_list[0]
            prob_label_with_context = (
                "true" if baseline_stats["logP_true"] > baseline_stats["logP_false"] else "false"
            )
            generated_eq_probs_with_context = (
                prob_label_with_context == gen_norm_with_context
                if gen_norm_with_context is not None
                else None
            )

            relevant = (prob_label_with_context != prob_label)

            row["per_model"][mid] = {
                "relevant": relevant,
                "generated_norm_without_context": gen_norm,
                "generated_eq_probs_without_context": generated_eq_probs_without_context,
                "probs_without_context": {
                    **tf_stats,
                    "label": prob_label,
                    "full_logps_without_context": full_logps_without_context,
                },
                "prob_label_with_context": prob_label_with_context,
                "full_logps_with_context": full_logps_with_context,
                "gen_norm_with_context": gen_norm_with_context,
                "generated_eq_probs_with_context": generated_eq_probs_with_context,
            }

        # שומרת checkpoint אחרי כל מודל
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

        model_config.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()