from __future__ import annotations

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
def load_model_and_tok(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


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
    ap.add_argument("--download_url",action="store_true", default=False)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3"],
    )
    args = ap.parse_args()

    examples = load_examples(args.input)
    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)

    # Load each model once
    models: Dict[str, Tuple[Any, Any]] = {}
    for mid in args.models:
        model, tok = load_model_and_tok(mid, args.device)
        models[mid] = (model, tok)

    report: Dict[str, Any] = {"meta": {"device": args.device,"batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,"models": args.models,"candidate_strings": {"true": " true", "false": " false"},},
        "results": [],}

    for i, ex in enumerate(examples):
        q = ex["query"]
        expected_norm = norm_expected(ex["expected_answer"])
        url = ex.get("contradicting_url")

        row = {"idx": i,"query": q,"expected_answer_raw": ex["expected_answer"],
            "expected_answer_norm": expected_norm,"contradicting_url": url,"prompt": "","per_model": {},}
        context_path = ex.get("context_path")

        for mid, (model, tok) in models.items():
            prompt = build_chat_prompt(tok, q, context="")
            gen_text = generate_answer(model, tok, prompt, max_new_tokens=args.max_new_tokens)
            try:
                gen_norm = normalize_true_false(gen_text)
            except Exception:
                gen_norm = None

            # 2) compute_probs (expects list of prompts)
            stats_list, full_logps_without_context = compute_probs(model=model, tok=tok, prompts=[prompt], device=model.device,
                                                                   expected_result=None, batch_size=1,
                                                                   masked_context_list=None, return_full_logp=True,
                                                                   file_name=f"compute_probs_baseline_without_context_{mid.replace('/', '__')}_idx{i}.txt",
                                                                   detect_flip_to_true=(expected_norm == "false"), )
            tf_stats = stats_list[0]
            # tf_stats keys: logP_true, logP_false, log_odds, p_true :contentReference[oaicite:3]{index=3}

            # derive label for convenience
            prob_label = "true" if tf_stats["logP_true"] > tf_stats["logP_false"] else "false"
            correct_by_probs = (prob_label == expected_norm) if expected_norm in ("true", "false") else None
            correct_by_gen = (gen_norm == expected_norm) if (gen_norm is not None and expected_norm in ("true", "false")) else None


            #Evaluate question with full context

            docs = load_documents_any(context_path)
            full_context = combine_document_text(docs)
            prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)
            baseline_prompt = prompt_template.format(context=full_context, question=q)

            baseline_stats_list, full_logps_with_context = compute_probs(model=model, tok=tok, prompts=[baseline_prompt], device=model.device,
                                                                         expected_result=None, batch_size=1,
                                                                         masked_context_list=None, return_full_logp=True,
                                                                         file_name=f"compute_probs_baseline_with_context_{mid.replace('/', '__')}_idx{i}.txt",
                                                                         detect_flip_to_true=(expected_norm == "false"), )

            baseline_stats = baseline_stats_list[0]
            prob_label_with_context = "true" if baseline_stats["logP_true"] > baseline_stats["logP_false"] else "false"

            relevant= prob_label_with_context != prob_label

            row["per_model"][mid] = {
                "relevant":relevant,
                "generated_text_without_context": gen_text,"generated_norm_without_context": gen_norm,"probs_without_context": {**tf_stats,
                    "label": prob_label,
                    "full_logps_without_context": full_logps_without_context,  # list with one p_true value when return_full_logp=True
                 },"correct_by_probs_without_context": correct_by_probs,"correct_by_generation": correct_by_gen,
                "prob_label_with_context": prob_label_with_context,
                "full_logps_without_context":full_logps_with_context,
            }
            context_path = ex.get("context_path")
            if not context_path:
                raise ValueError(f"Example {i} has no context_path")

        report["results"].append(row)

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()