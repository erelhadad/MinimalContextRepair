from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import torch
import numpy as np

try:
    from context_cite import ContextCiter
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ContextCiter = None

# --- make `import RagAdaptation...` work even if you run this file directly ---
_THIS_FILE = Path(__file__).resolve()
_PKG_DIR = _THIS_FILE.parents[1]          # .../RagAdaptation
_PROJECT_ROOT = _PKG_DIR.parent           # .../RAG_EXP
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RagAdaptation.compute_probs_updated import compute_probs
from RagAdaptation.core.documents import combine_document_text, load_documents_any
from RagAdaptation.core.paths import DATA_DIR
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE, TF_RAG_TEMPLATE_A2T
from RagAdaptation.baseline.bruteforce_common import tokenize_context_with_offsets
from RagAdaptation.core.prompting import ChatPromptTemplate
from RagAdaptation.baseline.partitioner import TokenContextPartitioner
from RagAdaptation.core.models import get_hf_scorer

from RagAdaptation.methods.common import (
    get_at2_token_scores,
    map_at2_scores_to_base_via_sources,
    mask_context_spans_same_length,
)

_HF_TOK = None
_HF_MODEL = None


def _find_token_indices_by_substring(
    full_text: str,
    substring: str,
    offsets_mapping: Sequence[Tuple[int, int]],
    start_search_at: int = 0,
):
    """
    Returns:
      (token_indices, relative_offsets_in_substring, end_char_pos_in_full_text)

    - token_indices: indices in the tokenized full_text
    - relative_offsets_in_substring: [(start,end)] relative to substring start
    """
    begin = full_text.find(substring, start_search_at)
    if begin < 0:
        raise ValueError(
            "Could not locate substring inside the prompt. Check template / uniqueness."
        )
    end = begin + len(substring)

    tok_indices: List[int] = []
    rel_offsets: List[Tuple[int, int]] = []
    for i, (s, e) in enumerate(offsets_mapping):
        if e <= s:
            continue
        if s >= begin and e <= end:
            tok_indices.append(i)
            rel_offsets.append((int(s - begin), int(e - begin)))

    return tok_indices, rel_offsets, end


def _map_scores_by_char_overlap(
    base_offsets: List[Tuple[int, int]],
    cur_offsets: List[Tuple[int, int]],
    cur_scores: np.ndarray,
) -> np.ndarray:
    """
    Map scores from the current prompt's context-tokenization back to the
    base context token spans (tokenization of the original full_context),
    using character-overlap weighted averaging.
    """
    n = len(base_offsets)
    m = len(cur_offsets)
    base_scores = np.zeros(n, dtype=np.float32)

    i = 0
    j = 0
    while i < n and j < m:
        bs, be = base_offsets[i]
        cs, ce = cur_offsets[j]

        if ce <= bs:
            j += 1
            continue
        if be <= cs:
            i += 1
            continue

        overlap = min(be, ce) - max(bs, cs)
        if overlap > 0:
            base_scores[i] += float(overlap) * float(cur_scores[j])

        if ce < be:
            j += 1
        else:
            i += 1

    lengths = np.array([max(1, e - s) for (s, e) in base_offsets], dtype=np.float32)
    return base_scores / lengths


def _attention_scores_mapped_to_base(
    *,
    hf_model,
    hf_tok,
    hf_device,
    prompt_template: ChatPromptTemplate,
    masked_context: str,
    query: str,
    base_offsets: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Compute attention scores on the current masked prompt, then map to stable base_offsets.
    Scoring: last-layer attention, averaged over heads, summed from query tokens -> context tokens.
    """
    full_prompt = prompt_template.format(context=masked_context, question=query)

    enc_full = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )

    offsets_full = enc_full["offset_mapping"]
    if hasattr(offsets_full, "tolist"):
        offsets_full = offsets_full.tolist()

    ctx_token_indices, ctx_rel_offsets, after_ctx = _find_token_indices_by_substring(
        full_prompt, masked_context, offsets_full, start_search_at=0
    )
    q_token_indices, _, _ = _find_token_indices_by_substring(
        full_prompt, query, offsets_full, start_search_at=after_ctx
    )

    enc = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )
    enc = {k: v.to(hf_device) for k, v in enc.items()}

    with torch.no_grad():
        out = hf_model(
            **enc,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
        )

    attn_layers = out.attentions
    if attn_layers is None:
        raise ValueError("No attentions returned (output_attentions=True required).")

    last = attn_layers[-1][0]   # [heads, seq, seq]
    attn_avg = last.mean(dim=0) # [seq, seq]

    q_idx = torch.tensor(q_token_indices, device=attn_avg.device, dtype=torch.long)
    c_idx = torch.tensor(ctx_token_indices, device=attn_avg.device, dtype=torch.long)

    q_to_c = attn_avg.index_select(0, q_idx).index_select(1, c_idx)  # [|Q|, |C|]
    scores_ctx = q_to_c.sum(dim=0).detach().float().cpu().numpy()    # [|C|]

    scores_base = _map_scores_by_char_overlap(base_offsets, ctx_rel_offsets, scores_ctx)

    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores_base


def _contextcite_scores_mapped_to_base(
    *,
    hf_model,
    hf_tok,
    masked_context: str,
    query: str,
    base_offsets: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Recompute ContextCite attributions on the CURRENT masked context,
    then map them to stable base_offsets.
    """
    if ContextCiter is None:
        raise ModuleNotFoundError("context_cite is required for context_cite recompute mode.")

    token_partitioner = TokenContextPartitioner(
        context=masked_context,
        tokenizer=hf_tok,
        ablate_mode="blank",
    )

    cc = ContextCiter(
        hf_model,
        hf_tok,
        masked_context,
        query,
        prompt_template=TF_RAG_TEMPLATE_A2T,
        partitioner=token_partitioner,
    )

    scores_cur = np.asarray(cc.get_attributions(), dtype=np.float32)

    _, cur_offsets = tokenize_context_with_offsets(masked_context, hf_tok)

    if len(scores_cur) != len(cur_offsets):
        raise ValueError(
            f"ContextCite scores len={len(scores_cur)} but current ctx tokens len={len(cur_offsets)}"
        )

    scores_base = _map_scores_by_char_overlap(base_offsets, cur_offsets, scores_cur)
    return scores_base

def _at2_scores_mapped_to_base(*,hf_model,hf_tok,
    masked_context: str,
    query: str,base_offsets: List[Tuple[int, int]],
    score_estimator_path, generate_kwargs: dict,
) -> np.ndarray:
    """
    Recompute AT2 attributions on the CURRENT masked context,
    then robustly map them back to the stable base_offsets.

    Important:
    - Do NOT retokenize masked_context here and require exact length equality.
    - Use AT2's own returned `sources` as the authority.
    """
    scores_cur, _gen, sources = get_at2_token_scores(
        full_context=masked_context,query=query,
        hf_model=hf_model,hf_tok=hf_tok,
        score_estimator_path=score_estimator_path,
        generate_kwargs=generate_kwargs,
    )

    scores_cur = np.asarray(scores_cur, dtype=np.float32)

    if len(scores_cur) != len(sources):
        raise ValueError(
            f"AT2 scores len={len(scores_cur)} != AT2 sources len={len(sources)}"
        )

    scores_base = map_at2_scores_to_base_via_sources(
        context=masked_context,
        source_pieces=sources,
        scores=scores_cur,
        base_offsets=base_offsets,
        max_lookahead=64,
        max_merge_pieces=4,
        whitespace_flex=True,
    )
    return scores_base


def _write_adaptive_log(
    path: str,
    *,title: str,
    query: str,
    full_context: str,
    base_offsets: List[Tuple[int, int]],
    order: List[int],
    scores_at_pick: List[float],
    masked_stats: List[dict],
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== {title} ===\n")
        f.write(f"Query: {query}\n")
        f.write(f"Total base context tokens considered: {len(base_offsets)}\n")
        f.write(f"Steps executed: {len(order)}\n\n")

        f.write("Adaptive masking order (first 100):\n")
        for rank, idx in enumerate(order[:100]):
            s, e = base_offsets[int(idx)]
            token_text = full_context[s:e].replace("\n", " ")
            f.write(
                f"{rank + 1:03d}. base_pos={int(idx):4d}, span[{s}:{e}]='{token_text}' "
                f"| score_at_pick={scores_at_pick[rank]:.6f}\n"
            )

        f.write("\n--- Per-step stats ---\n\n")
        prev_p_true = None
        for step, idx in enumerate(order):
            s, e = base_offsets[int(idx)]
            token_text = full_context[s:e].replace("\n", " ")
            stats = masked_stats[step]
            p_true = float(stats["p_true"])
            delta_p = None if prev_p_true is None else p_true - prev_p_true
            prev_p_true = p_true

            f.write(f"Step {step + 1:03d}: newly masked token at base_pos={int(idx)}\n")
            f.write(f"  span[{s}:{e}] = '{token_text}'\n")
            f.write(f"  score_at_pick = {scores_at_pick[step]:.6f}\n")
            f.write(
                f"  logP_true  = {stats['logP_true']:.6f}\n"
                f"  logP_false = {stats['logP_false']:.6f}\n"
                f"  log_odds   = {stats['log_odds']:.6f}\n"
                f"  p_true     = {p_true:.6f}\n"
            )

            if delta_p is not None:
                f.write(f"  Δp_true (vs prev) = {delta_p:+.6f}\n")
            f.write("\n")


def mask_by_order_recompute(
    *,
    full_context: str,
    query: str,
    hf_model,
    hf_tok,
    hf_device,
    max_steps: Optional[int] = 2000,
    batch_size: int = 2,
    score_mode: str = "attention",
    compute_probs_file_name: str = "attention_recompute_output_compute_probs.txt",
    log_path: Optional[str] = "greedy_token_masking_attention_recompute.txt",
    score_estimator_path=None,
    generate_kwargs=None,
    p_true_flipping: bool = False,
    true_variants=None,
    false_variants=None,
    masking_iteration=1,
    stop_scores_abs: Optional[float] = None,
    save_logs:bool=True,
    stop_on_flip:bool=False

):
    """
    Adaptive greedy masking:
    At step t:
      1) compute scores on the CURRENT masked prompt
      2) choose highest-score UNMASKED base token
      3) add it to masked set
      4) record the new masked prompt for later scoring
    """

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)
    if score_mode in ("context_cite", "at2"):
        prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE_A2T)

    _, base_offsets = tokenize_context_with_offsets(full_context, hf_tok)
    n = len(base_offsets)

    if max_steps is None:
        max_steps = n
    max_steps = int(min(max_steps, n))

    masking_iteration = max(1, int(masking_iteration))

    masked_flags = np.zeros(n, dtype=bool)
    masked_spans: List[Tuple[int, int]] = []

    order: List[int] = []
    scores_at_pick: List[float] = []

    masked_prompts: List[str] = []
    masked_context_list: List[str] = []

    keep_running = True

    while len(order) < max_steps and keep_running:
        cur_context = mask_context_spans_same_length(full_context, masked_spans)

        if score_mode == "attention":
            scores_base = _attention_scores_mapped_to_base(
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                prompt_template=prompt_template,
                masked_context=cur_context,
                query=query,
                base_offsets=base_offsets,
            )
        elif score_mode == "context_cite":
            scores_base = _contextcite_scores_mapped_to_base(
                hf_model=hf_model,
                hf_tok=hf_tok,
                masked_context=cur_context,
                query=query,
                base_offsets=base_offsets,
            )
        elif score_mode == "at2":
            scores_base = _at2_scores_mapped_to_base(
                hf_model=hf_model,
                hf_tok=hf_tok,
                masked_context=cur_context,
                query=query,
                base_offsets=base_offsets,
                score_estimator_path=score_estimator_path,
                generate_kwargs=generate_kwargs or {
                    "max_new_tokens": 128,
                    "do_sample": False,
                },
            )
        else:
            raise ValueError(
                f"Unknown score_mode={score_mode}. Use 'attention', 'context_cite', or 'at2'."
            )

        scores_base = scores_base.astype(np.float32, copy=False)
        scores_base[masked_flags] = -np.inf

        remaining_budget = max_steps - len(order)
        if remaining_budget <= 0:
            break

        top_k = min(masking_iteration, remaining_budget, n)
        if top_k <= 0:
            break

        top_idx = np.argpartition(scores_base, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores_base[top_idx])[::-1]]

        for index in top_idx:
            pick = int(index)
            pick_score = float(scores_base[pick])

            if not np.isfinite(pick_score):
                keep_running = False
                break

            if stop_scores_abs is not None and pick_score <= stop_scores_abs:
                keep_running = False
                break

            if masked_flags[pick]:
                continue

            masked_flags[pick] = True
            order.append(pick)
            scores_at_pick.append(pick_score)
            masked_spans.append(base_offsets[pick])

            new_context = mask_context_spans_same_length(full_context, masked_spans)
            masked_context_list.append(new_context)

            if score_mode in ("context_cite", "at2"):
                masked_prompts.append(
                    prompt_template.format(context=new_context, query=query)
                )
            else:
                masked_prompts.append(
                    prompt_template.format(context=new_context, question=query)
                )

        if len(order) > 0 and (len(order) == 1 or len(order) % 25 == 0):
            print(
                f"[adaptive] masked={len(order)}/{max_steps} "
                f"last_pick={order[-1]} last_score={scores_at_pick[-1]:.6f}"
            )

    os.makedirs(os.path.dirname(compute_probs_file_name) or ".", exist_ok=True)

    if masked_prompts:
        masked_stats, masked_logps = compute_probs(
            hf_model,
            hf_tok,
            masked_prompts,
            hf_device,
            None,
            batch_size=batch_size,
            detect_flip_to_true=p_true_flipping,
            true_variants=true_variants,
            false_variants=false_variants,
            masked_context_list=masked_context_list,
            return_full_logp=True,
            file_name=compute_probs_file_name,
            save_file=save_logs,
            stop_on_flip=stop_on_flip
        )
    else:
        masked_stats, masked_logps = [], []

    if save_logs and log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        _write_adaptive_log(
            log_path,
            title=f"Adaptive greedy masking (recompute each {masking_iteration} step/s)",
            query=query,
            full_context=full_context,
            base_offsets=base_offsets,
            order=order,
            scores_at_pick=scores_at_pick,
            masked_stats=masked_stats,
        )

    return masked_stats, masked_logps, order, scores_at_pick



def main():
    query = "Is being vegetarian considered healthy?"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="Is sugar considered an addictive substance?",
    )
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--score_mode", type=str, default="attention")
    parser.add_argument("--document_histogram", action="store_true")
    parser.add_argument("--out_dir", type=str, default="attention_recompute_results")
    parser.add_argument("--doc_name", type=str, default="Is_sugar_addictive_text_only_no_header.pdf")
    args = parser.parse_args()

    doc_path = DATA_DIR / args.doc_name
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found at: {doc_path}")

    docs = load_documents_any(doc_path)
    full_context = combine_document_text(docs)

    hf_model, hf_tok, hf_device = get_hf_scorer()

    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)
    baseline_prompt = prompt_template.format(context=full_context, question=query)

    os.makedirs(args.out_dir, exist_ok=True)

    print("Adaptive attention-guided masking (recompute scores each step)")
    masked_stats, masked_logps, order, scores_at_pick = mask_by_order_recompute(
        full_context=full_context,
        query=query,
        hf_model=hf_model,
        hf_tok=hf_tok,
        hf_device=hf_device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        score_mode=args.score_mode,
        compute_probs_file_name=os.path.join(
            args.out_dir,
            f"{args.score_mode}_recompute_output_compute_probs.txt",
        ),
        log_path=os.path.join(
            args.out_dir,
            f"greedy_token_masking_{args.score_mode}_recompute.txt",
        ),
    )

    if args.document_histogram:
        create_p_true_function(
            masked_logps,
            out_dir=args.out_dir,
            filename=f"{args.score_mode}_p_true_histogram.png",
        )

    print(f"[DONE] steps_run={len(order)} (requested max_steps={args.max_steps})")


if __name__ == "__main__":
    main()