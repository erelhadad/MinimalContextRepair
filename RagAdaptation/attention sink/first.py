from __future__ import annotations

from typing import Optional, Sequence, List, Tuple, Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate

from RagAdaptation.baseline.bruteforce_common import tokenize_context_with_offsets
from RagAdaptation.baseline.mask_iter_recompute_attention import _attention_scores_mapped_to_base
from RagAdaptation.dataset_creation.make_flip_benchmark import _join_hotpot_context
from RagAdaptation.methods.common import (
    get_attention_scores,
    find_token_indices_by_substring,
    mask_context_spans_same_length,
)
from RagAdaptation.core.model_config import ModelConfig
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE


def mask_by_order_inspect(
    full_context: str,
    query: str,
    model_con: ModelConfig,
    *,
    scores: Optional[Sequence[torch.Tensor]] = None,
    rng: Optional[np.random.Generator] = None,
    source_offsets: Optional[List[Tuple[int, int]]] = None,
    force_class_prompt: Optional[bool] = None,
):
    """
    Inspect the token order induced by a scoring function, without actually running
    the full masking pipeline.

    This is a light-weight inspection version of `mask_by_order`.
    It returns:
      1. ordered_offsets  - masking spans sorted by descending score
      2. scores_vec       - score per context token/span
      3. order            - permutation of token/span indices
      4. ctx_rel_offsets  - the offsets used for the ranking

    Notes
    -----
    - If `scores` is a list/tuple of attention tensors, we interpret it as regular
      attention output and compute question -> context attention on the last layer.
    - If `scores` is already a numeric vector, we use it directly.
    - If `scores is None`, we return a random order.
    """
    hf_model, hf_tok, hf_device = model_con.load()

    full_prompt = model_con.format_prompt(
        question=query,
        context=full_context,
        context_cite_at2_formating=False,
    )

    enc_full = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        padding=False,
    )

    offsets_full = enc_full["offset_mapping"]
    if hasattr(offsets_full, "tolist"):
        offsets_full = offsets_full.tolist()

    # Locate context tokens inside the full prompt
    ctx_token_indices, ctx_rel_offsets_prompt, after_ctx = find_token_indices_by_substring(
        full_prompt,
        full_context,
        offsets_full,
        start_search_at=0,
    )

    ctx_rel_offsets = ctx_rel_offsets_prompt
    scores_vec: Optional[np.ndarray] = None

    # Kept for compatibility with the original structure
    _ = force_class_prompt

    if scores is not None:
        # Case 1: raw attention tensors
        if isinstance(scores, (list, tuple)) and len(scores) > 0 and torch.is_tensor(scores[0]):
            q_token_indices, _, _ = find_token_indices_by_substring(
                full_prompt,
                query,
                offsets_full,
                start_search_at=after_ctx,
            )

            last = scores[-1]                # [batch, heads, seq, seq]
            last_avg = last[0].mean(dim=0)  # [seq, seq]

            # Question -> context attention block
            sub = last_avg[np.ix_(q_token_indices, ctx_token_indices)]

            # We keep your change: use SUM over question tokens, not mean
            scores_vec = sub.sum(axis=0).detach().float().cpu().numpy()

            order = np.argsort(scores_vec)[::-1]

        # Case 2: already-computed scores
        else:
            if torch.is_tensor(scores):
                scores_vec = scores.detach().float().cpu().numpy()
            else:
                scores_vec = np.asarray(scores, dtype=np.float32)

            if source_offsets is not None:
                ctx_rel_offsets = source_offsets

            if len(scores_vec) != len(ctx_rel_offsets):
                raise ValueError(
                    f"Provided non-attention scores length {len(scores_vec)} "
                    f"!= number of masking spans {len(ctx_rel_offsets)}"
                )

            order = np.argsort(scores_vec)[::-1]

    # Case 3: random inspection
    else:
        if rng is None:
            rng = np.random.default_rng()
        order = rng.permutation(len(ctx_rel_offsets))
        scores_vec = None

    ordered_offsets = [ctx_rel_offsets[i] for i in order]
    return ordered_offsets, scores_vec, order, ctx_rel_offsets


def mask_by_order_recompute_inspect(
    *,
    full_context: str,
    query: str,
    hf_model,
    hf_tok,
    hf_device,
    max_steps: Optional[int] = 5000,
    masking_iteration: int = 1,
    verbose: bool = False,
):
    """
    Inspect recompute-attention behavior.

    At each iteration:
      1. recompute attention scores on the CURRENT masked context
      2. select top-k unmasked tokens
      3. record their indices, scores, and text spans

    Returns
    -------
    order : List[int]
        Global token indices (with respect to `base_offsets`) in the order they were picked.
    scores_at_pick : List[float]
        Score assigned to each picked token at the moment it was selected.
    top_k_at_each_iteration : List[List[Dict]]
        Rich per-iteration records including token text and offsets.
    base_offsets : List[Tuple[int, int]]
        Stable token offsets of the original full_context.
    """
    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)
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
    keep_running = True

    top_k_at_each_iteration: List[List[Dict[str, Any]]] = []

    while len(order) < max_steps and keep_running:
        cur_context = mask_context_spans_same_length(full_context, masked_spans)

        scores_base = _attention_scores_mapped_to_base(
            hf_model=hf_model,
            hf_tok=hf_tok,
            hf_device=hf_device,
            prompt_template=prompt_template,
            masked_context=cur_context,
            query=query,
            base_offsets=base_offsets,
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

        current_k: List[Dict[str, Any]] = []

        for index in top_idx:
            pick = int(index)
            pick_score = float(scores_base[pick])

            if not np.isfinite(pick_score):
                keep_running = False
                break

            if masked_flags[pick]:
                continue

            s, e = base_offsets[pick]
            token_text = full_context[s:e]

            rec = {
                "token_idx": pick,
                "score": pick_score,
                "char_span": (int(s), int(e)),
                "token_text": token_text,
            }

            if verbose:
                print(f"token_idx={pick} span=({s},{e}) score={pick_score:.6f} text={token_text!r}")

            masked_flags[pick] = True
            order.append(pick)
            scores_at_pick.append(pick_score)
            masked_spans.append(base_offsets[pick])
            current_k.append(rec)

        top_k_at_each_iteration.append(current_k)

    return order, scores_at_pick, top_k_at_each_iteration, base_offsets


def extract_hotpot_supporting_facts(row: dict) -> List[Dict[str, Any]]:
    """
    Extract supporting facts from a HotpotQA row as explicit text records.

    Returns a list of dicts:
      {
        "title": ...,
        "sent_id": ...,
        "text": ...,
        "uid": ...
      }

    Supports both:
      - new HF dict format
      - older list-of-pairs format
    """
    context_field = row.get("context")
    supporting_facts = row.get("supporting_facts")

    if context_field is None or supporting_facts is None:
        return []

    # Normalize context
    if isinstance(context_field, dict):
        titles = context_field.get("title", [])
        sentences_lists = context_field.get("sentences", [])
    else:
        titles = [x[0] for x in context_field]
        sentences_lists = [x[1] for x in context_field]

    # Normalize supporting facts
    if isinstance(supporting_facts, dict):
        sf_pairs = list(zip(supporting_facts.get("title", []), supporting_facts.get("sent_id", [])))
    else:
        sf_pairs = [(x[0], x[1]) for x in supporting_facts]

    out: List[Dict[str, Any]] = []
    for sf_title, sf_sent_id in sf_pairs:
        sf_title = str(sf_title)
        sf_sent_id = int(sf_sent_id)

        for title, sents in zip(titles, sentences_lists):
            if str(title) != sf_title:
                continue
            if 0 <= sf_sent_id < len(sents):
                sent_text = str(sents[sf_sent_id]).strip()
                out.append(
                    {
                        "title": sf_title,
                        "sent_id": sf_sent_id,
                        "text": sent_text,
                        "uid": f"{sf_title}_{sf_sent_id}",
                    }
                )
            break

    return out


def supporting_facts_scores(supporting_facts: List[Dict[str, Any]],
    offsets: List[Tuple[int, int]],context: str,scores,):
    """
    Compute scores for each supporting fact, and also for each token inside the fact.

    Parameters
    ----------
    supporting_facts : List[Dict]
        Output of `extract_hotpot_supporting_facts`.
    offsets : List[(start, end)]
        Offsets of the tokenization used by `scores`.
    context : str
        Full context text in which we search for the supporting fact sentence.
    scores : array-like
        Score per token/span.

    Returns
    -------
    List[Dict]
        One record per supporting fact with both aggregate and token-level details.
    """
    scores = np.asarray(scores, dtype=np.float32)
    results: List[Dict[str, Any]] = []
    search_pos = 0

    for fact in supporting_facts:
        fact_text = fact["text"].strip()

        try:
            tok_idx, rel_offsets, end_pos = find_token_indices_by_substring(
                full_text=context,
                substring=fact_text,
                offsets_mapping=offsets,
                start_search_at=search_pos,
            )
        except ValueError:
            results.append(
                {
                    **fact,
                    "found": False,
                    "token_indices": [],
                    "token_details": [],
                    "sum_score": 0.0,
                    "mean_score": None,
                    "max_score": None,
                }
            )
            continue

        if not tok_idx:
            results.append(
                {
                    **fact,
                    "found": False,
                    "token_indices": [],
                    "token_details": [],
                    "sum_score": 0.0,
                    "mean_score": None,
                    "max_score": None,
                }
            )
            search_pos = end_pos
            continue

        vals = scores[tok_idx]

        token_details = []
        for idx in tok_idx:
            s, e = offsets[idx] #start and end of token indecies within the full context
            token_details.append(
                {
                    "token_idx": int(idx),
                    "token_text": context[s:e],
                    "char_span": (int(s), int(e)),
                    "score": float(scores[idx]),
                }
            )

        results.append(
            {
                **fact,
                "found": True,
                "token_indices": [int(x) for x in tok_idx],
                "token_details": token_details,
                "sum_score": float(vals.sum()),
                "mean_score": float(vals.mean()),
                "max_score": float(vals.max()),
                "relative_offsets_inside_fact": [(int(s), int(e)) for s, e in rel_offsets],
            }
        )

        # Move the search cursor forward so repeated identical substrings
        # are less likely to be matched to the wrong occurrence.
        search_pos = end_pos

    return results


def inspect_attention(
    query: str,
    model_con: ModelConfig,
    context: str,
    hotpot: bool = True,
    supporting_facts: Optional[List[Dict[str, Any]]] = None,
    masking_iteration: int = 5,
):
    """
    Main inspection entry point.

    This function computes:
      1. regular attention scores on the full prompt
      2. recompute-attention picks across iterations
      3. optional supporting-fact aggregate scores
      4. token-level scores inside each supporting fact

    Returns a dictionary with everything needed for focused inspection.
    """
    # Load model once
    hf_model, hf_tok, hf_device = model_con.load()

    # ---- regular attention ----
    baseline_prompt = model_con.format_prompt(
        question=query,
        context=context,
        context_cite_at2_formating=False,
    )
    attn = get_attention_scores(hf_model, hf_tok, hf_device, baseline_prompt)

    ordered_offsets_regular, scores_vec_regular, order_regular, ctx_offsets_regular = mask_by_order_inspect(
        full_context=context,
        query=query,
        model_con=model_con,
        scores=attn,
    )

    # ---- recompute attention ----
    order_recompute, scores_at_pick_recompute, top_k_at_each_iteration_recompute, base_offsets_recompute = (
        mask_by_order_recompute_inspect(
            full_context=context,
            query=query,
            hf_model=hf_model,
            hf_tok=hf_tok,
            hf_device=hf_device,
            masking_iteration=masking_iteration,
        )
    )

    out: Dict[str, Any] = {
        "query": query,
        "context": context,
        "regular_attention": {
            "ordered_offsets": ordered_offsets_regular,
            "scores_vec": None if scores_vec_regular is None else scores_vec_regular.tolist(),
            "order": [int(x) for x in order_regular],
            "offsets": [(int(s), int(e)) for s, e in ctx_offsets_regular],
        },
        "recompute_attention": {
            "order": [int(x) for x in order_recompute],
            "scores_at_pick": [float(x) for x in scores_at_pick_recompute],
            "top_k_at_each_iteration": top_k_at_each_iteration_recompute,
            "base_offsets": [(int(s), int(e)) for s, e in base_offsets_recompute],
        },
    }

    # ---- optional supporting facts analysis ----
    if hotpot and supporting_facts and scores_vec_regular is not None:
        out["supporting_facts_regular"] = supporting_facts_scores(
            supporting_facts=supporting_facts,
            offsets=ctx_offsets_regular,
            context=context,
            scores=scores_vec_regular,
        )

    return out


def iter_examples(context_mode: str = "full", split: str = "validation"):
    """
    Iterate over HotpotQA yes/no examples.

    Yields the original row together with:
      - question
      - context_text (rendered by _join_hotpot_context)
      - extracted supporting fact texts
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    for row in ds:
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

        extracted_supporting_facts = extract_hotpot_supporting_facts(row)

        yield {
            "row": row,
            "question": question,
            "context_text": context_text,
            "supporting_facts": extracted_supporting_facts,
        }


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
#
model_con = ModelConfig("mistralai/Mistral-7B-Instruct-v0.3")
for ex in iter_examples(context_mode="full", split="validation"):
    result = inspect_attention(
         query=ex["question"],
         model_con=model_con,
         context=ex["context_text"],
         hotpot=True,
         supporting_facts=ex["supporting_facts"],
         masking_iteration=5,
     )

    print(result["supporting_facts_regular"])

    stop=input("stop?")
    if stop=="stop":
        break

