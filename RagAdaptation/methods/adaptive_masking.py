
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import RagAdaptation.core.model_config as model_config
from RagAdaptation.compute_probs_updated import compute_probs
from RagAdaptation.core.prompting import (
    InterventionMode,
    coerce_intervention_mode,
    split_context_to_word_units,
    build_context_with_word_interventions_metadata,
    _filter_word_order_by_available_intervention,
)
from RagAdaptation.core.replacements import (
    ReplacementResolver,
    build_replacement_map_for_order,
    filter_replacement_order_semex,
)
from RagAdaptation.methods.common import (
    _get_mask_prompt_template,
    dump_masked_prompts_json,
    find_token_indices_by_substring,
    mask_context_spans_same_length,
)


def _target_progress(stat: Dict[str, Any], *, flip_to_true: bool) -> float:
    p_true = float(stat["p_true"])
    return p_true if flip_to_true else (1.0 - p_true)


def _progress_delta(prev_stat: Optional[Dict[str, Any]],
    cur_stat: Dict[str, Any],
    *,flip_to_true: bool,) -> Optional[float]:

    if prev_stat is None:
        return None
    return _target_progress(cur_stat, flip_to_true=flip_to_true) - _target_progress(
        prev_stat, flip_to_true=flip_to_true
    )


def _is_flip(stat: Dict[str, Any], *, flip_to_true: bool) -> bool:
    p_true = float(stat["p_true"])
    return p_true > 0.5 if flip_to_true else p_true < 0.5

def _build_single_word_intervention_prompt(
    *,
    pieces,
    word_units,
    selected_word_ids: Sequence[int],
    query: str,
    change_template_contextCite: bool,
    intervention_mode: InterventionMode,
    replacement_map=None,
):
    prompt_template = _get_mask_prompt_template(change_template_contextCite)
    intervened_context, metadata = build_context_with_word_interventions_metadata(
        pieces=pieces,
        word_units=word_units,
        selected_word_ids={int(i) for i in selected_word_ids},
        mode=intervention_mode,
        replacement_map=replacement_map,
    )

    if change_template_contextCite:
        prompt = prompt_template.format(context=intervened_context, query=query)
    else:
        prompt = prompt_template.format(context=intervened_context, question=query)

    return prompt, intervened_context, metadata


def _build_single_masked_prompt(*,
    document: str,query: str,spans: Sequence[Tuple[int, int]],change_template_contextCite: bool,
):
    prompt_template = _get_mask_prompt_template(change_template_contextCite)
    masked_context = mask_context_spans_same_length(document, spans)

    if change_template_contextCite:
        prompt = prompt_template.format(context=masked_context, query=query)
    else:
        prompt = prompt_template.format(context=masked_context, question=query)

    return prompt, masked_context


def _write_adaptive_trace(path: Optional[str], trace: List[Dict[str, Any]]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)


def _write_adaptive_filter_metadata(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _lookup_replacement_for_trace(
    replacement_map: Optional[Dict[Any, str]],
    *,
    chosen_idx: int,
    word_unit: Any,
) -> Optional[str]:
    if replacement_map is None:
        return None

    candidate_keys = [
        chosen_idx,
        str(chosen_idx),
        getattr(word_unit, "word", None),
        str(getattr(word_unit, "word", "")),
    ]

    for key in candidate_keys:
        if key is None:
            continue
        if key in replacement_map:
            return replacement_map[key]

    return None


def _describe_chosen_unit_for_trace(
    *,
    full_context: str,
    mode: InterventionMode,
    chosen_idx: int,
    ctx_rel_offsets: Sequence[Tuple[int, int]],
    word_units: Optional[Sequence[Any]],
    replacement_map: Optional[Dict[Any, str]],
) -> Dict[str, Any]:
    """
    Return human-readable information about the unit selected at this step.

    Important:
    - Read text from full_context, not from masked_context.
    - For word interventions, chosen_idx is a word id.
    - For token masking, chosen_idx is a token/span id.
    """
    chosen_idx = int(chosen_idx)

    if mode == InterventionMode.MASK_TOKEN:
        start, end = ctx_rel_offsets[chosen_idx]
        start, end = int(start), int(end)
        original_text = full_context[start:end]

        return {
            "chosen_unit_type": "token_span",
            "chosen_span": [start, end],
            "chosen_text": original_text,
            "chosen_text_visible": original_text.replace("\n", "\\n"),
            "replacement_text": None,
        }

    if word_units is None:
        raise RuntimeError("word_units is required for word-level trace description")

    unit = word_units[chosen_idx]
    start, end = int(unit.start), int(unit.end)

    # Prefer the stored word text if available, but slice from full_context as a fallback.
    original_text = getattr(unit, "text", None)
    if original_text is None:
        original_text = full_context[start:end]

    replacement_text = _lookup_replacement_for_trace(
        replacement_map,
        chosen_idx=chosen_idx,
        word_unit=unit,
    )

    return {
        "chosen_unit_type": "word",
        "chosen_word_id": chosen_idx,
        "chosen_span": [start, end],
        "chosen_text": str(original_text),
        "chosen_text_visible": str(original_text).replace("\n", "\\n"),
        "piece_index": getattr(unit, "piece_index", None),
        "replacement_text": replacement_text,
    }

def _score_candidate_prompts(*,hf_model,hf_tok,hf_device,prompts: Sequence[str],
    true_variants: Sequence[str],false_variants: Sequence[str],):
    stats, logps = compute_probs(
        hf_model, hf_tok,list(prompts),hf_device,
        None,
        batch_size=max(1, min(4, len(prompts))),
        return_full_logp=True,file_name="adaptive_tie_break_tmp.txt",
        detect_flip_to_true=False,
        true_variants=list(true_variants),false_variants=list(false_variants),
        save_file=False,stop_on_flip=False,
    )
    return stats, logps


def _choose_next_idx_with_ptrue_tie(
    *,remaining: Sequence[int],
    scores_vec: np.ndarray,
    candidate_prompt_builder: Callable[[int], Tuple[str, str]],
    hf_model,hf_tok,hf_device,true_variants: Sequence[str],false_variants: Sequence[str],
    flip_to_true: bool,tie_abs_gap: float,tie_max_candidates: int,
):
    if not remaining:
        raise ValueError("remaining must be non-empty")

    scored = sorted(remaining, key=lambda i: float(scores_vec[i]), reverse=True)
    best_idx = int(scored[0])
    best_score = float(scores_vec[best_idx])
    keep_scoring= 0
    bucket: List[int] = []
    for idx in scored:
        keep_scoring += 1
        idx = int(idx)
        gap_abs = abs(best_score - float(scores_vec[idx]))
        if gap_abs <= tie_abs_gap:
            bucket.append(idx)
        if len(bucket) >= max(1, tie_max_candidates):
            break

    if len(bucket) <= 1:
        return best_idx, {"used_ptrue_tie": False,"candidate_indices": [best_idx],"candidate_scores": [best_score],}

    prompts: List[str] = []
    masked_contexts: List[str] = []
    for idx in bucket:
        prompt, masked_context = candidate_prompt_builder(int(idx))
        prompts.append(prompt)
        masked_contexts.append(masked_context)

    cand_stats, _cand_logps = _score_candidate_prompts(
        hf_model=hf_model,hf_tok=hf_tok,hf_device=hf_device,
        prompts=prompts,true_variants=true_variants,false_variants=false_variants,
    )

    best_pos = max(
        range(len(bucket)),
        key=lambda pos: _target_progress(cand_stats[pos], flip_to_true=flip_to_true),
    )

    return int(bucket[best_pos]), {"used_ptrue_tie": True,
        "candidate_indices": [int(i) for i in bucket],
        "candidate_scores": [float(scores_vec[i]) for i in bucket],
        "candidate_progress": [ float(_target_progress(st, flip_to_true=flip_to_true)) for st in cand_stats],
        "winner_index": int(bucket[best_pos]),}

# masked by order adaptive combined
def _minmax_normalize_scores(scores: Sequence[float],*,eps: float = 1e-12,) -> np.ndarray:

    """
    Normalize a score vector to [0, 1].

    Important:
      - This preserves ranking.
      - It makes epsilon/tau comparable across attribution methods.
      - If all scores are equal, return 0.5 for every entry so the method treats
        the scores as uninformative rather than forcing recompute forever.
    """
    arr = np.asarray(scores, dtype=np.float32)

    if arr.size == 0:
        return arr

    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, 0.5, dtype=np.float32)

    finite_vals = arr[finite]
    mn = float(np.min(finite_vals))
    mx = float(np.max(finite_vals))

    clean = arr.copy()
    clean[~np.isfinite(clean)] = mn

    denom = mx - mn
    if denom <= eps:
        return np.full_like(clean, 0.5, dtype=np.float32)

    return ((clean - mn) / denom).astype(np.float32)


def mask_by_order_adaptive_combined(full_context: str,query: str,
    model_con: model_config.ModelConfig,*,scores: Optional[Sequence[torch.Tensor]] = None,
    rng: Optional[np.random.Generator] = None,
    compute_probs_file_name: str = "output_compute_probs.txt",
    p_true_flipping: bool = False,dump_json_path: Optional[str] = None,
    dump_policy: str = "flip",dump_window: int = 1,
    source_offsets: Optional[List[Tuple[int, int]]] = None,
    force_class_prompt: Optional[bool] = None,
    baseline_stats: Optional[Dict[str, Any]] = None,
    save_logs: bool = True,stop_on_flip: bool = False,
    enable_eps_recompute: bool = False,recompute_scores_fn: Optional[Callable[[str], np.ndarray]] = None,
    adaptive_trace_path: Optional[str] = None,
    k: int = 3,epsilon: float = 1e-3, tau: float = 0.01,
    intervention_mode: InterventionMode = InterventionMode.MASK_TOKEN,
    replacement_map: Optional[Dict[Any, str]] = None,
    replacement_resolver: Optional[ReplacementResolver] = None,
):
    """
    Combined adaptive masking strategy.

    Semantics:
      1. Maintain a current score ordering over context tokens.
      2. At each step, take the current best remaining token.
      3. If its score is below epsilon, recompute scores on the current masked context.
      4. After recompute, re-sort the remaining tokens.
      5. Look at the current top-k remaining tokens.
      6. If their score spread is <= tau, choose among them by p_true lookahead.
      7. Otherwise, choose the current best-scoring token.
    """

    if scores is None:
        raise ValueError(
            "mask_by_order_adaptive_combined requires scores. "
            "Random/rng mode is not implemented for the combined epsilon/tau strategy."
        )

    if k <= 0:
        raise ValueError("k must be positive")

    mode = coerce_intervention_mode(intervention_mode)
    hf_model, hf_tok, hf_device = model_con.load()
    true_variants = model_con.get_true_variants()
    false_variants = model_con.get_false_variants()

    if enable_eps_recompute and recompute_scores_fn is None:
        raise ValueError("enable_eps_recompute=True requires recompute_scores_fn")

    full_prompt = model_con.format_prompt(
        question=query,context=full_context,
        context_cite_at2_formating=False,
    )

    enc_full = hf_tok(
        full_prompt,add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,padding=False,
    )

    offsets_full = enc_full["offset_mapping"]
    if hasattr(offsets_full, "tolist"):
        offsets_full = offsets_full.tolist()

    ctx_token_indices, ctx_rel_offsets_prompt, after_ctx = find_token_indices_by_substring(
        full_prompt,full_context,offsets_full,
        start_search_at=0,
    )

    ctx_rel_offsets = ctx_rel_offsets_prompt
    pieces = None
    word_units = None
    change_template_contextCite = False

    if force_class_prompt is not None:
        change_template_contextCite = bool(force_class_prompt)

    # ------------------------------------------------------------
    # Initial score extraction
    # ------------------------------------------------------------
    if isinstance(scores, (list, tuple)) and len(scores) > 0 and torch.is_tensor(scores[0]):
        # Raw attention tensor case.
        q_token_indices, _, _ = find_token_indices_by_substring(
            full_prompt,
            query,
            offsets_full,
            start_search_at=after_ctx,
        )

        last = scores[-1]              # typically last layer attention
        last_avg = last[0].mean(dim=0) # [seq, seq], averaged over heads

        q_idx = torch.as_tensor(q_token_indices, device=last_avg.device, dtype=torch.long)
        c_idx = torch.as_tensor(ctx_token_indices, device=last_avg.device, dtype=torch.long)

        sub = last_avg.index_select(0, q_idx).index_select(1, c_idx)
        scores_vec = sub.sum(dim=0).detach().float().cpu().numpy().astype(np.float32)

        change_template_contextCite = False

    else:
        # Already-computed token/source scores case.
        if torch.is_tensor(scores):
            scores_vec = scores.detach().float().cpu().numpy().astype(np.float32)
        else:
            scores_vec = np.asarray(scores, dtype=np.float32)

        if source_offsets is not None:
            ctx_rel_offsets = source_offsets

        if len(scores_vec) != len(ctx_rel_offsets):
            raise ValueError(
                f"Provided non-attention scores length {len(scores_vec)} "
                f"!= number of masking spans {len(ctx_rel_offsets)}")

        if force_class_prompt is None:
            change_template_contextCite = True
        else:
            change_template_contextCite = bool(force_class_prompt)

    if mode != InterventionMode.MASK_TOKEN:
        pieces, word_units = split_context_to_word_units(full_context)
        word_offsets = [(int(u.start), int(u.end)) for u in word_units]
        if source_offsets is None and len(scores_vec) == len(word_units):
            ctx_rel_offsets = word_offsets

    #normalization
    scores_vec = _minmax_normalize_scores(scores_vec)

    if len(scores_vec) == 0:
        raise ValueError("scores_vec is empty; cannot run adaptive masking")

    # Initial static order.
    order = np.argsort(scores_vec)[::-1]

    if mode != InterventionMode.MASK_TOKEN and word_units is not None:
        excluded_units: List[Dict[str, Any]] = []
        candidate_filter: Dict[str, Any] = {
            "replacement_semex_filter_enabled": False,
            "excluded_candidates": 0,
        }

        if mode in {
            InterventionMode.REPLACEMENT_NEUTRAL_WORD,
            InterventionMode.REPLACEMENT_ANTONYM_WORD,
        }:
            semex_filter_enabled = bool(getattr(replacement_resolver, "semex_filter_enabled", True))
            candidate_filter["replacement_semex_filter_enabled"] = semex_filter_enabled
            if semex_filter_enabled:
                filtered_order, _filtered_scores, semex_excluded, semex_meta = filter_replacement_order_semex(
                    context=full_context,
                    word_units=word_units,
                    ordered_word_ids=[int(i) for i in order],
                    pick_scores=[float(scores_vec[int(i)]) for i in order],
                    spacy_model=str(getattr(replacement_resolver, "semex_spacy_model", "en_core_web_sm")),
                )
                order = np.asarray(filtered_order, dtype=np.int64)
                excluded_units.extend(semex_excluded)
                candidate_filter.update(semex_meta)

            replacement_map = build_replacement_map_for_order(
                context=full_context,
                query=query,
                word_units=word_units,
                ordered_word_ids=[int(i) for i in order],
                mode=mode,
                replacement_map=replacement_map,
                resolver=replacement_resolver,
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                model_id=model_con.model_id,
            )

        filtered_order, _filtered_scores, availability_excluded = _filter_word_order_by_available_intervention(
            word_units=word_units,
            order=[int(i) for i in order],
            pick_scores=[float(scores_vec[int(i)]) for i in order],
            mode=mode,
            replacement_map=replacement_map,
            return_exclusions=True,
        )
        excluded_units.extend(availability_excluded)
        candidate_filter["excluded_candidates"] = int(len(excluded_units))
        candidate_filter["available_candidates"] = int(len(filtered_order))
        order = np.asarray(filtered_order, dtype=np.int64)
    else:
        excluded_units = []
        candidate_filter = {
            "replacement_semex_filter_enabled": False,
            "excluded_candidates": 0,
            "available_candidates": int(len(order)),
        }

    selected: set[int] = set()
    current_order: List[int] = [int(i) for i in order]
    selected_order: List[int] = []
    scores_at_pick: List[float] = []

    masked_spans: List[Tuple[int, int]] = []
    intervention_metadata: List[Dict[str, Any]] = []
    masked_prompts: List[str] = []
    masked_contexts: List[str] = []
    masked_stats: List[Dict[str, Any]] = []
    masked_logps: List[float] = []
    trace: List[Dict[str, Any]] = []

    step = 0

    while True:
        step += 1

        # No re-sort here unless scores were recomputed.
        remaining = [int(i) for i in current_order if int(i) not in selected]
        if not remaining:
            break

        best_idx_before_recompute = int(remaining[0])
        best_score_before_recompute = float(scores_vec[best_idx_before_recompute])

        if step == 1:
            epsilon =epsilon * best_score_before_recompute


        recompute_triggered_this_step = False

        # --------------------------------------------------------
        # Epsilon recompute trigger
        # --------------------------------------------------------
        if enable_eps_recompute and best_score_before_recompute < epsilon and step > 1:
            current_context_for_recompute = masked_contexts[-1] if masked_contexts else full_context

            new_scores = recompute_scores_fn(current_context_for_recompute)
            new_scores = np.asarray(new_scores, dtype=np.float32)

            if len(new_scores) != len(ctx_rel_offsets):
                raise ValueError(
                    f"Recomputed scores len={len(new_scores)} "
                    f"!= masking spans len={len(ctx_rel_offsets)}"
                )

            scores_vec = _minmax_normalize_scores(new_scores)
            recompute_triggered_this_step = True

            # Scores changed, so now we must re-sort the remaining candidates.
            remaining = [int(i) for i in current_order if int(i) not in selected]
            remaining.sort(key=lambda i: float(scores_vec[i]), reverse=True)
            epsilon= float(scores_vec[int(remaining[0])]) * 0.3

            current_order = list(selected_order) + remaining

        # After optional recompute, remaining is guaranteed to match current scores.
        best_idx = int(remaining[0])
        best_score = float(scores_vec[best_idx])

        # --------------------------------------------------------
        # Tau p_true tie trigger over current top-k
        # --------------------------------------------------------
        topk = remaining[: min(k, len(remaining))]
        topk_scores = [float(scores_vec[i]) for i in topk]

        use_ptrue_tie = (
            len(topk) > 1
            and (max(topk_scores) - min(topk_scores)) <= tau
        )

        def build_candidate_prompt(candidate_idx: int) -> Tuple[str, str]:
            candidate_order = list(selected_order) + [int(candidate_idx)]
            if mode == InterventionMode.MASK_TOKEN:
                spans = [ctx_rel_offsets[i] for i in candidate_order]
                return _build_single_masked_prompt(
                    document=full_context,
                    query=query,
                    spans=spans,
                    change_template_contextCite=change_template_contextCite,
                )
            if pieces is None or word_units is None:
                raise RuntimeError("word intervention state was not initialized")
            prompt, context, _metadata = _build_single_word_intervention_prompt(
                pieces=pieces,
                word_units=word_units,
                selected_word_ids=candidate_order,
                query=query,
                change_template_contextCite=change_template_contextCite,
                intervention_mode=mode,
                replacement_map=replacement_map,
            )
            return prompt, context

        if use_ptrue_tie:
            next_idx, tie_info = _choose_next_idx_with_ptrue_tie(
                remaining=topk,scores_vec=scores_vec,
                candidate_prompt_builder=build_candidate_prompt,
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                true_variants=true_variants,
                false_variants=false_variants,
                flip_to_true=p_true_flipping,
                tie_abs_gap=tau,tie_max_candidates=len(topk),
            )
        else:
            next_idx = best_idx
            tie_info = {
                "used_ptrue_tie": False,
                "candidate_indices": [int(i) for i in topk],
                "candidate_scores": [float(s) for s in topk_scores],
                "winner_index": int(next_idx),
            }

        # --------------------------------------------------------
        # Commit chosen index
        # --------------------------------------------------------
        selected.add(int(next_idx))
        selected_order.append(int(next_idx))
        scores_at_pick.append(float(scores_vec[next_idx]))

        chosen_trace_info = _describe_chosen_unit_for_trace(
            full_context=full_context,
            mode=mode,
            chosen_idx=int(next_idx),
            ctx_rel_offsets=ctx_rel_offsets,
            word_units=word_units,
            replacement_map=replacement_map,
        )

        step_intervention_metadata = None

        if mode == InterventionMode.MASK_TOKEN:
            masked_spans.append(ctx_rel_offsets[next_idx])
            prompt, masked_context = _build_single_masked_prompt(
                document=full_context,
                query=query,
                spans=masked_spans,
                change_template_contextCite=change_template_contextCite,
            )
        else:
            if pieces is None or word_units is None:
                raise RuntimeError("word intervention state was not initialized")

            prompt, masked_context, metadata = _build_single_word_intervention_prompt(
                pieces=pieces,
                word_units=word_units,
                selected_word_ids=selected_order,
                query=query,
                change_template_contextCite=change_template_contextCite,
                intervention_mode=mode,
                replacement_map=replacement_map,
            )

            step_intervention_metadata = {str(k): v for k, v in metadata.items()}
            intervention_metadata.append(step_intervention_metadata)

        stats_chunk, logps_chunk = compute_probs(
            hf_model,hf_tok,[prompt],hf_device,None,
            batch_size=1,return_full_logp=True,
            file_name=compute_probs_file_name,detect_flip_to_true=p_true_flipping,
            true_variants=true_variants,false_variants=false_variants,
            save_file=False,stop_on_flip=False,)

        cur_stat = stats_chunk[0]
        cur_logp = logps_chunk[0]
        cur_stat["step_index"] = step

        masked_prompts.append(prompt)
        masked_contexts.append(masked_context)
        masked_stats.append(cur_stat)
        masked_logps.append(cur_logp)

        trace.append(
            {
                "intervention_mode": mode.name,
                "step": len(selected_order),
                "chosen_idx": int(next_idx),

                **chosen_trace_info,

                "score_at_pick": float(scores_at_pick[-1]),
                "p_true": float(cur_stat["p_true"]),
                "log_odds": float(cur_stat["log_odds"]),
                "target_progress": float(_target_progress(cur_stat, flip_to_true=p_true_flipping)),
                "recompute_triggered": bool(recompute_triggered_this_step),
                "best_idx_before_recompute": int(best_idx_before_recompute),
                "best_score_before_recompute": float(best_score_before_recompute),
                "best_idx_before_pick": int(best_idx),
                "best_score_before_pick": float(best_score),
                "topk_indices": [int(i) for i in topk],
                "topk_scores": [float(s) for s in topk_scores],
                "used_ptrue_tie": bool(use_ptrue_tie),
                "tie": tie_info,

                # Full intervention metadata for the whole selected set at this step.
                "interventions": step_intervention_metadata,
                "excluded_candidate_count": int(len(excluded_units)),
            }
        )

        if stop_on_flip and _is_flip(cur_stat, flip_to_true=p_true_flipping):
            break

    # ------------------------------------------------------------
    # Optional dump
    # ------------------------------------------------------------
    if dump_json_path and save_logs:
        if baseline_stats is None:
            baseline_stats_list, _ = compute_probs(
                hf_model,hf_tok,[full_prompt],
                hf_device,expected_result=None,
                batch_size=1,return_full_logp=True,
                file_name=compute_probs_file_name + ".baseline_tmp",
                detect_flip_to_true=p_true_flipping,
                true_variants=true_variants,
                false_variants=false_variants,
                save_file=False,stop_on_flip=False,
            )
            baseline_stats = baseline_stats_list[0]

        dump_masked_prompts_json(
            dump_json_path, query=query,baseline_prompt=full_prompt,
            baseline_stats=baseline_stats,
            masked_prompts=masked_prompts,
            masked_stats=masked_stats,
            masked_context_list=masked_contexts,
            order=selected_order,scores_at_pick=scores_at_pick,
            policy=dump_policy,window=dump_window,
            excluded_units=excluded_units,
            candidate_filter=candidate_filter,
        )

    if adaptive_trace_path and save_logs:
        _write_adaptive_trace(adaptive_trace_path, trace)
        _write_adaptive_filter_metadata(
            str(Path(adaptive_trace_path).with_name("replacement_filter.json")),
            {
                "intervention_mode": mode.name,
                "candidate_filter": candidate_filter,
                "excluded_units": excluded_units,
            },
        )

    return masked_stats, masked_logps, selected_order, scores_at_pick, excluded_units, candidate_filter
