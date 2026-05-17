
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import RagAdaptation.core.model_config as model_config
from RagAdaptation.compute_probs_updated import compute_probs
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
    scores_vec: np.ndarray,masked_spans: Sequence[Tuple[int, int]],
    ctx_rel_offsets: Sequence[Tuple[int, int]],
    full_context: str,query: str,change_template_contextCite: bool,
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
        #adding each iteration the next index in the bucket with the already masked promts but no overlapping betweern the
        #indcies within the bucket it self
        spans = list(masked_spans) + [ctx_rel_offsets[idx]]
        prompt, masked_context = _build_single_masked_prompt(document=full_context,query=query,spans=spans,
            change_template_contextCite=change_template_contextCite,)

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


def mask_by_order_adaptive(full_context: str,query: str,model_con: model_config.ModelConfig,
    *,scores: Optional[Sequence[torch.Tensor]] = None,
    rng: Optional[np.random.Generator] = None,
    compute_probs_file_name: str = "output_compute_probs.txt",
    p_true_flipping: bool = False,dump_json_path: Optional[str] = None,
    dump_policy: str = "flip",dump_window: int = 1,
    source_offsets: Optional[List[Tuple[int, int]]] = None,
    force_class_prompt: Optional[bool] = None,
    baseline_stats: Optional[Dict[str, Any]] = None,
    stop_scores_relative: Optional[float] = 0,
    save_logs: bool = True,stop_on_flip: bool = False,
    enable_ptrue_tie: bool = False,tie_abs_gap: float = 0.0,tie_rel_gap: float = 0.0,tie_max_candidates: int = 2,
    enable_eps_recompute: bool = False,
    recompute_epsilon: float = 0.0, recompute_patience: int = 1,recompute_scores_fn: Optional[Callable[[str], np.ndarray]] = None,
    adaptive_trace_path: Optional[str] = None,):

    hf_model, hf_tok, hf_device = model_con.load()
    true_variants = model_con.get_true_variants()
    false_variants = model_con.get_false_variants()

    if enable_eps_recompute and recompute_scores_fn is None:
        raise ValueError("enable_eps_recompute=True requires recompute_scores_fn")

    full_prompt = model_con.format_prompt(question=query,context=full_context,context_cite_at2_formating=False,)

    enc_full = hf_tok(full_prompt,add_special_tokens=False,return_offsets_mapping=True,truncation=True,padding=False,)

    offsets_full = enc_full["offset_mapping"]
    if hasattr(offsets_full, "tolist"):
        offsets_full = offsets_full.tolist()

    ctx_token_indices, ctx_rel_offsets_prompt, after_ctx = find_token_indices_by_substring(
        full_prompt, full_context, offsets_full, start_search_at=0)

    ctx_rel_offsets = ctx_rel_offsets_prompt
    scores_vec: Optional[np.ndarray] = None
    change_template_contextCite = False

    if force_class_prompt is not None:
        change_template_contextCite = bool(force_class_prompt)

    if scores is not None:
        if isinstance(scores, (list, tuple)) and len(scores) > 0 and torch.is_tensor(scores[0]):
            q_token_indices, _, _ = find_token_indices_by_substring(full_prompt, query, offsets_full, start_search_at=after_ctx)
            last = scores[-1]
            last_avg = last[0].mean(dim=0)
            sub = last_avg[np.ix_(q_token_indices, ctx_token_indices)]
            scores_vec = sub.sum(axis=0).detach().float().cpu().numpy()
            order = np.argsort(scores_vec)[::-1]
            change_template_contextCite = False
        else:
            if torch.is_tensor(scores):
                scores_vec = scores.detach().float().cpu().numpy()
            else:
                scores_vec = np.asarray(scores, dtype=np.float32)

            if source_offsets is not None:
                ctx_rel_offsets = source_offsets

            if len(scores_vec) != len(ctx_rel_offsets):
                raise ValueError(
                    f"Provided non-attention scores length {len(scores_vec)} != number of masking spans {len(ctx_rel_offsets)}"
                )

            order = np.argsort(scores_vec)[::-1]
            if force_class_prompt is None:
                change_template_contextCite = True
            else:
                change_template_contextCite = bool(force_class_prompt)
    else:
        if rng is None:
            rng = np.random.default_rng()
        order = rng.permutation(len(ctx_rel_offsets))
        scores_vec = None

    if scores_vec is not None:
        scores_vec = np.asarray(scores_vec, dtype=np.float32)
        max_val = float(np.max(scores_vec)) if scores_vec.size else None
    else:
        max_val = None

    if max_val is not None and stop_scores_relative is not None:
        threshold = max_val * stop_scores_relative
        order = [int(i) for i in order if scores_vec[i] >= threshold]
    else:
        order = [int(i) for i in order]

    selected: set[int] = set()
    current_order: List[int] = list(order)
    selected_order: List[int] = []
    scores_at_pick: List[float] = []
    masked_spans: List[Tuple[int, int]] = []
    masked_prompts: List[str] = []
    masked_contexts: List[str] = []
    masked_stats: List[Dict[str, Any]] = []
    masked_logps: List[float] = []
    trace: List[Dict[str, Any]] = []

    plateau_count = 0
    prev_stat: Optional[Dict[str, Any]] = None
    step=0
    while True:
        step += 1
        remaining = [int(i) for i in current_order if int(i) not in selected]
        if not remaining:
            break

        if enable_ptrue_tie and scores_vec is not None:
            next_idx, tie_info = _choose_next_idx_with_ptrue_tie(remaining=remaining,scores_vec=scores_vec,
                masked_spans=masked_spans,ctx_rel_offsets=ctx_rel_offsets,
                full_context=full_context,query=query,change_template_contextCite=change_template_contextCite,
                hf_model=hf_model,hf_tok=hf_tok,hf_device=hf_device,true_variants=true_variants,false_variants=false_variants,
                flip_to_true=p_true_flipping,tie_abs_gap=tie_abs_gap,tie_max_candidates=tie_max_candidates,)
        else:
            next_idx = int(remaining[0])
            tie_info = {"used_ptrue_tie": False,"candidate_indices": [next_idx],
                "candidate_scores": None if scores_vec is None else [float(scores_vec[next_idx])],
            }

        selected.add(next_idx)
        selected_order.append(next_idx)
        if scores_vec is not None:
            scores_at_pick.append(float(scores_vec[next_idx]))
        else:
            scores_at_pick.append(float("nan"))

        masked_spans.append(ctx_rel_offsets[next_idx])
        prompt, masked_context = _build_single_masked_prompt(document=full_context,query=query,spans=masked_spans,
            change_template_contextCite=change_template_contextCite,)

        stats_chunk, logps_chunk = compute_probs(hf_model,hf_tok,
            [prompt],hf_device,None,
            batch_size=1,return_full_logp=True,file_name=compute_probs_file_name,detect_flip_to_true=p_true_flipping,
            true_variants=true_variants,false_variants=false_variants,save_file=False,stop_on_flip=False,)

        cur_stat = stats_chunk[0]
        cur_logp = logps_chunk[0]
        cur_stat["step_index"]=step

        masked_prompts.append(prompt)
        masked_contexts.append(masked_context)
        masked_stats.append(cur_stat)
        masked_logps.append(cur_logp)

        delta = _progress_delta(prev_stat, cur_stat, flip_to_true=p_true_flipping)
        recompute_triggered = False

        if enable_eps_recompute and delta is not None:
            if delta < recompute_epsilon:
                plateau_count += 1
            else:
                plateau_count = 0

            # adding recompute if the delta of the p_true is too minimal
            if plateau_count >= recompute_patience:
                new_scores = recompute_scores_fn(masked_context)
                new_scores = np.asarray(new_scores, dtype=np.float32)
                if len(new_scores) != len(ctx_rel_offsets):
                    raise ValueError(f"Recomputed scores len={len(new_scores)} != masking spans len={len(ctx_rel_offsets)}"
                    )
                scores_vec = new_scores
                remaining_after_pick = [i for i in remaining if i != next_idx and i not in selected]
                if stop_scores_relative is not None and scores_vec.size:
                    max_val = float(np.max(scores_vec))
                    threshold = max_val * stop_scores_relative
                    remaining_after_pick = [i for i in remaining_after_pick if float(scores_vec[i]) >= threshold]
                remaining_after_pick.sort(key=lambda i: float(scores_vec[i]), reverse=True)
                current_order = list(selected_order) + remaining_after_pick
                recompute_triggered = True
                plateau_count = 0

        trace.append(
            {   "step": len(selected_order),"chosen_idx": int(next_idx),
                "score_at_pick": None if scores_vec is None else float(scores_at_pick[-1]),
                "p_true": float(cur_stat["p_true"]),"log_odds": float(cur_stat["log_odds"]),
                "target_progress": float(_target_progress(cur_stat, flip_to_true=p_true_flipping)),
                "delta_progress": None if delta is None else float(delta),
                "tie": tie_info,
                "recompute_triggered": recompute_triggered,
                "plateau_count_after_step": int(plateau_count),
            }
        )

        prev_stat = cur_stat

        if stop_on_flip and _is_flip(cur_stat, flip_to_true=p_true_flipping):
            break

    if dump_json_path and save_logs:
        if baseline_stats is None:
            baseline_stats = compute_probs(hf_model,hf_tok,
                [full_prompt],
                hf_device,expected_result=None,batch_size=1,
                return_full_logp=True,file_name=compute_probs_file_name + ".baseline_tmp",
                detect_flip_to_true=p_true_flipping,
                true_variants=true_variants,false_variants=false_variants,
                save_file=False,stop_on_flip=False,)[0][0]

        dump_masked_prompts_json(dump_json_path,query=query,baseline_prompt=full_prompt,
            baseline_stats=baseline_stats,masked_prompts=masked_prompts,masked_stats=masked_stats,
            masked_context_list=masked_contexts,order=selected_order,scores_at_pick=scores_at_pick,
            policy=dump_policy,window=dump_window,
        )

    if adaptive_trace_path and save_logs:
        _write_adaptive_trace(adaptive_trace_path, trace)

    return masked_stats, masked_logps, selected_order, scores_at_pick



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

        #normalization
    scores_vec = _minmax_normalize_scores(scores_vec)

    if len(scores_vec) == 0:
        raise ValueError("scores_vec is empty; cannot run adaptive masking")

    # Initial static order.
    order = np.argsort(scores_vec)[::-1]

    selected: set[int] = set()
    current_order: List[int] = [int(i) for i in order]
    selected_order: List[int] = []
    scores_at_pick: List[float] = []

    masked_spans: List[Tuple[int, int]] = []
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

        if use_ptrue_tie:
            next_idx, tie_info = _choose_next_idx_with_ptrue_tie(
                remaining=topk,scores_vec=scores_vec,
                masked_spans=masked_spans,ctx_rel_offsets=ctx_rel_offsets,
                full_context=full_context,
                query=query,
                change_template_contextCite=change_template_contextCite,
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

        masked_spans.append(ctx_rel_offsets[next_idx])

        prompt, masked_context = _build_single_masked_prompt(document=full_context,query=query,spans=masked_spans,
            change_template_contextCite=change_template_contextCite,
        )

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
                "step": len(selected_order),
                "chosen_idx": int(next_idx),
                "score_at_pick": float(scores_at_pick[-1]),
                "p_true": float(cur_stat["p_true"]),
                "log_odds": float(cur_stat["log_odds"]),
                "target_progress": float(_target_progress(cur_stat, flip_to_true=p_true_flipping) ),
                "recompute_triggered": bool(recompute_triggered_this_step),
                "best_idx_before_recompute": int(best_idx_before_recompute),
                "best_score_before_recompute": float(best_score_before_recompute),
                "best_idx_before_pick": int(best_idx),
                "best_score_before_pick": float(best_score),"topk_indices": [int(i) for i in topk],
                "topk_scores": [float(s) for s in topk_scores],
                "used_ptrue_tie": bool(use_ptrue_tie),"tie": tie_info,
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
        )

    if adaptive_trace_path and save_logs:
        _write_adaptive_trace(adaptive_trace_path, trace)

    return masked_stats, masked_logps, selected_order, scores_at_pick