from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from context_cite import ContextCiter
except ModuleNotFoundError:  # pragma: no cover
    ContextCiter = None

from RagAdaptation.baseline.bruteforce_common import tokenize_context_with_offsets
from RagAdaptation.baseline.mask_iter_recompute_attention import (
    _at2_scores_mapped_to_base,
    _attention_scores_mapped_to_base,
    _contextcite_scores_mapped_to_base,
)
from RagAdaptation.baseline.partitioner import TokenContextPartitioner, WordContextPartitioner
from RagAdaptation.core.artifacts import method_dir, plots_dir, write_json
from RagAdaptation.core.model_config import ModelConfig
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.core.timing import TimingRecorder
from RagAdaptation.core.prompting import (
    ChatPromptTemplate,
    InterventionMode,
    coerce_intervention_mode,
    split_context_to_word_units,
)
from RagAdaptation.core.replacements import ReplacementResolver
from RagAdaptation.methods.adaptive_masking import mask_by_order_adaptive_combined
from RagAdaptation.methods.at2 import get_at2_scores_for_intervention_mode
from RagAdaptation.methods.common import (
    attention_token_scores_to_word_scores,
    find_token_indices_by_substring,
    get_attention_scores,
    get_at2_token_scores,
    map_at2_scores_to_base_via_sources,
)
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE, TF_RAG_TEMPLATE_A2T


_DEFAULT_COMBINED_K = 5
_DEFAULT_COMBINED_EPSILON = 0.6
_DEFAULT_COMBINED_TAU = 0.01


def _finalize_method_result(*, method_name: str, out_dir: str, masked_logps, payload: Dict[str, Any], save_logs=False):
    if save_logs:
        create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename=f"{method_name}_p_true.png")
    return payload


def _token_context_offsets(*, model_con: ModelConfig, hf_tok, full_context: str, query: str):
    full_prompt = model_con.format_prompt(
        question=query,
        context=full_context,
        context_cite_at2_formating=False,
        empty=False,
    )
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
    _ctx_token_indices, base_offsets, _after_ctx = find_token_indices_by_substring(
        full_prompt,
        full_context,
        offsets_full,
        start_search_at=0,
    )
    return base_offsets


def _word_offsets(full_context: str):
    _pieces, word_units = split_context_to_word_units(full_context)
    return word_units, [(int(u.start), int(u.end)) for u in word_units]


def run_attention_combined_method(
    *,
    model_con: ModelConfig,
    out_dir: str,
    baseline_prompt: str,
    baseline_stats,
    full_context: str,
    query: str,
    p_true_flipping: bool,
    dump_policy: str,
    dump_window: int,
    save_logs: bool = True,
    stop_on_flip: bool = False,
    k: int = _DEFAULT_COMBINED_K,
    epsilon: float = _DEFAULT_COMBINED_EPSILON,
    tau: float = _DEFAULT_COMBINED_TAU,
    intervention_mode=None,
    replacement_resolver: ReplacementResolver | None = None,
):
    hf_model, hf_tok, hf_device = model_con.load()
    mode = coerce_intervention_mode(intervention_mode or InterventionMode.MASK_TOKEN)
    timer = TimingRecorder()
    with timer.section("attributions_scores"):
        token_scores = get_attention_scores(
            hf_model,
            hf_tok,
            hf_device,
            full_prompt=baseline_prompt,
            full_context=full_context,
            query=query,
        )

    if mode == InterventionMode.MASK_TOKEN:
        scores = token_scores
        source_offsets = None
        recompute_base_offsets = _token_context_offsets(
            model_con=model_con,
            hf_tok=hf_tok,
            full_context=full_context,
            query=query,
        )
    else:
        word_units, source_offsets = _word_offsets(full_context)
        scores = attention_token_scores_to_word_scores(
            full_context=full_context,
            hf_tok=hf_tok,
            token_scores=token_scores,
            word_units=word_units,
            reduction="sum",
        )
        recompute_base_offsets = source_offsets

    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)

    def recompute_scores_fn(current_context: str) -> np.ndarray:
        if mode == InterventionMode.MASK_TOKEN:
            return _attention_scores_mapped_to_base(
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                prompt_template=prompt_template,
                masked_context=current_context,
                query=query,
                base_offsets=recompute_base_offsets,
            )

        current_prompt = model_con.format_prompt(
            question=query,
            context=current_context,
            context_cite_at2_formating=False,
            empty=False,
        )
        cur_token_scores = get_attention_scores(
            hf_model,
            hf_tok,
            hf_device,
            full_prompt=current_prompt,
            full_context=current_context,
            query=query,
        )
        # Keep the original WordUnit list.  MASK_WORD replaces words with
        # same-length spaces, so character offsets remain stable, but splitting
        # current_context again would drop already-masked words and shrink the
        # score vector.  Aggregating current token scores into the original
        # word spans gives zero/near-zero for masked words and preserves 1:1
        # alignment with the original search units.
        cur_scores = attention_token_scores_to_word_scores(
            full_context=current_context,
            hf_tok=hf_tok,
            token_scores=cur_token_scores,
            word_units=word_units,
            reduction="sum",
        )
        if len(cur_scores) != len(recompute_base_offsets):
            raise ValueError(
                f"Recomputed attention word scores len={len(cur_scores)} "
                f"!= original word count {len(recompute_base_offsets)}"
            )
        return cur_scores

    method_name = "attention_combined"
    method_path = method_dir(out_dir, method_name)
    with timer.section("masking_search_compute_probs"):
        masked_stats, masked_logps, order, scores_at_pick, excluded_units, candidate_filter = mask_by_order_adaptive_combined(
            full_context,
            query,
            model_con=model_con,
            scores=scores,
            compute_probs_file_name=str(method_path / "compute_probs.txt"),
            p_true_flipping=p_true_flipping,
            dump_json_path=str(method_path / "dump.json"),
            dump_policy=dump_policy,
            dump_window=dump_window,
            source_offsets=source_offsets,
            force_class_prompt=False,
            baseline_stats=baseline_stats,
            save_logs=save_logs,
            stop_on_flip=stop_on_flip,
            enable_eps_recompute=True,
            recompute_scores_fn=recompute_scores_fn,
            adaptive_trace_path=str(method_path / "adaptive_trace.json"),
            k=k,
            epsilon=epsilon,
            tau=tau,
            intervention_mode=mode,
            replacement_resolver=replacement_resolver,
        )

    return _finalize_method_result(
        method_name=method_name,
        out_dir=out_dir,
        masked_logps=masked_logps,
        payload={
            "masked_stats": masked_stats,
            "masked_logps": masked_logps,
            "order": order,
            "scores_at_pick": scores_at_pick,
            "combined_k": k,
            "combined_epsilon": epsilon,
            "combined_tau": tau,
            "excluded_units": excluded_units,
            "candidate_filter": candidate_filter,
            "timing": timer.to_dict(),
        },
        save_logs=save_logs,
    )


def run_context_cite_combined_method(
    *,
    model_con: ModelConfig,
    out_dir: str,
    baseline_stats,
    full_context: str,
    query: str,
    p_true_flipping: bool,
    dump_policy: str,
    dump_window: int,
    save_logs: bool = True,
    stop_on_flip: bool = False,
    k: int = _DEFAULT_COMBINED_K,
    epsilon: float = _DEFAULT_COMBINED_EPSILON,
    tau: float = _DEFAULT_COMBINED_TAU,
    intervention_mode=None,
    replacement_resolver: ReplacementResolver | None = None,
):
    if ContextCiter is None:
        raise ModuleNotFoundError("context_cite is required for the context_cite method.")

    hf_model, hf_tok, _hf_device = model_con.load()
    mode = coerce_intervention_mode(intervention_mode or InterventionMode.MASK_TOKEN)
    partitioner_cls = TokenContextPartitioner if mode == InterventionMode.MASK_TOKEN else WordContextPartitioner
    partitioner = partitioner_cls(context=full_context, tokenizer=hf_tok, ablate_mode="blank")
    timer = TimingRecorder()

    with timer.section("attributions_scores"):
        cc = ContextCiter(
            hf_model,
            hf_tok,
            full_context,
            query,
            prompt_template=TF_RAG_TEMPLATE_A2T,
            partitioner=partitioner,
        )
        raw_results = np.asarray(cc.get_attributions(), dtype=np.float32)
    contextcite_offsets = list(partitioner._spans)
    if len(raw_results) != len(contextcite_offsets):
        raise ValueError(
            f"ContextCite scores len={len(raw_results)} != partition spans len={len(contextcite_offsets)}"
        )

    def recompute_scores_fn(current_context: str) -> np.ndarray:
        if mode == InterventionMode.MASK_TOKEN:
            return _contextcite_scores_mapped_to_base(
                hf_model=hf_model,
                hf_tok=hf_tok,
                masked_context=current_context,
                query=query,
                base_offsets=contextcite_offsets,
            )

        # Same issue as attention recompute: after MASK_WORD, splitting the
        # current context again loses the blanked words.  Force ContextCite to
        # use the original stable word spans over the same-length current text.
        cur_partitioner = WordContextPartitioner(context=current_context, tokenizer=hf_tok, ablate_mode="blank")
        cur_partitioner._cache["spans"] = list(contextcite_offsets)
        cur_cc = ContextCiter(
            hf_model,
            hf_tok,
            current_context,
            query,
            prompt_template=TF_RAG_TEMPLATE_A2T,
            partitioner=cur_partitioner,
        )
        cur_scores = np.asarray(cur_cc.get_attributions(), dtype=np.float32)
        if len(cur_scores) != len(contextcite_offsets):
            raise ValueError(
                f"Recomputed ContextCite word scores len={len(cur_scores)} "
                f"!= original word count {len(contextcite_offsets)}"
            )
        return cur_scores

    method_name = "context_cite_combined"
    method_path = method_dir(out_dir, method_name)
    with timer.section("masking_search_compute_probs"):
        masked_stats, masked_logps, order, scores_at_pick, excluded_units, candidate_filter = mask_by_order_adaptive_combined(
            full_context,
            query,
            model_con=model_con,
            scores=raw_results,
            compute_probs_file_name=str(method_path / "compute_probs.txt"),
            p_true_flipping=p_true_flipping,
            dump_json_path=str(method_path / "dump.json"),
            dump_policy=dump_policy,
            dump_window=dump_window,
            source_offsets=contextcite_offsets,
            force_class_prompt=True,
            baseline_stats=baseline_stats,
            save_logs=save_logs,
            stop_on_flip=stop_on_flip,
            enable_eps_recompute=True,
            recompute_scores_fn=recompute_scores_fn,
            adaptive_trace_path=str(method_path / "adaptive_trace.json"),
            k=k,
            epsilon=epsilon,
            tau=tau,
            intervention_mode=mode,
            replacement_resolver=replacement_resolver,
        )

    return _finalize_method_result(
        method_name=method_name,
        out_dir=out_dir,
        masked_logps=masked_logps,
        payload={
            "masked_stats": masked_stats,
            "masked_logps": masked_logps,
            "order": order,
            "scores_at_pick": scores_at_pick,
            "combined_k": k,
            "combined_epsilon": epsilon,
            "combined_tau": tau,
            "excluded_units": excluded_units,
            "candidate_filter": candidate_filter,
            "timing": timer.to_dict(),
        },
        save_logs=save_logs,
    )


def run_at2_combined_method(
    *,
    model_con: ModelConfig,
    out_dir: str,
    baseline_stats,
    model_id: str,
    full_context: str,
    query: str,
    p_true_flipping: bool,
    dump_policy: str,
    dump_window: int,
    save_logs: bool = True,
    stop_on_flip: bool = False,
    k: int = _DEFAULT_COMBINED_K,
    epsilon: float = _DEFAULT_COMBINED_EPSILON,
    tau: float = _DEFAULT_COMBINED_TAU,
    intervention_mode=None,
    replacement_resolver: ReplacementResolver | None = None,
    prefer_at2_word_scorer: bool = False,
    running_env:str="local"
):
    hf_model, hf_tok, _hf_device = model_con.load()
    mode = coerce_intervention_mode(intervention_mode or InterventionMode.MASK_TOKEN)
    timer = TimingRecorder()

    with timer.section("attributions_scores"):
        scores_base, base_offsets, gen, sources, est_path, scorer_kind = get_at2_scores_for_intervention_mode(
            model_id=model_id,
            full_context=full_context,
            query=query,
            hf_model=hf_model,
            hf_tok=hf_tok,
            intervention_mode=mode,
            prefer_word_scorer=prefer_at2_word_scorer,
            running_env=running_env
        )

    base_word_units = None
    if mode != InterventionMode.MASK_TOKEN:
        _base_pieces, base_word_units = split_context_to_word_units(full_context)

    method_name = "at2_combined"
    method_path = method_dir(out_dir, method_name)
    if save_logs:
        with timer.section("write_logs"):
            write_json(
                method_path / f"unit_scores_{mode.name}.json",
                {
                    "model": model_id,
                    "estimator": str(est_path),
                    "scorer_kind": scorer_kind,
                    "generation": gen,
                    "raw_sources": [
                        {"unit": i, "unit_text": sources[i]}
                        for i in range(len(sources))
                    ],
                    "mapped_scores_to_base_offsets": [
                        {
                            "base_unit_idx": i,
                            "span": [int(s), int(e)],
                            "unit_text": full_context[s:e],
                            "mapped_score": float(scores_base[i]),
                        }
                        for i, (s, e) in enumerate(base_offsets)
                    ],
                },
            )

    def recompute_scores_fn(current_context: str) -> np.ndarray:
        if mode == InterventionMode.MASK_TOKEN:
            return _at2_scores_mapped_to_base(
                hf_model=hf_model,
                hf_tok=hf_tok,
                masked_context=current_context,
                query=query,
                base_offsets=base_offsets,
                score_estimator_path=est_path,
                generate_kwargs={"max_new_tokens": 20, "do_sample": False},
            )

        if base_word_units is None:
            raise RuntimeError("AT2 word recompute state was not initialized")

        # Recompute AT2 on the current same-length masked context, but map the
        # scores back into the original word offsets.  Do not call
        # get_at2_scores_for_intervention_mode() here, because it re-splits
        # current_context and therefore drops words that were masked to spaces.
        word_est_path = None
        valid_word_estimator = False
        try:
            from RagAdaptation.methods.at2 import AT2_ESTIMATOR_BY_MODEL_WORD, _valid_estimator_path
            word_est_path = AT2_ESTIMATOR_BY_MODEL_WORD.get(model_id)
            valid_word_estimator = _valid_estimator_path(word_est_path)
        except Exception:
            word_est_path = None
            valid_word_estimator = False

        if prefer_at2_word_scorer and valid_word_estimator:
            raw_scores, _cur_gen, cur_sources = get_at2_token_scores(
                full_context=current_context,
                query=query,
                hf_model=hf_model,
                hf_tok=hf_tok,
                score_estimator_path=word_est_path,
                generate_kwargs={"max_new_tokens": 20, "do_sample": False},
                source_type="word",
            )
            cur_scores = map_at2_scores_to_base_via_sources(
                context=current_context,
                source_pieces=cur_sources,
                scores=raw_scores,
                base_offsets=base_offsets,
                max_lookahead=64,
                max_merge_pieces=4,
                whitespace_flex=True,
            )
        else:
            raw_scores, _cur_gen, cur_sources = get_at2_token_scores(
                full_context=current_context,
                query=query,
                hf_model=hf_model,
                hf_tok=hf_tok,
                score_estimator_path=est_path,
                generate_kwargs={"max_new_tokens": 20, "do_sample": False},
            )
            _, cur_token_offsets = tokenize_context_with_offsets(current_context, hf_tok)
            cur_token_scores = map_at2_scores_to_base_via_sources(
                context=current_context,
                source_pieces=cur_sources,
                scores=raw_scores,
                base_offsets=cur_token_offsets,
                max_lookahead=64,
                max_merge_pieces=4,
                whitespace_flex=True,
            )
            cur_scores = attention_token_scores_to_word_scores(
                full_context=current_context,
                hf_tok=hf_tok,
                token_scores=cur_token_scores,
                word_units=base_word_units,
                reduction="sum",
            )

        if len(cur_scores) != len(base_offsets):
            raise ValueError(
                f"Recomputed AT2 word scores len={len(cur_scores)} != original word count {len(base_offsets)}"
            )
        return np.asarray(cur_scores, dtype=np.float32)

    with timer.section("masking_search_compute_probs"):
        masked_stats, masked_logps, order, scores_at_pick, excluded_units, candidate_filter = mask_by_order_adaptive_combined(
            full_context,
            query,
            model_con=model_con,
            scores=scores_base,
            compute_probs_file_name=str(method_path / "compute_probs.txt"),
            p_true_flipping=p_true_flipping,
            dump_json_path=str(method_path / "dump.json"),
            dump_policy=dump_policy,
            dump_window=dump_window,
            source_offsets=base_offsets,
            force_class_prompt=True,
            baseline_stats=baseline_stats,
            save_logs=save_logs,
            stop_on_flip=stop_on_flip,
            enable_eps_recompute=True,
            recompute_scores_fn=recompute_scores_fn,
            adaptive_trace_path=str(method_path / "adaptive_trace.json"),
            k=k,
            epsilon=epsilon,
            tau=tau,
            intervention_mode=mode,
            replacement_resolver=replacement_resolver,
        )

    return _finalize_method_result(
        method_name=method_name,
        out_dir=out_dir,
        masked_logps=masked_logps,
        payload={
            "masked_stats": masked_stats,
            "masked_logps": masked_logps,
            "order": order,
            "scores_at_pick": scores_at_pick,
            "generation": gen,
            "estimator": str(est_path),
            "scorer_kind": scorer_kind,
            "combined_k": k,
            "combined_epsilon": epsilon,
            "combined_tau": tau,
            "excluded_units": excluded_units,
            "candidate_filter": candidate_filter,
            "timing": timer.to_dict(),
        },
        save_logs=save_logs,
    )
