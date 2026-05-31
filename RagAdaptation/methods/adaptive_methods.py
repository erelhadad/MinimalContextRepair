
from __future__ import annotations

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
from RagAdaptation.methods.adaptive_masking import mask_by_order_adaptive , mask_by_order_adaptive_combined
from RagAdaptation.methods.at2 import AT2_ESTIMATOR_BY_MODEL, AT2_ESTIMATOR_BY_MODEL_WORD
from RagAdaptation.methods.common import (
    find_token_indices_by_substring,
    get_attention_scores,
    get_at2_token_scores,
    map_at2_scores_to_base_via_sources,
)
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE, TF_RAG_TEMPLATE_A2T
from RagAdaptation.core.prompting import ChatPromptTemplate
from RagAdaptation.core.models import get_hf_scorer_single_device


_DEFAULT_TIE_ABS_GAP = 1e-4
_DEFAULT_TIE_REL_GAP = 0.01
_DEFAULT_TIE_MAX_CANDIDATES = 2
_DEFAULT_RECOMPUTE_EPS = 0.01
_DEFAULT_RECOMPUTE_PATIENCE = 2

_DEFAULT_COMBINED_K = 5
_DEFAULT_COMBINED_EPSILON = 0.6
_DEFAULT_COMBINED_TAU = 0.01


from RagAdaptation.core.prompting import (
    InterventionMode,
    coerce_intervention_mode,
    split_context_to_word_units,
)
from RagAdaptation.methods.common import attention_token_scores_to_word_scores

def _finalize_method_result(*, method_name: str, out_dir: str, masked_logps, payload: Dict[str, Any], save_logs=False):
    if save_logs:
        create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename=f"{method_name}_p_true.png")
    return payload


def run_attention_ptrue_tie_method(*,model_con: ModelConfig,
    out_dir: str,baseline_prompt: str,baseline_stats,full_context: str,query: str,
    p_true_flipping: bool,dump_policy: str,dump_window: int,save_logs: bool = True,
    stop_on_flip: bool = False,):

    hf_model, hf_tok, hf_device = model_con.load()

    #the updated version of the function-hidden states usage
    attn = get_attention_scores( hf_model,hf_tok,hf_device,
        full_prompt=baseline_prompt,full_context=full_context,query=query,)

    method_name = "attention_ptrue_tie"
    method_path = method_dir(out_dir, method_name)
    masked_stats, masked_logps, order, scores_at_pick = mask_by_order_adaptive(
        full_context,query,model_con=model_con,scores=attn,compute_probs_file_name=str(method_path / "compute_probs.txt"),
        p_true_flipping=p_true_flipping,dump_json_path=str(method_path / "dump.json"),dump_policy=dump_policy,dump_window=dump_window,baseline_stats=baseline_stats,
        save_logs=save_logs,stop_on_flip=stop_on_flip,
        enable_ptrue_tie=True, tie_abs_gap=_DEFAULT_TIE_ABS_GAP, tie_rel_gap=_DEFAULT_TIE_REL_GAP,tie_max_candidates=_DEFAULT_TIE_MAX_CANDIDATES,
        adaptive_trace_path=str(method_path / "adaptive_trace.json"),
    )
    return _finalize_method_result(
        method_name=method_name,out_dir=out_dir,
        masked_logps=masked_logps,
        payload={
            "masked_stats": masked_stats,
            "masked_logps": masked_logps,
            "order": order,
            "scores_at_pick": scores_at_pick,
        },save_logs=save_logs
    )

#combinded methods iterations

def run_attention_combined_method(*, model_con: ModelConfig,out_dir: str,baseline_prompt: str,baseline_stats,full_context: str,query: str,p_true_flipping: bool,
    dump_policy: str,dump_window: int,save_logs: bool = True,stop_on_flip: bool = False,
    k: int = _DEFAULT_COMBINED_K,epsilon: float = _DEFAULT_COMBINED_EPSILON, tau: float = _DEFAULT_COMBINED_TAU,
    intervention_mode=None):
    """
    Attention + combined adaptive masking.

    Combined policy:
      - use attention scores as the initial ordering;
      - if current best remaining score < epsilon, recompute attention scores;
      - if top-k scores are within tau, choose the next index by p_true lookahead.
    """
    hf_model, hf_tok, hf_device = model_con.load()

    mode =intervention_mode

    if mode not in {InterventionMode.MASK_TOKEN, InterventionMode.MASK_WORD}:
        raise NotImplementedError(
            "attention_combined currently supports MASK_TOKEN and MASK_WORD. "
            "Replacement modes need a word-aware combined loop because replacements "
            "can change text length."
        )

    tokens_score = get_attention_scores(hf_model,hf_tok,hf_device,
        full_prompt=baseline_prompt,full_context=full_context,query=query,)

    if mode == InterventionMode.MASK_TOKEN:
        full_prompt = model_con.format_prompt(question=query,context=full_context,context_cite_at2_formating=False,empty=False,
        )

        enc_full = hf_tok(full_prompt,add_special_tokens=False,return_offsets_mapping=True,truncation=False,padding=False,
        )

        offsets_full = enc_full["offset_mapping"]
        if hasattr(offsets_full, "tolist"):
            offsets_full = offsets_full.tolist()

        _ctx_token_indices, base_offsets, _after_ctx = find_token_indices_by_substring(
            full_prompt,full_context,offsets_full, start_search_at=0,
        )
        source_offsets=None

    elif (mode == InterventionMode.MASK_WORD or intervention_mode==InterventionMode.REPLACEMENT_ANTONYM_WORD
          or intervention_mode==InterventionMode.REPLACEMENT_ANTONYM_WORD):
        _pieces, word_units = split_context_to_word_units(full_context)
        source_offsets = [(int(u.start), int(u.end)) for u in word_units]

        scores = attention_token_scores_to_word_scores(
            full_context=full_context,
            hf_tok=hf_tok,
            token_scores=tokens_score,
            word_units=word_units,
            reduction="sum",
        )

        recompute_base_offsets = source_offsets

    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)

    def recompute_scores_fn(masked_context: str) -> np.ndarray:
        return _attention_scores_mapped_to_base(
            hf_model=hf_model,
            hf_tok=hf_tok,
            hf_device=hf_device,
            prompt_template=prompt_template,
            masked_context=masked_context,
            query=query,
            base_offsets=recompute_base_offsets,
        )

    method_name = "attention_combined"
    method_path = method_dir(out_dir, method_name)

    masked_stats, masked_logps, order, scores_at_pick = mask_by_order_adaptive_combined(
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
        intervention_mode=intervention_mode
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
    intervention_mode=None
):
    """
    ContextCite + combined adaptive masking.

    Combined policy:
      - use ContextCite scores as the initial ordering;
      - if current best remaining score < epsilon, recompute ContextCite scores;
      - if top-k scores are within tau, choose the next index by p_true lookahead.
    """
    if ContextCiter is None:
        raise ModuleNotFoundError("context_cite is required for the context_cite method.")

    hf_model, hf_tok, _hf_device = model_con.load()

    mode = intervention_mode
    if mode==intervention_mode.MASK_TOKEN:
        partitioner = TokenContextPartitioner(
            context=full_context,
            tokenizer=hf_tok,
            ablate_mode="blank",
        )

    elif (mode==intervention_mode.MASK_WORD or intervention_mode==InterventionMode.REPLACEMENT_ANTONYM_WORD
          or intervention_mode==InterventionMode.REPLACEMENT_ANTONYM_WORD):
        partitioner= WordContextPartitioner(
        context = full_context,
        tokenizer = hf_tok,
        ablate_mode = "blank",
        )

    cc = ContextCiter(hf_model,hf_tok,full_context,query,prompt_template=TF_RAG_TEMPLATE_A2T,
        partitioner=partitioner,)

    raw_results = np.asarray(cc.get_attributions(), dtype=np.float32)
    contextcite_offsets = list(partitioner._spans)

    if len(raw_results) != len(contextcite_offsets):
        raise ValueError(
            f"ContextCite scores len={len(raw_results)} "
            f"!= partition spans len={len(contextcite_offsets)}"
        )


    def recompute_scores_fn(masked_context: str) -> np.ndarray:
        return _contextcite_scores_mapped_to_base(
            hf_model=hf_model,
            hf_tok=hf_tok,
            masked_context=masked_context,
            query=query,
            base_offsets=contextcite_offsets,
        )

    method_name = "context_cite_combined"
    method_path = method_dir(out_dir, method_name)

    masked_stats, masked_logps, order, scores_at_pick = mask_by_order_adaptive_combined(
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
        intervention_mode=intervention_mode
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
        },
        save_logs=save_logs,
    )


def run_at2_combined_method(*,model_con: ModelConfig,out_dir: str,baseline_stats,model_id: str,full_context: str,
    query: str,p_true_flipping: bool,dump_policy: str,dump_window: int,save_logs: bool = True, stop_on_flip: bool = False,
    k: int = _DEFAULT_COMBINED_K,epsilon: float = _DEFAULT_COMBINED_EPSILON, tau: float = _DEFAULT_COMBINED_TAU,
    intervention_mode=None
):
    """
    AT2 + combined adaptive masking.

    Combined policy:
      - use AT2 mapped token scores as the initial ordering;
      - if current best remaining score < epsilon, recompute AT2 scores;
      - if top-k scores are within tau, choose the next index by p_true lookahead.
    """


    hf_model_main, hf_tok_main, _hf_device_main = model_con.load()

    if intervention_mode == InterventionMode.MASK_TOKEN:
        est_path = AT2_ESTIMATOR_BY_MODEL.get(model_id)
        if est_path is None:
            raise ValueError(f"No AT2 estimator registered for model={model_id}")

        scores, gen, sources = get_at2_token_scores(
            full_context=full_context,
            query=query,
            hf_model=hf_model_main,
            hf_tok=hf_tok_main,
            score_estimator_path=est_path,
            generate_kwargs={"max_new_tokens": 20, "do_sample": False},
        )

        _, base_offsets = tokenize_context_with_offsets(full_context, hf_tok_main)

    elif (intervention_mode ==InterventionMode.MASK_WORD or intervention_mode==InterventionMode.REPLACEMENT_ANTONYM_WORD
          or intervention_mode==InterventionMode.REPLACEMENT_ANTONYM_WORD):
        est_path=AT2_ESTIMATOR_BY_MODEL_WORD.get(model_id)
        scores, gen, sources = get_at2_token_scores(
            full_context=full_context,
            query=query,
            hf_model=hf_model_main,
            hf_tok=hf_tok_main,
            score_estimator_path=est_path,
            generate_kwargs={"max_new_tokens": 20, "do_sample": False},
        )
        #need to convert it to word subtitution
        _, base_offsets = tokenize_context_with_offsets(full_context, hf_tok_main)


    scores_base = map_at2_scores_to_base_via_sources(
        context=full_context,
        source_pieces=sources,
        scores=scores,
        base_offsets=base_offsets,
        max_lookahead=64,
        max_merge_pieces=4,
        whitespace_flex=True,
    )

    method_name = "at2_combined"
    method_path = method_dir(out_dir, method_name)

    if save_logs:
        raw_at2_dump = [
            {
                "unit": i,
                "unit_text": sources[i],
                "score": float(scores[i]),
            }
            for i in range(len(scores))
        ]

        mapped_dump = [
            {
                "base_unit_idx": i,
                "span": [int(s), int(e)],
                "unit_text": full_context[s:e],
                "mapped_score": float(scores_base[i]),
            }
            for i, (s, e) in enumerate(base_offsets)
        ]

        write_json(
            method_path / f"unit_scores{intervention_mode}.json",
            {
                "model": model_id,
                "estimator": str(est_path),
                "generation": gen,
                "raw_at2_scores": raw_at2_dump,
                "mapped_scores_to_base_offsets": mapped_dump,
            },
        )

    def recompute_scores_fn(masked_context: str) -> np.ndarray:
        return _at2_scores_mapped_to_base(
            hf_model=hf_model_main,
            hf_tok=hf_tok_main,
            masked_context=masked_context,
            query=query,
            base_offsets=base_offsets,
            score_estimator_path=est_path,
            generate_kwargs={"max_new_tokens": 20, "do_sample": False},
        )

    masked_stats, masked_logps, order, scores_at_pick = mask_by_order_adaptive_combined(
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
        intervention_mode=intervention_mode
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
            "combined_k": k,
            "combined_epsilon": epsilon,
            "combined_tau": tau,
        },
        save_logs=save_logs,
    )