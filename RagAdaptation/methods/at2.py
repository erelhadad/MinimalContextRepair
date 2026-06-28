from __future__ import annotations

from pathlib import Path

from RagAdaptation.core.model_config import ModelConfig
from RagAdaptation.baseline.bruteforce_common import tokenize_context_with_offsets
from RagAdaptation.core.artifacts import method_dir, plots_dir, write_json
from RagAdaptation.core.models import get_hf_scorer_single_device
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.methods.common import (
    attention_token_scores_to_word_scores,
    get_at2_token_scores,
    mask_by_order,
    map_at2_scores_to_base_via_sources,
)
from RagAdaptation.core.timing import TimingRecorder
from RagAdaptation.core.prompting import InterventionMode, coerce_intervention_mode, split_context_to_word_units
'''
newton
AT2_ESTIMATOR_BY_MODEL = {
    "mistralai/Mistral-7B-Instruct-v0.3": Path(
        "/home/erel.hadad/artifacts/at2_estimators/mistralai_Mistral-7B-Instruct-v0.3_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
    "microsoft/Phi-3-mini-4k-instruct": Path(
        "/home/erel.hadad/artifacts/at2_estimators/microsoft_Phi-3-mini-4k-instruct_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
    "Qwen/Qwen3-4B-Instruct-2507":
    Path(
    "/home/erel.hadad/artifacts/at2_estimators/Qwen_Qwen3-4B-Instruct-2507_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"   
    ),
}


'''

AT2_ESTIMATOR_BY_MODEL_TOKEN = {
    "mistralai/Mistral-7B-Instruct-v0.3": Path(
        "/data/home/erel.hadad/RAG_EXP/outputs/mistralai_Mistral-7B-Instruct-v0.3_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
    "microsoft/Phi-3-mini-4k-instruct": Path(
        "/data/home/erel.hadad/RAG_EXP/outputs/microsoft_Phi-3-mini-4k-instruct_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
    "Qwen/Qwen3-4B-Instruct-2507":
    Path(
        "/data/home/erel.hadad/MinimalContextRepair/RagAdaptation/Qwen_Qwen3-4B-Instruct-2507_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
}

AT2_ESTIMATOR_BY_MODEL_NEWTON_WORD={
"mistralai/Mistral-7B-Instruct-v0.3": Path("/home/erel.hadad/MinimalContextRepair/outputs/WORDmistralai_Mistral-7B-Instruct-v0.3_databricks_databricks-dolly-15k_n1000_seed42_srcWord/estimators/default/score_estimator.pt"),
"microsoft/Phi-3-mini-4k-instruct": Path("/home/erel.hadad/MinimalContextRepair/outputs/WORDmicrosoft_Phi-3-mini-4k-instruct_databricks_databricks-dolly-15k_n1000_seed42_srcWord/estimators/default/score_estimator.pt"),
"Qwen/Qwen3-4B-Instruct-2507":
    Path("/home/erel.hadad/MinimalContextRepair/outputs/WORDQwen_Qwen3-4B-Instruct-2507_databricks_databricks-dolly-15k_n1000_seed42_srcWord/estimators/default/score_estimator.pt")
}

AT2_ESTIMATOR_BY_MODEL_LOCAL_WORD={
"mistralai/Mistral-7B-Instruct-v0.3": Path("/data/home/erel.hadad/MinimalContextRepair/outputs/WORDmistralai_Mistral-7B-Instruct-v0.3_databricks_databricks-dolly-15k_n1000_seed42_srcWord/estimators/default/score_estimator.pt"),
"microsoft/Phi-3-mini-4k-instruct": Path("/data/home/erel.hadad/MinimalContextRepair/outputs/WORDmicrosoft_Phi-3-mini-4k-instruct_databricks_databricks-dolly-15k_n1000_seed42_srcWord/estimators/default/score_estimator.pt"),
"Qwen/Qwen3-4B-Instruct-2507":
    Path("/data/home/erel.hadad/MinimalContextRepair/outputs/WORDQwen_Qwen3-4B-Instruct-2507_databricks_databricks-dolly-15k_n1000_seed42_srcWord/estimators/default/score_estimator.pt")
}

def _valid_estimator_path(path: Path | None) -> bool:
    return path is not None and str(path) not in {"", "."} and Path(path).exists()


def get_at2_scores_for_intervention_mode(
    *,model_id: str,
    full_context: str,
    query: str, hf_model, hf_tok, intervention_mode,
    prefer_word_scorer: bool = False,
    running_env:str="local"
):
    mode = coerce_intervention_mode(intervention_mode)
    generate_kwargs = {"max_new_tokens": 20, "do_sample": False}

    if mode == InterventionMode.MASK_TOKEN:
        est_path = AT2_ESTIMATOR_BY_MODEL_TOKEN.get(model_id)
        if est_path is None:
            raise ValueError(f"No AT2 estimator registered for model={model_id}")

        scores, gen, sources = get_at2_token_scores(
            full_context=full_context,
            query=query,
            hf_model=hf_model,
            hf_tok=hf_tok,
            score_estimator_path=est_path,
            generate_kwargs=generate_kwargs,
        )
        _, base_offsets = tokenize_context_with_offsets(full_context, hf_tok)
        scores_base = map_at2_scores_to_base_via_sources(
            context=full_context,
            source_pieces=sources,
            scores=scores,
            base_offsets=base_offsets,
            max_lookahead=64,
            max_merge_pieces=4,
            whitespace_flex=True,
        )
        return scores_base, base_offsets, gen, sources, est_path, "token"

    _pieces, word_units = split_context_to_word_units(full_context)
    word_offsets = [(int(u.start), int(u.end)) for u in word_units]
    word_est_path=""
    if prefer_word_scorer:
        if running_env=="newton":
            word_est_path = AT2_ESTIMATOR_BY_MODEL_NEWTON_WORD.get(model_id)
        else:#local or else
            word_est_path = AT2_ESTIMATOR_BY_MODEL_LOCAL_WORD.get(model_id)

    if prefer_word_scorer and _valid_estimator_path(word_est_path):
        scores, gen, sources = get_at2_token_scores(
            full_context=full_context,
            query=query,
            hf_model=hf_model,
            hf_tok=hf_tok,
            score_estimator_path=word_est_path,
            generate_kwargs=generate_kwargs,
            source_type="word",
        )
        scores_base = map_at2_scores_to_base_via_sources(
            context=full_context,
            source_pieces=sources,
            scores=scores,
            base_offsets=word_offsets,
            max_lookahead=64,
            max_merge_pieces=4,
            whitespace_flex=True,
        )
        if len(scores_base) != len(word_offsets):
            raise ValueError("AT2 word scorer did not align to WordUnit spans.")
        return scores_base, word_offsets, gen, sources, word_est_path, "word"

    est_path = AT2_ESTIMATOR_BY_MODEL_TOKEN.get(model_id)
    if est_path is None:
        raise ValueError(f"No AT2 estimator registered for model={model_id}")

    scores, gen, sources = get_at2_token_scores(
        full_context=full_context,
        query=query,
        hf_model=hf_model,
        hf_tok=hf_tok,
        score_estimator_path=est_path,
        generate_kwargs=generate_kwargs,
    )
    _, token_offsets = tokenize_context_with_offsets(full_context, hf_tok)
    token_scores = map_at2_scores_to_base_via_sources(
        context=full_context,
        source_pieces=sources,
        scores=scores,
        base_offsets=token_offsets,
        max_lookahead=64,
        max_merge_pieces=4,
        whitespace_flex=True,
    )
    word_scores = attention_token_scores_to_word_scores(
        full_context=full_context,
        hf_tok=hf_tok,
        token_scores=token_scores,
        word_units=word_units,
        reduction="sum",
    )
    return word_scores, word_offsets, gen, sources, est_path, "token_aggregated_to_word"


def run_at2_method(
    *,model_con: ModelConfig,out_dir: str,
    baseline_stats,
    model_id: str,
    full_context: str,
    query: str,
    p_true_flipping: bool,
    dump_policy: str,
    dump_window: int,
    save_logs: bool = True,
    save_plots: bool = False,
    stop_on_flip: bool = False,
    intervention_mode=None,
    replacement_resolver=None,
    prefer_at2_word_scorer: bool = False,
):
    hf_model_main, hf_tok_main, hf_device_main = model_con.load()
    timer = TimingRecorder()
    with timer.section("attributions_scores"):
        scores_base, base_offsets, gen, sources, est_path, scorer_kind = get_at2_scores_for_intervention_mode(
            model_id=model_id,
            full_context=full_context,
            query=query,
            hf_model=hf_model_main,
            hf_tok=hf_tok_main,
            intervention_mode=intervention_mode,
            prefer_word_scorer=prefer_at2_word_scorer,
        )

    method_path = method_dir(out_dir, "at2")

    if save_logs:
        with timer.section("write_logs"):
            raw_at2_dump = [
                {"token_idx": i, "token_text": sources[i], "score": float(scores_base[i])}
                for i in range(len(scores_base))
            ]
            mapped_dump = [
                {
                    "base_token_idx": i,
                    "span": [int(s), int(e)],
                    "token_text": full_context[s:e],
                    "mapped_score": float(scores_base[i]),
                }
                for i, (s, e) in enumerate(base_offsets)
            ]

            write_json(
                method_path / "scores.json",
                {
                    "model": model_id,
                    "estimator": str(est_path),
                    "scorer_kind": scorer_kind,
                    "generation": gen,
                    "raw_at2_scores": raw_at2_dump,
                    "mapped_scores_to_base_offsets": mapped_dump,
                },
            )
    with timer.section("masking_search_compute_probs"):
        masked_stats, masked_logps = mask_by_order(
        full_context,
        query,
        model_con=model_con,
        scores=scores_base,
        compute_probs_file_name=str(method_path / "flip_log.txt"),
        p_true_flipping=p_true_flipping,
        dump_json_path=str(method_path / "mask_trace.json"),
        dump_policy=dump_policy,
        dump_window=dump_window,
        source_offsets=base_offsets,
        force_class_prompt=True,
        baseline_stats=baseline_stats,
        save_logs=save_logs,
        stop_on_flip=stop_on_flip,
        intervention_mode=intervention_mode,
        replacement_resolver=replacement_resolver,
    )

    if save_plots:
        with timer.section("plot"):
            create_p_true_function(
                masked_logps,
                out_dir=str(plots_dir(out_dir)),
                filename="at2_p_true.png",
            )

    return {
        "generation_init": gen,
        "masked_stats": masked_stats,
        "masked_logps": masked_logps,
        "timing": timer.to_dict(),
        "estimator": str(est_path),
        "scorer_kind": scorer_kind,
    }
