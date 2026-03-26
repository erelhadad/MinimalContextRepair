from __future__ import annotations

from pathlib import Path

from RagAdaptation.core.model_config import ModelConfig
from RagAdaptation.baseline.bruteforce_common import tokenize_context_with_offsets
from RagAdaptation.core.artifacts import method_dir, plots_dir, write_json
from RagAdaptation.core.models import get_hf_scorer_single_device
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.methods.common import (
    get_at2_token_scores,
    mask_by_order,
    map_at2_scores_to_base_via_sources,
)

AT2_ESTIMATOR_BY_MODEL = {
    "mistralai/Mistral-7B-Instruct-v0.3": Path(
        "/data/home/erel.hadad/RAG_EXP/outputs/mistralai_Mistral-7B-Instruct-v0.3_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
    "microsoft/Phi-3-mini-4k-instruct": Path(
        "/data/home/erel.hadad/RAG_EXP/outputs/microsoft_Phi-3-mini-4k-instruct_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
    "Qwen/Qwen3-4B-Instruct-2507":
    Path(
        "/data/home/erel.hadad/RAG_EXP/outputs/microsoft_Phi-3-mini-4k-instruct_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
}


def run_at2_method(
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
):
    est_path = AT2_ESTIMATOR_BY_MODEL.get(model_id)
    if est_path is None:
        raise ValueError(f"No AT2 estimator registered for model={model_id}")

    hf_model_at2, hf_tok_at2, hf_device_at2 = get_hf_scorer_single_device(
        model_id=model_id,
        device="cuda:0",
    )

    scores, gen, sources = get_at2_token_scores(
        full_context=full_context,
        query=query,
        hf_model=hf_model_at2,
        hf_tok=hf_tok_at2,
        score_estimator_path=est_path,
        generate_kwargs={"max_new_tokens": 128, "do_sample": False},
    )

    # Stable base tokenization of the original context
    _, base_offsets = tokenize_context_with_offsets(full_context, hf_tok_at2)

    # Robust mapping from AT2 source pieces back to the tokenizer base offsets
    scores_base = map_at2_scores_to_base_via_sources(
        context=full_context,
        source_pieces=sources,
        scores=scores,
        base_offsets=base_offsets,
        max_lookahead=64,
        max_merge_pieces=4,
        whitespace_flex=True,
    )

    method_path = method_dir(out_dir, "at2")

    if save_logs:
        raw_at2_dump = [
            {"token_idx": i, "token_text": sources[i], "score": float(scores[i])}
            for i in range(len(scores))
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
            method_path / "token_scores.json",
            {
                "model": model_id,
                "estimator": str(est_path),
                "generation": gen,
                "raw_at2_scores": raw_at2_dump,
                "mapped_scores_to_base_offsets": mapped_dump,
            },
        )

    masked_stats, masked_logps = mask_by_order(
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
    )


    create_p_true_function(
            masked_logps,
            out_dir=str(plots_dir(out_dir)),
            filename="at2_p_true.png",
        )

    return {
        "masked_stats": masked_stats,
        "masked_logps": masked_logps,
        "generation": gen,
        "estimator": str(est_path),
    }