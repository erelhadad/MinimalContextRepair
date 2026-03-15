from __future__ import annotations

import json
from pathlib import Path
from RagAdaptation.core.model_config import ModelConfig
from RagAdaptation.baseline.bruteforce_common import tokenize_context_with_offsets
from RagAdaptation.core.artifacts import method_dir, plots_dir, write_json
from RagAdaptation.core.models import get_hf_scorer_single_device
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.methods.common import get_at2_token_scores, mask_by_order

AT2_ESTIMATOR_BY_MODEL = {
    "mistralai/Mistral-7B-Instruct-v0.3": Path(
        "/data/home/erel.hadad/RAG_EXP/outputs/mistralai_Mistral-7B-Instruct-v0.3_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
    "microsoft/Phi-3-mini-4k-instruct": Path(
        "/data/home/erel.hadad/RAG_EXP/outputs/microsoft_Phi-3-mini-4k-instruct_databricks_databricks-dolly-15k_n1000_seed42_srcToken/estimators/default/score_estimator.pt"
    ),
}


def run_at2_method(*,model_con:ModelConfig, out_dir: str, baseline_stats, model_id: str, full_context: str, query: str, p_true_flipping: bool, dump_policy: str, dump_window: int,):
    est_path = AT2_ESTIMATOR_BY_MODEL.get(model_id)
    if est_path is None:
        raise ValueError(f"No AT2 estimator registered for model={model_id}")

    hf_model_at2, hf_tok_at2, hf_device_at2 = get_hf_scorer_single_device(model_id=model_id, device="cuda:0")
    scores, gen, sources = get_at2_token_scores(
        full_context=full_context,
        query=query,
        hf_model=hf_model_at2,
        hf_tok=hf_tok_at2,
        score_estimator_path=est_path,
        generate_kwargs={"max_new_tokens": 128, "do_sample": False},
    )

    _, at2_offsets = tokenize_context_with_offsets(full_context, hf_tok_at2)
    if len(scores) != len(at2_offsets):
        raise ValueError(f"AT2 scores len={len(scores)} != tokenizer offsets len={len(at2_offsets)}")

    at2_dump = [{"token_idx": i, "token_text": sources[i], "score": float(scores[i])} for i in range(len(scores))]
    method_path = method_dir(out_dir, "at2")
    write_json(
        method_path / "token_scores.json",
        {
            "model": model_id,
            "estimator": str(est_path),
            "generation": gen,
            "scores": at2_dump,
        },
    )

    masked_stats, masked_logps = mask_by_order(
        full_context,
        query,
        model_con=model_con,
        scores=scores,
        compute_probs_file_name=str(method_path / "compute_probs.txt"),
        p_true_flipping=p_true_flipping,
        dump_json_path=str(method_path / "dump.json"),
        dump_policy=dump_policy,
        dump_window=dump_window,
        source_offsets=at2_offsets,
        force_class_prompt=True,
        baseline_stats=baseline_stats,
    )
    create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename="at2_p_true.png")
    return {"masked_stats": masked_stats, "masked_logps": masked_logps, "generation": gen, "estimator": str(est_path)}
