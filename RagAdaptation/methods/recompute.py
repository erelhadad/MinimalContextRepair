from __future__ import annotations

from RagAdaptation.core.artifacts import method_dir, plots_dir
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.methods.at2 import AT2_ESTIMATOR_BY_MODEL


def run_recompute_method(*, out_dir: str, rec_method: str, model_id: str, full_context: str, query: str, hf_model, hf_tok, hf_device, p_true_flipping: bool, true_variants, false_variants):
    from RagAdaptation.baseline.mask_iter_recompute_attention import mask_by_order_recompute
    from RagAdaptation.core.models import get_hf_scorer_single_device

    if rec_method == "attention":
        method_name = "recompute_attention"
        masked_stats, masked_logps, order, scores_at_pick = mask_by_order_recompute(
            full_context=full_context,
            query=query,
            hf_model=hf_model,
            hf_tok=hf_tok,
            hf_device=hf_device,
            batch_size=2,
            score_mode="attention",
            compute_probs_file_name=str(method_dir(out_dir, method_name) / "compute_probs.txt"),
            log_path=str(method_dir(out_dir, method_name) / "log.txt"),
            p_true_flipping=p_true_flipping,
            true_variants=true_variants,
            false_variants=false_variants,
        )
        create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename="recompute_attention_p_true.png")
        return method_name, {"masked_stats": masked_stats, "masked_logps": masked_logps, "order": order, "scores_at_pick": scores_at_pick}

    if rec_method == "context_cite":
        method_name = "recompute_context_cite"
        masked_stats, masked_logps, order, scores_at_pick = mask_by_order_recompute(
            full_context=full_context,
            query=query,
            hf_model=hf_model,
            hf_tok=hf_tok,
            hf_device=hf_device,
            batch_size=2,
            score_mode="context_cite",
            compute_probs_file_name=str(method_dir(out_dir, method_name) / "compute_probs.txt"),
            log_path=str(method_dir(out_dir, method_name) / "log.txt"),
            p_true_flipping=p_true_flipping,
            true_variants=true_variants,
            false_variants=false_variants,
        )
        create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename="recompute_context_cite_p_true.png")
        return method_name, {"masked_stats": masked_stats, "masked_logps": masked_logps, "order": order, "scores_at_pick": scores_at_pick}

    if rec_method == "at2":
        method_name = "recompute_at2"
        est_path = AT2_ESTIMATOR_BY_MODEL.get(model_id)
        if est_path is None:
            raise ValueError(f"No AT2 estimator registered for model={model_id}")
        hf_model_at2, hf_tok_at2, hf_device_at2 = get_hf_scorer_single_device(model_id=model_id, device="cuda:0")
        masked_stats, masked_logps, order, scores_at_pick = mask_by_order_recompute(
            full_context=full_context,
            query=query,
            hf_model=hf_model_at2,
            hf_tok=hf_tok_at2,
            hf_device=hf_device_at2,
            batch_size=2,
            score_mode="at2",
            compute_probs_file_name=str(method_dir(out_dir, method_name) / "compute_probs.txt"),
            log_path=str(method_dir(out_dir, method_name) / "log.txt"),
            score_estimator_path=est_path,
            generate_kwargs={"max_new_tokens": 128, "do_sample": False},
            p_true_flipping=p_true_flipping,
            true_variants=true_variants,
            false_variants=false_variants,
        )
        create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename="recompute_at2_p_true.png")
        return method_name, {"masked_stats": masked_stats, "masked_logps": masked_logps, "order": order, "scores_at_pick": scores_at_pick, "estimator": str(est_path)}

    raise ValueError(f"Unknown recompute method: {rec_method}")
