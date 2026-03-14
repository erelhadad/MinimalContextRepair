from __future__ import annotations

from typing import Any, Dict

from RagAdaptation.core.artifacts import method_dir, plots_dir
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.methods.common import get_attention_scores, mask_by_order


def run_attention_method(*, out_dir: str, baseline_prompt: str, baseline_stats: Dict[str, Any], full_context: str, query: str, hf_model, hf_tok, hf_device, p_true_flipping: bool, dump_policy: str, dump_window: int, true_variants, false_variants, k: int = 1):
    attn = get_attention_scores(hf_model, hf_tok, hf_device, baseline_prompt)
    method_path = method_dir(out_dir, "attention")
    masked_stats, masked_logps = mask_by_order(
        full_context,
        query,
        k,
        hf_model,
        hf_tok,
        hf_device,
        scores=attn,
        compute_probs_file_name=str(method_path / "compute_probs.txt"),
        p_true_flipping=p_true_flipping,
        dump_json_path=str(method_path / "dump.json"),
        dump_policy=dump_policy,
        dump_window=dump_window,
        true_variants=true_variants,
        false_variants=false_variants,
        baseline_stats=baseline_stats,
    )
    create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename="attention_p_true.png")
    return {"masked_stats": masked_stats, "masked_logps": masked_logps}
