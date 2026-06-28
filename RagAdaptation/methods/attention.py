from __future__ import annotations

from typing import Any, Dict

from RagAdaptation.core.artifacts import method_dir, plots_dir
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.core.timing import TimingRecorder
from RagAdaptation.methods.common import attention_token_scores_to_word_scores, get_attention_scores, mask_by_order
from RagAdaptation.core.model_config import ModelConfig
from RagAdaptation.core.prompting import InterventionMode, coerce_intervention_mode, split_context_to_word_units

def run_attention_method(*,model_con:ModelConfig, out_dir: str, baseline_prompt: str, baseline_stats: Dict[str, Any], full_context: str, query: str, p_true_flipping: bool, dump_policy: str, dump_window: int,save_logs:bool=True, save_plots: bool=False, stop_on_flip: bool=False,
                         intervention_mode=None,replacement_resolver=None):
    hf_model, hf_tok, hf_device= model_con.load()
    mode = coerce_intervention_mode(intervention_mode or InterventionMode.MASK_TOKEN)
    timer = TimingRecorder()
    with timer.section("attributions_scores"):
        attn = get_attention_scores(
            hf_model,
            hf_tok,
            hf_device,
            full_prompt=baseline_prompt,
            full_context=full_context,
            query=query,
        )
    source_offsets = None
    if mode != InterventionMode.MASK_TOKEN:
        _pieces, word_units = split_context_to_word_units(full_context)
        source_offsets = [(int(u.start), int(u.end)) for u in word_units]
        attn = attention_token_scores_to_word_scores(
            full_context=full_context,
            hf_tok=hf_tok,
            token_scores=attn,
            word_units=word_units,
            reduction="sum",
        )

    method_path = method_dir(out_dir, "attention")
    with timer.section("masking_search_compute_probs"):
        masked_stats, masked_logps = mask_by_order(
            full_context,
            query,
            model_con=model_con,
            scores=attn,
            compute_probs_file_name=str(method_path / "flip_log.txt"),
            p_true_flipping=p_true_flipping,
            dump_json_path=str(method_path / "mask_trace.json"),
            dump_policy=dump_policy,
            dump_window=dump_window,
            source_offsets=source_offsets,
            baseline_stats=baseline_stats,
            save_logs = save_logs, stop_on_flip = stop_on_flip,
            intervention_mode=mode,
            replacement_resolver=replacement_resolver,
        )

    if save_plots:
        with timer.section("plot"):
            create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename="attention_p_true.png")
    return {"masked_stats": masked_stats, "masked_logps": masked_logps, "timing": timer.to_dict()}
