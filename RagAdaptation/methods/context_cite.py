from __future__ import annotations

import numpy as np

try:
    from context_cite import ContextCiter
except ModuleNotFoundError:  # pragma: no cover
    ContextCiter = None

from RagAdaptation.baseline.partitioner import TokenContextPartitioner
from RagAdaptation.core.artifacts import method_dir, plots_dir
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.methods.common import mask_by_order
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE_A2T
from RagAdaptation.core.model_config import ModelConfig

def run_context_cite_method(*,model_con:ModelConfig, out_dir: str, baseline_stats, full_context: str, query: str, p_true_flipping: bool, dump_policy: str, dump_window: int):
    if ContextCiter is None:
        raise ModuleNotFoundError("context_cite is required for the context_cite method.")

    hf_model, hf_tok, hf_device= model_con.load()
    token_partitioner = TokenContextPartitioner(context=full_context, tokenizer=hf_tok, ablate_mode="blank")
    cc = ContextCiter(
        hf_model,
        hf_tok,
        full_context,
        query,
        prompt_template=TF_RAG_TEMPLATE_A2T,
        partitioner=token_partitioner,
    )
    raw_results = np.asarray(cc.get_attributions(), dtype=np.float32)
    contextcite_offsets = list(token_partitioner._spans)
    if len(raw_results) != len(contextcite_offsets):
        raise ValueError(
            f"ContextCite scores len={len(raw_results)} != partition spans len={len(contextcite_offsets)}"
        )

    method_path = method_dir(out_dir, "context_cite")
    masked_stats, masked_logps = mask_by_order(
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
        baseline_stats=baseline_stats,
    )
    create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename="context_cite_p_true.png")
    return {"masked_stats": masked_stats, "masked_logps": masked_logps}
