from __future__ import annotations

import numpy as np

from RagAdaptation.core.artifacts import method_dir, plots_dir
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.methods.common import mask_by_order


def run_random_method(*, out_dir: str, baseline_stats, full_context: str, query: str, hf_model, hf_tok, hf_device, seeds: list[int], p_true_flipping: bool, dump_policy: str, dump_window: int, true_variants, false_variants, k: int = 1):
    results = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        method_path = method_dir(out_dir, "random", seed=seed)
        masked_stats, masked_logps = mask_by_order(
            full_context,
            query,
            k,
            hf_model,
            hf_tok,
            hf_device,
            scores=None,
            rng=rng,
            compute_probs_file_name=str(method_path / "compute_probs.txt"),
            p_true_flipping=p_true_flipping,
            dump_json_path=str(method_path / "dump.json"),
            dump_policy=dump_policy,
            dump_window=dump_window,
            true_variants=true_variants,
            false_variants=false_variants,
            baseline_stats=baseline_stats,
        )
        results[str(seed)] = {"masked_stats": masked_stats, "masked_logps": masked_logps}
        create_p_true_function(masked_logps, out_dir=str(plots_dir(out_dir)), filename=f"random_seed_{seed}_p_true.png")
    return results
