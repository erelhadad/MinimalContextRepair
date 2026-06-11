from __future__ import annotations

import numpy as np
import time

from RagAdaptation.core.artifacts import method_dir, plots_dir
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.core.timing import TimingRecorder
from RagAdaptation.methods.common import mask_by_order
from RagAdaptation.core.model_config import ModelConfig

def run_random_method(*,model_con:ModelConfig, out_dir: str, baseline_stats, full_context: str, query: str, seeds: list[int], p_true_flipping: bool, dump_policy: str, dump_window: int, save_logs:bool=True, stop_on_flip:bool=False,
                      intervention_mode=None,replacement_resolver=None):
    results = {}
    for seed in seeds:
        timer = TimingRecorder()
        start_time= time.perf_counter_ns()
        rng = np.random.default_rng(seed)
        method_path = method_dir(out_dir, "random", seed=seed)
        with timer.section("masking_search_compute_probs"):
            masked_stats, masked_logps = mask_by_order(
                full_context,
                query,
                model_con=model_con,
                scores=None,
                rng=rng,
                compute_probs_file_name=str(method_path / "compute_probs.txt"),
                p_true_flipping=p_true_flipping,
                dump_json_path=str(method_path / "dump.json"),
                dump_policy=dump_policy,
                dump_window=dump_window,
                baseline_stats=baseline_stats,
                save_logs=save_logs,
                stop_on_flip=True,
                intervention_mode=intervention_mode,
                replacement_resolver=replacement_resolver,
            )
        seed_time= time.perf_counter_ns()-start_time
        results[str(seed)] = {"masked_stats": masked_stats, "masked_logps": masked_logps,"time": seed_time,"timing": timer.to_dict()}
    return results
