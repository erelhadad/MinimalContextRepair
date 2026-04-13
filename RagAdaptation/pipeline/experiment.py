from __future__ import annotations

import os
from typing import List, Optional

from RagAdaptation.compute_probs_updated import compute_probs
from RagAdaptation.core.artifacts import method_dir, write_json
from RagAdaptation.core.models import get_hf_scorer
from RagAdaptation.core.prompting import ChatPromptTemplate
from RagAdaptation.methods import (
    run_attention_method,run_at2_method,
    run_context_cite_method,
    run_random_method,run_recompute_method,)

import RagAdaptation.core.model_config as Model_Config
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE



def run_full_pipeline(*,model_id: str,
    query: str,full_context: str,
    methods: List[str],
    seeds: Optional[List[int]] = None,
    out_dir: str = "runs",
    detect_flip_to_true: bool = False,
    dump_policy: str = "flip",
    dump_window: int = 1,
    recompute: Optional[List[str]] = None,
    skip_recompute: List[int]= None,
    save_logs: bool= True,
    stop_on_flip: bool= False,
                      ):

    model_config = Model_Config.ModelConfig(model_id)
    os.makedirs(out_dir, exist_ok=True)

    hf_model, hf_tok, hf_device = model_config.load()
    true_variants, false_variants = model_config.true_variants, model_config.false_variants
    prompt_template = model_config.get_prompt_template()
    baseline_prompt =  model_config.format_prompt(
        question=query,
        context=full_context,
        context_cite_at2_formating=False,
    )

    baseline_dir = method_dir(out_dir, "baseline")
    baseline_stats_list, _ = compute_probs(
        hf_model,hf_tok,
        [baseline_prompt],
        hf_device,
        None,
        batch_size=1,
        return_full_logp=True,
        file_name=str(baseline_dir / f"compute_probs_{model_id}.txt"),
        detect_flip_to_true=detect_flip_to_true,
        true_variants=true_variants,
        false_variants=false_variants,
        save_file=save_logs,
        stop_on_flip=stop_on_flip,
    )
    baseline_stats = baseline_stats_list[0]

    results = {
        "model_id": model_id,
        "query": query,
        "p_true_flipping": detect_flip_to_true,
        "baseline": {"prompt": baseline_prompt, "stats": baseline_stats},
        "methods": {},
    }

    seeds = seeds or [0]
    recompute = recompute or []

    for method_name in methods:
        if method_name == "baseline":
            continue
        try:
            if method_name == "attention":
                    results["methods"]["attention"] = run_attention_method(model_con=model_config,out_dir=out_dir,
                        baseline_prompt=baseline_prompt,baseline_stats=baseline_stats,
                        full_context=full_context,query=query,
                        p_true_flipping=detect_flip_to_true,
                        dump_policy=dump_policy,dump_window=dump_window,save_logs=save_logs,
                        stop_on_flip=stop_on_flip,)

            elif method_name == "random":
                results["methods"]["random"] = run_random_method(
                    model_con=model_config,
                    out_dir=out_dir,baseline_stats=baseline_stats,
                    full_context=full_context,
                    query=query,seeds=seeds,
                    p_true_flipping=detect_flip_to_true,
                    dump_policy=dump_policy,dump_window=dump_window,
                    save_logs=save_logs,
                    stop_on_flip=stop_on_flip,
                )

            elif method_name == "context_cite":
                results["methods"]["context_cite"] = run_context_cite_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    baseline_stats=baseline_stats,
                    full_context=full_context,
                    query=query,p_true_flipping=detect_flip_to_true,
                    dump_policy=dump_policy,dump_window=dump_window,
                    save_logs=save_logs,
                    stop_on_flip=stop_on_flip,
                )
            elif method_name == "at2":
                results["methods"]["at2"] = run_at2_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    baseline_stats=baseline_stats,
                    model_id=model_id,
                    full_context=full_context,
                    query=query,
                    p_true_flipping=detect_flip_to_true,
                    dump_policy=dump_policy,
                    dump_window=dump_window,save_logs=save_logs,
                    stop_on_flip=stop_on_flip,)
            else:
                raise ValueError(f"Unknown method: {method_name}")
        except Exception as e:
            results["methods"][method_name] = {
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e),
            }


    if skip_recompute is not None and 1 in skip_recompute:
        for rec_method in recompute:
            try:
                result_name, payload = run_recompute_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    rec_method=rec_method,
                    model_id=model_id,
                    full_context=full_context,
                    query=query,
                    p_true_flipping=detect_flip_to_true,save_logs=save_logs,
                    stop_on_flip=stop_on_flip,
                )
                results["methods"][result_name] = payload
            except Exception as e:
                if rec_method == "at2":
                    results["methods"]["recompute_at2"] = {"error": str(e), "status": "failed"}
                else:
                    raise

    elif skip_recompute is not None:
        for val in skip_recompute:
            for rec_method in recompute:
                try:
                    result_name, payload = run_recompute_method(
                        model_con=model_config,
                        out_dir=out_dir,rec_method=rec_method,model_id=model_id,
                        full_context=full_context,query=query,
                        p_true_flipping=detect_flip_to_true,
                        skip_recompute=val,
                        save_logs=save_logs,
                        stop_on_flip=stop_on_flip,
                    )
                    results["methods"][f"{result_name}_SR{val}"] = payload
                except Exception as e:
                    if rec_method == "at2":
                        results["methods"]["recompute_at2"] = {"error": str(e), "status": "failed"}
                    else:
                        raise

    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = "_".join(methods) if methods else "none"
    rec_tag = "_".join(recompute) if recompute else "none"

    out_path = os.path.join(
        out_dir,
        f"pipeline_result_methods_{method_tag}_recompute_{rec_tag}_{stamp}.json"
    )
    print(f"[done] saved {out_path}")
    write_json(out_path, results)
    return out_path



