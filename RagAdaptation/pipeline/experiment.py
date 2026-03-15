from __future__ import annotations

import os
from typing import List, Optional

from RagAdaptation.compute_probs_updated import compute_probs
from RagAdaptation.core.artifacts import method_dir, write_json
from RagAdaptation.core.models import get_hf_scorer
from RagAdaptation.core.prompting import ChatPromptTemplate
from RagAdaptation.methods import (
    run_attention_method,
    run_at2_method,
    run_context_cite_method,
    run_random_method,
    run_recompute_method,
)
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE


def run_full_pipeline(*,model_id: str,
    query: str,full_context: str,
    methods: List[str],
    seeds: Optional[List[int]] = None,
    out_dir: str = "runs",
    detect_flip_to_true: bool = False,
    dump_policy: str = "flip",
    dump_window: int = 1,
    true_variants=None,false_variants=None,
    recompute: Optional[List[str]] = None,
    skip_recompute: int= 1):
    os.makedirs(out_dir, exist_ok=True)

    hf_model, hf_tok, hf_device = get_hf_scorer(model_id)

    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)
    baseline_prompt = prompt_template.format(context=full_context, question=query)

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
        if method_name == "attention":
            results["methods"]["attention"] = run_attention_method(out_dir=out_dir,
                baseline_prompt=baseline_prompt,baseline_stats=baseline_stats,
                full_context=full_context,query=query,
                hf_model=hf_model,hf_tok=hf_tok,hf_device=hf_device,
                p_true_flipping=detect_flip_to_true,
                dump_policy=dump_policy,dump_window=dump_window,
                true_variants=true_variants,false_variants=false_variants,)

        elif method_name == "random":
            results["methods"]["random"] = run_random_method(
                out_dir=out_dir,baseline_stats=baseline_stats,
                full_context=full_context,
                query=query,hf_model=hf_model,hf_tok=hf_tok,
                hf_device=hf_device,seeds=seeds,
                p_true_flipping=detect_flip_to_true,
                dump_policy=dump_policy,dump_window=dump_window,
                true_variants=true_variants,false_variants=false_variants,)

        elif method_name == "context_cite":
            results["methods"]["context_cite"] = run_context_cite_method(
                out_dir=out_dir,
                baseline_stats=baseline_stats,
                full_context=full_context,
                query=query,hf_model=hf_model,hf_tok=hf_tok,
                hf_device=hf_device,p_true_flipping=detect_flip_to_true,
                dump_policy=dump_policy,dump_window=dump_window,
                true_variants=true_variants,false_variants=false_variants,
            )
        elif method_name == "at2":
            results["methods"]["at2"] = run_at2_method(
                out_dir=out_dir,
                baseline_stats=baseline_stats,
                model_id=model_id,
                full_context=full_context,
                query=query,
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                p_true_flipping=detect_flip_to_true,
                dump_policy=dump_policy,
                dump_window=dump_window,
                true_variants=true_variants,
                false_variants=false_variants,
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

    for rec_method in recompute:
        try:
            result_name, payload = run_recompute_method(
                out_dir=out_dir,
                rec_method=rec_method,
                model_id=model_id,
                full_context=full_context,
                query=query,
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                p_true_flipping=detect_flip_to_true,
                true_variants=true_variants,
                false_variants=false_variants,
            )
            results["methods"][result_name] = payload
        except Exception as e:
            if rec_method == "at2":
                results["methods"]["recompute_at2"] = {"error": str(e), "status": "failed"}
            else:
                raise


    if skip_recompute!=1:
        for rec_method in recompute:
            try:
                result_name, payload = run_recompute_method(
                    out_dir=out_dir,rec_method=rec_method,model_id=model_id,
                    full_context=full_context,query=query,
                    hf_model=hf_model,hf_tok=hf_tok,hf_device=hf_device,
                    p_true_flipping=detect_flip_to_true,
                    true_variants=true_variants,false_variants=false_variants,
                    skip_recompute=skip_recompute,
                )
                results["methods"][result_name+skip_recompute] = payload
            except Exception as e:
                if rec_method == "at2":
                    results["methods"]["recompute_at2"] = {"error": str(e), "status": "failed"}
                else:
                    raise

    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    method_tag = "_".join(methods) if methods else "none"
    rec_tag = "_".join(recompute) if recompute else "none"

    out_path = os.path.join(out_dir, f"summary_methods_{method_tag}_recompute_{rec_tag}_{stamp}.json")
    write_json(out_path, results)

    compat_path = os.path.join(out_dir, f"pipeline_result_methods_{method_tag}_recompute_{rec_tag}_{stamp}.json")
    write_json(compat_path, results)

    return out_path
