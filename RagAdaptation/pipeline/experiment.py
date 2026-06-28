
from __future__ import annotations
from datetime import datetime
import os
from typing import List, Optional

from RagAdaptation.compute_probs_updated import compute_probs
from RagAdaptation.core.artifacts import write_json
from RagAdaptation.methods import (
    run_attention_method,
    run_at2_method,
    run_context_cite_method,
    run_random_method,
    run_recompute_method,
    run_attention_flow_method,
    run_attention_combined_method,
    run_context_cite_combined_method,
    run_at2_combined_method,
)
import RagAdaptation.core.model_config as Model_Config
from RagAdaptation.core.timing import TimingRecorder
from RagAdaptation.core.prompting import InterventionMode, coerce_intervention_mode
from RagAdaptation.core.replacements import ReplacementResolver

def run_full_pipeline(*, model_id: str,query: str, full_context: str,
    methods: List[str],seeds: Optional[List[int]] = None, out_dir: str = "runs",
    detect_flip_to_true: bool = False, dump_policy: str = "flip", dump_window: int = 1,
    recompute: Optional[List[str]] = None, skip_recompute: List[int] = None, save_logs: bool = False,
    save_plots: bool = False,
    stop_on_flip: bool = False,
    tau: float = 0.01,epsilon: float = 0.6,k: int = 5,
    intervention_mode:InterventionMode=1, replacement_cache: str | None = None,
    neutral_model: str | None = None,  # Deprecated no-op; retained for older callers.
    conceptnet_min_weight: float = 1.0,
    replacement_semex_filter: bool = True,
    replacement_semex_spacy_model: str = "en_core_web_sm",
    prefer_at2_word_scorer: bool = False,running_env:str="local",
    use_yes_no_variants: bool = False,
    use_model_default_tuning: bool = False,
):

    model_config = Model_Config.ModelConfig(
        model_id,
        use_yes_no_variants=use_yes_no_variants,
    )
    os.makedirs(out_dir, exist_ok=True)

    hf_model, hf_tok, hf_device = model_config.load()
    true_variants, false_variants = model_config.true_variants, model_config.false_variants
    baseline_prompt = model_config.format_prompt(question=query,context=full_context,context_cite_at2_formating=False,
    )

    timing = TimingRecorder()

    with timing.section("baseline_compute_probs"):
        baseline_stats_list, _ = compute_probs(
            hf_model, hf_tok,
            [baseline_prompt],
            hf_device, None,
            batch_size=1,
            return_full_logp=True,
            file_name="baseline_flip_log.txt",
            detect_flip_to_true=detect_flip_to_true,
            true_variants=true_variants,
            false_variants=false_variants,
            save_file=False,
            stop_on_flip=stop_on_flip,
        )

    baseline_stats = baseline_stats_list[0]
    mode = coerce_intervention_mode(intervention_mode)
    replacement_resolver = ReplacementResolver(
        cache_path=replacement_cache,
        conceptnet_min_weight=conceptnet_min_weight,
        semex_filter_enabled=replacement_semex_filter,
        semex_spacy_model=replacement_semex_spacy_model,
    )

    results = {
        "model_id": model_id,
        "query": query,
        "intervention_mode": mode.name,
        "replacement_model_id": model_id,
        "replacement_semex_filter": bool(replacement_semex_filter),
        "replacement_semex_spacy_model": replacement_semex_spacy_model,
        "p_true_flipping": detect_flip_to_true,
        "baseline": {
            "prompt": baseline_prompt,
            "stats": baseline_stats,
            "timing": timing.to_dict(),
        },
        "methods": {},
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = "_".join(methods) if methods else "none"
    rec_tag = "_".join(recompute or []) if recompute else "none"
    partial_path = os.path.join(out_dir,f"pipeline_result_methods_{method_tag}_recompute_{rec_tag}_{stamp}.partial.json",)

    def save_partial():
        write_json(partial_path, results)

    save_partial()

    seeds = seeds or [0]
    recompute = recompute or []
    import time

    for method_name in methods:
        if running_env=="local" and use_model_default_tuning:
            if model_id == "Qwen/Qwen3-4B-Instruct-2507":
                if method_name == "at2_combined":
                    k = 3
                    epsilon =0.6
                    tau = 0.05
                if method_name == "attention_combined":
                    k= 5
                    epsilon= 0.6
                    tau = 0.05
                if method_name == "context_cite_combined":
                    k = 5
                    epsilon = 0.7
                    tau = 0.05

            if model_id == "microsoft/Phi-3-mini-4k-instruct":
                if method_name == "at2_combined":
                    k = 5
                    epsilon = 0.5
                    tau = 0.2
                if method_name == "attention_combined":
                    k= 5
                    epsilon = 0.6
                    tau = 0.2
                if method_name == "context_cite_combined":
                    k = 5
                    epsilon = 0.5
                    tau = 0.2
            if model_id == "mistralai/Mistral-7B-Instruct-v0.3":
                if method_name == "at2_combined":
                    k = 5
                    epsilon =0.5
                    tau = 0.2
                if method_name == "attention_combined":
                    k= 3
                    epsilon= 0.5
                    tau = 0.2
                if method_name == "context_cite_combined": #approximation
                    k = 5
                    epsilon = 0.6
                    tau = 0.2
        elif running_env=="newton" and use_model_default_tuning:
                if model_id == "Qwen/Qwen3-4B-Instruct-2507":
                    if method_name == "at2_combined":
                        k = 5
                        epsilon = 0.5
                        tau = 0.2
                    if method_name == "attention_combined":
                        k = 5
                        epsilon = 0.6
                        tau = 0.2
                    if method_name == "context_cite_combined":
                        k = 5
                        epsilon = 0.6
                        tau = 0.2

                if model_id == "microsoft/Phi-3-mini-4k-instruct":
                    if method_name == "at2_combined":
                        k = 5
                        epsilon = 0.5
                        tau = 0.2
                    if method_name == "attention_combined":
                        k = 3
                        epsilon = 0.5
                        tau = 0.1
                    if method_name == "context_cite_combined":
                        k = 5
                        epsilon = 0.6
                        tau = 0.2
                if model_id == "mistralai/Mistral-7B-Instruct-v0.3":
                    if method_name == "at2_combined":
                        k = 4
                        epsilon = 0.6
                        tau = 0.2
                    if method_name == "attention_combined":
                        k = 4
                        epsilon = 0.5
                        tau = 0.1
                    if method_name == "context_cite_combined":  # approximation
                        k = 3
                        epsilon = 0.5
                        tau = 0.1
        elif use_model_default_tuning:
            print("there isn't any env mentioned therefore I couldn't match finetune parameter.")


        if method_name == "baseline":
            continue
        try:
            if method_name == "attention":
                method_timer = TimingRecorder()

                with method_timer.section("method_total"):
                    results["methods"]["attention"] = run_attention_method(
                        model_con=model_config,
                        out_dir=out_dir,
                        baseline_prompt=baseline_prompt,
                        baseline_stats=baseline_stats,
                        full_context=full_context,
                        query=query,
                        p_true_flipping=detect_flip_to_true,
                        dump_policy=dump_policy,
                        dump_window=dump_window,
                        save_logs=save_logs,
                        save_plots=save_plots,
                        stop_on_flip=stop_on_flip,
                        intervention_mode=mode,
                        replacement_resolver=replacement_resolver,
                    )
                results["methods"]["attention"].setdefault("timing", {})
                results["methods"]["attention"]["timing"]["outer"] = method_timer.to_dict()

                save_partial()

            elif method_name == "attention_flow":
                method_time = time.perf_counter()
                results["methods"]["attention_flow"] = run_attention_flow_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    baseline_stats=baseline_stats,
                    full_context=full_context,
                    query=query,
                    p_true_flipping=detect_flip_to_true,
                    dump_policy=dump_policy,
                    dump_window=dump_window,
                    save_logs=save_logs,
                    save_plots=save_plots,
                    stop_on_flip=stop_on_flip,
                    intervention_mode=mode,
                )
                results["methods"]["attention_flow"]["time"] = time.perf_counter() - method_time
                save_partial()

            elif method_name == "random":
                results["methods"]["random"] = run_random_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    baseline_stats=baseline_stats,
                    full_context=full_context,
                    query=query,
                    seeds=seeds,
                    p_true_flipping=detect_flip_to_true,
                    dump_policy=dump_policy,
                    dump_window=dump_window,
                    save_logs=save_logs,
                    save_plots=save_plots,
                    stop_on_flip=stop_on_flip,
                    intervention_mode=mode,
                    replacement_resolver=replacement_resolver,
                )
                save_partial()

            elif method_name == "context_cite":
                method_timer = TimingRecorder()

                with method_timer.section("method_total"):
                    results["methods"]["context_cite"] = run_context_cite_method(
                        model_con=model_config,out_dir=out_dir,
                        baseline_stats=baseline_stats,
                        full_context=full_context,
                        query=query,p_true_flipping=detect_flip_to_true,
                        dump_policy=dump_policy,dump_window=dump_window,
                        save_logs=save_logs,save_plots=save_plots,stop_on_flip=stop_on_flip,
                        intervention_mode=mode,
                        replacement_resolver=replacement_resolver)

                results["methods"]["context_cite"].setdefault("timing", {})
                results["methods"]["context_cite"]["timing"]["outer"] = method_timer.to_dict()
                save_partial()

            elif method_name == "at2":
                method_timer= TimingRecorder()
                with method_timer.section("method_total"):

                    results["methods"]["at2"] = run_at2_method(
                        model_con=model_config,
                        out_dir=out_dir,
                        baseline_stats=baseline_stats,
                        model_id=model_id,
                        full_context=full_context,
                        query=query,
                        p_true_flipping=detect_flip_to_true,
                        dump_policy=dump_policy,
                        dump_window=dump_window,
                        save_logs=save_logs,
                        save_plots=save_plots,
                        stop_on_flip=stop_on_flip,
                        intervention_mode=mode,
                        replacement_resolver=replacement_resolver,
                        prefer_at2_word_scorer=prefer_at2_word_scorer,
                    )
                results["methods"]["at2"].setdefault("timing", {})
                results["methods"]["at2"]["timing"]["outer"] = method_timer.to_dict()
                save_partial()

            # combined beam and adaptive recompute methods
            elif method_name == "attention_combined":
                method_timer = TimingRecorder()

                with method_timer.section("method_total"):
                    results["methods"]["attention_combined"] = run_attention_combined_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    baseline_prompt=baseline_prompt,
                    baseline_stats=baseline_stats,
                    full_context=full_context,
                    query=query,
                    p_true_flipping=detect_flip_to_true,
                    dump_policy=dump_policy,
                    dump_window=dump_window,
                    save_logs=save_logs,
                    save_plots=save_plots,
                    stop_on_flip=stop_on_flip,
                    tau=tau,
                    epsilon=epsilon,
                    k=k,
                    intervention_mode=mode,
                    replacement_resolver=replacement_resolver,
                )
                results["methods"]["attention_combined"].setdefault("timing", {})
                results["methods"]["attention_combined"]["timing"]["outer"] = method_timer.to_dict()

                save_partial()

            elif method_name == "context_cite_combined":
                method_timer = TimingRecorder()
                with method_timer.section("method_total"):
                    results["methods"]["context_cite_combined"] = run_context_cite_combined_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    baseline_stats=baseline_stats,
                    full_context=full_context,
                    query=query,
                    p_true_flipping=detect_flip_to_true,
                    dump_policy=dump_policy,
                    dump_window=dump_window,
                    save_logs=save_logs,
                    save_plots=save_plots,
                    stop_on_flip=stop_on_flip,
                    tau=tau,
                    epsilon=epsilon,
                    k=k,
                    intervention_mode=mode,
                    replacement_resolver=replacement_resolver,

                )
                results["methods"]["context_cite_combined"].setdefault("timing", {})
                results["methods"]["context_cite_combined"]["timing"]["outer"] = method_timer.to_dict()["total_sec"]
                save_partial()

            elif method_name == "at2_combined":
                method_timer = TimingRecorder()
                with method_timer.section("method_total"):
                    results["methods"]["at2_combined"] = run_at2_combined_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    baseline_stats=baseline_stats,
                    model_id=model_id,
                    full_context=full_context,
                    query=query,
                    p_true_flipping=detect_flip_to_true,
                    dump_policy=dump_policy,
                    dump_window=dump_window,
                    save_logs=save_logs,
                    save_plots=save_plots,
                    stop_on_flip=stop_on_flip,
                    tau=tau,
                    epsilon=epsilon,
                    k=k,
                    intervention_mode=mode,
                    replacement_resolver=replacement_resolver,
                    prefer_at2_word_scorer=prefer_at2_word_scorer,
                    running_env=running_env
                )
                results["methods"]["at2_combined"].setdefault("timing", {})
                results["methods"]["at2_combined"]["timing"]["outer"] = method_timer.to_dict()["total_sec"]
                save_partial()

            else:
                raise ValueError(f"Unknown method: {method_name}")


        except Exception as e:
            results["methods"][method_name] = {
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e),
            }
            save_partial()

    if skip_recompute is not None and 1 in skip_recompute:
        for rec_method in recompute:
            try:
                method_timer = TimingRecorder()
                with method_timer.section("method_total"):
                    result_name, payload = run_recompute_method(
                    model_con=model_config,
                    out_dir=out_dir,
                    rec_method=rec_method,
                    model_id=model_id,
                    full_context=full_context,
                    query=query,
                    p_true_flipping=detect_flip_to_true,
                    save_logs=save_logs,
                    save_plots=save_plots,
                    stop_on_flip=stop_on_flip,
                    intervention_mode=mode,
                    replacement_resolver=replacement_resolver,
                )
                results["methods"][result_name] = payload
                results["methods"][result_name].setdefault("timing", {})
                results["methods"][result_name]["timing"]["outer"] = method_timer.to_dict()
                save_partial()
            except Exception as e:
                if rec_method == "at2":
                    results["methods"]["recompute_at2"] = {"error": str(e), "status": "failed"}
                else:
                    raise
                save_partial()

    elif skip_recompute is not None:
        for val in skip_recompute:
            for rec_method in recompute:
                try:
                    method_timer=TimingRecorder()
                    with method_timer.section("method_total"):
                        result_name, payload = run_recompute_method(
                        model_con=model_config,
                        out_dir=out_dir,
                        rec_method=rec_method,
                        model_id=model_id,
                        full_context=full_context,
                        query=query,
                        p_true_flipping=detect_flip_to_true,
                        skip_recompute=val,
                        save_logs=save_logs,
                        save_plots=save_plots,
                        stop_on_flip=stop_on_flip,
                        intervention_mode=mode,
                        replacement_resolver=replacement_resolver,
                    )
                    key = f"{result_name}_SR{val}"
                    results["methods"][key] = payload
                    results["methods"][key].setdefault("timing", {})
                    results["methods"][key]["timing"]["inner"] = method_timer.to_dict()
                    save_partial()
                except Exception as e:
                    if rec_method == "at2":
                        results["methods"]["recompute_at2"] = {"error": str(e), "status": "failed"}
                    else:
                        raise
                    save_partial()

    final_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        out_dir,
        f"pipeline_result_methods_{method_tag}_recompute_{rec_tag}_{final_stamp}.json",
    )
    write_json(out_path, results)
    try:
        os.remove(partial_path)
    except FileNotFoundError:
        pass

    print(f"[done] saved {out_path}")
    return out_path
