from __future__ import annotations

import argparse
from pathlib import Path

from RagAdaptation.core.paths import CACHE_DIR, RUNS_DIR
from RagAdaptation.pipeline.config import PipelineConfig
from RagAdaptation.pipeline.runner import run_dataset


'''
 find . -type d -name "methods" -prune -exec rm -rf {} +

newton
qwen: python -m RagAdaptation.run_pipeline --input ./outputs/reports/dataset_creation/hotpot_yesno__validation__qwen/report_flip_only__Qwen__Qwen3-4B-Instruct-2507.json --out_dir combined_word_qwen --running_env "local" --prefer_at2_word_scorer --intervention_mode mask_word --models "Qwen/Qwen3-4B-Instruct-2507" --stop_at_flip --save_logs --methods "attention_combined" "context_cite_combined" "at2_combined"
newton reports:

/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Phi_full_dataset_reports/report_flip_only__microsoft__Phi-3-mini-4k-instruct_with_context_lengths.json



/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Phi_full_dataset_reports/report_flip_only__Qwen__Qwen3-4B-Instruct-2507_with_context_lengths.json



/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Mistral_full_dataset_reports/report_flip_only__mistralai__Mistral-7B-Instruct-v0.3_with_context_lengths.json

brit combined word

qwen:  
 python -m RagAdaptation.run_pipeline --input /data/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/hotpot_yesno__validation__all__full/report_flip_only__Qwen__Qwen3-4B-Instruct-2507.json --out_dir "antonym_replacements/qwen" --running_env "local" --prefer_at2_word_scorer --intervention_mode ANTONYM_WORD --models "Qwen/Qwen3-4B-Instruct-2507" --stop_at_flip --save_logs --methods "attention_combined" "context_cite_combined" "at2_combined" --examples_range 0 200 --disable_replacement_semex_filter

minstral: 
 python -m RagAdaptation.run_pipeline --models "mistralai/Mistral-7B-Instruct-v0.3" --methods "attention_combined" "context_cite_combined" "at2_combined" --save_logs --stop_at_flip --out_dir "antonym_replacements/mins" --input  /data/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/hotpot_yesno__validation__all__full/report_flip_only__mistralai__Mistral-7B-Instruct-v0.3.json  --running_env "local" --prefer_at2_word_scorer --intervention_mode ANTONYM_WORD  --examples_range 0 200 --disable_replacement_semex_filter

micro:
 python -m RagAdaptation.run_pipeline --models microsoft/Phi-3-mini-4k-instruct --methods "attention_combined" "context_cite_combined" "at2_combined" --save_logs --stop_at_flip --out_dir "antonym_replacements/micro" --input /data/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/hotpot_yesno__validation__all__full/report_flip_only__microsoft__Phi-3-mini-4k-instruct.json --running_env "local" --prefer_at2_word_scorer --intervention_mode mask_token --examples_range 0 10 


'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="report.json that contains field of relevant example")
    ap.add_argument("--out_dir", default=str(RUNS_DIR))
    ap.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3",
                 "Qwen/Qwen3-4B-Instruct-2507"],
    )
    ap.add_argument("--methods", nargs="+", default=["attention", "random", "context_cite","at2"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 10, 20, 40])
    ap.add_argument("--context_field", default="context")
    ap.add_argument("--recompute", nargs="+", default=[])
    ap.add_argument("--skip_recompute",nargs="*", type=int, default=[])
    ap.add_argument("--skip_examples", nargs="*", type=int, default=[])
    ap.add_argument("--save_logs",  action="store_true")
    ap.add_argument("--stop_at_flip", action="store_true")
    ap.add_argument("--examples_range", nargs=2, type=int,help="Range of examples to run")
    ap.add_argument("--tau",type=float, default=0.01)
    ap.add_argument("--epsilon",type=float, default=0.6)
    ap.add_argument("--k",type=int, default=5)
    ap.add_argument("--intervention_mode", type=str, default="mask_token")
    ap.add_argument("--use_yes_no_variants", action="store_true")
    ap.add_argument("--replacement_cache", type=str, default=str(CACHE_DIR / "replacement_cache.json"))
    ap.add_argument("--neutral_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--conceptnet_min_weight", type=float, default=1.0)
    ap.add_argument("--disable_replacement_semex_filter", action="store_true")
    ap.add_argument("--replacement_semex_spacy_model", type=str, default="en_core_web_sm")
    ap.add_argument("--prefer_at2_word_scorer", action="store_true")
    ap.add_argument("--running_env", type=str, default="local")
    ap.add_argument("--max_scoring_prompt_batch_size", type=int)
    ap.add_argument("--scoring_row_batch_size", type=int)
    ap.add_argument("--streaming_chunk_size", type=int, default=32)
    ap.add_argument("--cleanup_every_batches", type=int, default=0)
    ap.add_argument("--aggressive_cleanup", action="store_true")
    ap.add_argument("--unload_models_between_runs", action="store_true")
    ap.add_argument("--attention_native_seq_len_limit", type=int, default=512)
    args = ap.parse_args()

    config = PipelineConfig(
        input_path=Path(args.input),
        output_root=Path(args.out_dir),
        models=list(args.models),
        methods=list(args.methods),
        seeds=list(args.seeds),
        recompute=list(args.recompute),
        context_field=args.context_field,
        skip_example_indices=list(args.skip_examples),
        skip_recompute=args.skip_recompute,
        save_logs=args.save_logs,
        stop_at_flip=args.stop_at_flip,
        examples_range=args.examples_range,
        tau=args.tau
        ,epsilon=args.epsilon,
        k=args.k,
        intervention_mode=args.intervention_mode,
        use_yes_no_variants=args.use_yes_no_variants,
        replacement_cache=Path(args.replacement_cache),
        neutral_model=args.neutral_model,
        conceptnet_min_weight=args.conceptnet_min_weight,
        replacement_semex_filter=not args.disable_replacement_semex_filter,
        replacement_semex_spacy_model=args.replacement_semex_spacy_model,
        prefer_at2_word_scorer=args.prefer_at2_word_scorer,
        running_env=args.running_env,
        max_scoring_prompt_batch_size=args.max_scoring_prompt_batch_size,
        scoring_row_batch_size=args.scoring_row_batch_size,
        streaming_chunk_size=args.streaming_chunk_size,
        cleanup_every_batches=args.cleanup_every_batches,
        aggressive_cleanup=args.aggressive_cleanup,
        unload_models_between_runs=args.unload_models_between_runs,
        attention_native_seq_len_limit=args.attention_native_seq_len_limit,
    )

    run_root = run_dataset(config)
    print(f"[ok] wrote organized run to {run_root}")


if __name__ == "__main__":
    main()
