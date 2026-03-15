from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from RagAdaptation.core.artifacts import create_run_root, example_dir, model_dir, write_example_inputs, write_manifest
from RagAdaptation.core.documents import combine_document_text, load_documents_any
from RagAdaptation.pipeline.config import PipelineConfig
from RagAdaptation.prompts_format import normalize_true_false


def load_items(path: str | Path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON format")


def norm_expected(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    return normalize_true_false(str(x))


def build_manifest(config: PipelineConfig, items_count: int) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(config.input_path),
        "output_root": str(config.output_root),
        "models": list(config.models),
        "methods": list(config.methods),
        "seeds": list(config.seeds),
        "recompute": list(config.recompute),
        "items_count": int(items_count),
        "context_field": config.context_field,
        "skip_example_indices": list(config.skip_example_indices),
    }


def run_dataset(config: PipelineConfig, *, run_pipeline_fn: Callable[..., str] | None = None) -> Path:
    items = load_items(config.input_path)
    run_root = create_run_root(config.output_root)
    write_manifest(run_root, build_manifest(config, len(items)))

    if run_pipeline_fn is None:
        from RagAdaptation.pipeline.experiment import run_full_pipeline as run_pipeline_fn

    for ex_i, ex in enumerate(items):
        if ex_i in set(config.skip_example_indices):
            continue

        query = ex.get("query") or ex.get("question")
        if query is None:
            raise ValueError(f"Example {ex_i} has no query/question")

        expected = ex.get("expected_answer_norm") or ex.get("expected_answer")
        # THE ANSWER THAT THE MODEL GAVE WITH THE CONTEXT
        expected_norm = norm_expected(expected)

        context_path = ex.get("context_path")
        if not context_path:
            raise ValueError(f"Example {ex_i} has no context_path")

        docs = load_documents_any(context_path)
        full_context = combine_document_text(docs)


        ex_dir = example_dir(run_root, ex_i)
        write_example_inputs(ex_dir, example_payload=ex, context_text=full_context)


        for model_id in config.models:
            if not ex["per_model"][model_id]["relevant"]:
                print(f"Example {ex_i} has no relevant run for the model {model_id}")
                continue

            #meaning are we in the search for flipping from false to true
            detect_flip_to_true = ex["per_model"][model_id]["prob_label_with_context"] == "false"
            print(f"p_true_fliiping is for model:{model_id} is: ", detect_flip_to_true,"\n")

            mdl_dir = model_dir(ex_dir, model_id)
            run_pipeline_fn(model_id=model_id,
                query=query,
                full_context=full_context,
                methods=config.methods,
                seeds=config.seeds,
                out_dir=str(mdl_dir),
                detect_flip_to_true=detect_flip_to_true,
                dump_policy="flip",
                dump_window=1,
                true_variants=config.true_variants,
                false_variants=config.false_variants,
                recompute=config.recompute,
                skip_recompute=config.skip_recompute
            )

    return run_root
