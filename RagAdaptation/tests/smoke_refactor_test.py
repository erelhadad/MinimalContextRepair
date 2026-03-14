from __future__ import annotations

import json
import tempfile
from pathlib import Path

from RagAdaptation.pipeline.config import PipelineConfig
from RagAdaptation.pipeline.runner import run_dataset


def fake_run_pipeline_fn(**kwargs):
    out_dir = Path(kwargs["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": kwargs["model_id"],
        "query": kwargs["query"],
        "methods": kwargs["methods"],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_dir / "summary.json")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ctx = tmp / "ctx.txt"
        ctx.write_text("Line one.\nLine two.", encoding="utf-8")
        inp = tmp / "examples.json"
        inp.write_text(json.dumps([
            {
                "query": "Is this a test?",
                "expected_answer_norm": "true",
                "context_path": str(ctx),
            }
        ]), encoding="utf-8")

        config = PipelineConfig(
            input_path=inp,
            output_root=tmp / "runs",
            models=["org/model-a"],
            methods=["random"],
            seeds=[7],
            recompute=[],
        )
        run_root = run_dataset(config, run_pipeline_fn=fake_run_pipeline_fn)

        assert (run_root / "manifest.json").exists()
        ex_dir = run_root / "examples" / "ex0000"
        assert (ex_dir / "input.json").exists()
        assert (ex_dir / "context.txt").read_text(encoding="utf-8") == "Line one.\nLine two."
        mdl_dir = ex_dir / "models" / "org__model-a"
        assert (mdl_dir / "summary.json").exists()

        manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["items_count"] == 1
        assert manifest["models"] == ["org/model-a"]

        print("smoke_refactor_test: OK")


if __name__ == "__main__":
    main()
