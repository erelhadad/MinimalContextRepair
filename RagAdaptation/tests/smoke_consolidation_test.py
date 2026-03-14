from __future__ import annotations

from pathlib import Path
import tempfile

from RagAdaptation.core.documents import combine_document_text, load_documents_any
from RagAdaptation.core.artifacts import create_run_root, example_dir, model_dir, method_dir, plots_dir, write_json
from RagAdaptation.core.plotting import create_p_true_function
from RagAdaptation.document_handling.database_hadling import load_documents_any as legacy_load_documents_any


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ctx = tmp / "ctx.txt"
        ctx.write_text("alpha\nbeta", encoding="utf-8")

        docs = load_documents_any(ctx)
        assert combine_document_text(docs) == "alpha\nbeta"
        docs2 = legacy_load_documents_any(str(ctx))
        assert combine_document_text(docs2) == "alpha\nbeta"

        run_root = create_run_root(tmp / "runs", "demo")
        ex_dir = example_dir(run_root, 3)
        mdl_dir = model_dir(ex_dir, "org/model")
        mdir = method_dir(mdl_dir, "random", seed=7)
        pdir = plots_dir(mdl_dir)
        write_json(mdir / "summary.json", {"ok": True})
        assert (mdir / "summary.json").exists()

        plot_path = create_p_true_function([0.2, 0.4, 0.9], out_dir=pdir, filename="curve.png")
        assert Path(plot_path).exists()

        print("smoke_consolidation_test: OK")


if __name__ == "__main__":
    main()
