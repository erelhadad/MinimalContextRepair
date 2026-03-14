from __future__ import annotations

import tempfile
from pathlib import Path

from RagAdaptation.baseline import bruteforce as brute_wrapper
from RagAdaptation.baseline.bruteforce_common import (
    bruteforce_output_dir,
    create_masked_prompts,
    plot_histograms,
    resolve_document_path,
    tokenize_context_with_offsets,
)


class DummyTok:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, truncation=False):
        parts = text.split()
        ids = list(range(len(parts)))
        if not return_offsets_mapping:
            return {"input_ids": ids}
        offsets = []
        pos = 0
        for part in parts:
            start = text.find(part, pos)
            end = start + len(part)
            offsets.append((start, end))
            pos = end
        return {"input_ids": ids, "offset_mapping": offsets}


def main():
    tok = DummyTok()
    ctx = "alpha beta gamma"
    ids, offsets = tokenize_context_with_offsets(ctx, tok)
    assert ids == [0, 1, 2]
    prompts, masked = create_masked_prompts(ctx, "Is this a test?", offsets, k=1)
    assert len(prompts) == 3
    assert len(masked) == 3

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "plots"
        plot_path = plot_histograms([0.1, 0.5, 0.7], out_dir=out_dir, tag="demo")
        assert Path(plot_path).exists()

    assert callable(brute_wrapper.main)
    assert callable(brute_wrapper.run_exp)

    sugar_path = resolve_document_path("sugar")
    assert sugar_path.name == "Is_sugar_addictive_text_only_no_header.pdf"

    out = bruteforce_output_dir("demo-test")
    assert out.name == "demo-test"
    print("smoke_bruteforce_consolidation_test: OK")


if __name__ == "__main__":
    main()
