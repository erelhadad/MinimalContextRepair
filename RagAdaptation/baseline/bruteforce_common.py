from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from RagAdaptation.core.paths import DATA_DIR, LOGS_DIR
from RagAdaptation.core.artifacts import ensure_dir, sanitize_name
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE


TIME_LIMIT_SECONDS = 2 * 60 * 60  # 2 hours


def tokenize_context_with_offsets(context: str, tok) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Return tokenizer ids and character offsets for each token in the original context.
    Requires a fast HF tokenizer.
    """
    enc = tok(
        context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = enc["input_ids"]
    off = enc["offset_mapping"]
    if hasattr(off, "tolist"):
        off = off.tolist()
    offsets: List[Tuple[int, int]] = [(int(s), int(e)) for (s, e) in off]
    return [int(x) for x in ids], offsets


def _merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    spans2 = [(int(s), int(e)) for (s, e) in spans if int(s) < int(e)]
    spans2.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    for s, e in spans2:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def mask_context_spans(context: str, spans: Sequence[Tuple[int, int]]) -> str:
    """Return context where each span is replaced with a single space."""
    merged = _merge_spans(spans)
    out = context
    for s, e in reversed(merged):
        out = out[:s] + " " + out[e:]
    return out


def create_masked_prompts(document: str, query: str, offsets: List[Tuple[int, int]], k: int = 1):
    """Return prompts and masked contexts for every k-token combination."""
    document_size = len(offsets)
    combinations_indices = combinations(range(document_size), k)
    batch: List[str] = []
    masked_context_list: List[str] = []
    prompt_template = TF_RAG_TEMPLATE

    for indices in combinations_indices:
        spans_of_indices = [offsets[j] for j in indices]
        masked_context = mask_context_spans(document, spans_of_indices)
        masked_context_list.append(masked_context)
        prompt_full = prompt_template.format(context=masked_context, question=query)
        batch.append(prompt_full)

    return batch, masked_context_list


def plot_histograms(
    m_logp: list,
    out_dir: str | Path = "plots_Ptrue",
    tag: str = "values of the p_t for each masked tokens prompts",
):
    """Plot histogram of P(True) values for masked prompts."""
    out_dir = ensure_dir(out_dir)
    m = np.array(m_logp, dtype=float)
    if m.size == 0:
        return str(out_dir / f"Ptrue_hist_{sanitize_name(tag)}.png")

    plt.figure()
    plt.xlim(m.min(), m.max())
    plt.hist(m, bins=180, alpha=0.7, label="masked")
    plt.xlabel("Probability of the token True to be generated")
    plt.ylabel("Masked prompts count")
    plt.legend()
    plt.title(f"Histogram of {tag}")
    out_path = out_dir / f"Ptrue_hist_{sanitize_name(tag)}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return str(out_path)


def generate_answer(model, tokenizer, prompt, max_new_tokens=10, device=None):
    if device is None:
        device = model.device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    answer_ids = out_ids[0, enc["input_ids"].shape[1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return answer_text.strip()


def get_tf_candidate_ids(tok):
    """Return single-token candidate ids for common true/false variants."""
    candidates = [" true", " false", " True", " False", "true", "false", "True", "False"]
    cand_ids = {}
    for s in candidates:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            cand_ids[s] = ids[0]
    return cand_ids


def resolve_document_path(document: str) -> Path:
    if document == "sugar":
        return DATA_DIR / "Is_sugar_addictive_text_only_no_header.pdf"
    if document == "vegan":
        return DATA_DIR / "vegan.txt"
    candidate = Path(document)
    if candidate.exists():
        return candidate
    candidate = DATA_DIR / document
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve document path from: {document}")


def bruteforce_output_dir(tag: str = "default") -> Path:
    return ensure_dir(LOGS_DIR / "bruteforce" / sanitize_name(tag))
