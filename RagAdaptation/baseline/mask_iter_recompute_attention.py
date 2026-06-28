from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from RagAdaptation.methods.attention_flow import _get_augmented_attention_mats_auto, _build_sparse_flow_graph
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import torch
import numpy as np

try:
    from context_cite import ContextCiter
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ContextCiter = None

# --- make `import RagAdaptation...` work even if you run this file directly ---
_THIS_FILE = Path(__file__).resolve()
_PKG_DIR = _THIS_FILE.parents[1]          # .../RagAdaptation
_PROJECT_ROOT = _PKG_DIR.parent           # .../RAG_EXP
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RagAdaptation.compute_probs_updated import compute_probs
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE, TF_RAG_TEMPLATE_A2T
from RagAdaptation.baseline.bruteforce_common import tokenize_context_with_offsets
from RagAdaptation.core.prompting import ChatPromptTemplate
from RagAdaptation.core.prompting import (
    InterventionMode,
    build_context_with_word_interventions_metadata,
    coerce_intervention_mode,
    split_context_to_word_units,
    _filter_word_order_by_available_intervention,
)
from RagAdaptation.core.replacements import (
    ReplacementResolver,
    build_replacement_map_for_order,
    filter_replacement_order_semex,
)
from RagAdaptation.baseline.partitioner import TokenContextPartitioner

from RagAdaptation.methods.common import (
    get_at2_token_scores,
    map_at2_scores_to_base_via_sources,
    mask_context_spans_same_length,
    _rewrite_chunked_step_metadata,
    _write_compute_probs_flip_log,
    _write_masking_checkpoint,
    _infer_attention_model_type,
    _project_qk_last_layer,
)

_HF_TOK = None
_HF_MODEL = None


def _find_token_indices_by_substring(
    full_text: str,substring: str,offsets_mapping: Sequence[Tuple[int, int]],start_search_at: int = 0,):
    """
    Returns:
      (token_indices, relative_offsets_in_substring, end_char_pos_in_full_text)

    - token_indices: indices in the tokenized full_text
    - relative_offsets_in_substring: [(start,end)] relative to substring start
    """
    begin = full_text.find(substring, start_search_at)
    if begin < 0:
        raise ValueError(
            "Could not locate substring inside the prompt. Check template / uniqueness."
        )
    end = begin + len(substring)

    tok_indices: List[int] = []
    rel_offsets: List[Tuple[int, int]] = []
    for i, (s, e) in enumerate(offsets_mapping):
        if e <= s:
            continue
        if s >= begin and e <= end:
            tok_indices.append(i)
            rel_offsets.append((int(s - begin), int(e - begin)))

    return tok_indices, rel_offsets, end


def _map_scores_by_char_overlap(
    base_offsets: List[Tuple[int, int]],
    cur_offsets: List[Tuple[int, int]],
    cur_scores: np.ndarray,
) -> np.ndarray:
    """
    Map scores from the current prompt's context-tokenization back to the
    base context token spans (tokenization of the original full_context),
    using character-overlap weighted averaging.
    """
    n = len(base_offsets)
    m = len(cur_offsets)
    base_scores = np.zeros(n, dtype=np.float32)

    i = 0
    j = 0
    while i < n and j < m:
        bs, be = base_offsets[i]
        cs, ce = cur_offsets[j]

        if ce <= bs:
            j += 1
            continue
        if be <= cs:
            i += 1
            continue

        overlap = min(be, ce) - max(bs, cs)
        if overlap > 0:
            base_scores[i] += float(overlap) * float(cur_scores[j])

        if ce < be:
            j += 1
        else:
            i += 1

    lengths = np.array([max(1, e - s) for (s, e) in base_offsets], dtype=np.float32)
    return base_scores / lengths


def _aggregate_token_scores_to_spans(
    *,
    target_offsets: Sequence[Tuple[int, int]],
    token_offsets: Sequence[Tuple[int, int]],
    token_scores: np.ndarray,
    reduction: str = "sum",
) -> np.ndarray:
    token_scores = np.asarray(token_scores, dtype=np.float32)
    out = np.zeros(len(target_offsets), dtype=np.float32)

    for out_i, (s, e) in enumerate(target_offsets):
        vals = []
        s, e = int(s), int(e)
        for tok_i, (ts, te) in enumerate(token_offsets):
            if tok_i >= len(token_scores):
                break
            ts, te = int(ts), int(te)
            if te <= ts:
                continue
            if min(e, te) > max(s, ts):
                vals.append(float(token_scores[tok_i]))

        if not vals:
            out[out_i] = 0.0
        elif reduction == "sum":
            out[out_i] = float(np.sum(vals))
        elif reduction == "mean":
            out[out_i] = float(np.mean(vals))
        elif reduction == "max":
            out[out_i] = float(np.max(vals))
        else:
            raise ValueError(f"Unsupported reduction={reduction!r}")

    return out


def _build_word_intervention_context_and_offsets(
    *,
    pieces: Sequence[str],
    word_units,
    selected_word_ids: Sequence[int],
    mode: InterventionMode,
    replacement_map: Optional[Mapping[Any, str]],
) -> Tuple[str, List[Tuple[int, int]]]:
    selected = {int(i) for i in selected_word_ids}
    context, metadata = build_context_with_word_interventions_metadata(
        pieces=pieces,
        word_units=word_units,
        selected_word_ids=selected,
        mode=mode,
        replacement_map=replacement_map,
    )

    current_pieces = list(pieces)
    unit_by_piece = {int(unit.piece_index): unit for unit in word_units}
    for wid, meta in metadata.items():
        unit = word_units[int(wid)]
        replacement = meta.get("replacement")
        if replacement is not None:
            current_pieces[int(unit.piece_index)] = str(replacement)

    word_offsets: List[Tuple[int, int]] = [(0, 0)] * len(word_units)
    pos = 0
    for piece_i, piece in enumerate(current_pieces):
        piece_text = str(piece)
        start = pos
        end = start + len(piece_text)
        unit = unit_by_piece.get(int(piece_i))
        if unit is not None:
            word_offsets[int(unit.word)] = (start, end)
        pos = end

    if pos != len(context):
        raise RuntimeError("Internal error: rebuilt word context length mismatch.")

    return context, word_offsets


def _attention_scores_mapped_to_base(
    *,
    hf_model,
    hf_tok,
    hf_device,
    prompt_template: ChatPromptTemplate,
    masked_context: str,
    query: str,
    base_offsets: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Compute attention scores on the current masked prompt, then map to stable
    base_offsets. Scoring stays the same:
      - last layer
      - average across heads
      - question -> context block
      - sum over question tokens

    Implementation detail:
      - do NOT use output_attentions=True
      - reconstruct only the needed last-layer attention block from hidden states
    """
    full_prompt = prompt_template.format(context=masked_context, question=query)

    enc_full = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )

    offsets_full = enc_full["offset_mapping"]
    if hasattr(offsets_full, "tolist"):
        offsets_full = offsets_full.tolist()

    ctx_token_indices, ctx_rel_offsets, after_ctx = _find_token_indices_by_substring(
        full_prompt, masked_context, offsets_full, start_search_at=0
    )
    q_token_indices, _, _ = _find_token_indices_by_substring(
        full_prompt, query, offsets_full, start_search_at=after_ctx
    )

    enc = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )
    enc = {k: v.to(hf_device) for k, v in enc.items()}

    with torch.no_grad():
        out = hf_model(
            **enc,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )

    hidden_states = out.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden states.")

    model_type = _infer_attention_model_type(hf_model)
    query_states, key_states, causal_mask, head_dim = _project_qk_last_layer(
        hf_model,
        hidden_states,
        model_type=model_type,
    )

    q_start = q_token_indices[0]
    q_end = q_token_indices[-1] + 1

    query_states = query_states[:, :, q_start:q_end, :]
    causal_mask = causal_mask[:, :, q_start:q_end, :]

    attn_scores = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / np.sqrt(float(head_dim))
    attn_scores = attn_scores + causal_mask
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)

    last_layer_q_to_all = attn_weights[0].mean(dim=0)  # [|Q|, seq]

    c_idx = torch.tensor(
        ctx_token_indices,
        device=last_layer_q_to_all.device,
        dtype=torch.long,
    )
    q_to_c = last_layer_q_to_all.index_select(1, c_idx)  # [|Q|, |C|]
    scores_ctx = q_to_c.sum(dim=0).detach().float().cpu().numpy().astype(
        np.float32, copy=False
    )

    scores_base = _map_scores_by_char_overlap(base_offsets, ctx_rel_offsets, scores_ctx)

    del (
        out,
        hidden_states,
        query_states,
        key_states,
        causal_mask,
        attn_scores,
        attn_weights,
        last_layer_q_to_all,
        q_to_c,
        c_idx,
        enc,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores_base


def _attention_token_scores_for_context(
    *,
    hf_model,
    hf_tok,
    hf_device,
    prompt_template: ChatPromptTemplate,
    masked_context: str,
    query: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    full_prompt = prompt_template.format(context=masked_context, question=query)

    enc_full = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )

    offsets_full = enc_full["offset_mapping"]
    if hasattr(offsets_full, "tolist"):
        offsets_full = offsets_full.tolist()

    ctx_token_indices, ctx_rel_offsets, after_ctx = _find_token_indices_by_substring(
        full_prompt, masked_context, offsets_full, start_search_at=0
    )
    q_token_indices, _, _ = _find_token_indices_by_substring(
        full_prompt, query, offsets_full, start_search_at=after_ctx
    )

    enc = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )
    enc = {k: v.to(hf_device) for k, v in enc.items()}

    with torch.no_grad():
        out = hf_model(
            **enc,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )

    hidden_states = out.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden states.")

    model_type = _infer_attention_model_type(hf_model)
    query_states, key_states, causal_mask, head_dim = _project_qk_last_layer(
        hf_model,
        hidden_states,
        model_type=model_type,
    )

    q_start = q_token_indices[0]
    q_end = q_token_indices[-1] + 1

    query_states = query_states[:, :, q_start:q_end, :]
    causal_mask = causal_mask[:, :, q_start:q_end, :]

    attn_scores = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / np.sqrt(float(head_dim))
    attn_scores = attn_scores + causal_mask
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)

    last_layer_q_to_all = attn_weights[0].mean(dim=0)
    c_idx = torch.tensor(
        ctx_token_indices,
        device=last_layer_q_to_all.device,
        dtype=torch.long,
    )
    q_to_c = last_layer_q_to_all.index_select(1, c_idx)
    scores_ctx = q_to_c.sum(dim=0).detach().float().cpu().numpy().astype(
        np.float32, copy=False
    )

    del (
        out,
        hidden_states,
        query_states,
        key_states,
        causal_mask,
        attn_scores,
        attn_weights,
        last_layer_q_to_all,
        q_to_c,
        c_idx,
        enc,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores_ctx, [(int(s), int(e)) for s, e in ctx_rel_offsets]


def _contextcite_token_scores_for_context(
    *,
    hf_model,
    hf_tok,
    masked_context: str,
    query: str,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    if ContextCiter is None:
        raise ModuleNotFoundError("context_cite is required for context_cite recompute mode.")

    token_partitioner = TokenContextPartitioner(
        context=masked_context,
        tokenizer=hf_tok,
        ablate_mode="blank",
    )

    cc = ContextCiter(
        hf_model,
        hf_tok,
        masked_context,
        query,
        prompt_template=TF_RAG_TEMPLATE_A2T,
        partitioner=token_partitioner,
    )

    scores_cur = np.asarray(cc.get_attributions(), dtype=np.float32)
    cur_offsets = [(int(s), int(e)) for s, e in token_partitioner._spans]

    if len(scores_cur) != len(cur_offsets):
        raise ValueError(
            f"ContextCite scores len={len(scores_cur)} but current ctx spans len={len(cur_offsets)}"
        )

    return scores_cur, cur_offsets


def _at2_token_scores_for_context(
    *,
    hf_model,
    hf_tok,
    masked_context: str,
    query: str,
    score_estimator_path,
    generate_kwargs: dict,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    scores_cur, _gen, sources = get_at2_token_scores(
        full_context=masked_context,
        query=query,
        hf_model=hf_model,
        hf_tok=hf_tok,
        score_estimator_path=score_estimator_path,
        generate_kwargs=generate_kwargs,
    )
    scores_cur = np.asarray(scores_cur, dtype=np.float32)
    _, cur_offsets = tokenize_context_with_offsets(masked_context, hf_tok)
    scores_token = map_at2_scores_to_base_via_sources(
        context=masked_context,
        source_pieces=sources,
        scores=scores_cur,
        base_offsets=cur_offsets,
        max_lookahead=64,
        max_merge_pieces=4,
        whitespace_flex=True,
    )
    return scores_token.astype(np.float32, copy=False), cur_offsets


def _contextcite_scores_mapped_to_base(
    *,
    hf_model,
    hf_tok,
    masked_context: str,
    query: str,
    base_offsets: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Recompute ContextCite attributions on the CURRENT masked context,
    then map them to stable base_offsets.
    """
    if ContextCiter is None:
        raise ModuleNotFoundError("context_cite is required for context_cite recompute mode.")

    token_partitioner = TokenContextPartitioner(
        context=masked_context,
        tokenizer=hf_tok,
        ablate_mode="blank",
    )

    cc = ContextCiter(
        hf_model,
        hf_tok,
        masked_context,
        query,
        prompt_template=TF_RAG_TEMPLATE_A2T,
        partitioner=token_partitioner,
    )

    scores_cur = np.asarray(cc.get_attributions(), dtype=np.float32)

    _, cur_offsets = tokenize_context_with_offsets(masked_context, hf_tok)

    if len(scores_cur) != len(cur_offsets):
        raise ValueError(
            f"ContextCite scores len={len(scores_cur)} but current ctx tokens len={len(cur_offsets)}"
        )

    scores_base = _map_scores_by_char_overlap(base_offsets, cur_offsets, scores_cur)
    return scores_base

def _at2_scores_mapped_to_base(*,hf_model,hf_tok,
    masked_context: str,
    query: str,base_offsets: List[Tuple[int, int]],
    score_estimator_path, generate_kwargs: dict,
) -> np.ndarray:
    """
    Recompute AT2 attributions on the CURRENT masked context,
    then robustly map them back to the stable base_offsets.

    Important:
    - Do NOT retokenize masked_context here and require exact length equality.
    - Use AT2's own returned `sources` as the authority.
    """
    scores_cur, _gen, sources = get_at2_token_scores(
        full_context=masked_context,query=query,
        hf_model=hf_model,hf_tok=hf_tok,
        score_estimator_path=score_estimator_path,
        generate_kwargs=generate_kwargs,
    )

    scores_cur = np.asarray(scores_cur, dtype=np.float32)

    if len(scores_cur) != len(sources):
        raise ValueError(
            f"AT2 scores len={len(scores_cur)} != AT2 sources len={len(sources)}"
        )

    scores_base = map_at2_scores_to_base_via_sources(
        context=masked_context,
        source_pieces=sources,
        scores=scores_cur,
        base_offsets=base_offsets,
        max_lookahead=64,
        max_merge_pieces=4,
        whitespace_flex=True,
    )
    return scores_base


def _write_adaptive_log(
    path: str,
    *,title: str,
    query: str,
    full_context: str,
    base_offsets: List[Tuple[int, int]],
    order: List[int],
    scores_at_pick: List[float],
    masked_stats: List[dict],
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== {title} ===\n")
        f.write(f"Query: {query}\n")
        f.write(f"Total base context tokens considered: {len(base_offsets)}\n")
        f.write(f"Steps executed: {len(order)}\n\n")

        f.write("Adaptive masking order (first 100):\n")
        for rank, idx in enumerate(order[:100]):
            s, e = base_offsets[int(idx)]
            token_text = full_context[s:e].replace("\n", " ")
            f.write(
                f"{rank + 1:03d}. base_pos={int(idx):4d}, span[{s}:{e}]='{token_text}' "
                f"| score_at_pick={scores_at_pick[rank]:.6f}\n"
            )

        f.write("\n--- Per-step stats ---\n\n")
        prev_p_true = None
        for step, idx in enumerate(order):
            s, e = base_offsets[int(idx)]
            token_text = full_context[s:e].replace("\n", " ")
            stats = masked_stats[step]
            p_true = float(stats["p_true"])
            delta_p = None if prev_p_true is None else p_true - prev_p_true
            prev_p_true = p_true

            f.write(f"Step {step + 1:03d}: newly masked token at base_pos={int(idx)}\n")
            f.write(f"  span[{s}:{e}] = '{token_text}'\n")
            f.write(f"  score_at_pick = {scores_at_pick[step]:.6f}\n")
            f.write(
                f"  logP_true  = {stats['logP_true']:.6f}\n"
                f"  logP_false = {stats['logP_false']:.6f}\n"
                f"  log_odds   = {stats['log_odds']:.6f}\n"
                f"  p_true     = {p_true:.6f}\n"
            )

            if delta_p is not None:
                f.write(f"  Δp_true (vs prev) = {delta_p:+.6f}\n")
            f.write("\n")


def _flush_recompute_prompt_chunk_until_flip(
    *,
    hf_model,
    hf_tok,
    hf_device,
    prompt_chunk: List[str],
    batch_size: int,
    p_true_flipping: bool,
    true_variants,
    false_variants,
    compute_probs_file_name: str,
    step_offset: int,
    masked_prompts_acc: List[str],
    masked_stats_acc: List[dict],
    masked_logps_acc: List[float],
):
    """
    Score a chunk of already-created masked prompts and stop early if compute_probs
    finds a flip.

    This preserves the scoring semantics of the old implementation and only changes
    prompt handling: we no longer need to materialize the full prompt trajectory
    before scoring.
    """
    if not prompt_chunk:
        return False, step_offset

    chunk_stats, chunk_logps = compute_probs(
        hf_model,
        hf_tok,
        prompt_chunk,
        hf_device,
        None,
        batch_size=batch_size,
        detect_flip_to_true=p_true_flipping,
        true_variants=true_variants,
        false_variants=false_variants,
        masked_context_list=None,
        return_full_logp=True,
        file_name=compute_probs_file_name,
        save_file=False,
        stop_on_flip=True,
    )

    _rewrite_chunked_step_metadata(chunk_stats, step_offset=step_offset)

    effective_steps = len(chunk_stats)
    masked_prompts_acc.extend(prompt_chunk[:effective_steps])
    masked_stats_acc.extend(chunk_stats)
    if chunk_logps is not None:
        masked_logps_acc.extend(chunk_logps[:effective_steps])

    step_offset += effective_steps
    stopped_early = effective_steps < len(prompt_chunk)
    return stopped_early, step_offset

def _attention_flow_scores_mapped_to_base(
    *,
    hf_model,
    hf_tok,
    hf_device,
    prompt_template: ChatPromptTemplate,
    masked_context: str,
    query: str,
    base_offsets: List[Tuple[int, int]],
    topk_per_row: int = 8,
    seq_len_limit: int = 384,
) -> np.ndarray:
    """
    Recompute attention-flow scores on the CURRENT masked prompt,
    then map them back to the stable base_offsets.
    """
    full_prompt = prompt_template.format(context=masked_context, question=query)

    enc_full = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )
    offsets_full = enc_full["offset_mapping"]
    if hasattr(offsets_full, "tolist"):
        offsets_full = offsets_full.tolist()

    ctx_token_indices, ctx_rel_offsets, after_ctx = _find_token_indices_by_substring(
        full_prompt, masked_context, offsets_full, start_search_at=0
    )
    q_token_indices, _, _ = _find_token_indices_by_substring(
        full_prompt, query, offsets_full, start_search_at=after_ctx
    )

    enc = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )
    enc = {k: v.to(hf_device) for k, v in enc.items()}

    # same backend logic you used for the new attention_flow method:
    mats, _attn_mode = _get_augmented_attention_mats_auto(
        hf_model=hf_model,
        hf_tok=hf_tok,
        hf_device=hf_device,
        full_prompt=full_prompt,
    )
    mode=None
    seq_len = mats[0].shape[0]
    if nx is None or seq_len > seq_len_limit:
        # same fallback behavior as your method:
        joint = mats[0]
        for l in range(1, len(mats)):
            joint = mats[l] @ joint

        q_scores = joint[np.asarray(q_token_indices, dtype=np.int64)]
        scores_ctx = q_scores[:, np.asarray(ctx_token_indices, dtype=np.int64)].mean(axis=0)
        mode="rollout_fallback"
    else:
        mode="max_flow"
        G = _build_sparse_flow_graph(
            mats,
            source_nodes=[int(q) for q in q_token_indices],
            topk_per_row=topk_per_row,
        )

        super_source = ("src", -1)
        scores_ctx = np.zeros(len(ctx_token_indices), dtype=np.float32)

        for out_i, ctx_tok in enumerate(ctx_token_indices):
            sink = (-1, int(ctx_tok))
            H = G.copy()
            super_sink = ("sink", -1)
            H.add_edge(sink, super_sink, capacity=1.0)
            try:
                flow_val, _ = nx.maximum_flow(H, super_source, super_sink)
            except Exception:
                flow_val = 0.0
            scores_ctx[out_i] = float(flow_val)

    scores_base = _map_scores_by_char_overlap(base_offsets, ctx_rel_offsets, scores_ctx)
    return scores_base.astype(np.float32, copy=False) , mode

def mask_by_order_recompute(
    *,full_context: str,query: str,
    hf_model,hf_tok,hf_device,max_steps: Optional[int] = 5000,
    batch_size: int = 2,score_mode: str = "attention",
    compute_probs_file_name: str = "attention_recompute_output_compute_probs.txt",
    log_path: Optional[str] = "greedy_token_masking_attention_recompute.txt",
    score_estimator_path=None,generate_kwargs=None,p_true_flipping: bool = False,
    true_variants=None,false_variants=None,masking_iteration=1,
    stop_scores_abs: Optional[float] = None,
    save_logs:bool=True,stop_on_flip:bool=False,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 25,
    intervention_mode: InterventionMode | str = InterventionMode.MASK_TOKEN,
    replacement_map: Optional[Mapping[Any, str]] = None,
    replacement_resolver: Optional[ReplacementResolver] = None,
    model_id: Optional[str] = None,
):
    """
    Adaptive greedy masking:
    At step t:
      1) compute scores on the CURRENT masked prompt
      2) choose highest-score UNMASKED base token
      3) add it to masked set
      4) record the new masked prompt for later scoring
    """

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)
    if score_mode in ("context_cite", "at2"):
        prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE_A2T)

    mode = coerce_intervention_mode(intervention_mode)
    _, base_offsets = tokenize_context_with_offsets(full_context, hf_tok)
    pieces = None
    word_units = None
    original_word_offsets: Optional[List[Tuple[int, int]]] = None
    candidate_word_ids: Optional[List[int]] = None
    excluded_units: List[Dict[str, Any]] = []
    candidate_filter: Dict[str, Any] = {
        "replacement_semex_filter_enabled": False,
        "excluded_candidates": 0,
    }

    if mode == InterventionMode.MASK_TOKEN:
        n = len(base_offsets)
    else:
        pieces, word_units = split_context_to_word_units(full_context)
        original_word_offsets = [(int(u.start), int(u.end)) for u in word_units]
        candidate_word_ids = [int(u.word) for u in word_units]
        n = len(word_units)

        if mode in {
            InterventionMode.REPLACEMENT_NEUTRAL_WORD,
            InterventionMode.REPLACEMENT_ANTONYM_WORD,
        }:
            semex_filter_enabled = bool(getattr(replacement_resolver, "semex_filter_enabled", True))
            candidate_filter["replacement_semex_filter_enabled"] = semex_filter_enabled
            if semex_filter_enabled:
                candidate_word_ids, _scores, semex_excluded, semex_meta = filter_replacement_order_semex(
                    context=full_context,
                    word_units=word_units,
                    ordered_word_ids=candidate_word_ids,
                    pick_scores=None,
                    spacy_model=str(getattr(replacement_resolver, "semex_spacy_model", "en_core_web_sm")),
                )
                excluded_units.extend(semex_excluded)
                candidate_filter.update(semex_meta)

            replacement_map = build_replacement_map_for_order(
                context=full_context,
                query=query,
                word_units=word_units,
                ordered_word_ids=candidate_word_ids,
                mode=mode,
                replacement_map=replacement_map,
                resolver=replacement_resolver,
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                model_id=model_id,
            )

        candidate_word_ids, _scores, availability_excluded = _filter_word_order_by_available_intervention(
            word_units=word_units,
            order=candidate_word_ids,
            pick_scores=None,
            mode=mode,
            replacement_map=replacement_map,
            return_exclusions=True,
        )
        excluded_units.extend(availability_excluded)
        candidate_filter["excluded_candidates"] = int(len(excluded_units))
        candidate_filter["available_candidates"] = int(len(candidate_word_ids))

    if max_steps is None:
        max_steps = n
    max_steps = int(min(max_steps, n))

    masking_iteration = max(1, int(masking_iteration))

    masked_flags = np.zeros(n, dtype=bool)
    masked_spans: List[Tuple[int, int]] = []

    order: List[int] = []
    scores_at_pick: List[float] = []

    masked_prompts: List[str] = []
    masked_context_list: List[str] = []

    stream_chunk_size = 32
    pending_prompts: List[str] = []
    streamed_prompts: List[str] = []
    streamed_stats: List[dict] = []
    streamed_logps: List[float] = []
    step_offset = 0

    keep_running = True

    def save_recompute_checkpoint(status: str, stage: str) -> None:
        if not checkpoint_path:
            return

        _write_masking_checkpoint(checkpoint_path,
            {"status": status,"stage": stage,"score_mode": score_mode,
                "intervention_mode": mode.name,
                "query": query,"p_true_flipping": bool(p_true_flipping),
                "masking_iteration": int(masking_iteration),
                "max_steps": int(max_steps),"num_base_tokens": int(n),"steps_selected": int(len(order)),
                "steps_scored": int(len(streamed_stats) if stop_on_flip else 0),
                "pending_prompts": int(len(pending_prompts)) if stop_on_flip else 0,
                "order": [int(x) for x in order],
                "scores_at_pick": [float(x) for x in scores_at_pick],
                "excluded_units": excluded_units,
                "candidate_filter": candidate_filter,
                "masked_stats": streamed_stats if stop_on_flip else [],
                "masked_logps": streamed_logps if stop_on_flip else [],
            },
        )

    save_recompute_checkpoint("started", "recompute_start")

    while len(order) < max_steps and keep_running:
        if mode == InterventionMode.MASK_TOKEN:
            cur_context = mask_context_spans_same_length(full_context, masked_spans)

            if score_mode == "attention":
                scores_base = _attention_scores_mapped_to_base(
                    hf_model=hf_model,
                    hf_tok=hf_tok,
                    hf_device=hf_device,
                    prompt_template=prompt_template,
                    masked_context=cur_context,
                    query=query,
                    base_offsets=base_offsets,
                )
            elif score_mode=="attention_flow":
                scores_base , attention_flow_mode = _attention_flow_scores_mapped_to_base(
                    hf_model=hf_model,
                    hf_tok=hf_tok,
                    hf_device=hf_device,
                    prompt_template=prompt_template,
                    masked_context=cur_context,
                    query=query,
                    base_offsets=base_offsets,
                )

            elif score_mode == "context_cite":
                scores_base = _contextcite_scores_mapped_to_base(
                    hf_model=hf_model,
                    hf_tok=hf_tok,
                    masked_context=cur_context,
                    query=query,
                    base_offsets=base_offsets,
                )
            elif score_mode == "at2":
                scores_base = _at2_scores_mapped_to_base(
                    hf_model=hf_model,
                    hf_tok=hf_tok,
                    masked_context=cur_context,
                    query=query,
                    base_offsets=base_offsets,
                    score_estimator_path=score_estimator_path,
                    generate_kwargs=generate_kwargs or {
                        "max_new_tokens": 128,
                        "do_sample": False,
                    },
                )
            else:
                raise ValueError(
                    f"Unknown score_mode={score_mode}. Use 'attention', 'attention_flow', 'context_cite', or 'at2'."
                )
        else:
            if pieces is None or word_units is None:
                raise RuntimeError("Word intervention mode was selected without word units.")

            cur_context, current_word_offsets = _build_word_intervention_context_and_offsets(
                pieces=pieces,
                word_units=word_units,
                selected_word_ids=order,
                mode=mode,
                replacement_map=replacement_map,
            )

            if score_mode == "attention":
                token_scores, token_offsets = _attention_token_scores_for_context(
                    hf_model=hf_model,
                    hf_tok=hf_tok,
                    hf_device=hf_device,
                    prompt_template=prompt_template,
                    masked_context=cur_context,
                    query=query,
                )
                scores_base = _aggregate_token_scores_to_spans(
                    target_offsets=current_word_offsets,
                    token_offsets=token_offsets,
                    token_scores=token_scores,
                    reduction="sum",
                )
            elif score_mode == "attention_flow":
                scores_base, attention_flow_mode = _attention_flow_scores_mapped_to_base(
                    hf_model=hf_model,
                    hf_tok=hf_tok,
                    hf_device=hf_device,
                    prompt_template=prompt_template,
                    masked_context=cur_context,
                    query=query,
                    base_offsets=current_word_offsets,
                )
            elif score_mode == "context_cite":
                token_scores, token_offsets = _contextcite_token_scores_for_context(
                    hf_model=hf_model,
                    hf_tok=hf_tok,
                    masked_context=cur_context,
                    query=query,
                )
                scores_base = _aggregate_token_scores_to_spans(
                    target_offsets=current_word_offsets,
                    token_offsets=token_offsets,
                    token_scores=token_scores,
                    reduction="sum",
                )
            elif score_mode == "at2":
                token_scores, token_offsets = _at2_token_scores_for_context(
                    hf_model=hf_model,
                    hf_tok=hf_tok,
                    masked_context=cur_context,
                    query=query,
                    score_estimator_path=score_estimator_path,
                    generate_kwargs=generate_kwargs or {
                        "max_new_tokens": 128,
                        "do_sample": False,
                    },
                )
                scores_base = _aggregate_token_scores_to_spans(
                    target_offsets=current_word_offsets,
                    token_offsets=token_offsets,
                    token_scores=token_scores,
                    reduction="sum",
                )
            else:
                raise ValueError(
                    f"Unknown score_mode={score_mode}. Use 'attention', 'attention_flow', 'context_cite', or 'at2'."
                )

        scores_base = scores_base.astype(np.float32, copy=False)
        scores_base[masked_flags] = -np.inf
        if candidate_word_ids is not None:
            unavailable = np.ones(n, dtype=bool)
            unavailable[np.asarray(candidate_word_ids, dtype=np.int64)] = False
            scores_base[unavailable] = -np.inf

        remaining_budget = max_steps - len(order)
        if remaining_budget <= 0:
            break

        top_k = min(masking_iteration, remaining_budget, n)
        if top_k <= 0:
            break

        top_idx = np.argpartition(scores_base, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores_base[top_idx])[::-1]]

        for index in top_idx:
            pick = int(index)
            pick_score = float(scores_base[pick])

            if not np.isfinite(pick_score):
                keep_running = False
                break

            if stop_scores_abs is not None and pick_score <= stop_scores_abs:
                keep_running = False
                break

            if masked_flags[pick]:
                continue

            masked_flags[pick] = True
            order.append(pick)
            scores_at_pick.append(pick_score)
            if mode == InterventionMode.MASK_TOKEN:
                masked_spans.append(base_offsets[pick])
                new_context = mask_context_spans_same_length(full_context, masked_spans)
            else:
                if pieces is None or word_units is None:
                    raise RuntimeError("Word intervention mode was selected without word units.")
                new_context, _current_word_offsets = _build_word_intervention_context_and_offsets(
                    pieces=pieces,
                    word_units=word_units,
                    selected_word_ids=order,
                    mode=mode,
                    replacement_map=replacement_map,
                )

            if score_mode in ("context_cite", "at2"):
                new_prompt = prompt_template.format(context=new_context, query=query)
            else:
                new_prompt = prompt_template.format(context=new_context, question=query)

            if stop_on_flip:
                pending_prompts.append(new_prompt)
                if len(pending_prompts) >= stream_chunk_size:
                    stopped_early, step_offset = _flush_recompute_prompt_chunk_until_flip(
                        hf_model=hf_model,
                        hf_tok=hf_tok,
                        hf_device=hf_device,
                        prompt_chunk=pending_prompts,
                        batch_size=batch_size,
                        p_true_flipping=p_true_flipping,
                        true_variants=true_variants,
                        false_variants=false_variants,
                        compute_probs_file_name=compute_probs_file_name,
                        step_offset=step_offset,
                        masked_prompts_acc=streamed_prompts,
                        masked_stats_acc=streamed_stats,
                        masked_logps_acc=streamed_logps,
                    )
                    pending_prompts = []

                    save_recompute_checkpoint(
                        "stopped_early" if stopped_early else "running",
                        "recompute_scored_chunk",
                    )

                    if stopped_early:
                        keep_running = False
                        break
            else:
                masked_context_list.append(new_context)
                masked_prompts.append(new_prompt)

        if len(order) > 0 and (len(order) == 1 or len(order) % 25 == 0):
            print(
                f"[adaptive] masked={len(order)}/{max_steps} "
                f"last_pick={order[-1]} last_score={scores_at_pick[-1]:.6f}"
            )
            if checkpoint_path and len(order) % max(1, checkpoint_every) == 0:
                save_recompute_checkpoint("running", "recompute_order_progress")

    os.makedirs(os.path.dirname(compute_probs_file_name) or ".", exist_ok=True)

    if stop_on_flip:
        if pending_prompts and keep_running:
            _stopped_early, step_offset = _flush_recompute_prompt_chunk_until_flip(
                hf_model=hf_model,
                hf_tok=hf_tok,
                hf_device=hf_device,
                prompt_chunk=pending_prompts,
                batch_size=batch_size,
                p_true_flipping=p_true_flipping,
                true_variants=true_variants,
                false_variants=false_variants,
                compute_probs_file_name=compute_probs_file_name,
                step_offset=step_offset,
                masked_prompts_acc=streamed_prompts,
                masked_stats_acc=streamed_stats,
                masked_logps_acc=streamed_logps,
            )

        masked_stats, masked_logps = streamed_stats, streamed_logps
        effective_steps = len(masked_stats)
        order = order[:effective_steps]
        scores_at_pick = scores_at_pick[:effective_steps]

        if save_logs:
            _write_compute_probs_flip_log(
                compute_probs_file_name,
                masked_prompts=streamed_prompts,
                masked_stats=masked_stats,
            )
    else:
        if masked_prompts:
            masked_stats, masked_logps = compute_probs(
                hf_model,
                hf_tok,
                masked_prompts,
                hf_device,
                None,
                batch_size=batch_size,
                detect_flip_to_true=p_true_flipping,
                true_variants=true_variants,
                false_variants=false_variants,
                masked_context_list=masked_context_list,
                return_full_logp=True,
                file_name=compute_probs_file_name,
                save_file=save_logs,
                stop_on_flip=False
            )
        else:
            masked_stats, masked_logps = [], []

    if save_logs and log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_offsets = base_offsets
        if mode != InterventionMode.MASK_TOKEN:
            log_offsets = original_word_offsets or []
        _write_adaptive_log(
            log_path,
            title=f"Adaptive greedy masking (recompute each {masking_iteration} step/s, {mode.name})",
            query=query,
            full_context=full_context,
            base_offsets=log_offsets,
            order=order,
            scores_at_pick=scores_at_pick,
            masked_stats=masked_stats,
        )

    if checkpoint_path:
        _write_masking_checkpoint(
            checkpoint_path,
            {
                "status": "completed",
                "stage": "recompute_completed",
                "score_mode": score_mode,
                "intervention_mode": mode.name,
                "query": query,
                "p_true_flipping": bool(p_true_flipping),
                "masking_iteration": int(masking_iteration),
                "max_steps": int(max_steps),
                "num_base_tokens": int(n),
                "steps_selected": int(len(order)),
                "steps_scored": int(len(masked_stats)),
                "order": [int(x) for x in order],
                "scores_at_pick": [float(x) for x in scores_at_pick],
                "excluded_units": excluded_units,
                "candidate_filter": candidate_filter,
                "masked_stats": masked_stats,
                "masked_logps": masked_logps,
            },
        )

    return masked_stats, masked_logps, order, scores_at_pick


