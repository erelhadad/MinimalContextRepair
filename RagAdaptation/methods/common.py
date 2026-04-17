from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys
import numpy as np
import torch
import RagAdaptation.core.model_config as model_config


_THIS_FILE = Path(__file__).resolve()
_PKG_DIR = _THIS_FILE.parents[1]  # .../RagAdaptation
_PROJECT_ROOT = _PKG_DIR.parent  # .../RAG_EXP
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from RagAdaptation.compute_probs_updated import compute_probs
from RagAdaptation.core.prompting import ChatPromptTemplate
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE, TF_RAG_TEMPLATE_A2T


def find_token_indices_by_substring(
    full_text: str,
    substring: str,
    offsets_mapping: Sequence[Tuple[int, int]],
    start_search_at: int = 0,
):
    begin = full_text.find(substring, start_search_at)
    if begin < 0:
        raise ValueError(
            "Could not locate substring inside the prompt. Check template / text uniqueness."
        )

    end = begin + len(substring)

    tok_indices: List[int] = []
    rel_offsets: List[Tuple[int, int]] = []
    for i, (s, e) in enumerate(offsets_mapping):
        if e <= s:
            continue
        if s >= begin and e <= end:
            tok_indices.append(i)
            rel_offsets.append((s - begin, e - begin))

    return tok_indices, rel_offsets, end


def mask_context_spans_same_length(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    spans2 = [
        (int(s), int(e))
        for (s, e) in spans
        if s is not None and e is not None and 0 <= s < e <= len(text)
    ]
    if not spans2:
        return text

    spans2.sort(key=lambda x: x[0])
    merged: List[List[int]] = []
    for s, e in spans2:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    out = text
    for s, e in reversed(merged):
        out = out[:s] + (" " * (e - s)) + out[e:]
    return out


def _get_mask_prompt_template(change_template_contextCite: bool):
    if change_template_contextCite:
        return ChatPromptTemplate.from_template(TF_RAG_TEMPLATE_A2T)
    return ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)



def iter_masked_prompts_iterative_chunks(
    document: str,
    query: str,
    offsets: List[Tuple[int, int]],
    *,
    k: int = 1,
    change_template_contextCite: bool = False,
    chunk_size: int = 32,
):
    """
    Yield masked prompts in chunks while preserving the exact cumulative masking logic
    of the original implementation.

    Semantics are unchanged:
      step i masks offsets[0], ..., offsets[i]

    The only difference is execution style: prompts are built lazily in small chunks
    instead of materializing the full masking trajectory up front.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    prompt_template = _get_mask_prompt_template(change_template_contextCite)
    masked_spans: List[Tuple[int, int]] = []
    prompt_chunk: List[str] = []
    context_chunk: List[str] = []

    for i in range(len(offsets)):
        masked_spans.extend(offsets[i : i + k])
        masked_context = mask_context_spans_same_length(document, masked_spans)
        context_chunk.append(masked_context)

        if change_template_contextCite:
            prompt_chunk.append(prompt_template.format(context=masked_context, query=query))
        else:
            prompt_chunk.append(prompt_template.format(context=masked_context, question=query))

        if len(prompt_chunk) >= chunk_size:
            yield prompt_chunk, context_chunk
            prompt_chunk = []
            context_chunk = []

    if prompt_chunk:
        yield prompt_chunk, context_chunk



def create_masked_prompts_iterative(
    document: str,
    query: str,
    offsets: List[Tuple[int, int]],
    k: int = 1,
    change_template_contextCite: bool = False,
):
    """
    Compatibility wrapper that preserves the old return value by materializing the
    chunked iterator.
    """
    batch: List[str] = []
    masked_context_list: List[str] = []
    for prompt_chunk, context_chunk in iter_masked_prompts_iterative_chunks(
        document,
        query,
        offsets,
        k=k,
        change_template_contextCite=change_template_contextCite,
        chunk_size=max(1, len(offsets) or 1),
    ):
        batch.extend(prompt_chunk)
        masked_context_list.extend(context_chunk)

    return batch, masked_context_list



def _infer_attention_model_type(hf_model) -> str:
    name = str(getattr(hf_model, "name_or_path", "")).lower()
    if "mistral" in name:
        return "mistral"
    if "llama" in name:
        return "llama"
    if "phi-3" in name or "phi3" in name or "phi_3" in name:
        return "phi3"
    if "qwen" in name:
        return "qwen2"
    if "gemma" in name:
        return "gemma"
    raise ValueError(f"Unsupported model for attention extraction: {hf_model.name_or_path}")


def _get_transformers_attention_helpers(model_type: str):
    import transformers.models

    model_module_name = {
        "llama": "llama",
        "mistral": "mistral",
        "phi3": "phi3",
        "qwen2": "qwen2",
        "gemma": "gemma2",
    }[model_type]

    model_module = getattr(transformers.models, model_module_name)
    modeling_module = getattr(model_module, f"modeling_{model_module_name}")
    return modeling_module.apply_rotary_pos_emb, modeling_module.repeat_kv


def _get_llm_core_and_layers(hf_model):
    core = getattr(hf_model, "model", None)
    if core is None or not hasattr(core, "layers"):
        raise ValueError("Expected a decoder-only HF model with `model.layers`.")
    return core, core.layers


def _build_causal_mask_for_hidden_states(hf_model, hidden_states):
    """
    Build the same style of causal mask used by the AT2 attention extraction path,
    but only once for the current prompt length.
    """
    input_embeds = hidden_states[0]
    _, seq_len, _ = input_embeds.shape
    device = input_embeds.device
    dtype = getattr(hf_model, "dtype", input_embeds.dtype)

    position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)

    attention_mask = torch.ones(seq_len, seq_len + 1, device=device, dtype=dtype)
    attention_mask = torch.triu(attention_mask, diagonal=1)
    attention_mask *= torch.finfo(dtype).min
    attention_mask = attention_mask[None, None]  # [1, 1, seq, seq+1]
    return position_ids, attention_mask


def _project_qk_last_layer(
    hf_model,
    hidden_states,
    *,
    model_type: str,
):
    """
    Reconstruct Q/K for the LAST attention layer from hidden states, without asking
    the full model to materialize output_attentions=True.

    This follows the AT2 idea: compute only what we need from hidden states instead of
    storing the full attention tensors for every layer.
    """
    core, layers = _get_llm_core_and_layers(hf_model)
    layer = layers[-1]
    self_attn = layer.self_attn

    layer_input = hidden_states[-2]
    layer_input = layer.input_layernorm(layer_input)

    bsz, q_len, _ = layer_input.size()
    cfg = core.config
    num_attention_heads = cfg.num_attention_heads
    num_key_value_heads = cfg.num_key_value_heads
    head_dim = self_attn.head_dim

    if model_type in ("llama", "mistral", "qwen2", "gemma"):
        query_states = self_attn.q_proj(layer_input)
        key_states = self_attn.k_proj(layer_input)
    elif model_type == "phi3":
        qkv = self_attn.qkv_proj(layer_input)
        query_pos = num_attention_heads * head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + num_key_value_heads * head_dim]
    else:
        raise ValueError(f"Unsupported model_type={model_type!r}")

    query_states = query_states.view(bsz, q_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    if model_type == "gemma":
        if hasattr(self_attn, "q_norm") and self_attn.q_norm is not None:
            query_states = self_attn.q_norm(query_states)
        if hasattr(self_attn, "k_norm") and self_attn.k_norm is not None:
            key_states = self_attn.k_norm(key_states)

    position_ids, attention_mask = _build_causal_mask_for_hidden_states(hf_model, hidden_states)

    if hasattr(core, "rotary_emb_local") and getattr(self_attn, "is_sliding", False):
        position_embeddings = core.rotary_emb_local(layer_input, position_ids)
    else:
        position_embeddings = core.rotary_emb(layer_input, position_ids)

    cos, sin = position_embeddings
    apply_rotary_pos_emb, repeat_kv = _get_transformers_attention_helpers(model_type)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    return query_states, key_states, causal_mask, head_dim


def get_attention_scores(
    hf_model,
    hf_tok,
    hf_device,
    *,
    full_prompt: str,
    full_context: str,
    query: str,
):
    """
    Compute context-token attention scores without materializing output_attentions=True.

    Scoring rule stays the same as before:
      - use the LAST layer
      - average across heads
      - take question -> context attention
      - sum over question tokens

    The difference is only the extraction path:
    we reconstruct the relevant last-layer attention block from hidden states, in the
    style used by AT2, instead of storing the full attention tensor for the whole model.
    """
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

    ctx_token_indices, _ctx_rel_offsets, after_ctx = find_token_indices_by_substring(
        full_prompt,
        full_context,
        offsets_full,
        start_search_at=0,
    )
    q_token_indices, _, _ = find_token_indices_by_substring(
        full_prompt,
        query,
        offsets_full,
        start_search_at=after_ctx,
    )

    if not ctx_token_indices:
        raise ValueError("Could not find any context tokens inside the full prompt.")
    if not q_token_indices:
        raise ValueError("Could not find any query tokens inside the full prompt.")

    enc = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
        padding=False,
    )
    enc = {k: v.to(hf_device) for k, v in enc.items()}

    with torch.inference_mode():
        out = hf_model(
            **enc,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
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

    attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(float(head_dim))
    attn_scores = attn_scores + causal_mask
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)

    last_layer_q_to_all = attn_weights[0].mean(dim=0)  # [|Q|, seq]
    c_idx = torch.tensor(ctx_token_indices, device=last_layer_q_to_all.device, dtype=torch.long)
    q_to_c = last_layer_q_to_all.index_select(1, c_idx)
    scores_vec = q_to_c.sum(dim=0).detach().float().cpu().numpy().astype(np.float32, copy=False)

    del out, hidden_states, query_states, key_states, causal_mask, attn_scores, attn_weights, last_layer_q_to_all, q_to_c, c_idx, enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores_vec

def _label_from_stats(st: Dict[str, Any]) -> str:
    return "true" if float(st["p_true"]) > 0.5 else "false"


def _first_flip_idx(
    baseline_stats: Dict[str, Any],
    masked_stats: List[Dict[str, Any]],
) -> Optional[int]:
    base_lab = _label_from_stats(baseline_stats)
    for i, st in enumerate(masked_stats):
        if _label_from_stats(st) != base_lab:
            return i
    return None


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _rewrite_chunked_step_metadata(masked_stats: List[Dict[str, Any]], *, step_offset: int) -> None:
    """
    Adjust chunk-local step indices returned by compute_probs to global masking steps.
    """
    for local_i, st in enumerate(masked_stats):
        global_step = step_offset + local_i + 1
        st["step_index"] = global_step
        if st.get("is_flipped"):
            st["first_flip_index"] = global_step



def _write_compute_probs_flip_log(
    file_name: str,
    *,
    masked_prompts: Sequence[str],
    masked_stats: Sequence[Dict[str, Any]],
) -> None:
    """
    Recreate the lightweight compute_probs flip log for the chunked execution path.

    The original compute_probs implementation only writes a short file when a flip is
    found, so reproducing that behavior is enough to stay compatible with current usage.
    """
    if not file_name:
        return

    flip_idx = None
    for i, st in enumerate(masked_stats):
        if st.get("is_flipped"):
            flip_idx = i
            break

    if flip_idx is None:
        return

    p = Path(file_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    st = masked_stats[flip_idx]
    step0 = int(st.get("step_index", flip_idx + 1)) - 1

    with p.open("w", encoding="utf-8") as f:
        f.write(
            f"[{step0}] logP_true={float(st['logP_true']):.4f} "
            f"logP_false={float(st['logP_false']):.4f} "
            f"log_odds={float(st['log_odds']):.4f} "
            f"p_true={float(st['p_true']):.6f}\n\n"
        )
        f.write(f"After {step0} iterations we had converted\n")
        f.write(f"The prompt:\n{masked_prompts[flip_idx]}\n")



def _compute_probs_streaming_until_flip(
    *,
    document: str,
    query: str,
    ordered_offsets: List[Tuple[int, int]],
    change_template_contextCite: bool,
    hf_model,
    hf_tok,
    hf_device,
    true_variants,
    false_variants,
    compute_probs_file_name: str,
    p_true_flipping: bool,
    save_logs: bool,
    chunk_size: int = 32,
):
    """
    Stream masked prompts chunk-by-chunk until compute_probs finds a flip.

    Important semantic note:
      - This path is used only for stop_on_flip=True.
      - The masking logic is identical to the original implementation.
      - We change only prompt handling: lazy chunked creation instead of eager full-list
        materialization.
    """
    masked_prompts_acc: List[str] = []
    masked_contexts_acc: List[str] = []
    masked_stats_acc: List[Dict[str, Any]] = []
    masked_logps_acc: List[float] = []

    step_offset = 0
    stopped_early = False

    for prompt_chunk, context_chunk in iter_masked_prompts_iterative_chunks(
        document,
        query,
        ordered_offsets,
        change_template_contextCite=change_template_contextCite,
        chunk_size=chunk_size,
    ):
        chunk_stats, chunk_logps = compute_probs(
            hf_model,
            hf_tok,
            prompt_chunk,
            hf_device,
            None,
            batch_size=2,
            return_full_logp=True,
            file_name=compute_probs_file_name,
            detect_flip_to_true=p_true_flipping,
            true_variants=true_variants,
            false_variants=false_variants,
            save_file=False,
            stop_on_flip=True,
        )

        _rewrite_chunked_step_metadata(chunk_stats, step_offset=step_offset)

        effective_steps = len(chunk_stats)
        masked_prompts_acc.extend(prompt_chunk[:effective_steps])
        masked_contexts_acc.extend(context_chunk[:effective_steps])
        masked_stats_acc.extend(chunk_stats)
        if chunk_logps is not None:
            masked_logps_acc.extend(chunk_logps[:effective_steps])

        step_offset += effective_steps

        if effective_steps < len(prompt_chunk):
            stopped_early = True
            break

    if save_logs:
        _write_compute_probs_flip_log(
            compute_probs_file_name,
            masked_prompts=masked_prompts_acc,
            masked_stats=masked_stats_acc,
        )

    return masked_prompts_acc, masked_contexts_acc, masked_stats_acc, masked_logps_acc, stopped_early



def dump_masked_prompts_json(
    out_path: str,
    *,
    query: str,
    baseline_prompt: str,
    baseline_stats: Dict[str, Any],
    masked_prompts: Sequence[str],
    masked_stats: Sequence[Dict[str, Any]],
    masked_context_list: Optional[Sequence[str]] = None,
    order: Optional[Sequence[int]] = None,
    scores_at_pick: Optional[Sequence[float]] = None,
    policy: str = "flip",
    window: int = 1,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n_steps = min(len(masked_prompts), len(masked_stats))
    if masked_context_list is not None:
        n_steps = min(n_steps, len(masked_context_list))
    if order is not None:
        n_steps = min(n_steps, len(order))
    if scores_at_pick is not None:
        n_steps = min(n_steps, len(scores_at_pick))

    masked_prompts = masked_prompts[:n_steps]
    masked_stats = masked_stats[:n_steps]
    if masked_context_list is not None:
        masked_context_list = masked_context_list[:n_steps]
    if order is not None:
        order = order[:n_steps]
    if scores_at_pick is not None:
        scores_at_pick = scores_at_pick[:n_steps]

    flip_idx = _first_flip_idx(baseline_stats, list(masked_stats))
    idxs: List[int] = []

    if policy == "all":
        idxs = list(range(n_steps))
    else:
        if flip_idx is None:
            idxs = [0] if n_steps > 0 else []
        else:
            lo = max(0, flip_idx - window)
            hi = min(n_steps - 1, flip_idx + window)
            idxs = sorted(set([0] + list(range(lo, hi + 1))))

    masked_entries = []
    for i in idxs:
        ent = {"step": int(i + 1), "prompt": masked_prompts[i], "stats": masked_stats[i]}
        if masked_context_list is not None:
            ent["masked_context"] = masked_context_list[i]
        if order is not None:
            ent["newly_masked_base_pos"] = int(order[i])
        if scores_at_pick is not None:
            ent["score_at_pick"] = float(scores_at_pick[i])
        masked_entries.append(ent)

    payload = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "query": query,
        "baseline": {
            "prompt": baseline_prompt,
            "stats": baseline_stats,
        },
        "masked": masked_entries,
        "flip": {
            "found": flip_idx is not None,
            "flip_step": None if flip_idx is None else int(flip_idx + 1),
            "baseline_label": _label_from_stats(baseline_stats),
            "flip_label": None if flip_idx is None else _label_from_stats(masked_stats[flip_idx]),
        },
        "integrity": {
            "baseline_prompt_sha1": _sha1(baseline_prompt),
            "n_masked_total": int(n_steps),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def mask_by_order(
    full_context: str,
    query: str,
    model_con: model_config.ModelConfig = None,
    *,
    scores: Optional[Sequence[torch.Tensor]] = None,
    rng: Optional[np.random.Generator] = None,
    compute_probs_file_name: str = "output_compute_probs.txt",
    p_true_flipping=False,
    dump_json_path: Optional[str] = None,
    dump_policy: str = "flip",
    dump_window: int = 1,
    source_offsets: Optional[List[Tuple[int, int]]] = None,
    force_class_prompt: Optional[bool] = None,
    baseline_stats: Optional[Dict[str, Any]] = None,
    stop_scores_relative: Optional[float] = 0,
    save_logs:bool=True,
    stop_on_flip:bool =False,
):
    hf_model, hf_tok, hf_device = model_con.load()
    true_variants = model_con.get_true_variants()
    false_variants = model_con.get_false_variants()

    full_prompt = model_con.format_prompt(
        question=query,
        context=full_context,
        context_cite_at2_formating=False,
    )

    enc_full = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        padding=False,
    )

    offsets_full = enc_full["offset_mapping"]
    if hasattr(offsets_full, "tolist"):
        offsets_full = offsets_full.tolist()

    ctx_token_indices, ctx_rel_offsets_prompt, after_ctx = find_token_indices_by_substring(
        full_prompt, full_context, offsets_full, start_search_at=0
    )

    ctx_rel_offsets = ctx_rel_offsets_prompt
    scores_vec: Optional[np.ndarray] = None
    change_template_contextCite = False

    if force_class_prompt is not None:
        change_template_contextCite = bool(force_class_prompt)

    if scores is not None:
        if isinstance(scores, (list, tuple)) and len(scores) > 0 and torch.is_tensor(scores[0]):
            q_token_indices, _, _ = find_token_indices_by_substring(
                full_prompt, query, offsets_full, start_search_at=after_ctx
            )

            last = scores[-1]
            last_avg = last[0].mean(dim=0)
            sub = last_avg[np.ix_(q_token_indices, ctx_token_indices)]
            #changed to sum
            scores_vec = sub.sum(axis=0).detach().float().cpu().numpy()
            order = np.argsort(scores_vec)[::-1]
            change_template_contextCite = False
        else:
            if torch.is_tensor(scores):
                scores_vec = scores.detach().float().cpu().numpy()
            else:
                scores_vec = np.asarray(scores, dtype=np.float32)

            if source_offsets is not None:
                ctx_rel_offsets = source_offsets

            if len(scores_vec) != len(ctx_rel_offsets):
                raise ValueError(
                    f"Provided non-attention scores length {len(scores_vec)} != number of masking spans {len(ctx_rel_offsets)}"
                )

            order = np.argsort(scores_vec)[::-1]
            if force_class_prompt is None:
                change_template_contextCite = True
            else:
                change_template_contextCite = bool(force_class_prompt)
    else:
        if rng is None:
            rng = np.random.default_rng()
        order = rng.permutation(len(ctx_rel_offsets))
        scores_vec = None

    if scores_vec is not None:
        max_val = float(np.max(scores_vec))
    else:
        max_val = None

    if max_val is not None and stop_scores_relative is not None:
        threshold = max_val * stop_scores_relative
        order = [i for i in order if scores_vec[i] >= threshold]

    ordered_offsets = [ctx_rel_offsets[i] for i in order]

    if stop_on_flip:
        masked_prompts, masked_context_list, masked_stats, masked_logps, _stopped_early = _compute_probs_streaming_until_flip(
            document=full_context,
            query=query,
            ordered_offsets=ordered_offsets,
            change_template_contextCite=change_template_contextCite,
            hf_model=hf_model,
            hf_tok=hf_tok,
            hf_device=hf_device,
            true_variants=true_variants,
            false_variants=false_variants,
            compute_probs_file_name=compute_probs_file_name,
            p_true_flipping=p_true_flipping,
            save_logs=save_logs,
        )
    else:
        masked_prompts, masked_context_list = create_masked_prompts_iterative(
            full_context,
            query,
            ordered_offsets,
            change_template_contextCite=change_template_contextCite,
        )

        masked_stats, masked_logps = compute_probs(
            hf_model,
            hf_tok,
            masked_prompts,
            hf_device,
            None,
            batch_size=2,
            return_full_logp=True,
            file_name=compute_probs_file_name,
            detect_flip_to_true=p_true_flipping,
            true_variants=true_variants,
            false_variants=false_variants,
            save_file=save_logs,
            stop_on_flip=False,
        )
        # keep parallel arrays aligned if compute_probs stopped early on flip
        effective_steps = len(masked_stats)
        if effective_steps < len(masked_prompts):
            masked_prompts = masked_prompts[:effective_steps]
            masked_context_list = masked_context_list[:effective_steps]

    effective_steps = len(masked_stats)
    order = list(order)[:effective_steps]
    if scores_vec is not None:
        scores_vec = np.asarray(scores_vec, dtype=np.float32)

    if dump_json_path and save_logs:
        pick_scores = None if scores_vec is None else [float(scores_vec[i]) for i in order]
        if baseline_stats is None:
            baseline_stats = compute_probs(
                hf_model,
                hf_tok,
                [full_prompt],
                hf_device,
                expected_result=None,
                batch_size=1,
                return_full_logp=True,
                file_name=compute_probs_file_name + ".baseline_tmp",
                detect_flip_to_true=p_true_flipping,
                true_variants=true_variants,
                false_variants=false_variants,
            )[0][0]
        dump_masked_prompts_json(
            dump_json_path,
            query=query,
            baseline_prompt=full_prompt,
            baseline_stats=baseline_stats,
            masked_prompts=masked_prompts,
            masked_stats=masked_stats,
            masked_context_list=masked_context_list,
            order=order,
            scores_at_pick=pick_scores,
            policy=dump_policy,
            window=dump_window,
        )

    return masked_stats, masked_logps


def _piece_matches_at(
    context: str,
    piece: str,
    start: int,
    *,
    whitespace_flex: bool = True,
) -> bool:
    end = start + len(piece)
    if start < 0 or end > len(context):
        return False

    window = context[start:end]
    for pc, cc in zip(piece, window):
        if pc == cc:
            continue
        if whitespace_flex and pc.isspace() and cc.isspace():
            continue
        return False
    return True


def build_offsets_from_source_pieces(
    context: str,
    source_pieces,
    *,
    max_lookahead: int = 64,
    whitespace_flex: bool = True,
):
    offsets = []
    pos = 0
    n = len(context)

    for i, piece in enumerate(source_pieces):
        if piece is None:
            raise ValueError(f"source_pieces[{i}] is None")
        if piece == "":
            raise ValueError(f"source_pieces[{i}] is empty")

        idx = None
        matched_piece = None
        candidates = [piece]

        stripped = piece.lstrip()
        if stripped != piece:
            if pos == 0 or (pos < n and not context[pos].isspace()):
                candidates.append(stripped)

        for cand in candidates:
            if _piece_matches_at(context, cand, pos, whitespace_flex=whitespace_flex):
                idx = pos
                matched_piece = cand
                break

        if idx is None:
            for cand in candidates:
                last_start = min(n - len(cand), pos + max_lookahead)
                for cand_pos in range(pos, last_start + 1):
                    if _piece_matches_at(context, cand, cand_pos, whitespace_flex=whitespace_flex):
                        idx = cand_pos
                        matched_piece = cand
                        break
                if idx is not None:
                    break

        if idx is None or matched_piece is None:
            sample = context[pos : min(n, pos + 120)]
            raise ValueError(
                f"Could not align source piece #{i} to context.\n"
                f"piece={piece!r}\n"
                f"search_start={pos}\n"
                f"max_lookahead={max_lookahead}\n"
                f"context_sample_after_pos={sample!r}"
            )

        end = idx + len(matched_piece)
        offsets.append((idx, end))
        pos = end

    return offsets


#at2 habdling
def _align_source_text_to_context(
    context: str,
    piece: str,
    pos: int,
    *,
    max_lookahead: int = 64,
    whitespace_flex: bool = True,
):
    """
    Try to align one source string (or a merged group of source strings)
    onto `context`, starting near `pos`.

    Returns:
        (start, end, matched_text) or None
    """
    if piece is None:
        return None
    if piece == "":
        return None

    n = len(context)
    candidates = [piece]

    stripped = piece.lstrip()
    if stripped != piece:
        if pos == 0 or (pos < n and not context[pos].isspace()):
            candidates.append(stripped)

    for cand in candidates:
        if _piece_matches_at(context, cand, pos, whitespace_flex=whitespace_flex):
            return pos, pos + len(cand), cand

    for cand in candidates:
        if len(cand) > n:
            continue
        last_start = min(n - len(cand), pos + max_lookahead)
        for cand_pos in range(pos, last_start + 1):
            if _piece_matches_at(context, cand, cand_pos, whitespace_flex=whitespace_flex):
                return cand_pos, cand_pos + len(cand), cand

    return None


def _length_weighted_mean(values, lengths) -> float:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return 0.0

    w = np.asarray([max(1, int(x)) for x in lengths], dtype=np.float32)
    denom = float(w.sum())
    if denom <= 0:
        return float(vals.mean())
    return float((vals * w).sum() / denom)


def map_at2_scores_to_base_via_sources(
    *,
    context: str,
    source_pieces,
    scores,
    base_offsets,
    max_lookahead: int = 64,
    max_merge_pieces: int = 4,
    whitespace_flex: bool = True,
) -> np.ndarray:
    """
    Robust AT2 recompute mapper.

    Goal:
      Map AT2 token-level scores, indexed by `source_pieces`, back to the
      stable `base_offsets` of the original full context.

    Why this exists:
      In recompute mode, AT2 source pieces and a fresh tokenizer pass on the
      masked context may disagree by a local split/merge. We therefore align
      AT2's own returned `source_pieces` directly to the masked context.

    Strategy:
      - walk left-to-right through the masked context
      - try to align 1 source piece
      - if that fails, try merging the next 2..max_merge_pieces source pieces
      - assign a length-weighted average score to the matched substring
      - accumulate score as per-character "mass"
      - finally average character mass over each base token span
    """
    scores = np.asarray(scores, dtype=np.float32)

    if len(source_pieces) != len(scores):
        raise ValueError(
            f"AT2 source/count mismatch: len(source_pieces)={len(source_pieces)} "
            f"!= len(scores)={len(scores)}"
        )

    char_mass = np.zeros(len(context), dtype=np.float32)

    pos = 0
    i = 0
    n_src = len(source_pieces)

    while i < n_src:
        matched = False
        max_k = min(max_merge_pieces, n_src - i)

        for k in range(1, max_k + 1):
            group_pieces = list(source_pieces[i : i + k])
            merged_text = "".join(group_pieces)

            aligned = _align_source_text_to_context(
                context,
                merged_text,
                pos,
                max_lookahead=max_lookahead,
                whitespace_flex=whitespace_flex,
            )
            if aligned is None:
                continue

            start, end, _matched_text = aligned
            if end > start:
                group_lengths = [len(p) if p is not None else 1 for p in group_pieces]
                group_score = _length_weighted_mean(scores[i : i + k], group_lengths)
                char_mass[start:end] += group_score

            pos = end
            i += k
            matched = True
            break

        if not matched:
            # Fail-soft:
            # do not kill the whole example because of one weird local mismatch.
            # Skip this source piece and let later pieces try to align from the same pos.
            i += 1

    out = np.zeros(len(base_offsets), dtype=np.float32)
    for j, (s, e) in enumerate(base_offsets):
        if e > s:
            out[j] = float(char_mass[s:e].mean())
        else:
            out[j] = 0.0

    return out



def get_at2_token_scores(
    *,full_context: str,
    query: str,hf_model,hf_tok,
    score_estimator_path: str | Path,
    generate_kwargs: dict,):

    from at2.tasks import SimpleContextAttributionTask
    from at2 import AT2Attributor

    task = SimpleContextAttributionTask(
        context=full_context,
        query=query,
        model=hf_model,
        tokenizer=hf_tok,
        source_type="token",
        generate_kwargs=generate_kwargs,
    )

    score_estimator_path = Path(score_estimator_path)
    attributor = AT2Attributor.from_path(task, score_estimator_path)

    gen = task.generation
    start, end = 0, len(gen)
    scores = attributor.get_attribution_scores(start=start, end=end)

    return np.asarray(scores), gen, list(task.sources)
