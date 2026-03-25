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


def create_masked_prompts_iterative(
    document: str,
    query: str,
    offsets: List[Tuple[int, int]],
    k: int = 1,
    change_template_contextCite: bool = False,
):
    if change_template_contextCite:
        prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE_A2T)
    else:
        prompt_template = ChatPromptTemplate.from_template(TF_RAG_TEMPLATE)

    batch: List[str] = []
    masked_context_list: List[str] = []
    masked_spans: List[Tuple[int, int]] = []

    for i in range(len(offsets)):
        masked_spans.extend(offsets[i : i + k])
        masked_context = mask_context_spans_same_length(document, masked_spans)
        masked_context_list.append(masked_context)
        if change_template_contextCite:
            batch.append(prompt_template.format(context=masked_context, query=query))
        else:
            batch.append(prompt_template.format(context=masked_context, question=query))

    return batch, masked_context_list


def get_attention_scores(hf_model, hf_tok, hf_device, full_prompt: str):
    enc = hf_tok(
        full_prompt,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    enc = {k: v.to(hf_device) for k, v in enc.items()}

    with torch.no_grad():
        out = hf_model(
            **enc,
            output_attentions=True,
            return_dict=True,
        )

    attention_scores = out.attentions
    if attention_scores is None:
        raise ValueError(f"Model  did not return attentions.")

    return attention_scores


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
    window: int = 1,) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    flip_idx = _first_flip_idx(baseline_stats, list(masked_stats))
    idxs: List[int] = []

    if policy == "all":
        idxs = list(range(len(masked_prompts)))
    else:
        if flip_idx is None:
            idxs = [0] if len(masked_prompts) > 0 else []
        else:
            lo = max(0, flip_idx - window)
            hi = min(len(masked_prompts) - 1, flip_idx + window)
            idxs = sorted(set([0] + list(range(lo, hi + 1))))

    masked_entries = []
    for i in idxs:
        ent = {"step": int(i + 1), "prompt": masked_prompts[i], "stats": masked_stats[i]}
        if masked_context_list is not None:
            ent["masked_context"] = masked_context_list[i]
        if order is not None and i < len(order):
            ent["newly_masked_base_pos"] = int(order[i])
        if scores_at_pick is not None and i < len(scores_at_pick):
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
            "n_masked_total": int(len(masked_prompts)),
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
        batch_size=8,
        return_full_logp=True,
        file_name=compute_probs_file_name,
        detect_flip_to_true=p_true_flipping,
        true_variants=true_variants,
        false_variants=false_variants,
        save_file=save_logs,
        stop_on_flip=stop_on_flip,
    )

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
