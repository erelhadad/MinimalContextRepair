from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
from RagAdaptation.core.memory import MemoryConfig, cleanup_memory, get_memory_config
from RagAdaptation.prompts_format import normalize_true_false

import torch


def _stable_sigmoid_from_logodds(log_odds: float) -> float:
    if log_odds >= 0:
        z = math.exp(-log_odds)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(log_odds)
        return z / (1.0 + z)


def _single_token_variant_ids(tok, variants: Sequence[str]) -> List[Tuple[str, int]]:
    """
    Return [(variant_str, token_id)] for variants that tokenize to exactly 1 token.
    Uses add_special_tokens=False (as in your baseline compute_probs).
    """
    out: List[Tuple[str, int]] = []
    for s in variants:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.append((s, ids[0]))
    return out

def _variant_token_seqs(tok, variants: Sequence[str]) -> List[Tuple[str, List[int]]]:
    """
    Tokenize each string variant with add_special_tokens=False.
    Returns:
      [(variant_str, token_id_list), ...]
    Notes:
    - multi-token variants are allowed
    - empty-token variants are skipped
    - duplicate tokenizations are deduplicated so the same sequence is not
      counted twice in logsumexp
    """
    out: List[Tuple[str, List[int]]] = []
    seen: set[Tuple[int, ...]] = set()

    for s in variants:
        ids = tok(s, add_special_tokens=False).input_ids
        ids = [int(x) for x in ids]
        if len(ids) == 0:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)

        out.append((s, ids))
    return out


@torch.inference_mode()
def _score_variants_sequential(model,prompt_id_lists: List[List[int]],variant_token_seqs: List[Tuple[str, List[int]]],device,
    *,row_batch_size: int,pad_token_id: int, cleanup_every_batches: int = 0,
    aggressive_cleanup: bool = False,) -> torch.Tensor:
    """
    Score full variant sequences autoregressively.

    For each prompt P and variant of true/false v = [t1, ..., tm], we score:
      log p(v | P) = sum_k log p(t_k | P, t_<k) // meaning the summation over all tokens t_k that corresponds to the true/false partition
    Returns:
      scores of shape [num_prompts, num_variants]
    """
    num_prompts = len(prompt_id_lists)
    num_variants = len(variant_token_seqs)

    scores = torch.full((num_prompts, num_variants),
        fill_value=float("-inf"),dtype=torch.float32,)

    if num_prompts == 0 or num_variants == 0:
        return scores

    rows: List[List[int]] = []
    rows_map: List[Tuple[int, int, int, int]] = []
    batches_seen = 0

    def flush_rows() -> None:
        nonlocal rows, rows_map, batches_seen
        if not rows:
            return
        max_len = max(len(ids) for ids in rows)

        input_ids = torch.full(
            (len(rows), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,device=device,)
        attention_mask = torch.zeros(
            (len(rows), max_len),
            dtype=torch.long, device=device,)

        #acommdate attention_mask and tokens units ids
        for r, ids in enumerate(rows):
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            input_ids[r, : len(ids)] = ids_t
            attention_mask[r, : len(ids)] = 1

        out = model(input_ids=input_ids,attention_mask=attention_mask,
            return_dict=True,output_attentions=False,
            output_hidden_states=False,use_cache=False,)

        for local_r, (p_idx, v_idx, prompt_len, variant_len) in enumerate(rows_map):
            # logits[:, i, :] predicts token at position i+1.
            start_pos = prompt_len - 1
            end_pos = start_pos + variant_len
            logits_slice = out.logits[local_r, start_pos:end_pos, :]
            labels = input_ids[local_r, prompt_len : prompt_len + variant_len]
            token_logp = torch.log_softmax(logits_slice, dim=-1).gather(
                1, labels.unsqueeze(-1)
            )
            seq_lp = token_logp.sum()
            scores[p_idx, v_idx] = float(seq_lp.detach().cpu().item())

        del input_ids, attention_mask, out
        rows = []
        rows_map = []
        batches_seen += 1
        if cleanup_every_batches > 0 and batches_seen % cleanup_every_batches == 0:
            cleanup_memory(aggressive=aggressive_cleanup)

    for p_idx, prompt_ids in enumerate(prompt_id_lists):
        if len(prompt_ids) == 0:
            raise ValueError("Encountered an empty prompt tokenization; cannot score next tokens.")
        for v_idx, (_, variant_ids) in enumerate(variant_token_seqs):
            rows.append(prompt_ids + variant_ids)
            rows_map.append((p_idx, v_idx, len(prompt_ids), len(variant_ids)))
            if len(rows) >= row_batch_size:
                flush_rows()

    flush_rows()

    return scores

from RagAdaptation.core.prompting import InterventionMode

@torch.no_grad()
def compute_probs( model,tok,prompts: List[str],
    device, expected_result, batch_size: int, masked_context_list=None,
    return_full_logp: bool = True,
    file_name: str = "output_compute_probe_brute_force.txt",
    detect_flip_to_true: bool = False,
    true_variants: Optional[List[str]] = None,
    false_variants: Optional[List[str]] = None,
    reduction: str = "logsumexp",  # "logsumexp" or "max"
    save_file : bool = True, stop_on_flip  : bool = False,
    intervention_mode:InterventionMode=1,
    memory_config: MemoryConfig | None = None):

    """
    General class scorer for binary true/false answers.

    This version supports BOTH:
      - single-token variants
      - multi-token variants

    It scores the full token sequence for each variant autoregressively, then aggregates
    within each class using:
      - logsumexp (default), or
      - max

    The public output format is kept compatible with the rest of your pipeline.
    """

    if true_variants is None:
        true_variants = [" true", "true", " True", "True", " TRUE", "TRUE"]
    if false_variants is None:
        false_variants = [" false", "false", " False", "False", " FALSE", "FALSE"]

    true_cands = _variant_token_seqs(tok, true_variants)
    false_cands = _variant_token_seqs(tok, false_variants)

    if len(true_cands) == 0 or len(false_cands) == 0:
        raise ValueError(
            "No usable tokenized candidates for one of the classes.\n"
            f"true_variants={true_variants}\n"
            f"false_variants={false_variants}\n"
            f"true_cands={true_cands}\n"
            f"false_cands={false_cands}\n"
        )

    all_cands = true_cands + false_cands
    num_true = len(true_cands)

    pad_token_id = tok.pad_token_id
    if pad_token_id is None:
        pad_token_id = tok.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

    results: List[Dict[str, Any]] = []
    p_true_values: List[float] = []
    first_flip_index: Optional[int] = None
    log_f = None

    if save_file:
        p = Path(file_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        log_f = p.open("w", encoding="utf-8")

    mem = get_memory_config(memory_config)
    prompt_batch_size = mem.prompt_batch_size(batch_size)
    # Each prompt expands into answer variants, so this is usually the limiter.
    row_batch_size = mem.row_batch_size(prompt_batch_size)
    flag_flip_stop=False
    n=len(prompts)

    for i in range(0, n , prompt_batch_size):
            chunk = prompts[i : i + prompt_batch_size]

            prompt_id_lists = [
                [int(x) for x in tok(prompt, add_special_tokens=False).input_ids]
                for prompt in chunk
            ]

            all_scores = _score_variants_sequential(
                model=model,prompt_id_lists=prompt_id_lists,
                variant_token_seqs=all_cands,
                device=device,row_batch_size=row_batch_size,pad_token_id=pad_token_id,
                cleanup_every_batches=mem.cleanup_every_batches,
                aggressive_cleanup=mem.aggressive_cleanup,
            )  # [B, K_true + K_false]

            true_logps = all_scores[:, :num_true]   # [B, Kt]
            false_logps = all_scores[:, num_true:]  # [B, Kf]

            if reduction == "logsumexp":
                lp_true = torch.logsumexp(true_logps, dim=1)   # [B]
                lp_false = torch.logsumexp(false_logps, dim=1) # [B]
            elif reduction == "max":
                lp_true, _ = true_logps.max(dim=1)
                lp_false, _ = false_logps.max(dim=1)
            else:
                raise ValueError("reduction must be 'logsumexp' or 'max'")

            best_true_idx = torch.argmax(true_logps, dim=1)
            best_false_idx = torch.argmax(false_logps, dim=1)

            for j in range(len(chunk)):
                log_true_val = float(lp_true[j].item())
                log_false_val = float(lp_false[j].item())
                log_odds = log_true_val - log_false_val
                p_true_out_of_true_and_false = _stable_sigmoid_from_logodds(log_odds)

                res = {
                    "logP_true": log_true_val,
                    "logP_false": log_false_val,
                    "log_odds": float(log_odds),
                    "p_true": float(p_true_out_of_true_and_false),
                    "reduction": reduction,
                    "true_variants_used": [s for s, _ in true_cands],
                    "false_variants_used": [s for s, _ in false_cands],
                    "best_true_variant": true_cands[int(best_true_idx[j].item())][0],
                    "best_false_variant": false_cands[int(best_false_idx[j].item())][0],
                    "step_index": i + j+1,
                    }

                # Same flip semantics as your current pipeline:
                # - detect_flip_to_true=True  => looking for false -> true
                # - detect_flip_to_true=False => looking for true -> false
                flip_cond = (
                    ((detect_flip_to_true is False) and (log_true_val < log_false_val))
                    or
                    ((detect_flip_to_true is True) and (log_false_val < log_true_val))
                )

                res["is_flipped"]=flip_cond
                if flip_cond:
                    first_flip_index = i + j
                    flag_seq_flip= True
                    seq_flip = int(n / 100)
                    pass_no_flip = 10

                    # Generating the full response - if we report on this response we want to be sure that the actual response had changed
                    # enc = tok(prompts[i+j], return_tensors="pt", add_special_tokens=False).to(model.device)
                    #
                    # out_ids = model.generate(
                    #     enc,
                    #     max_new_tokens=20,
                    #     do_sample=False,
                    #     temperature=0.0,
                    #     eos_token_id=tok.eos_token_id,
                    #     pad_token_id=tok.pad_token_id,
                    # )
                    # gen_ids = out_ids[0, enc["input_ids"].shape[1]:]
                    # generated_response= tok.decode(gen_ids, skip_special_tokens=True).strip()
                    # norm_answer=normalize_true_false(generated_response)
                    # generate_and_ptrue_agree= (norm_answer =="true" and p_true_out_of_true_and_false>0.5 ) or (norm_answer =="false" and p_true_out_of_true_and_false<0.5)

                    if (stop_on_flip):
                            #and generate_and_ptrue_agree):
                        flag_flip_stop=True

                    res["first_flip_index"] = first_flip_index  + 1
                    # res["generated_response"] = generated_response
                    # res["generate_and_ptrue_agree"] = generate_and_ptrue_agree

                    # even if we dont save for the file we still return the results
                    if save_file and log_f is not None:
                        log_f.write(
                            f"[{i + j}] logP_true={log_true_val:.4f} "
                            f"logP_false={log_false_val:.4f} "
                            f"log_odds={log_odds:.4f} "
                            f"p_true={p_true_out_of_true_and_false:.6f}\n\n"
                        )
                        log_f.write(f"After {i + j} iterations we had converted\n")
                        log_f.write(f"The prompt:\n{prompts[i + j]}\n")

                # elif flip_cond:
                #     seq_flip-=1
                #     pass_no_flip=10
                #     flag_seq_flip = True
                #     if seq_flip==0:
                #         flag_flip_stop = True
                #
                # else:
                #     if flag_seq_flip:
                #         if pass_no_flip==0:
                #             seq_flip = int(n / 100)
                #             flag_seq_flip=False
                #         else:
                #             pass_no_flip -=1


                if return_full_logp:
                    p_true_values.append(float(p_true_out_of_true_and_false))

                results.append(res)
                #we already found a flip then stop the full iteration
                if flag_flip_stop:
                    if torch.cuda.is_available():
                        cleanup_memory(aggressive=mem.aggressive_cleanup)
                    if log_f is not None:
                        log_f.close()
                    return results, (p_true_values if return_full_logp else None)

    if log_f is not None:
        log_f.close()


    cleanup_memory(aggressive=mem.aggressive_cleanup)

    return results, (p_true_values if return_full_logp else None)
