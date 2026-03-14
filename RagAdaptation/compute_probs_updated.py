from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

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


@torch.no_grad()
def compute_probs(model, tok,
                  prompts: List[str], device,
                  expected_result, batch_size: int,
                  masked_context_list=None, return_full_logp: bool = True,
                  file_name: str = "output_compute_probe_brute_force.txt",
                  detect_flip_to_true: bool = False,
                  # NEW:
                  true_variants: Optional[List[str]] = None,
                  false_variants: Optional[List[str]] = None,
                  reduction: str = "logsumexp",  # "logsumexp" or "max"
                  ):
    """
    Updated compute_probs:
      - supports multiple true/false variants
      - aggregates within each class using LOGSUMEXP (default) or MAX
      - keeps flip detection behavior consistent with your original compute_probs:
          flip_cond = (not p_true_flipping and logP_true > logP_false) or
                      (p_true_flipping and logP_false > logP_true)
        and writes conversion info into file_name.

    IMPORTANT: Still next-token scoring, so we only use variants that are single tokens.
    """
    if true_variants is None:
        true_variants = [" true","true", "True", "TRUE"]
    if false_variants is None:
        false_variants = [" false","false", "False", "FALSE"]

    true_cands = _single_token_variant_ids(tok, true_variants)
    false_cands = _single_token_variant_ids(tok, false_variants)

    if len(true_cands) == 0 or len(false_cands) == 0:
        raise ValueError(
            "No usable single-token candidates for one of the classes.\n"
            f"true_variants={true_variants}\n"
            f"false_variants={false_variants}\n"
            f"true_cands(single-token)={true_cands}\n"
            f"false_cands(single-token)={false_cands}\n"
            "Fix: change variants or add multi-token sequential scoring fallback."
        )

    true_ids = torch.tensor([tid for _, tid in true_cands], device=device, dtype=torch.long)
    false_ids = torch.tensor([tid for _, tid in false_cands], device=device, dtype=torch.long)

    results: List[Dict[str, Any]] = []
    full_logps = []  # keep same meaning as your code: list of p_true values

    # Track first flip (optional but useful)
    first_flip_index: Optional[int] = None

    p = Path(file_name)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = tok(list(chunk),
                return_tensors="pt",padding=True,
                truncation=False,add_special_tokens=False,)
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc,return_dict=True,output_attentions=False,output_hidden_states=False,use_cache=False,)

            attn = enc["attention_mask"]              # [B, L]
            last_pos = attn.sum(dim=1) - 1            # [B]
            logits = out.logits                       # [B, L, V]
            b_idx = torch.arange(logits.size(0), device=device)
            last_logits = logits[b_idx, last_pos, :]  # [B, V]
            logp = torch.log_softmax(last_logits, dim=-1)

            # candidate logps
            true_logps = logp.index_select(dim=1, index=true_ids)    # [B, Kt]
            false_logps = logp.index_select(dim=1, index=false_ids)  # [B, Kf]

            if reduction == "logsumexp":
                lp_true = torch.logsumexp(true_logps, dim=1)   # [B]
                lp_false = torch.logsumexp(false_logps, dim=1) # [B]
            elif reduction == "max":
                lp_true, _ = true_logps.max(dim=1)
                lp_false, _ = false_logps.max(dim=1)
            else:
                raise ValueError("reduction must be 'logsumexp' or 'max'")

            # for reporting: best variants (argmax within each class)
            best_true_idx = torch.argmax(true_logps, dim=1)
            best_false_idx = torch.argmax(false_logps, dim=1)

            for j in range(len(chunk)):
                log_true_val = float(lp_true[j].item())
                log_false_val = float(lp_false[j].item())
                log_odds = log_true_val - log_false_val
                p_true_out_of_true_and_false = _stable_sigmoid_from_logodds(log_odds)

                res = { "logP_true": log_true_val,"logP_false": log_false_val,
                    "log_odds": float(log_odds),"p_true": float(p_true_out_of_true_and_false),
                    "reduction": reduction,"true_variants_used": [s for s, _ in true_cands],
                    "false_variants_used": [s for s, _ in false_cands],
                    "best_true_variant": true_cands[int(best_true_idx[j].item())][0],
                    "best_false_variant": false_cands[int(best_false_idx[j].item())][0],}

                # --- Flip detection (same condition as your original code) ---
                flip_cond = ((detect_flip_to_true is False) and (log_true_val < log_false_val)) or \
                            ((detect_flip_to_true is True) and (log_false_val < log_true_val))

                if flip_cond and first_flip_index is None:
                    first_flip_index = i + j
                    f.write(f"[{i + j}] logP_true={log_true_val:.4f} logP_false={log_false_val:.4f} "
                        f"log_odds={log_odds:.4f} p_true={p_true_out_of_true_and_false:.6f}\n\n")
                    f.write(f"After {i + j} iterations we had converted\n")
                    f.write(f"The prompt:\n{prompts[i + j]}\n")

                if return_full_logp:
                    full_logps.append(float(p_true_out_of_true_and_false))

                results.append(res)

        del out
        torch.cuda.empty_cache()
    if len(results) > 0:
        results[0]["first_flip_index"] = first_flip_index

    return results, (full_logps if return_full_logp else None)