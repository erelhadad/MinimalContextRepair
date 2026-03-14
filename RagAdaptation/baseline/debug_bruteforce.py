import hashlib
import json
import torch
import math

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _encode_like_scoring(tok, prompts, device):
    # match compute_probs: padding=True, truncation=False
    enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}

@torch.no_grad()
def score_one_prompt_next_token(model, tok, device, prompt: str):
    enc = _encode_like_scoring(tok, [prompt], device)
    out = model(**enc, return_dict=True, output_attentions=False, output_hidden_states=False, use_cache=False)

    attn = enc["attention_mask"]
    last_pos = attn.sum(dim=1) - 1  # [1]
    logits = out.logits  # [1, L, V]
    last_logits = logits[0, last_pos.item(), :]  # [V]
    logp = torch.log_softmax(last_logits, dim=-1)

    true_id = tok(" True", add_special_tokens=False).input_ids
    false_id = tok(" False", add_special_tokens=False).input_ids
    if len(true_id) != 1 or len(false_id) != 1:
        raise ValueError(f'" True"/" False" not single tokens: {true_id=} {false_id=}')
    true_id, false_id = true_id[0], false_id[0]

    logP_true = float(logp[true_id].item())
    logP_false = float(logp[false_id].item())
    p_true_tf = float(1.0 / (1.0 + math.exp(logP_false - logP_true)))

    return {
        "logP_true": logP_true,
        "logP_false": logP_false,
        "log_odds": logP_true - logP_false,
        "p_true": p_true_tf,
        "top_tokens": topk_next_tokens(tok, logp, k=10),
    }

def topk_next_tokens(tok, logp_vec: torch.Tensor, k: int = 10):
    vals, idx = torch.topk(logp_vec, k)
    idx = idx.detach().cpu().tolist()
    vals = vals.detach().cpu().tolist()
    out = []
    for tid, lp in zip(idx, vals):
        out.append({
            "token_id": int(tid),
            "token_str": tok.decode([int(tid)]),
            "logp": float(lp),
        })
    return out

@torch.no_grad()
def generate_answer_like_scoring(model, tok, device, prompt: str, max_new_tokens=10):
    enc = _encode_like_scoring(tok, [prompt], device)
    out_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    # strip prompt tokens
    prompt_len = enc["input_ids"].shape[1]
    answer_ids = out_ids[0, prompt_len:]
    return tok.decode(answer_ids, skip_special_tokens=True).strip()

def debug_dump_flip(out_path: str, *, prompt: str, scored: dict, rescored: dict, gen_text: str):
    payload = {
        "prompt_sha1": _sha1(prompt),
        "prompt_preview": prompt[:400],
        "scored": scored,
        "rescored": rescored,
        "generated_text": gen_text,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
