from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import torch as ch

_HF_SCORER_CACHE: dict[str, tuple[Any, Any, Any]] = {}
_HF_SINGLE_DEVICE_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}



import gc
import torch

def unload_hf_model(model_id: str, device: str | None = None):
    global _HF_SCORER_CACHE, _HF_SINGLE_DEVICE_CACHE

    cached = _HF_SCORER_CACHE.pop(model_id, None)
    if cached is not None:
        model, tok, dev = cached
        del model, tok, dev

    if device is not None:
        cached2 = _HF_SINGLE_DEVICE_CACHE.pop((model_id, str(device)), None)
        if cached2 is not None:
            model, tok, dev = cached2
            del model, tok, dev

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "RagAdaptation.core.models requires the transformers package installed."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def get_hf_scorer(model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
    """
    Load a Hugging Face causal LM and fast tokenizer once per model_id and reuse them
    for next-token probability scoring.
    """
    cached = _HF_SCORER_CACHE.get(model_id)
    if cached is not None:
        return cached

    AutoModelForCausalLM, AutoTokenizer = _require_transformers()

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        output_attentions=True,
    ).eval()

    cached = (model, tok, model.device)
    _HF_SCORER_CACHE[model_id] = cached
    return cached


def get_hf_scorer_single_device(
    model_id: str,
    device: Union[str, ch.device] = "cuda:0",
    torch_dtype: Optional[ch.dtype] = None,
):
    """
    Load a Hugging Face causal LM onto a single explicit device. Useful for AT2 code paths
    that must avoid model sharding.
    """
    dev_key = str(device)
    cache_key = (model_id, dev_key)
    cached = _HF_SINGLE_DEVICE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if torch_dtype is None:
        if str(device).startswith("cuda"):
            torch_dtype = ch.bfloat16 if ch.cuda.is_available() else ch.float32
        else:
            torch_dtype = ch.float32

    AutoModelForCausalLM, AutoTokenizer = _require_transformers()

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    ).to(device).eval()

    cached = (model, tok, model.device)
    _HF_SINGLE_DEVICE_CACHE[cache_key] = cached
    return cached
