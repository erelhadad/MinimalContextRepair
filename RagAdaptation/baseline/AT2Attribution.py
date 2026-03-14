import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from at2.tasks import SimpleContextAttributionTask
from at2.utils import get_model_and_tokenizer
from at2 import AT2Trainer

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
#microsoft/Phi-3-mini-4k-instruct
#mistralai/Mistral-7B-Instruct-v0.3
ATTN_IMPL = "eager"
GENERATE_KWARGS = {"max_new_tokens": 128, "do_sample": False}
TRAIN_DATASET = "databricks/databricks-dolly-15k"
SEED = 42
N_TRAIN = 1000

# Optional cache locations (good on shared servers)
os.environ["HF_HOME"] = "/data2/hf"
os.environ["HF_DATASETS_CACHE"] = "/data2/hf/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/data2/hf/transformers"

# ----------------------------
# Load model
# ----------------------------
model, tokenizer = get_model_and_tokenizer(MODEL_NAME,attn_implementation=ATTN_IMPL,)


if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id


# ----------------------------
# Robust max length
# ----------------------------
def reasonable_model_max_len(tok, fallback: int = 4096) -> int:
    m = getattr(tok, "model_max_length", None)
    if m is None or m > 100_000:
        return fallback
    return int(m)

MODEL_MAX_TOKENS = reasonable_model_max_len(tokenizer, fallback=4096)

# Prefer the model's true positional limit if available (often the ground truth)
cfg_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
if cfg_max is not None:
    MODEL_MAX_TOKENS = min(MODEL_MAX_TOKENS, int(cfg_max))

MAX_NEW = int(GENERATE_KWARGS.get("max_new_tokens", 128))
SAFETY = 32
MAX_INPUT_TOKENS = MODEL_MAX_TOKENS - MAX_NEW - SAFETY

model.eval()

# ----------------------------
# Token-length safe filtering
# ----------------------------
def reasonable_model_max_len(tok, fallback: int = 4096) -> int:
    m = getattr(tok, "model_max_length", None)
    if m is None or m > 100_000:
        return fallback
    return int(m)

MODEL_MAX_TOKENS = reasonable_model_max_len(tokenizer, fallback=4096)
RESERVED_FOR_QUERY = 512
MAX_CONTEXT_TOKENS = max(256, MODEL_MAX_TOKENS - RESERVED_FOR_QUERY)

def full_prompt_len(example) -> int:
    prompt = f"Context: {example['context']}\n\nQuery: {example['instruction']}"
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Key trick: pass a huge max_length so HF tokenizer won't compare against model_max_length and warn
    ids = tokenizer(
        chat_prompt,
        add_special_tokens=False,
        truncation=False,
        max_length=1_000_000,
    )["input_ids"]
    return len(ids)

def truncate_context_to_fit(example):
    # Build a "base" prompt with empty context to estimate overhead from template + query
    base_prompt = f"Context: \n\nQuery: {example['instruction']}"
    base_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": base_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    base_ids = tokenizer(
        base_chat, add_special_tokens=False, truncation=False, max_length=1_000_000
    )["input_ids"]
    overhead = len(base_ids)

    # Remaining budget for context tokens (approx)
    budget_for_ctx = MAX_INPUT_TOKENS - overhead
    if budget_for_ctx <= 0:
        example["context"] = ""
        return example

    ctx_prefix = "Context: "
    ctx_ids = tokenizer(
        ctx_prefix + example["context"],
        add_special_tokens=False,
        truncation=False,
        max_length=1_000_000,
    )["input_ids"]

    if len(ctx_ids) > budget_for_ctx:
        ctx_ids = ctx_ids[:budget_for_ctx]
        ctx_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)
        if ctx_text.startswith(ctx_prefix):
            ctx_text = ctx_text[len(ctx_prefix):]
        example["context"] = ctx_text

    return example

def filter_fn(example):
    if example.get("context") is None:
        return False
    if example.get("category") not in ["summarization", "closed_qa", "information_extraction"]:
        return False
    return full_prompt_len(example) <= MAX_INPUT_TOKENS

# ----------------------------
# Load -> filter -> shuffle -> select
# ----------------------------
raw = load_dataset(TRAIN_DATASET, split="train[:4000]")
ds = raw.filter(filter_fn).shuffle(seed=SEED)
ds = ds.map(truncate_context_to_fit)   # <-- add this line
ds = ds.select(range(min(N_TRAIN, ds.num_rows)))

print(f"[dataset] raw={raw.num_rows} filtered_used={ds.num_rows} max_ctx_tokens={MAX_CONTEXT_TOKENS}")

# ----------------------------
# Task builder (token-level sources)
# ----------------------------
def task_from_example(example, model, tokenizer):
    return SimpleContextAttributionTask(
        context=example["context"],
        query=example["instruction"],
        model=model,tokenizer=tokenizer,
        source_type="token",generate_kwargs=GENERATE_KWARGS,)

# ----------------------------
# Unique save_path (where training artifacts are written)
# ----------------------------
run_id = f"{MODEL_NAME.replace('/', '_')}_{TRAIN_DATASET.replace('/', '_')}_n{ds.num_rows}_seed{SEED}_srcToken"
save_path = Path("outputs") / run_id

# Clean run (optional)
shutil.rmtree(save_path, ignore_errors=True)

# ----------------------------
# TRAINING PIPELINE (this is the training part)
# ----------------------------
trainer = AT2Trainer(
    save_path=save_path,
    dataset=ds,
    model=model,
    tokenizer=tokenizer,
    task_from_example=task_from_example,
)

trainer.generate()
trainer.compute_features_and_outputs()
trainer.train(save_name="default")

print(f"[done] estimator saved under: {save_path / 'estimators' / 'default'}")