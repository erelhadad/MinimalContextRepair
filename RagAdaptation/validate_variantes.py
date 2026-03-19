
from RagAdaptation.core.model_config import ModelConfig
from RagAdaptation.baseline.bruteforce_common import tokenize_context_with_offsets
models_dict= {
        # current models
        "microsoft/Phi-3-mini-4k-instruct": {
            "prompt_style": "rag",
            "true_variants": ["true", "True", "TRUE"],
            "false_variants": ["false", "False", "FALSE"],
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "prompt_style": "rag",
            "true_variants": ["true", "True", "TRUE", " true"],
            "false_variants": ["false", "False", "FALSE", " false"],
        },

        # new models you want to add
        "Qwen/Qwen2.5-3B-Instruct": {
            "prompt_style": "rag",
            "true_variants": ["true", "True", "TRUE", " true"],
            "false_variants": ["false", "False", "FALSE", " false"],
        },

        # future reasoning-capable branch
        "Qwen/Qwen3-4B-Instruct-2507": {
            "prompt_style": "rag",
            "supports_reasoning_toggle": True,
            "default_reasoning": False,
            "true_variants": ["true", "True", "TRUE", " true"],
            "false_variants": ["false", "False", "FALSE", " false"],
        },
        "Qwen/Qwen3-4B-Thinking-2507": {
            "prompt_style": "rag",
            "supports_reasoning_toggle": True,
            "default_reasoning": True,
            "true_variants": ["true", "True", "TRUE", " true"],
            "false_variants": ["false", "False", "FALSE", " false"],
        },
    }



for model_name, model_det in models_dict.items():
    print(f"{model_name}:")
    model,tok,_= ModelConfig(model_name).load()
    totlist= model_det["true_variants"]+model_det["false_variants"]
    for var in totlist:
        enc=tok(
        var,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
        print(f"for var {var} enc value is: ",enc)