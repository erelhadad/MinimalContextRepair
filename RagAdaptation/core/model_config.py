from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from RagAdaptation.core.models import get_hf_scorer, get_hf_scorer_single_device
from RagAdaptation.prompts_format import TF_RAG_TEMPLATE, TF_RAG_TEMPLATE_A2T, TF_NO_CONTEXT_TEMPLATE


@dataclass
class ModelConfig:
    
    model_id: str

    # prompt / scoring behavior
    prompt_style: str = "rag"          # "rag" or "a2t"
    true_variants: List[str] = field(default_factory=lambda: ["true", "True", "TRUE"])
    false_variants: List[str] = field(default_factory=lambda: ["false", "False", "FALSE"])

    # future reasoning support
    supports_reasoning_toggle: bool = False
    default_reasoning: bool = False

    # loading behavior
    prefer_single_device_loader: bool = False
    device: Optional[str] = None

    # optional extra kwargs for future extension
    load_kwargs: Dict[str, Any] = field(default_factory=dict)

    # loaded objects
    hf_model: Any = field(init=False, default=None)
    tok_hf: Any = field(init=False, default=None)
    hf_device: Any = field(init=False, default=None)

    # -------- registry --------
    MODEL_REGISTRY: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # current models
        "microsoft/Phi-3-mini-4k-instruct": {
            "prompt_style": "rag",
            "true_variants": ["true", "True", "TRUE", " true"],
            "false_variants": ["false", "False", "FALSE", " false"],
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
        "meta-llama/Llama-3.2-3B-Instruct": {
            "prompt_style": "rag",
            "true_variants": ["true", "True", "TRUE", " true"],
            "false_variants": ["false", "False", "FALSE", " false"],
        },
        "google/gemma-2-2b-it": {
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
    })

    def __post_init__(self) -> None:
        spec = self.MODEL_REGISTRY.get(self.model_id, {})
        for key, value in spec.items():
            setattr(self, key, value)

    # -------- loading --------
    def load(self):
        """
        Load the HF model/tokenizer/device once and cache them on the object.
        """
        if self.hf_model is not None and self.tok_hf is not None and self.hf_device is not None:
            return self.hf_model, self.tok_hf, self.hf_device

        if self.prefer_single_device_loader:
            dev = self.device or "cuda:0"
            self.hf_model, self.tok_hf, self.hf_device = get_hf_scorer_single_device(
                model_id=self.model_id,
                device=dev,
            )
        else:
            # In your project this should return (model, tok, device)
            self.hf_model, self.tok_hf, self.hf_device = get_hf_scorer(model_id=self.model_id)

        return self.hf_model, self.tok_hf, self.hf_device

    # -------- prompt building --------
    def get_prompt_template(self, context_cite_at2: bool = False) -> str:
        '''for_attribution == true -> query'''
        if context_cite_at2:
            return TF_RAG_TEMPLATE_A2T 
        return TF_RAG_TEMPLATE


    def format_prompt(self,*,question: str, context: str,
        context_cite_at2_formating: bool = False,empty:bool=False) -> str:
        if context=="" or empty:
            return TF_NO_CONTEXT_TEMPLATE.format(question=question)
        template = self.get_prompt_template(context_cite_at2=context_cite_at2_formating)

        if context_cite_at2_formating:
            # TF_RAG_TEMPLATE_A2T expects {query}
            return template.format(context=context, query=question)

        # TF_RAG_TEMPLATE expects {question}
        return template.format(context=context, question=question)


    # -------- scoring helpers --------
    def get_true_variants(self) -> List[str]:
        return list(self.true_variants)

    def get_false_variants(self) -> List[str]:
        return list(self.false_variants)

    def compute_probs_kwargs(self) -> Dict[str, Any]:
        return {"true_variants": self.get_true_variants(),
            "false_variants": self.get_false_variants(),
        }

    # -------- future reasoning toggle --------
    def validate_reasoning_request(self, reasoning: bool) -> None:
        if reasoning and not self.supports_reasoning_toggle:
            raise ValueError(
                f"Model {self.model_id} does not support a reasoning toggle in ModelConfig."
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_style": self.prompt_style,
            "supports_reasoning_toggle": self.supports_reasoning_toggle,
            "default_reasoning": self.default_reasoning,
            "true_variants": self.true_variants,
            "false_variants": self.false_variants,
            "device": self.device,
        }
    # --------- unloading
    def unload(self):
        from RagAdaptation.core.models import unload_hf_model

        device_str = None if self.hf_device is None else str(self.hf_device)

        self.hf_model = None
        self.tok_hf = None
        self.hf_device = None

        unload_hf_model(self.model_id, device=device_str)
