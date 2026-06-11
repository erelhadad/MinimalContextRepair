from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, List, Tuple, Optional

from RagAdaptation.core.paths import RUNS_DIR
from RagAdaptation.core.paths import CACHE_DIR
from RagAdaptation.core.memory import MemoryConfig


@dataclass
class PipelineConfig:
    input_path: Path
    output_root: Path = RUNS_DIR
    models: List[str] = field(default_factory=list)
    methods:  List[str] = field(default_factory=lambda: ["attention", "random", "context_cite"])
    seeds: List[int] = field(default_factory=lambda: [0, 10, 20, 40])
    recompute:  List[str] = field(default_factory=list)
    true_variants:  List[str] = field(default_factory=lambda: ["true", "True", "TRUE"])
    false_variants: List[str] = field(default_factory=lambda: ["false", "False", "FALSE"])
    context_field: str = "context"
    skip_example_indices:  List[int] = field(default_factory=list)
    skip_recompute: List[int] = None
    save_logs :bool = True
    stop_at_flip :bool = True
    examples_range: Tuple[int,Optional[int]] = None
    tau:float = 0.5
    epsilon:float = 1e-2
    k:int = 5
    intervention_mode:str="mask_token"
    use_yes_no_variants: bool = False
    replacement_cache: Path = CACHE_DIR / "replacement_cache.json"
    neutral_model: str = "gpt-4o-mini"
    conceptnet_min_weight: float = 1.0
    replacement_semex_filter: bool = True
    replacement_semex_spacy_model: str = "en_core_web_sm"
    prefer_at2_word_scorer: bool = False
    running_env: str = "local"
    max_scoring_prompt_batch_size: Optional[int] = None
    scoring_row_batch_size: Optional[int] = None
    streaming_chunk_size: int = 32
    cleanup_every_batches: int = 0
    aggressive_cleanup: bool = False
    unload_models_between_runs: bool = False
    attention_native_seq_len_limit: int = 512

    def memory_config(self) -> MemoryConfig:
        return MemoryConfig(
            max_scoring_prompt_batch_size=self.max_scoring_prompt_batch_size,
            scoring_row_batch_size=self.scoring_row_batch_size,
            streaming_chunk_size=self.streaming_chunk_size,
            cleanup_every_batches=self.cleanup_every_batches,
            aggressive_cleanup=self.aggressive_cleanup,
            unload_models_between_runs=self.unload_models_between_runs,
            attention_native_seq_len_limit=self.attention_native_seq_len_limit,
        )

