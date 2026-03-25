from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, List

from RagAdaptation.core.paths import RUNS_DIR


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

