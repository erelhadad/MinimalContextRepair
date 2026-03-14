from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from RagAdaptation.core.paths import RUNS_DIR


def sanitize_name(value: str) -> str:
    value = value.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "item"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def method_dir(base_dir: str | Path, method: str, *, seed: int | None = None) -> Path:
    p = Path(base_dir) / "methods" / sanitize_name(method)
    if seed is not None:
        p = p / f"seed_{seed}"
    return ensure_dir(p)


def plots_dir(base_dir: str | Path) -> Path:
    return ensure_dir(Path(base_dir) / "plots")


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


def write_text(path: str | Path, content: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(content, encoding='utf-8')
    return p


def create_run_root(output_root: str | Path, run_name: str | None = None) -> Path:
    root = Path(output_root) if output_root is not None else RUNS_DIR
    if run_name:
        root = root / sanitize_name(run_name)
    return ensure_dir(root)


def example_dir(run_root: str | Path, ex_idx: int) -> Path:
    return ensure_dir(Path(run_root) / "examples" / f"ex{ex_idx:04d}")


def model_dir(example_dir_path: str | Path, model_id: str) -> Path:
    return ensure_dir(Path(example_dir_path) / "models" / sanitize_name(model_id))


def write_manifest(run_root: str | Path, manifest: dict[str, Any]) -> Path:
    return write_json(Path(run_root) / "manifest.json", manifest)


def write_example_inputs(example_dir_path: str | Path, *, example_payload: dict[str, Any], context_text: str) -> None:
    write_json(Path(example_dir_path) / "input.json", example_payload)
    write_text(Path(example_dir_path) / "context.txt", context_text)
