from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Any


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def filter_examples_by_ex_idx(examples: List[dict], wanted: Iterable[int]) -> List[dict]:
    wanted_set = set(wanted)
    return [ex for ex in examples if ex.get("ex_idx") in wanted_set]


def extract_examples(blob: Any, path: Path) -> List[dict]:
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        if "results" in blob and isinstance(blob["results"], list):
            return blob["results"]
        if "examples" in blob and isinstance(blob["examples"], list):
            return blob["examples"]
    raise ValueError(
        f"Could not extract examples list from {path}. "
        f"Top-level type={type(blob).__name__}, "
        f"keys={list(blob.keys()) if isinstance(blob, dict) else 'N/A'}"
    )


def rebuild_like_source(original: Any, filtered: List[dict]) -> Any:
    if isinstance(original, list):
        return filtered
    if isinstance(original, dict):
        rebuilt = dict(original)
        if "results" in rebuilt and isinstance(rebuilt["results"], list):
            rebuilt["results"] = filtered
            return rebuilt
        if "examples" in rebuilt and isinstance(rebuilt["examples"], list):
            rebuilt["examples"] = filtered
            return rebuilt
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create small sanity-check input files from flip-only reports."
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="Directory containing the report_flip_only__*.json files",
    )
    parser.add_argument(
        "--mistral_examples",
        type=int,
        nargs="+",
        default=[13, 30, 91],
        help="ex_idx values to keep for Mistral sanity run",
    )
    parser.add_argument(
        "--phi3_examples",
        type=int,
        nargs="+",
        default=[98],
        help="ex_idx values to keep for Phi-3 sanity run",
    )
    args = parser.parse_args()

    mistral_src = args.base_dir / "report_flip_only__mistralai__Mistral-7B-Instruct-v0.3.json"
    phi3_src = args.base_dir / "report_flip_only__microsoft__Phi-3-mini-4k-instruct.json"

    mistral_dst = args.base_dir / (
        "report_flip_only__mistral__sanity_" + "_".join(map(str, args.mistral_examples)) + ".json"
    )
    phi3_dst = args.base_dir / (
        "report_flip_only__phi3__sanity_" + "_".join(map(str, args.phi3_examples)) + ".json"
    )

    if not mistral_src.exists():
        raise FileNotFoundError(f"Mistral source file not found: {mistral_src}")
    if not phi3_src.exists():
        raise FileNotFoundError(f"Phi-3 source file not found: {phi3_src}")

    mistral_data = load_json(mistral_src)
    phi3_data = load_json(phi3_src)

    mistral_examples = extract_examples(mistral_data, mistral_src)
    phi3_examples = extract_examples(phi3_data, phi3_src)

    mistral_filtered = filter_examples_by_ex_idx(mistral_examples, args.mistral_examples)
    phi3_filtered = filter_examples_by_ex_idx(phi3_examples, args.phi3_examples)

    save_json(mistral_dst, rebuild_like_source(mistral_data, mistral_filtered))
    save_json(phi3_dst, rebuild_like_source(phi3_data, phi3_filtered))

    print(f"[ok] wrote {mistral_dst} with {len(mistral_filtered)} examples")
    print(f"[ok] wrote {phi3_dst} with {len(phi3_filtered)} examples")

    missing_mistral = sorted(set(args.mistral_examples) - {x.get('ex_idx') for x in mistral_filtered})
    missing_phi3 = sorted(set(args.phi3_examples) - {x.get('ex_idx') for x in phi3_filtered})

    if missing_mistral:
        print(f"[warn] Mistral missing ex_idx values: {missing_mistral}")
    if missing_phi3:
        print(f"[warn] Phi-3 missing ex_idx values: {missing_phi3}")


if __name__ == "__main__":
    main()