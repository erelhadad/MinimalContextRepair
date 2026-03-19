from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUMMARY_FILENAMES = ("summary.json", "pipeline_result.json")
SUMMARY_STEM_PREFIXES = ("summary_methods_", "pipeline_result_methods_")
SUMMARY_FILENAME_RE = re.compile(
    r"^(summary|pipeline_result)_methods_(?P<methods>.+?)_recompute_(?P<recompute>.+?)_(?P<stamp>\d{8}_\d{6})\.json$"
)
MANIFEST_FILENAMES = ("manifest.json",)
BASE_METHOD_ORDER = [
    "attention",
    "recompute_attention",
    "context_cite",
    "recompute_context_cite",
    "at2",
    "recompute_at2",
    "random",
]
PLOT_PAIRS = [
    ("attention", "recompute_attention"),
    ("context_cite", "recompute_context_cite"),
    ("at2", "recompute_at2"),
]
PLOT_BASENAMES = {
    "attention": "attention_p_true.png",
    "recompute_attention": "recompute_attention_p_true.png",
    "context_cite": "context_cite_p_true.png",
    "recompute_context_cite": "recompute_context_cite_p_true.png",
    "at2": "at2_p_true.png",
    "recompute_at2": "recompute_at2_p_true.png",
}


@dataclass
class SummarySource:
    summary_path: Path
    model_dir: Path
    example_key: str
    model_id: str
    query: str
    methods: Dict[str, Any]
    baseline: Dict[str, Any] = field(default_factory=dict)
    p_true_flipping: bool = False


@dataclass
class ModelMethodResult:
    model_id: str
    model_name: str
    value: Optional[float]
    status: str
    detail: str
    source_summary: Path


@dataclass
class ExampleBundle:
    example_key: str
    query: str
    models: Dict[str, SummarySource] = field(default_factory=dict)


# -----------------------------
# General helpers
# -----------------------------

def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def pretty_model_name(model_id: str) -> str:
    low = model_id.lower()
    if "phi-3" in low or "phi_3" in low:
        return "Phi-3"
    if "mistral" in low:
        return "Mistral-7B"
    tail = model_id.split("/")[-1]
    tail = tail.replace("-", " ").replace("_", " ")
    return tail


def safe_name(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def infer_example_key(summary_path: Path) -> str:
    for part in reversed(summary_path.parts):
        m = re.search(r"(?:^|[_-])(example|ex)[_-]?(\d+)$", part, flags=re.IGNORECASE)
        if m:
            return f"example_{int(m.group(2)):03d}"
    if len(summary_path.parents) >= 2:
        return safe_name(summary_path.parents[1].name)
    return safe_name(summary_path.stem)


def choose_better_payload(old_payload: Any, new_payload: Any) -> Any:
    def score(payload: Any) -> Tuple[int, int]:
        if not isinstance(payload, dict):
            return (0, 0)
        if payload.get("status") == "failed" or "error" in payload:
            return (0, 0)
        if "masked_stats" in payload and isinstance(payload["masked_stats"], list):
            return (2, len(payload["masked_stats"]))
        if all(isinstance(v, dict) for v in payload.values()):
            total = 0
            for v in payload.values():
                total += len(v.get("masked_stats", [])) if isinstance(v, dict) else 0
            return (1, total)
        return (1, 0)

    return new_payload if score(new_payload) >= score(old_payload) else old_payload


def maybe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def is_summary_filename(name: str) -> bool:
    return name in SUMMARY_FILENAMES or bool(SUMMARY_FILENAME_RE.match(name))


def parse_summary_filename(path: Path) -> Dict[str, Any]:
    m = SUMMARY_FILENAME_RE.match(path.name)
    if not m:
        return {}

    def split_tag(tag: str) -> List[str]:
        if not tag or tag == "none":
            return []
        return [part for part in tag.split("_") if part]

    methods = split_tag(m.group("methods"))
    recompute = split_tag(m.group("recompute"))
    stamp = m.group("stamp")
    iso_stamp = None
    try:
        iso_stamp = datetime.strptime(stamp, "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        iso_stamp = None
    return {
        "kind": m.group(1),
        "methods": methods,
        "recompute": recompute,
        "stamp": stamp,
        "iso_stamp": iso_stamp,
    }


def _summary_dedupe_key(path: Path) -> Tuple[str, ...]:
    meta = parse_summary_filename(path)
    if meta:
        return (
            str(path.parent.resolve()),
            ",".join(meta.get("methods", [])),
            ",".join(meta.get("recompute", [])),
            str(meta.get("stamp", "")),
        )
    return (str(path.resolve()),)


def _summary_preference(path: Path) -> Tuple[int, float]:
    meta = parse_summary_filename(path)
    kind = str(meta.get("kind", ""))
    priority = 1 if kind == "summary" or path.name == "summary.json" else 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (priority, mtime)


def find_summary_files(run_roots: Iterable[Path]) -> List[Path]:
    found: Dict[Tuple[str, ...], Path] = {}

    def add_candidate(path: Path) -> None:
        if not is_summary_filename(path.name):
            return
        resolved = path.resolve()
        key = _summary_dedupe_key(resolved)
        prev = found.get(key)
        if prev is None or _summary_preference(resolved) >= _summary_preference(prev):
            found[key] = resolved

    for root in run_roots:
        root = root.resolve()
        if root.is_file():
            add_candidate(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            add_candidate(path)
    return sorted(found.values())


def load_summary_source(summary_path: Path) -> SummarySource:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    model_id = data.get("model_id", summary_path.parent.name)
    query = data.get("query", "")
    example_key = infer_example_key(summary_path)
    return SummarySource(
        summary_path=summary_path,
        model_dir=summary_path.parent,
        example_key=example_key,
        model_id=model_id,
        query=query,
        methods=data.get("methods", {}),
        baseline=data.get("baseline", {}),
        p_true_flipping=bool(data.get("p_true_flipping", False)),
    )


def merge_sources(sources: List[SummarySource]) -> Dict[str, ExampleBundle]:
    bundles: Dict[str, ExampleBundle] = {}
    for src in sources:
        bundle = bundles.setdefault(src.example_key, ExampleBundle(example_key=src.example_key, query=src.query))
        if not bundle.query and src.query:
            bundle.query = src.query

        existing = bundle.models.get(src.model_id)
        if existing is None:
            bundle.models[src.model_id] = src
            continue

        merged_methods = dict(existing.methods)
        for method_name, payload in src.methods.items():
            if method_name in merged_methods:
                merged_methods[method_name] = choose_better_payload(merged_methods[method_name], payload)
            else:
                merged_methods[method_name] = payload

        better_summary_path = src.summary_path if src.summary_path.stat().st_mtime >= existing.summary_path.stat().st_mtime else existing.summary_path
        better_model_dir = src.model_dir if src.summary_path.stat().st_mtime >= existing.summary_path.stat().st_mtime else existing.model_dir
        bundle.models[src.model_id] = SummarySource(
            summary_path=better_summary_path,
            model_dir=better_model_dir,
            example_key=src.example_key,
            model_id=src.model_id,
            query=src.query or existing.query,
            methods=merged_methods,
            baseline=src.baseline or existing.baseline,
            p_true_flipping=src.p_true_flipping or existing.p_true_flipping,
        )
    return bundles


# -----------------------------
# Manifest / run metadata helpers
# -----------------------------

def find_manifest_files(run_roots: Sequence[Path], summary_paths: Sequence[Path]) -> List[Path]:
    found: Dict[Path, Path] = {}

    def probe(start: Path) -> None:
        cur = start.resolve()
        chain = [cur] + list(cur.parents[:8])
        for base in chain:
            for name in MANIFEST_FILENAMES:
                cand = base / name
                if cand.exists() and cand.is_file():
                    found[cand.resolve()] = cand.resolve()

    for root in run_roots:
        if root.is_file():
            probe(root.parent)
        else:
            probe(root)
            for name in MANIFEST_FILENAMES:
                for path in root.rglob(name):
                    found[path.resolve()] = path.resolve()

    for summary_path in summary_paths:
        probe(summary_path.parent)

    return sorted(found)


def format_datetime_strings(values: Sequence[str]) -> str:
    cleaned = [v for v in values if v]
    if not cleaned:
        return "[fill manually]"
    parsed: List[Tuple[str, str]] = []
    for raw in cleaned:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            parsed.append((dt.isoformat(), raw))
        except ValueError:
            parsed.append((raw, raw))
    parsed.sort()
    uniq = []
    seen = set()
    for _, raw in parsed:
        if raw not in seen:
            uniq.append(raw)
            seen.add(raw)
    if len(uniq) == 1:
        return uniq[0]
    return f"{uniq[0]} .. {uniq[-1]}"


def sort_example_keys(example_keys: Sequence[str]) -> List[str]:
    def key_fn(x: str) -> Tuple[int, str]:
        m = re.search(r"(\d+)$", x)
        if m:
            return (int(m.group(1)), x)
        return (10**9, x)
    return sorted(example_keys, key=key_fn)


def summarize_example_keys(example_keys: Sequence[str]) -> str:
    ordered = sort_example_keys(example_keys)
    if not ordered:
        return "[fill manually]"
    if len(ordered) == 1:
        return f"{ordered[0]} (1 example)"
    return f"{ordered[0]} .. {ordered[-1]} ({len(ordered)} examples)"


def summarize_string_list(values: Iterable[str], empty: str = "[fill manually]") -> str:
    uniq = []
    seen = set()
    for v in values:
        if not v:
            continue
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return ", ".join(uniq) if uniq else empty


def _stats_list_length(payload: Any) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    stats = payload.get("masked_stats")
    if isinstance(stats, list):
        return len(stats)
    return None


def collect_binary_scoring_settings(bundles: Dict[str, ExampleBundle]) -> Dict[str, Any]:
    reductions = set()
    true_variants = set()
    false_variants = set()
    flip_directions = set()

    for bundle in bundles.values():
        for src in bundle.models.values():
            flip_directions.add("flip to true" if src.p_true_flipping else "flip to false")
            baseline_stats = src.baseline.get("stats", {}) if isinstance(src.baseline, dict) else {}
            if isinstance(baseline_stats, dict):
                red = baseline_stats.get("reduction")
                if red:
                    reductions.add(str(red))
                for s in baseline_stats.get("true_variants_used", []) or []:
                    true_variants.add(str(s))
                for s in baseline_stats.get("false_variants_used", []) or []:
                    false_variants.add(str(s))

    return {
        "reduction": sorted(reductions),
        "true_variants_used": sorted(true_variants),
        "false_variants_used": sorted(false_variants),
        "flip_direction": sorted(flip_directions),
    }


def collect_method_parameter_summary(bundles: Dict[str, ExampleBundle]) -> Dict[str, Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = {}

    def entry(method_name: str) -> Dict[str, Any]:
        return agg.setdefault(method_name, {
            "examples": set(),
            "models": set(),
            "ablation_steps": [],
            "order_lengths": [],
            "seed_counts": [],
            "estimators": set(),
            "failures": 0,
        })

    for bundle in bundles.values():
        for model_id, src in bundle.models.items():
            for method_name, payload in src.methods.items():
                e = entry(method_name)
                e["examples"].add(bundle.example_key)
                e["models"].add(pretty_model_name(model_id))

                if not isinstance(payload, dict):
                    continue
                if payload.get("status") == "failed" or "error" in payload:
                    e["failures"] += 1

                if method_name == "random":
                    e["seed_counts"].append(len(payload))
                    for seed_payload in payload.values():
                        n = _stats_list_length(seed_payload)
                        if n is not None:
                            e["ablation_steps"].append(n)
                else:
                    n = _stats_list_length(payload)
                    if n is not None:
                        e["ablation_steps"].append(n)
                    order = payload.get("order")
                    if isinstance(order, list):
                        e["order_lengths"].append(len(order))
                    estimator = payload.get("estimator")
                    if estimator:
                        e["estimators"].add(str(estimator))

    out: Dict[str, Dict[str, Any]] = {}
    for method_name, data in agg.items():
        ablation_steps = data["ablation_steps"]
        order_lengths = data["order_lengths"]
        seed_counts = data["seed_counts"]
        out[method_name] = {
            "examples_count": len(data["examples"]),
            "models": sorted(data["models"]),
            "ablation_steps_min": min(ablation_steps) if ablation_steps else None,
            "ablation_steps_max": max(ablation_steps) if ablation_steps else None,
            "ablation_steps_mean": (mean(ablation_steps) if ablation_steps else None),
            "order_length_min": min(order_lengths) if order_lengths else None,
            "order_length_max": max(order_lengths) if order_lengths else None,
            "seed_counts": sorted(set(seed_counts)),
            "estimators": sorted(data["estimators"]),
            "failures": int(data["failures"]),
        }
    return out


def method_param_lines(method_params: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for method_name in sorted(method_params, key=lambda m: method_sort_key(m)):
        mp = method_params[method_name]
        chunks = [f"examples={mp['examples_count']}"]
        if mp.get("models"):
            chunks.append("models=" + ", ".join(mp["models"]))
        if mp.get("ablation_steps_min") is not None:
            lo = mp["ablation_steps_min"]
            hi = mp["ablation_steps_max"]
            mu = mp["ablation_steps_mean"]
            chunks.append(f"observed ablation steps={lo}..{hi} (mean {mu:.2f})")
        if mp.get("order_length_min") is not None:
            chunks.append(f"order length={mp['order_length_min']}..{mp['order_length_max']}")
        if mp.get("seed_counts"):
            chunks.append("seed counts=" + ", ".join(str(x) for x in mp["seed_counts"]))
        if mp.get("estimators"):
            chunks.append("estimators=" + ", ".join(mp["estimators"]))
        if mp.get("failures"):
            chunks.append(f"failures={mp['failures']}")
        lines.append(f"{method_name}: " + "; ".join(chunks))
    return lines


def infer_prompt_template_line(observed_methods: Sequence[str]) -> str:
    methods = set(observed_methods)
    lines = ["baseline prompt inferred from current pipeline code: TF_RAG_TEMPLATE"]
    if {"context_cite", "recompute_context_cite", "at2", "recompute_at2"} & methods:
        lines.append("ContextCite / AT2 masking prompts inferred from current pipeline code: TF_RAG_TEMPLATE_A2T")
    return " | ".join(lines)


def infer_generation_settings_line(observed_methods: Sequence[str]) -> str:
    methods = set(observed_methods)
    chunks = [
        "regular pipeline uses compute_probs next-token scoring (not free-text generation)",
        "baseline compute_probs batch_size=1 (inferred from code)",
        "regular masking compute_probs batch_size=8 (inferred from code)",
    ]
    if {"recompute_attention", "recompute_context_cite", "recompute_at2"} & methods:
        chunks.append("recompute methods use compute_probs batch_size=2 (inferred from code)")
        chunks.append("recompute max_steps default=1000 and then capped to number of context spans (inferred from code)")
    if "recompute_at2" in methods:
        chunks.append("recompute_at2 score-estimator generate_kwargs={max_new_tokens=128, do_sample=False} (inferred from code)")
    return " | ".join(chunks)


def collect_main_outputs(bundles: Dict[str, ExampleBundle]) -> List[str]:
    outputs = set(["summary.json / pipeline_result.json"])
    for bundle in bundles.values():
        for src in bundle.models.values():
            for method_name in src.methods.keys():
                outputs.add(f"{method_name}/compute_probs.txt")
                if method_name == "random":
                    outputs.add("random/dump_seed*.json")
                else:
                    outputs.add(f"{method_name}/dump.json or log.txt (when relevant)")
                plot_name = PLOT_BASENAMES.get(method_name)
                if plot_name:
                    outputs.add(f"plots/{plot_name}")
    return sorted(outputs)


def collect_run_overview(
    run_roots: Sequence[Path],
    summary_paths: Sequence[Path],
    bundles: Dict[str, ExampleBundle],
) -> Dict[str, Any]:
    manifest_paths = find_manifest_files(run_roots, summary_paths)
    manifest_data = [maybe_load_json(p) or {} for p in manifest_paths]

    requested_models = []
    requested_methods = []
    requested_recompute = []
    requested_seeds = []
    input_paths = []
    context_fields = []
    created_at = []
    run_ids = []

    for path, data in zip(manifest_paths, manifest_data):
        created_at.append(str(data.get("created_at", "")))
        input_paths.append(str(data.get("input_path", "")))
        context_fields.append(str(data.get("context_field", "")))
        requested_models.extend([str(x) for x in data.get("models", [])])
        requested_methods.extend([str(x) for x in data.get("methods", [])])
        requested_recompute.extend([str(x) for x in data.get("recompute", [])])
        requested_seeds.extend([str(x) for x in data.get("seeds", [])])
        run_ids.append(path.parent.name)

    summary_file_meta = [parse_summary_filename(p) for p in summary_paths]
    summary_methods = []
    summary_recompute = []
    summary_dates = []
    summary_run_ids = []
    for p, meta in zip(summary_paths, summary_file_meta):
        if not meta:
            continue
        summary_methods.extend([str(x) for x in meta.get("methods", [])])
        summary_recompute.extend([str(x) for x in meta.get("recompute", [])])
        if meta.get("iso_stamp"):
            summary_dates.append(str(meta.get("iso_stamp")))
        summary_run_ids.append(f"{p.parent.name}:{meta.get('stamp', '')}")

    if not created_at:
        created_at.extend(summary_dates)
    if not run_ids:
        run_ids.extend(summary_run_ids)

    observed_models_raw = sorted({model_id for bundle in bundles.values() for model_id in bundle.models.keys()})
    observed_models_pretty = [pretty_model_name(m) for m in observed_models_raw]
    observed_methods = sorted({m for bundle in bundles.values() for src in bundle.models.values() for m in src.methods.keys()}, key=method_sort_key)

    binary_scoring = collect_binary_scoring_settings(bundles)
    method_params = collect_method_parameter_summary(bundles)
    outputs = collect_main_outputs(bundles)

    return {
        "run_roots": [str(p.resolve()) for p in run_roots],
        "manifest_files": [str(p) for p in manifest_paths],
        "run_ids": run_ids or [Path(run_roots[0]).resolve().name],
        "date": format_datetime_strings(created_at),
        "input_examples": summarize_example_keys(list(bundles.keys())),
        "input_paths": [p for p in input_paths if p],
        "models_requested": requested_models,
        "models_observed_raw": observed_models_raw,
        "models_observed_pretty": observed_models_pretty,
        "methods_requested": requested_methods or summary_methods,
        "methods_observed": observed_methods,
        "recompute_requested": requested_recompute or summary_recompute,
        "seeds_requested": requested_seeds,
        "context_fields": [c for c in context_fields if c],
        "prompt_template": infer_prompt_template_line(observed_methods),
        "generation_settings": infer_generation_settings_line(observed_methods),
        "binary_scoring": binary_scoring,
        "method_specific_parameters": method_params,
        "main_outputs": outputs,
        "notes": {
            "dump_policy": "flip (inferred from current runner code)",
            "dump_window": 1,
            "regular_k": 1,
        },
    }


# -----------------------------
# Flip extraction
# -----------------------------

def _is_flip(stats: Dict[str, Any], detect_flip_to_true: bool) -> bool:
    lp_true = float(stats.get("logP_true", float("nan")))
    lp_false = float(stats.get("logP_false", float("nan")))
    if math.isnan(lp_true) or math.isnan(lp_false):
        return False
    if detect_flip_to_true:
        return lp_false < lp_true
    return lp_true < lp_false


def extract_flip_step(masked_stats: List[Dict[str, Any]], detect_flip_to_true: bool) -> Optional[int]:
    if not masked_stats:
        return None
    first = masked_stats[0]
    first_idx = first.get("first_flip_index")
    if isinstance(first_idx, int):
        return first_idx + 1
    for i, st in enumerate(masked_stats):
        if _is_flip(st, detect_flip_to_true):
            return i + 1
    return None


def summarize_method_for_model(method_name: str, payload: Any, src: SummarySource) -> ModelMethodResult:
    model_name = pretty_model_name(src.model_id)
    detect_flip_to_true = src.p_true_flipping

    if not isinstance(payload, dict):
        return ModelMethodResult(src.model_id, model_name, None, "invalid", "payload is not a dict", src.summary_path)

    if payload.get("status") == "failed" or "error" in payload:
        detail = payload.get("error", payload.get("status", "failed"))
        return ModelMethodResult(src.model_id, model_name, None, "failed", str(detail), src.summary_path)

    if method_name == "random":
        seed_steps: List[int] = []
        failed_seeds: List[str] = []
        for seed, seed_payload in sorted(payload.items(), key=lambda kv: str(kv[0])):
            if not isinstance(seed_payload, dict):
                failed_seeds.append(str(seed))
                continue
            step = extract_flip_step(seed_payload.get("masked_stats", []), detect_flip_to_true)
            if step is None:
                failed_seeds.append(str(seed))
            else:
                seed_steps.append(step)

        if not seed_steps:
            detail = "no flip in any seed"
            if failed_seeds:
                detail += f" (failed/no-flip seeds: {', '.join(failed_seeds)})"
            return ModelMethodResult(src.model_id, model_name, None, "no_flip", detail, src.summary_path)

        mu = mean(seed_steps)
        sigma = pstdev(seed_steps) if len(seed_steps) > 1 else 0.0
        detail = f"{model_name}: {mu:.2f}±{sigma:.2f} over {len(seed_steps)} seeds"
        if failed_seeds:
            detail += f"; missing/no-flip seeds: {', '.join(failed_seeds)}"
        return ModelMethodResult(src.model_id, model_name, mu, "ok", detail, src.summary_path)

    step = extract_flip_step(payload.get("masked_stats", []), detect_flip_to_true)
    if step is None:
        return ModelMethodResult(src.model_id, model_name, None, "no_flip", f"{model_name}: no flip", src.summary_path)
    return ModelMethodResult(src.model_id, model_name, float(step), "ok", f"{model_name}: {step:g}", src.summary_path)


# -----------------------------
# Plot discovery/copy
# -----------------------------

def find_plot_for_method(model_dir: Path, method_name: str) -> Optional[Path]:
    basename = PLOT_BASENAMES.get(method_name)
    if not basename:
        return None
    direct = model_dir / basename
    if direct.exists():
        return direct
    hits = list(model_dir.rglob(basename))
    if hits:
        hits.sort(key=lambda p: len(p.parts))
        return hits[0]
    return None


def copy_plot(src: Path, dst_dir: Path, stem_prefix: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{stem_prefix}_{src.name}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


# -----------------------------
# LaTeX generation helpers
# -----------------------------

def method_sort_key(method_name: str) -> Tuple[int, str]:
    try:
        return (BASE_METHOD_ORDER.index(method_name), method_name)
    except ValueError:
        return (999, method_name)


def methods_for_bundle(bundle: ExampleBundle) -> List[str]:
    union = set()
    for src in bundle.models.values():
        union.update(src.methods.keys())
    return sorted(union, key=method_sort_key)


def differs_a_lot(values: List[float], abs_threshold: float, rel_threshold: float) -> bool:
    if len(values) < 2:
        return False
    span = max(values) - min(values)
    if span >= abs_threshold:
        return True
    denom = max(1.0, abs(mean(values)))
    return (span / denom) >= rel_threshold


def aggregate_method_row(bundle: ExampleBundle, method_name: str, abs_threshold: float, rel_threshold: float) -> Tuple[str, str, str, str]:
    per_model: List[ModelMethodResult] = []
    missing_models: List[str] = []

    for model_id, src in sorted(bundle.models.items(), key=lambda kv: pretty_model_name(kv[0])):
        payload = src.methods.get(method_name)
        if payload is None:
            missing_models.append(pretty_model_name(model_id))
            continue
        per_model.append(summarize_method_for_model(method_name, payload, src))

    numeric = [r.value for r in per_model if r.value is not None]
    status_notes = [r.detail for r in per_model if r.status != "ok"]
    ok_notes = [r.detail for r in per_model if r.status == "ok"]

    if not numeric:
        result_cell = "N/A"
        std_cell = "--"
        notes = status_notes or ["not available"]
    elif len(numeric) == 1:
        result_cell = f"{numeric[0]:.2f}"
        std_cell = "--"
        only_model = next(r.model_name for r in per_model if r.value is not None)
        notes = [f"only {only_model} available"]
        notes.extend(status_notes)
    else:
        result_cell = f"{mean(numeric):.2f}"
        std_cell = f"{pstdev(numeric):.2f}"
        notes = []
        if differs_a_lot(numeric, abs_threshold=abs_threshold, rel_threshold=rel_threshold):
            notes.extend(ok_notes)
        notes.extend(status_notes)

    if missing_models:
        notes.append("missing models: " + ", ".join(missing_models))

    notes_text = "; ".join(notes) if notes else ""
    if notes_text:
        notes_text = r"{\scriptsize " + latex_escape(notes_text) + "}"
    else:
        notes_text = ""

    return latex_escape(method_name), result_cell, std_cell, notes_text


def build_table_tex(bundle: ExampleBundle, abs_threshold: float, rel_threshold: float) -> str:
    rows = []
    for method_name in methods_for_bundle(bundle):
        method_cell, result_cell, std_cell, notes_cell = aggregate_method_row(bundle, method_name, abs_threshold, rel_threshold)
        rows.append(f"{method_cell} & {result_cell} & {std_cell} & {notes_cell} \\")

    query = latex_escape(bundle.query)
    body = "\n".join(rows) if rows else r"\multicolumn{4}{c}{No methods found} \\"
    return rf"""
\begin{{table}}[htbp]
\centering
\small
\begin{{tabular}}{{lccc}}
\hline
Method & Mean flip step & Std. across models & Notes \\
\hline
{body}
\hline
\end{{tabular}}
\caption{{Summary for {latex_escape(bundle.example_key)}. Query: {query}}}
\label{{tab:{safe_name(bundle.example_key)}}}
\end{{table}}
""".strip()


def build_plots_tex(bundle: ExampleBundle, out_dir: Path) -> str:
    fig_dir = out_dir / "figures" / safe_name(bundle.example_key)
    blocks: List[str] = []

    sorted_models = sorted(bundle.models.items(), key=lambda kv: pretty_model_name(kv[0]))
    for regular, recompute in PLOT_PAIRS:
        copied: List[Tuple[str, str, Path]] = []
        pair_has_any = False
        for model_id, src in sorted_models:
            model_label = pretty_model_name(model_id)
            reg_plot = find_plot_for_method(src.model_dir, regular)
            rec_plot = find_plot_for_method(src.model_dir, recompute)
            if reg_plot is not None:
                pair_has_any = True
                copied_path = copy_plot(reg_plot, fig_dir, f"{safe_name(model_label)}_{regular}")
                copied.append((model_label, regular, copied_path))
            if rec_plot is not None:
                pair_has_any = True
                copied_path = copy_plot(rec_plot, fig_dir, f"{safe_name(model_label)}_{recompute}")
                copied.append((model_label, recompute, copied_path))
        if not pair_has_any:
            continue

        subs = []
        for model_label, method_name, img_path in copied:
            rel = img_path.relative_to(out_dir)
            caption = latex_escape(f"{model_label} — {method_name}")
            subs.append(rf"""
\begin{{subfigure}}[t]{{0.48\textwidth}}
    \centering
    \includegraphics[width=\linewidth]{{{latex_escape(rel.as_posix())}}}
    \caption{{{caption}}}
\end{{subfigure}}
""".strip())

        method_caption = latex_escape(f"p_true plots for {regular} vs {recompute} ({bundle.example_key})")
        blocks.append(rf"""
\begin{{figure}}[htbp]
\centering
{chr(10).join(subs)}
\caption{{{method_caption}}}
\label{{fig:{safe_name(bundle.example_key)}_{safe_name(regular)}_{safe_name(recompute)}}}
\end{{figure}}
""".strip())

    return "\n\n".join(blocks)


def latex_kv_line(label: str, value: str) -> str:
    return rf"\noindent\textbf{{{latex_escape(label)}}} {latex_escape(value)}\\"


def latex_kv_list(label: str, values: Sequence[str], empty_value: str = "[fill manually]") -> str:
    value = summarize_string_list(values, empty=empty_value)
    return latex_kv_line(label, value)


def build_run_card_tex(run_overview: Dict[str, Any]) -> str:
    binary = run_overview.get("binary_scoring", {})
    method_lines = method_param_lines(run_overview.get("method_specific_parameters", {}))
    outputs = run_overview.get("main_outputs", [])

    run_id_value = summarize_string_list(run_overview.get("run_ids", []), empty="[fill manually]")
    date_value = str(run_overview.get("date", "[fill manually]"))
    input_examples = str(run_overview.get("input_examples", "[fill manually]"))

    models_value = summarize_string_list(
        run_overview.get("models_observed_pretty", []) or run_overview.get("models_requested", []),
        empty="[fill manually]",
    )
    methods_value = summarize_string_list(
        run_overview.get("methods_observed", []) or run_overview.get("methods_requested", []),
        empty="[fill manually]",
    )
    recompute_value = summarize_string_list(run_overview.get("recompute_requested", []), empty="none / [fill manually]")

    binary_parts = []
    if binary.get("flip_direction"):
        binary_parts.append("flip direction=" + ", ".join(binary["flip_direction"]))
    if binary.get("reduction"):
        binary_parts.append("reduction=" + ", ".join(binary["reduction"]))
    if binary.get("true_variants_used"):
        binary_parts.append("true_variants=" + ", ".join(binary["true_variants_used"]))
    if binary.get("false_variants_used"):
        binary_parts.append("false_variants=" + ", ".join(binary["false_variants_used"]))
    if run_overview.get("notes"):
        binary_parts.append(
            "dump_policy=" + str(run_overview["notes"].get("dump_policy", ""))
        )
        binary_parts.append(
            "dump_window=" + str(run_overview["notes"].get("dump_window", ""))
        )
    binary_value = " | ".join([p for p in binary_parts if p]) or "[fill manually]"

    method_items = "\n".join(rf"\item {latex_escape(line)}" for line in method_lines) if method_lines else r"\item [fill manually]"
    output_items = "\n".join(rf"\item {latex_escape(line)}" for line in outputs) if outputs else r"\item [fill manually]"

    return rf"""
\section*{{Run / Experiment Card}}

{latex_kv_line('Run ID:', run_id_value)}
{latex_kv_line('Date:', date_value)}
{latex_kv_line('Goal of run:', '[fill manually]')}
{latex_kv_line('Input examples:', input_examples)}
{latex_kv_line('Models:', models_value)}
{latex_kv_line('Methods:', methods_value)}
{latex_kv_line('Recompute setting:', recompute_value)}
{latex_kv_line('Prompt template:', str(run_overview.get('prompt_template', '[fill manually]')))}
{latex_kv_line('Generation settings:', str(run_overview.get('generation_settings', '[fill manually]')))}
{latex_kv_line('Binary scoring settings:', binary_value)}

\noindent\textbf{{Method-specific parameters:}}\\
\begin{{itemize}}[leftmargin=1.5em]
{method_items}
\end{{itemize}}

{latex_kv_line('What changed relative to previous run:', '[fill manually]')}

\noindent\textbf{{Main outputs inspected:}}\\
\begin{{itemize}}[leftmargin=1.5em]
{output_items}
\end{{itemize}}

{latex_kv_line('Main observation:', '[fill manually]')}
{latex_kv_line('Conclusion / next step:', '[fill manually]')}
""".strip() + "\n"


def build_example_tex(bundle: ExampleBundle, out_dir: Path, abs_threshold: float, rel_threshold: float) -> str:
    section_title = latex_escape(f"{bundle.example_key}: {bundle.query}")
    table_tex = build_table_tex(bundle, abs_threshold=abs_threshold, rel_threshold=rel_threshold)
    plots_tex = build_plots_tex(bundle, out_dir=out_dir)
    return rf"""
\section{{{section_title}}}

{table_tex}

{plots_tex}
""".strip() + "\n"


def build_master_tex(example_files: List[Path], include_run_card: bool = True) -> str:
    parts: List[str] = []
    if include_run_card:
        parts.append(r"\input{run_card.tex}")
    parts.extend([rf"\input{{{latex_escape(p.name)}}}" for p in sorted(example_files)])
    inputs = "\n\n".join(parts)
    return rf"""
\documentclass{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{float}}
\usepackage{{booktabs}}
\usepackage{{enumitem}}
\usepackage[T1]{{fontenc}}
\begin{{document}}

{inputs}

\end{{document}}
""".lstrip()


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect run_full_pipeline summaries (including timestamped partial-run summaries) and create LaTeX summary tables per example.")
    ap.add_argument("run_roots", nargs="+", help="One or more run roots (or example folders / summary.json files) to scan.")
    ap.add_argument("--out_dir", default="latex_summary", help="Directory where the .tex files, metadata, and copied figures will be written.")
    ap.add_argument("--diff-threshold-abs", type=float, default=5.0, help="Absolute threshold for deciding the two model values differ a lot.")
    ap.add_argument("--diff-threshold-rel", type=float, default=0.35, help="Relative threshold for deciding the two model values differ a lot.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_roots = [Path(p) for p in args.run_roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = find_summary_files(run_roots)
    if not summary_paths:
        raise SystemExit("No summary files were found (expected summary.json / pipeline_result.json or the newer summary_methods_* / pipeline_result_methods_* format).")

    sources = [load_summary_source(p) for p in summary_paths]
    bundles = merge_sources(sources)
    run_overview = collect_run_overview(run_roots, summary_paths, bundles)

    run_card_tex = build_run_card_tex(run_overview)
    run_card_path = out_dir / "run_card.tex"
    run_card_path.write_text(run_card_tex, encoding="utf-8")

    example_files: List[Path] = []
    for example_key, bundle in sorted(bundles.items()):
        tex = build_example_tex(
            bundle,
            out_dir=out_dir,
            abs_threshold=float(args.diff_threshold_abs),
            rel_threshold=float(args.diff_threshold_rel),
        )
        tex_path = out_dir / f"{safe_name(example_key)}.tex"
        tex_path.write_text(tex, encoding="utf-8")
        example_files.append(tex_path)

    master = build_master_tex(example_files, include_run_card=True)
    master_path = out_dir / "summary_report.tex"
    master_path.write_text(master, encoding="utf-8")

    manifest = {
        "run_roots": [str(p.resolve()) for p in run_roots],
        "summary_files_found": [str(p) for p in summary_paths],
        "examples_written": [p.name for p in example_files],
        "master_tex": str(master_path),
        "run_card_tex": str(run_card_path),
        "run_overview": run_overview,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(json.dumps(run_overview, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] wrote {len(example_files)} example tex files to {out_dir}")
    print(f"[ok] master file: {master_path}")
    print(f"[ok] run card: {run_card_path}")


if __name__ == "__main__":
    main()
