from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"
CACHE_DIR = OUTPUTS_DIR / "cache"
CHROMA_DIR = OUTPUTS_DIR / "chroma"
AT2_DIR = OUTPUTS_DIR / "at2"

'''paths for the boolq dataset'''
newton_dict ={
    "qwen":"/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Phi_full_dataset_reports/report_flip_only__Qwen__Qwen3-4B-Instruct-2507_with_context_lengths.json",
    "micro":"/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Phi_full_dataset_reports/report_flip_only__microsoft__Phi-3-mini-4k-instruct_with_context_lengths.json",
    "minstral":"/home/erel.hadad/MinimalContextRepair/outputs/reports/dataset_creation/Mistral_full_dataset_reports/report_flip_only__mistralai__Mistral-7B-Instruct-v0.3_with_context_lengths.json"
}
'''paths for the hotpot dataset'''
newton_brit_dict={
    "qwen":"",
    "micro":"",
    "minstral":"",
}
