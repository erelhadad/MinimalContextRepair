# RagAdaptation

A research codebase for analyzing **answer sensitivity in retrieval-augmented language models** through **minimal context deletion**.

The core idea of this project is simple: for a yes/no question answered by an LLM with retrieved context, identify the **smallest set of context tokens whose removal flips the model's answer** back to the answer obtained **without** the context.

This provides a counterfactual explanation of which parts of the retrieved context were most decisive for the model's prediction.

---

## Project goal

Retrieval-Augmented Generation (RAG) improves factual question answering by conditioning a language model on external context. However, model predictions can still be fragile: small changes in the retrieved evidence may alter the final answer.

This project studies that fragility in a controlled binary setting:

- the task is **yes/no question answering**
- we focus on examples where the model's answer **changes when context is added**
- we search for **minimal token deletions** that flip the context-conditioned answer back to the no-context answer

Formally, given a question `q`, a context `C`, and a model `M`, the goal is to find a maximal remaining sub-context `C* ⊆ C` such that:

`M_{C*}(q) = M(q)`

Equivalently, we want the **smallest deletion set** that changes the model's answer.

---

## What this repository contains

This repository implements an end-to-end experimental pipeline for:

1. **Building binary QA benchmarks** from datasets such as BoolQ and HotpotQA.
2. **Filtering to relevant examples** where the answer with context differs from the answer without context.
3. **Running masking-based attribution methods** over context tokens.
4. **Comparing methods** by how many tokens they need to mask before the answer flips.
5. **Producing reports, plots, and summaries** for analysis and writing.

---

## Main methods

The repository currently includes the following masking strategies:

- **Random** – masks context tokens in random order.
- **Attention** – uses question-to-context attention as a token-importance signal.
- **ContextCite** – uses ContextCite token attributions as the masking order.
- **AT2** – uses Attribution with Attention (AT2) token scores.
- **Recompute variants** – adaptive greedy masking that periodically recomputes scores on the partially masked prompt:
  - `recompute_attention`
  - `recompute_context_cite`
  - `recompute_at2`

The recompute variants are useful when token importance changes after earlier deletions.

---

## Binary scoring setup

This project does **not** rely only on the generated surface string.
Instead, for each prompt it computes the probability mass assigned to the two answer classes:

- **true**
- **false**

Multiple surface variants can be used for each class, and the code aggregates them with **logsumexp**. The reported score is:

`p_true = sigmoid(logP_true - logP_false)`

A **flip** is recorded when the induced binary label changes relative to the baseline prompt.

This scoring is implemented in `compute_probs_updated.py` and is used throughout the pipeline for consistency.

---

## Supported models

The code is organized around Hugging Face causal LMs.
The current model registry includes support for:

- `microsoft/Phi-3-mini-4k-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen3-4B-Instruct-2507`
- additional Qwen / Llama / Gemma entries in `core/model_config.py`

Some scripts also assume pre-trained **AT2 estimator checkpoints** are available for the relevant model.

---

## Repository structure

```text
RagAdaptation/
├── baseline/                 # brute-force and adaptive masking utilities
├── core/                     # paths, artifacts, model loading, prompting, plotting
├── dataset_creation/         # build benchmark examples and flip-only reports
├── document_handling/        # document loading / database helpers
├── methods/                  # attention, random, ContextCite, AT2, recompute wrappers
├── pipeline/                 # full pipeline orchestration and result summarization
├── tests/                    # smoke tests and small validation tests
├── compute_probs_updated.py  # class-based binary scorer
├── evaluate_questions.py     # evaluate no-context / with-context examples
├── run_pipeline.py           # CLI entry point for full masking experiments
└── document_run_pipeline_results.py / pipeline/report_results.py
                              # summarize experiment outputs
```

There are also experiment-specific folders and outputs, including:

- `outputs/reports/` – benchmark creation and evaluation reports
- `outputs/runs/` – organized per-example, per-model masking results
- `reports/` and `attention sink/` – additional analyses and writeups

---

## Installation

It is recommended to use a **Conda environment** with a PyTorch/CUDA setup compatible with your machine.

### 1. Create and activate an environment

```bash
conda create -n ragadapt python=3.11 -y
conda activate ragadapt
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Extra notes

This repository depends on external research packages such as:

- `transformers`
- `datasets`
- `context-cite`
- `at2`

If you plan to run AT2-based experiments, make sure that:

- the `at2` package is installed correctly
- the required estimator checkpoint exists
- the checkpoint path in `methods/at2.py` matches your local environment

---

## Quick start

### A. Build a benchmark from BoolQ

Create examples and evaluate them to find relevant flip cases:

```bash
python -m RagAdaptation.dataset_creation.make_flip_benchmark \
  --dataset boolq \
  --split validation \
  --limit 300
```

This creates raw examples, evaluation reports, enriched reports, and filtered reports such as:

- `examples.json`
- `eval_report.json`
- `report_any_flip.json`
- `report_all_models_flip.json`
- `report_flip_only__<model>.json`

### B. Run the masking pipeline on relevant examples

```bash
python -m RagAdaptation.run_pipeline \
  --input outputs/reports/dataset_creation/<run_name>/report_flip_only__microsoft__Phi-3-mini-4k-instruct.json \
  --stop_at_flip \
  --save_logs
```

By default, the pipeline can run:

- base methods: `attention`, `random`, `context_cite`, `at2`
- recompute methods: `attention`, `context_cite`, `at2`

You can also override models, methods, seeds, and recomputation intervals.

### C. Summarize results

After a run finishes, summarize results with a reporting script such as:

```bash
python RagAdaptation/document_run_pipeline_results.py
```

or the utilities under `RagAdaptation/pipeline/` depending on the format you want.

---

## Example workflow

A typical experimental workflow is:

1. Build a dataset-specific benchmark from BoolQ or HotpotQA.
2. Run `evaluate_questions.py` to detect examples where:
   - the no-context answer differs from the with-context answer
   - the example is therefore relevant for minimal deletion analysis
3. Run `run_pipeline.py` on the filtered report.
4. For each example/model pair:
   - compute the baseline probability
   - rank tokens using one of the attribution methods
   - iteratively mask tokens
   - stop once the answer flips (if `--stop_at_flip` is enabled)
5. Aggregate the results across examples and compare methods by:
   - success rate
   - average number of masked tokens until flip
   - average masked percentage of the context

---

## Output format

The pipeline writes outputs in an organized directory structure under `outputs/runs/`.

A typical run contains:

```text
outputs/runs/<run_name>/
├── manifest.json
└── examples/
    └── ex0000/
        ├── input.json
        ├── context.txt
        └── models/
            └── <model_name>/
                ├── methods/
                │   ├── baseline/
                │   ├── attention/
                │   ├── context_cite/
                │   ├── at2/
                │   └── ...
                ├── plots/
                └── pipeline_result_methods_...json
```

Method directories may include:

- `compute_probs.txt`
- `dump.json`
- `log.txt`
- token score dumps
- `p_true` plots

---

## Important scripts

### `dataset_creation/make_flip_benchmark.py`
Builds raw datasets and enriched reports, then filters them into flip-only benchmark files.

### `evaluate_questions.py`
Evaluates examples consistently using the same binary scoring logic later used in the masking pipeline.

### `run_pipeline.py`
Main CLI entry point for running masking methods on a benchmark report.

### `compute_probs_updated.py`
Implements the binary class scorer used to detect answer flips.

### `methods/`
Contains the concrete masking strategies.

### `pipeline/runner.py` and `pipeline/experiment.py`
Coordinate dataset iteration, per-model execution, and saving final run artifacts.

---

## Reproducibility notes

- The code uses deterministic-style scoring for binary decisions rather than relying only on free-form generation.
- Some scripts contain **absolute local paths** for checkpoints or report files; these may need to be changed before running on a new machine.
- AT2 experiments require external estimator checkpoints.
- GPU memory usage depends strongly on the chosen model, context length, and whether recomputation is enabled.

---

## Known limitations

- The repository currently focuses on **binary yes/no QA**.
- Some utilities are still research-oriented and assume a specific local directory layout.
- AT2 support may require additional patching or environment-specific fixes depending on the exact transformer version and hardware setup.
- Certain reporting scripts assume that external report files with context lengths already exist.

---

## Research context

This repository accompanies a project on:

**Minimal Context Deletions for Explaining Answer Sensitivity in Retrieval-Augmented Language Models**

The broader research question is how to identify the smallest decisive pieces of retrieved context that are responsible for a model's answer, and how different token-importance heuristics compare in finding those decisive pieces efficiently.

---

## Citation / acknowledgement

If you use or extend this codebase, please cite the relevant papers that inspired the methods, especially:

- ContextCite
- AT2 (Attribution with Attention)
- related work on counterfactual and attribution-based explanation for language models

---

## Author

**Erel Hadad**

If you are using this repository for a course or project submission, you may also want to add:

- course name
- supervisor / instructor
- semester and year
- short note about what part of the repository was implemented by you versus adapted from prior work

