# eval-backend-compare

Benchmarks inference output differences between **vLLM** and **HuggingFace Transformers** on the **MBPP+** coding benchmark. The goal is to identify which tasks produce different pass/fail outcomes depending on which backend is used.

## How it works

```
run_inference.py  →  run_eval.py  →  compare_results.py
(generate code)      (score code)     (Excel report)
```

1. Both backends run inference on the same MBPP+ tasks and produce a `samples.jsonl` file.
2. EvalPlus scores each file: a task passes only if both base and plus test suites pass.
3. The comparison script classifies every task into one of four buckets and exports an Excel report.

## Requirements

- Python 3.11
- CUDA-capable GPU (tested on A100 80 GB; minimum ~24 GB VRAM for a 7B model)
- [uv](https://docs.astral.sh/uv/) for environment management

## Setup

```bash
git clone <repo-url>
cd eval_backend_compare
uv sync          # installs all dependencies from uv.lock
```

No further steps needed. `uv sync` reads the lock file and reproduces the exact environment.

## Usage

### Run everything at once

```bash
bash run_all.sh
```

This executes all five steps in sequence and writes the final report to `outputs/comparison_report.xlsx`.

For a quick smoke test on the first 10 tasks:

```bash
bash run_all.sh --dry-run
```

### Run steps individually

```bash
# Step 1 – inference
uv run python run_inference.py --backend vllm
uv run python run_inference.py --backend transformers

# Step 2 – EvalPlus scoring
uv run python run_eval.py --backend vllm
uv run python run_eval.py --backend transformers

# Step 3 – comparison report
uv run python compare_results.py
```

### Override the model without editing config

```bash
uv run python run_inference.py --backend vllm --model /path/to/local/model
```

## Configuration

All runtime parameters are in `config.yaml`. Key fields:

| Field | Default | Description |
|---|---|---|
| `model.name_or_path` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HF hub ID or local path |
| `model.tensor_parallel_size` | `1` | Increase for multi-GPU (vLLM only) |
| `model.gpu_memory_utilization` | `0.5` | Fraction of GPU memory for vLLM |
| `model.dtype` | `bfloat16` | Model dtype |
| `generation.temperature` | `0.0` | `0.0` = greedy decoding (recommended for reproducibility) |
| `generation.max_new_tokens` | `512` | Max tokens per completion |
| `generation.transformers_batch_size` | `8` | Batch size for the Transformers backend |

Do not hardcode any of these values in Python files.

## Output

All outputs are written to `./outputs/` (gitignored):

| File | Description |
|---|---|
| `vllm_samples.jsonl` | Raw completions from vLLM |
| `transformers_samples.jsonl` | Raw completions from Transformers |
| `vllm_eval_results.json` | Per-task pass/fail from EvalPlus |
| `transformers_eval_results.json` | Per-task pass/fail from EvalPlus |
| `comparison_report.xlsx` | Final comparison report (5 sheets) |

The Excel report contains these sheets:

| Sheet | Contents |
|---|---|
| `Summary` | Overall pass rates, agreement rate, disagreement count |
| `vllm_only_pass` | Tasks where vLLM passes but Transformers fails |
| `transformers_only_pass` | Tasks where Transformers passes but vLLM fails |
| `both_pass` | Tasks both backends pass |
| `both_fail` | Tasks both backends fail |

## Testing

```bash
uv run pytest tests/
```

Tests cover prompt formatting and EvalPlus output parsing. All tests must pass before submitting changes.

## Project layout

```
eval_backend_compare/
├── config.yaml            # single source of truth for runtime params
├── run_inference.py       # inference entry point (vLLM + Transformers)
├── run_eval.py            # EvalPlus scoring entry point
├── compare_results.py     # diff analysis + Excel export
├── run_all.sh             # end-to-end pipeline script
├── utils/
│   ├── prompt.py          # prompt formatting shared by both backends
│   ├── io_utils.py        # JSONL read/write + config loading
│   └── eval_utils.py      # EvalPlus result parsing
└── tests/
    └── test_prompt.py     # unit tests
```
