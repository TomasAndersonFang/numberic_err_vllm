#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY-RUN MODE ==="
fi

echo "=== Step 1: vLLM inference ==="
uv run python run_inference.py --backend vllm $DRY_RUN

echo "=== Step 2: Transformers inference ==="
uv run python run_inference.py --backend transformers $DRY_RUN

echo "=== Step 3: EvalPlus scoring (vLLM) ==="
uv run python run_eval.py --backend vllm

echo "=== Step 4: EvalPlus scoring (Transformers) ==="
uv run python run_eval.py --backend transformers

echo "=== Step 5: Comparison report ==="
uv run python compare_results.py

echo "=== Done. Report saved to outputs/comparison_report.xlsx ==="
