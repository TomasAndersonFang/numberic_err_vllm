#!/usr/bin/env bash
# Run the full attribution-chain experiment: five inference modes,
# scoring for each, then the comparison report.
#
# Usage:
#   ./run_all.sh                          # run all five modes, full eval
#   ./run_all.sh --dry-run                # run all five modes on 10 tasks
#   ./run_all.sh --device 3               # pin to physical GPU 3
#   ./run_all.sh --dry-run --device 3     # both
#   ./run_all.sh --modes A_transformers_bs1,D_vllm_serial   # subset only
#
# Each mode is run independently; if one fails the script keeps going and
# the failure is summarized at the end. The comparison report runs as long
# as at least two modes have produced eval results.

set -uo pipefail

DRY_RUN=""
DEVICE=""
MODES_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --modes)
            MODES_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run] [--device N] [--modes M1,M2,...]"
            exit 1
            ;;
    esac
done

ALL_MODES=(
    "A_transformers_bs1"
    "B_transformers_bsN"
    "C_transformers_fa2"
    "D_vllm_serial"
    "E_vllm_default"
)

# Determine which modes to run.
if [[ -n "$MODES_ARG" ]]; then
    IFS=',' read -ra MODES <<< "$MODES_ARG"
else
    MODES=("${ALL_MODES[@]}")
fi

# Pin GPU if requested.
if [[ -n "$DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    echo "=== Using CUDA_VISIBLE_DEVICES=$DEVICE ==="
fi

if [[ -n "$DRY_RUN" ]]; then
    echo "=== DRY-RUN MODE (first 10 tasks only) ==="
fi

echo "=== Modes to run: ${MODES[*]} ==="
echo ""

# Track which modes finished inference vs eval successfully.
declare -a INFER_OK=()
declare -a INFER_FAIL=()
declare -a EVAL_OK=()
declare -a EVAL_FAIL=()

# vLLM tends to hold onto GPU memory briefly after exit; wait this many
# seconds between vLLM modes so the next launch sees a clean slate.
GPU_COOLDOWN=8

is_vllm_mode() {
    [[ "$1" == "D_vllm_serial" || "$1" == "E_vllm_default" ]]
}

run_inference_for_mode() {
    local mode="$1"
    echo ""
    echo "============================================================"
    echo "  INFERENCE: $mode"
    echo "============================================================"

    if uv run python run_inference.py --mode "$mode" $DRY_RUN; then
        INFER_OK+=("$mode")
        echo ">>> $mode inference OK"
    else
        INFER_FAIL+=("$mode")
        echo ">>> $mode inference FAILED (continuing with next mode)"
    fi

    # GPU cooldown after vLLM modes so the next process can claim memory.
    if is_vllm_mode "$mode"; then
        echo "  (sleeping ${GPU_COOLDOWN}s for GPU memory release)"
        sleep "$GPU_COOLDOWN"
    fi
}

run_eval_for_mode() {
    local mode="$1"
    echo ""
    echo "============================================================"
    echo "  EVAL: $mode"
    echo "============================================================"

    if uv run python run_eval.py --mode "$mode"; then
        EVAL_OK+=("$mode")
        echo ">>> $mode eval OK"
    else
        EVAL_FAIL+=("$mode")
        echo ">>> $mode eval FAILED (continuing with next mode)"
    fi
}

# Stage 1: inference for every mode.
echo ""
echo "############################################################"
echo "# STAGE 1: INFERENCE"
echo "############################################################"
for mode in "${MODES[@]}"; do
    run_inference_for_mode "$mode"
done

# Stage 2: eval only for modes whose inference succeeded.
echo ""
echo "############################################################"
echo "# STAGE 2: EVALUATION"
echo "############################################################"
for mode in "${INFER_OK[@]}"; do
    run_eval_for_mode "$mode"
done

# Stage 3: comparison report (proceeds as long as at least two modes
# made it through eval; the comparison script tolerates missing modes).
echo ""
echo "############################################################"
echo "# STAGE 3: COMPARISON REPORT"
echo "############################################################"
if [[ "${#EVAL_OK[@]}" -ge 2 ]]; then
    if uv run python run_comparison.py; then
        REPORT_STATUS="OK"
    else
        REPORT_STATUS="FAILED"
    fi
else
    echo "Skipping comparison: fewer than two modes completed eval."
    REPORT_STATUS="SKIPPED"
fi

# Final summary.
echo ""
echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
echo "Inference OK     (${#INFER_OK[@]}): ${INFER_OK[*]:-none}"
echo "Inference FAILED (${#INFER_FAIL[@]}): ${INFER_FAIL[*]:-none}"
echo "Eval      OK     (${#EVAL_OK[@]}): ${EVAL_OK[*]:-none}"
echo "Eval      FAILED (${#EVAL_FAIL[@]}): ${EVAL_FAIL[*]:-none}"
echo "Comparison report: $REPORT_STATUS"
echo ""

# Non-zero exit if anything failed, so CI / wrapper scripts can detect it.
if [[ "${#INFER_FAIL[@]}" -gt 0 || "${#EVAL_FAIL[@]}" -gt 0 || "$REPORT_STATUS" == "FAILED" ]]; then
    exit 1
fi
exit 0