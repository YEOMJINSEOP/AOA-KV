#!/bin/bash
# AOA-KV vs original Quest latency, same box, same settings, one CSV.
#
#   source env.sh
#   bash scripts/run_latency.sh --model longchat-7b-v1.5-32k --contexts 8192 --budgets 1024
#
# Both kernel trees must be built first:  bash setup.sh --build-kernels
# Results append to $RESULTS_DIR/latency/latency.csv and a comparison table is
# printed at the end.
set -euo pipefail

MODEL="longchat-7b-v1.5-32k"
CONTEXTS="8192"
BUDGETS="1024"
DECODE_LEN=256
ITERATION=10
WARMUP=3
PAGE_SIZE=16
GPU="0"
OUT=""
KERNELS="quest_orig aoa_kv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)      MODEL="$2"; shift 2 ;;
        --contexts)   CONTEXTS="$2"; shift 2 ;;
        --budgets)    BUDGETS="$2"; shift 2 ;;
        --decode_len) DECODE_LEN="$2"; shift 2 ;;
        --iteration)  ITERATION="$2"; shift 2 ;;
        --warmup)     WARMUP="$2"; shift 2 ;;
        --page_size)  PAGE_SIZE="$2"; shift 2 ;;
        --kernels)    KERNELS="${2//,/ }"; shift 2 ;;
        --gpu)        GPU="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        -h|--help)    sed -n '2,10p' "$0"; exit 0 ;;
        *)            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "${AOAKV_ROOT:-}" ]; then
    echo "Error: source env.sh first" >&2
    exit 1
fi
OUT="${OUT:-${RESULTS_DIR}/latency/latency.csv}"

# One GPU, exclusively: a second process on the same device makes every number
# noise. Checked, not assumed.
export CUDA_VISIBLE_DEVICES="$GPU"
used_mb=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used_mb" -gt 500 ]; then
    echo "WARNING: GPU $GPU already has ${used_mb} MiB in use — latency numbers will be noisy." >&2
fi

echo "=============================================="
echo "MODEL      = $MODEL"
echo "KERNELS    = $KERNELS"
echo "CONTEXTS   = $CONTEXTS"
echo "BUDGETS    = $BUDGETS"
echo "DECODE_LEN = $DECODE_LEN   ITERATION = $ITERATION (+${WARMUP} warmup)"
echo "PAGE_SIZE  = $PAGE_SIZE    GPU = $GPU"
echo "OUT        = $OUT"
echo "=============================================="

IFS=',' read -ra CONTEXT_LIST <<< "$CONTEXTS"
IFS=',' read -ra BUDGET_LIST <<< "$BUDGETS"

for CONTEXT in "${CONTEXT_LIST[@]}"; do
    for BUDGET in "${BUDGET_LIST[@]}"; do
        for KERNEL in $KERNELS; do
            echo
            echo "---------- ${KERNEL}  context=${CONTEXT}  budget=${BUDGET} ----------"
            python "${AOAKV_ROOT}/latency/bench_latency.py" \
                --kernel "${KERNEL}" \
                --model "${MODEL}" \
                --context_len "${CONTEXT}" \
                --token_budget "${BUDGET}" \
                --page_size "${PAGE_SIZE}" \
                --decode_len "${DECODE_LEN}" \
                --iteration "${ITERATION}" \
                --warmup "${WARMUP}" \
                --out "${OUT}"
        done
    done
done

echo
python "${AOAKV_ROOT}/latency/compare.py" --csv "${OUT}"
