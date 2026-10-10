#!/bin/bash
# RULER accuracy for one method. Run the three methods you want to compare and
# the eval step writes one score row per (task, seq_len, budget).
#
#   source env.sh
#   bash scripts/run_ruler.sh --model mistral-7b-instruct-v0.2 --method full
#   bash scripts/run_ruler.sh --model mistral-7b-instruct-v0.2 --method quest
#   bash scripts/run_ruler.sh --model mistral-7b-instruct-v0.2 --method aoakv
#
# Prepared data is shared across methods and budgets for a given model+seq_len,
# so the first run pays the preparation cost and later ones skip it.
set -euo pipefail

MODEL=""
METHOD_KIND=""
SEQ_LENS="4096,8192,16384"
BUDGETS="256,512,1024,2048"
TASKS_ARG=""
PAGE_SIZE=16
FIXED="true"
NUM_SAMPLES_ARG=""
GPU="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)        MODEL="$2"; shift 2 ;;
        --method)       METHOD_KIND="$2"; shift 2 ;;
        --seq_lens)     SEQ_LENS="$2"; shift 2 ;;
        --budgets)      BUDGETS="$2"; shift 2 ;;
        --tasks)        TASKS_ARG="$2"; shift 2 ;;
        --page_size)    PAGE_SIZE="$2"; shift 2 ;;
        --fixed)        FIXED="$2"; shift 2 ;;
        --num_samples)  NUM_SAMPLES_ARG="$2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,11p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "${AOAKV_ROOT:-}" ]; then
    echo "Error: source env.sh first" >&2
    exit 1
fi

[ -n "$MODEL" ] || { echo "Error: --model is required" >&2; exit 1; }
[ -n "$METHOD_KIND" ] || { echo "Error: --method is required (full|quest|aoakv)" >&2; exit 1; }

# --- method -> flags + result-directory name -------------------------------
# Names match the earlier runs (Full / Quest+fixed / Quest+fixed+VA) so old and
# new result trees stay comparable.
case "$METHOD_KIND" in
    full)
        QUEST="False"; VALUE_AWARE="False"; FIXED_FLAG="False"
        METHOD="Full"; BUDGETS="999999" ;;
    quest)
        QUEST="True"; VALUE_AWARE="False"
        METHOD="Quest" ;;
    aoakv)
        QUEST="True"; VALUE_AWARE="True"
        METHOD="Quest" ;;
    *)
        echo "Error: --method must be one of full|quest|aoakv" >&2; exit 1 ;;
esac
if [ "$METHOD_KIND" != "full" ]; then
    if [ "${FIXED,,}" = "true" ]; then
        FIXED_FLAG="True"; METHOD="${METHOD}+fixed"
    else
        FIXED_FLAG="False"
    fi
    [ "$VALUE_AWARE" = "True" ] && METHOD="${METHOD}+VA"
fi

cd "${AOAKV_ROOT}/RULER/scripts"
source config_models.sh
source config_tasks.sh

MODEL_CONFIG=$(MODEL_SELECT "${MODEL}")
[ -n "$MODEL_CONFIG" ] || { echo "Error: model ${MODEL} is not in config_models.sh" >&2; exit 1; }
IFS=":" read -r MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"

if [ -n "$TASKS_ARG" ]; then
    IFS=',' read -ra TASKS <<< "$TASKS_ARG"
else
    declare -n TASKS=synthetic
fi
[ -n "$NUM_SAMPLES_ARG" ] && NUM_SAMPLES="$NUM_SAMPLES_ARG"

export CUDA_VISIBLE_DEVICES="$GPU"
BENCHMARK=synthetic
ROOT_OUT="${RESULTS_DIR}/ruler/${MODEL}/${BENCHMARK}"

echo "=============================================="
echo "MODEL        = $MODEL  ($MODEL_PATH)"
echo "METHOD       = $METHOD   [quest=$QUEST fixed=$FIXED_FLAG value_aware=$VALUE_AWARE]"
echo "SEQ_LENS     = $SEQ_LENS"
echo "BUDGETS      = $BUDGETS"
echo "PAGE_SIZE    = $PAGE_SIZE"
echo "TASKS        = ${TASKS[*]}"
echo "NUM_SAMPLES  = $NUM_SAMPLES"
echo "BENCH_MODE   = $BENCH_MODE"
echo "GPU          = $CUDA_VISIBLE_DEVICES"
echo "OUT          = $ROOT_OUT"
echo "=============================================="

IFS=',' read -ra SEQ_LEN_LIST <<< "$SEQ_LENS"
IFS=',' read -ra BUDGET_LIST <<< "$BUDGETS"

for SEQ_LEN in "${SEQ_LEN_LIST[@]}"; do
    DATA_DIR="${ROOT_OUT}/${SEQ_LEN}/data"
    mkdir -p "$DATA_DIR"

    # --- 1. prepare (shared across methods/budgets) ---
    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir "${DATA_DIR}" \
            --benchmark ${BENCHMARK} \
            --task "${TASK}" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --tokenizer_type "${TOKENIZER_TYPE}" \
            --max_seq_length "${SEQ_LEN}" \
            --model_template_type "${MODEL_TEMPLATE_TYPE}" \
            --num_samples "${NUM_SAMPLES}" \
            ${REMOVE_NEWLINE_TAB}
    done

    for BUDGET in "${BUDGET_LIST[@]}"; do
        if [ "$METHOD_KIND" = "full" ]; then
            RUN_DIR="${ROOT_OUT}/${SEQ_LEN}/${METHOD}"
        else
            RUN_DIR="${ROOT_OUT}/${SEQ_LEN}/${METHOD}-${BUDGET}-${PAGE_SIZE}"
        fi
        PRED_DIR="${RUN_DIR}/pred"
        mkdir -p "$PRED_DIR"

        echo "---------- seq_len=${SEQ_LEN}  budget=${BUDGET}  ${METHOD} ----------"

        # --- 2. predict ---
        for TASK in "${TASKS[@]}"; do
            python pred/call_api.py \
                --data_dir "${DATA_DIR}" \
                --save_dir "${PRED_DIR}" \
                --benchmark ${BENCHMARK} \
                --task "${TASK}" \
                --server_type hf \
                --model_name_or_path "${MODEL_PATH}" \
                --temperature ${TEMPERATURE} \
                --top_k ${TOP_K} \
                --top_p ${TOP_P} \
                --batch_size 1 \
                --seq_len "${SEQ_LEN}" \
                --quest ${QUEST} --token_budget "${BUDGET}" --chunk_size ${PAGE_SIZE} \
                --fixed ${FIXED_FLAG} --value_aware ${VALUE_AWARE} \
                ${STOP_WORDS}
        done

        # --- 3. score ---
        python eval/evaluate.py \
            --data_dir "${PRED_DIR}" \
            --benchmark ${BENCHMARK} \
            --model "${MODEL}" \
            --method "${METHOD}" \
            --token_budget "${BUDGET}" \
            --max_seq_length "${SEQ_LEN}" \
            --bench_mode "${BENCH_MODE}" \
            --page_size ${PAGE_SIZE}
    done
done

echo "Done. Scores under ${ROOT_OUT}/<seq_len>/<method>/"
