#!/bin/bash
# LongBench accuracy for one method.
#
#   source env.sh
#   bash scripts/run_longbench.sh --model mistral-7b-instruct-v0.2 --method full
#   bash scripts/run_longbench.sh --model mistral-7b-instruct-v0.2 --method quest
#   bash scripts/run_longbench.sh --model mistral-7b-instruct-v0.2 --method aoakv
#
# Predictions go to $RESULTS_DIR/longbench/<model>/<task>/<method>/, and the
# eval step scores every task for one budget at a time.
set -euo pipefail

MODEL=""
METHOD_KIND=""
BUDGETS="256,512,1024,2048"
TASKS="hotpotqa,musique,2wikimqa,samsum,multifieldqa_en,qasper,multi_news,qmsum,trec,passage_retrieval_en,passage_count,triviaqa,lcc,repobench-p,gov_report,narrativeqa"
PAGE_SIZE=16
FIXED="true"
GPU="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2"; shift 2 ;;
        --method)    METHOD_KIND="$2"; shift 2 ;;
        --budgets)   BUDGETS="$2"; shift 2 ;;
        --tasks)     TASKS="$2"; shift 2 ;;
        --page_size) PAGE_SIZE="$2"; shift 2 ;;
        --fixed)     FIXED="$2"; shift 2 ;;
        --gpu)       GPU="$2"; shift 2 ;;
        -h|--help)   sed -n '2,10p' "$0"; exit 0 ;;
        *)           echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "${AOAKV_ROOT:-}" ]; then
    echo "Error: source env.sh first" >&2
    exit 1
fi
[ -n "$MODEL" ] || { echo "Error: --model is required" >&2; exit 1; }
[ -n "$METHOD_KIND" ] || { echo "Error: --method is required (full|quest|aoakv)" >&2; exit 1; }

case "$METHOD_KIND" in
    full)
        QUEST="False"; VALUE_AWARE="False"; FIXED_FLAG="False"
        METHOD="Full"; BUDGETS="999999" ;;
    quest)
        QUEST="True"; VALUE_AWARE="False"; METHOD="Quest" ;;
    aoakv)
        QUEST="True"; VALUE_AWARE="True"; METHOD="Quest" ;;
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

# LongBench's config keys spell a few models differently from RULER's. Accept
# the RULER spelling so both runners take the same --model value.
case "$MODEL" in
    qwen2.5-7b-instruct)   MODEL="qwen-2.5-7b-instruct" ;;
    qwen2.5-3b-instruct)   MODEL="qwen-2.5-3b-instruct" ;;
    longchat-7b-v1.5-32k)  MODEL="longchat-v1.5-7b-32k" ;;
esac

# The eval scripts spell their wandb switch "--bench_mode"; it gates nothing
# else (scores and CSVs are written either way), so derive it from WANDB_MODE.
if [ "${WANDB_MODE:-disabled}" = "disabled" ]; then WANDB_LOG="False"; else WANDB_LOG="True"; fi

export CUDA_VISIBLE_DEVICES="$GPU"
BASE_OUT="${RESULTS_DIR}/longbench/${MODEL}"
# pred.py passes this to from_pretrained(cache_dir=...) / load_dataset(cache_dir=...),
# both of which expect the HF *hub* cache layout ($HF_HOME/hub, i.e. models--org--name/).
# Pointing at $HF_HOME itself silently re-downloads model weights and datasets.
CACHE_DIR="${LONGBENCH_CACHE_DIR:-${HF_HUB_CACHE:-${HF_HOME}/hub}}"

echo "=============================================="
echo "MODEL      = $MODEL"
echo "METHOD     = $METHOD   [quest=$QUEST fixed=$FIXED_FLAG value_aware=$VALUE_AWARE]"
echo "BUDGETS    = $BUDGETS"
echo "PAGE_SIZE  = $PAGE_SIZE"
echo "WANDB      = $WANDB_MODE"
echo "GPU        = $CUDA_VISIBLE_DEVICES"
echo "OUT        = $BASE_OUT"
echo "=============================================="

cd "${AOAKV_ROOT}/evaluation/LongBench"
IFS=',' read -ra TASK_LIST <<< "$TASKS"
IFS=',' read -ra BUDGET_LIST <<< "$BUDGETS"

for BUDGET in "${BUDGET_LIST[@]}"; do
    echo "---------- budget=${BUDGET}  ${METHOD} ----------"
    for TASK in "${TASK_LIST[@]}"; do
        OUT_DIR="${BASE_OUT}/${TASK}/${METHOD}/"
        mkdir -p "$OUT_DIR"
        python -u pred.py \
            --model "${MODEL}" --task "${TASK}" \
            --cache_dir "${CACHE_DIR}" \
            --quest ${QUEST} --token_budget "${BUDGET}" --chunk_size ${PAGE_SIZE} \
            --fixed ${FIXED_FLAG} --value_aware ${VALUE_AWARE} \
            --out_dir "${OUT_DIR}"
    done

    python -u eval.py \
        --out_dir "${BASE_OUT}" \
        --model "${MODEL}" --tasks "${TASKS}" --method "${METHOD}" \
        --token_budget "${BUDGET}" \
        --bench_mode "${WANDB_LOG}"
done

echo "Done. Scores under ${BASE_OUT}/"
