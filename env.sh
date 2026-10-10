# Source this before anything else:  source env.sh
#
# Everything in this repo is addressed relative to AOAKV_ROOT, so a fresh box
# needs no path edits. Override any of these by exporting them beforehand.

AOAKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AOAKV_ROOT

# `evaluation.*`, `filtering.*` and `compat.*` are imported by the benchmark
# code; the repo root must be on PYTHONPATH for every run.
export PYTHONPATH="${AOAKV_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Model + dataset cache. Point this at a big disk (or a pre-warmed cache) to
# avoid re-downloading weights on every rented box.
export HF_HOME="${HF_HOME:-${AOAKV_ROOT}/.cache/huggingface}"

# Where predictions and scores land.
export RESULTS_DIR="${RESULTS_DIR:-${AOAKV_ROOT}/results}"

# wandb is off unless you ask for it:
#   export WANDB_MODE=online WANDB_API_KEY=... WANDB_PROJECT=aoakv
# Never commit a key — this repo deliberately contains none.
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-aoakv}"

# A stale PYTHONPATH pointing at another checkout of this code is the single
# easiest way to measure the wrong thing (see docs/LATENCY.md, "kernel loading
# trap"). Warn loudly if one is present.
_shadow=""
IFS=':' read -ra _pp <<< "${PYTHONPATH}"
for _d in "${_pp[@]}"; do
    [ -z "$_d" ] && continue
    [ "$_d" = "$AOAKV_ROOT" ] && continue
    if [ -f "$_d/evaluation/quest_attention.py" ]; then _shadow="$_shadow $_d"; fi
done
if [ -n "$_shadow" ]; then
    echo "WARNING: PYTHONPATH also contains another copy of this code:$_shadow"
    echo "         ${AOAKV_ROOT} comes first so it wins, but consider unsetting PYTHONPATH."
fi
unset _shadow _pp _d

echo "AOAKV_ROOT=${AOAKV_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "WANDB_MODE=${WANDB_MODE}  WANDB_PROJECT=${WANDB_PROJECT}"
