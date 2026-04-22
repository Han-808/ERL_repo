#!/usr/bin/env bash
# Submit one Slurm job per (method, model) pair.
# Each job gets 1 GPU and runs the full 90-task experiment sequentially.
#
# Usage:
#   bash slurm_run.sh                   # submit all jobs
#   bash slurm_run.sh --dry-run         # print sbatch commands without submitting

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
LOG_DIR="${SCRIPT_DIR}/log/slurm"
mkdir -p "$LOG_DIR"

DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

# ---- experiment config -------------------------------------------------------

MAX_STEPS=10
DISABLE_THINK=1

METHODS=(
  # "ace"           # full ACE: GT + test report + initial playbook
  # "ace_nogt"      # no ground truth
  # "ace_pb_empty"  # empty initial playbook
  # "ace_aed"
  # "hypothesis_v3" # ACE + open question curator
  # "hypothesis_v4" # ACE + open question curator
  "ace_once"
  "notebook_minimal"
)

MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-32B"
  "Qwen/Qwen3.5-27B"
  "Qwen/Qwen3.5-9B"
  "Qwen/Qwen3.5-4B"
)

# GPU memory requirements per model (to pick the right partition / gres)
# declare -A MODEL_MEM
# MODEL_MEM["Qwen/Qwen3-4B"]="20G"
# MODEL_MEM["Qwen/Qwen3-8B"]="20G"
# MODEL_MEM["Qwen/Qwen3-14B"]="40G"
# MODEL_MEM["Qwen/Qwen3-32B"]="80G"

PARTITION="gpu-h200"     # change to gpu-a100 / gpu-h200 as needed
TIME_LIMIT="6:00:00"
ACCOUNT="zlab"


# ------------------------------------------------------------------------------

for MODEL in "${MODELS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    # derive a short model tag: Qwen/Qwen3-8B -> qwen3-8b
    MODEL_TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||' | sed 's/qwen3/qwen3/')
    EXPERIMENT_NAME="${METHOD}-${MODEL_TAG}"
    DISABLE_THINK_FLAG=""
    if [[ "$DISABLE_THINK" -eq 1 ]]; then
      EXPERIMENT_NAME="${EXPERIMENT_NAME}-nothink"
      DISABLE_THINK_FLAG="--disable-thinking"
    fi
    JOB_NAME="${EXPERIMENT_NAME}"

    # pick number of GPUs based on model size
    NGPUS=1
    # if [[ "$MODEL" == *"32B"* ]]; then
    #   NGPUS=2
    # fi

    # Skip if output directory already exists
    OUTPUT_DIR="${SCRIPT_DIR}/outputs/${EXPERIMENT_NAME}"
    if [[ -d "$OUTPUT_DIR" ]]; then
      echo "Skipping $EXPERIMENT_NAME (output exists)"
      continue
    fi

    CMD=(
      sbatch
      --job-name="$JOB_NAME"
      --partition="$PARTITION"
      --account="$ACCOUNT"
      --nodes=1
      --ntasks=1
      --gpus-per-task="$NGPUS"
      --cpus-per-task=8
      --mem=200G
      --time="$TIME_LIMIT"
      --output="${LOG_DIR}/${JOB_NAME}_%j.out"
      --error="${LOG_DIR}/${JOB_NAME}_%j.err"
      --wrap="
        source '${PROJECT_DIR}/libs/ace/.venv/bin/activate'
        export PYTHONUNBUFFERED=1

        # Copy AppWorld data to node-local SSD to avoid GPFS I/O
        LOCAL_ROOT=/tmp/appworld-\${SLURM_JOB_ID}
        mkdir -p \"\${LOCAL_ROOT}\"
        cp -r '${PROJECT_DIR}/libs/ace-appworld/data' \"\${LOCAL_ROOT}/data\"
        export APPWORLD_ROOT=\"\${LOCAL_ROOT}\"

        # Cache Hugging Face models and datasets on local SSD as well
        export HF_HOME=/tmp/appworld-hf-cache
        mkdir -p \"\$HF_HOME\"

        python '${SCRIPT_DIR}/run.py' \
          --method '${METHOD}' \
          --model-name '${MODEL}' \
          --experiment-name '${EXPERIMENT_NAME}' \
          --max-steps ${MAX_STEPS} \
          ${DISABLE_THINK_FLAG}

        # Cleanup node-local data
        rm -rf \"\${LOCAL_ROOT}\"
      "
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] ${CMD[*]}"
    else
      JOB_ID=$("${CMD[@]}")
      echo "Submitted $JOB_NAME -> $JOB_ID"
    fi
  done
done
