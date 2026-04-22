#!/usr/bin/env bash
# Submit one Slurm job per (model, think_mode) pair.
# Usage:
#   bash slurm_run.sh            # submit all
#   bash slurm_run.sh --dry-run  # preview

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR="${SCRIPT_DIR}/log/slurm"
mkdir -p "$LOG_DIR"

DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

# ---- config ------------------------------------------------------------------
ENVS="both"
K=50
PARTITION="ckpt-all"
ACCOUNT="stf"
TIME_LIMIT="4:00:00"

MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

THINK_MODES=(0 1)   # 0=disable thinking, 1=enable thinking
# ------------------------------------------------------------------------------

for MODEL in "${MODELS[@]}"; do
  for THINK in "${THINK_MODES[@]}"; do

    MODEL_TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')

    THINK_FLAG=""
    THINK_SUFFIX="-think"
    if [[ "$THINK" -eq 0 ]]; then
      THINK_FLAG="--disable-thinking"
      THINK_SUFFIX="-nothink"
    fi

    EXPERIMENT_NAME="notebook_minimal-${MODEL_TAG}${THINK_SUFFIX}"
    OUTPUT_FILE="${SCRIPT_DIR}/results_ace_notebook_frozen_lake_${EXPERIMENT_NAME}.json"

    if [[ -f "$OUTPUT_FILE" ]]; then
      echo "Skipping $EXPERIMENT_NAME (output exists)"
      continue
    fi

    CMD=(
      sbatch
      --job-name="$EXPERIMENT_NAME"
      --partition="$PARTITION"
      --account="$ACCOUNT"
      --nodes=1
      --ntasks=1
      --gpus-per-task="l40:1"
      --cpus-per-task=8
      --mem=64G
      --time="$TIME_LIMIT"
      --output="${LOG_DIR}/${EXPERIMENT_NAME}_%j.out"
      --error="${LOG_DIR}/${EXPERIMENT_NAME}_%j.err"
      --wrap="
        set -e
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
        export PATH=/sw/cuda/12.4.1/bin:\$PATH
        export CUDA_HOME=/sw/cuda/12.4.1
        export LD_LIBRARY_PATH=/sw/cuda/12.4.1/lib64:\$LD_LIBRARY_PATH
        module load gcc/13.2.0
        source /tmp/sglang-conda/bin/activate
        export HF_HOME=/tmp/mohanc3-xdg-cache/huggingface
        export PYTHONUNBUFFERED=1

        echo 'Starting SGLang server for ${MODEL}...'
        sglang serve \
          --model-path '${MODEL}' \
          --host 0.0.0.0 \
          --port 30000 \
          --mem-fraction-static 0.8 \
          --disable-cuda-graph \
          --attention-backend triton \
          --sampling-backend pytorch &
        SERVER_PID=\$!

        echo 'Waiting for server to be ready...'
        for i in \$(seq 1 60); do
          sleep 5
          if curl -s http://localhost:30000/v1/models > /dev/null 2>&1; then
            echo 'Server ready after '\$((i*5))'s'
            break
          fi
          echo 'Still waiting... '\$((i*5))'s'
        done

        cat '${LOG_DIR}'/sglang_server_\${SLURM_JOB_ID}.log

        if ! curl -s http://localhost:30000/v1/models > /dev/null 2>&1; then
          echo 'ERROR: Server failed to start'
          kill \$SERVER_PID 2>/dev/null
          exit 1
        fi

        /tmp/sglang-conda/bin/python '${SCRIPT_DIR}/main_ace_notebook.py' \
          --env '${ENVS}' \
          --episodes ${K} \
          --model '${MODEL}' \
          --server http://localhost:30000/v1 \
          --experiment-name '${EXPERIMENT_NAME}' \
          ${THINK_FLAG}

        kill \$SERVER_PID 2>/dev/null
        echo 'Done: ${EXPERIMENT_NAME}'
      "
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] Job: $EXPERIMENT_NAME"
    else
      JOB_ID=$("${CMD[@]}")
      echo "Submitted $EXPERIMENT_NAME -> $JOB_ID"
    fi

  done
done
