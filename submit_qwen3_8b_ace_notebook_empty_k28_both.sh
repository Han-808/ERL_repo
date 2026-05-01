#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/gscratch/stf/mohanc3/projects/ERL_repo
UV=/gscratch/stf/mohanc3/uv-env/uv-bin/uv
SGLANG=/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang

MODEL=Qwen/Qwen3-8B
MODEL_TAG=qwen3-8b
EPISODES=28
ENV_NAME=both
METHODS=(ace notebook_minimal)
JOB_NAME="qwen3-8b-ace-notebook-empty-k28-both"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/runs"

echo "Preflight: ${REPO_DIR}"
test -f "${REPO_DIR}/run.py"
test -f "${REPO_DIR}/methods/ace.py"
test -f "${REPO_DIR}/methods/notebook_minimal.py"
test -f "${REPO_DIR}/environments/frozen_lake.py"

grep -Eq '"notebook_minimal".*_notebook_factory\("empty"\)' "${REPO_DIR}/run.py" || {
  echo "ERROR: notebook_minimal is not configured to use empty notebook."
  exit 1
}

grep -Eq 'MAX_N[[:space:]]*=[[:space:]]*5' "${REPO_DIR}/environments/frozen_lake.py" || {
  echo "ERROR: FrozenLake MAX_N is not 5."
  exit 1
}

if grep -R -E 'reward2|actions2|feedback2|gated|Attempt 2|build_attempt2' \
  "${REPO_DIR}/common.py" \
  "${REPO_DIR}/run.py" \
  "${REPO_DIR}/prompts.py" \
  "${REPO_DIR}/methods" \
  "${REPO_DIR}/environments" \
  "${REPO_DIR}/ace_notebook_pipeline.py" >/dev/null; then
  echo "ERROR: second-attempt code markers are still present."
  exit 1
fi

sbatch \
  --job-name="${JOB_NAME}" \
  --account=stf \
  --partition=gpu-l40s \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gpus=l40s:1 \
  --mem=64G \
  --time=6:00:00 \
  --output="${REPO_DIR}/logs/%x-%j.out" \
  --error="${REPO_DIR}/logs/%x-%j.err" \
  --wrap="
    set -euo pipefail

    cd ${REPO_DIR}

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export NO_PROXY=localhost,127.0.0.1
    export no_proxy=localhost,127.0.0.1

    export CUDA_HOME=/sw/cuda/12.4.1
    export CUDA_PATH=/sw/cuda/12.4.1
    export PATH=/sw/cuda/12.4.1/bin:/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin:\$PATH
    export LD_LIBRARY_PATH=/sw/cuda/12.4.1/lib64:\${LD_LIBRARY_PATH:-}

    export HF_HOME=/gscratch/stf/mohanc3/hf-cache
    export TRANSFORMERS_CACHE=/gscratch/stf/mohanc3/hf-cache
    export HF_DATASETS_CACHE=/gscratch/stf/mohanc3/hf-cache
    export WANDB_DIR=/gscratch/stf/mohanc3/wandb
    export XDG_CACHE_HOME=/gscratch/stf/mohanc3/xdg-cache
    export TORCH_EXTENSIONS_DIR=/gscratch/stf/mohanc3/torch-extensions
    export TVM_FFI_CACHE_DIR=/gscratch/stf/mohanc3/tvm-ffi-cache
    export PYTHONUNBUFFERED=1

    mkdir -p /gscratch/stf/mohanc3/hf-cache
    mkdir -p /gscratch/stf/mohanc3/wandb
    mkdir -p /gscratch/stf/mohanc3/xdg-cache
    mkdir -p /gscratch/stf/mohanc3/torch-extensions
    mkdir -p /gscratch/stf/mohanc3/tvm-ffi-cache

    export RUN_DIR=${REPO_DIR}/runs/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}
    mkdir -p \"\$RUN_DIR\" \"\$RUN_DIR/outputs\"

    echo \"Job: \$SLURM_JOB_NAME \$SLURM_JOB_ID\"
    echo \"Node: \$(hostname)\"
    echo \"Model: ${MODEL}\"
    echo \"Methods: ${METHODS[*]}\"
    echo \"Notebook variant: empty via notebook_minimal\"
    echo \"Env: ${ENV_NAME}\"
    echo \"Episodes: ${EPISODES}\"
    which nvcc
    nvcc --version

    ${SGLANG} serve \
      --model-path '${MODEL}' \
      --host 127.0.0.1 \
      --port 30000 \
      --mem-fraction-static 0.45 \
      --disable-cuda-graph \
      --attention-backend triton \
      --sampling-backend pytorch \
      > /dev/null 2>&1 &

    SERVER_PID=\$!

    cleanup() {
      kill \$SERVER_PID 2>/dev/null || true
    }
    trap cleanup EXIT

    echo \"Waiting for SGLang server...\"
    for i in \$(seq 1 120); do
      if curl --noproxy \"*\" -s http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
        echo \"Server ready after \$((i * 5)) seconds\"
        break
      fi
      sleep 5
    done

    if ! curl --noproxy \"*\" -s http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
      echo \"ERROR: SGLang server failed to start\"
      exit 1
    fi

    for METHOD in ${METHODS[*]}; do
      echo \"Running method: \$METHOD\"
      ${UV} run python ${REPO_DIR}/run.py \
        --method \"\$METHOD\" \
        --env '${ENV_NAME}' \
        --episodes ${EPISODES} \
        --model '${MODEL}' \
        --server http://127.0.0.1:30000/v1 \
        --outputs-dir \"\$RUN_DIR/outputs\" \
        --disable-thinking \
        2>&1 | tee \"\$RUN_DIR/\${METHOD}_${MODEL_TAG}_nothink_k${EPISODES}_${ENV_NAME}.log\"
    done

    echo \"Done. RUN_DIR=\$RUN_DIR\"
  "
