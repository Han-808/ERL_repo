#!/usr/bin/env bash
set -euo pipefail

# Submit a safe sharded single-turn pass@k eval for notebook_minimal.
#
# Default shape:
#   8 Slurm array tasks, 1 GPU each, Qwen3-14B, no thinking.
#   Shards 0-3 request L40 GPUs; shards 4-7 request L40S GPUs.
#   Each task evaluates a disjoint shard of notebook states for both
#   FrozenLake and Sokoban.
#
# Usage:
#   bash submit_single_turn_passk_qwen3_14b_nothink_array.sh --dry-run
#   bash submit_single_turn_passk_qwen3_14b_nothink_array.sh

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
A100_PARTITION="${A100_PARTITION:-gpu-a100}"
A100_GPU_REQUEST="${A100_GPU_REQUEST:-a100:1}"
L40_PARTITION="${L40_PARTITION:-gpu-l40}"
L40_GPU_REQUEST="${L40_GPU_REQUEST:-l40:1}"
L40S_PARTITION="${L40S_PARTITION:-gpu-l40s}"
L40S_GPU_REQUEST="${L40S_GPU_REQUEST:-l40s:1}"

# Use eight independent one-GPU shards by default: 4 on L40 and 4 on L40S.
# Override the per-GPU-type counts/concurrency at submit time if needed.
A100_SHARDS="${A100_SHARDS:-0}"
L40_SHARDS="${L40_SHARDS:-4}"
L40S_SHARDS="${L40S_SHARDS:-4}"
A100_ARRAY_MAX_CONCURRENT="${A100_ARRAY_MAX_CONCURRENT:-${A100_SHARDS}}"
L40_ARRAY_MAX_CONCURRENT="${L40_ARRAY_MAX_CONCURRENT:-${L40_SHARDS}}"
L40S_ARRAY_MAX_CONCURRENT="${L40S_ARRAY_MAX_CONCURRENT:-${L40S_SHARDS}}"
EXPECTED_NUM_SHARDS=$((A100_SHARDS + L40_SHARDS + L40S_SHARDS))
NUM_SHARDS="${NUM_SHARDS:-${EXPECTED_NUM_SHARDS}}"
if [[ "${NUM_SHARDS}" -ne "${EXPECTED_NUM_SHARDS}" ]]; then
  echo "ERROR: NUM_SHARDS (${NUM_SHARDS}) must equal A100_SHARDS + L40_SHARDS + L40S_SHARDS (${EXPECTED_NUM_SHARDS})." >&2
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3-14B}"
MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
ENV_NAME="${ENV_NAME:-both}"
NUM_STATES="${NUM_STATES:-40}"
SAMPLES_Y="${SAMPLES_Y:-8}"
GAMES_Z="${GAMES_Z:-20}"
BASE_SEED="${BASE_SEED:-20260509}"
ABLATION_ORIGINAL_VS_UPDATED="${ABLATION_ORIGINAL_VS_UPDATED:-1}"
UPDATER_OBJECTIVE_VARIANT="${UPDATER_OBJECTIVE_VARIANT:-baseline}"
ENABLE_PLOTS="${ENABLE_PLOTS:-0}"
FROZEN_LAKE_TRACE="${FROZEN_LAKE_TRACE:-}"
SOKOBAN_TRACE="${SOKOBAN_TRACE:-}"

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-128G}"
TIME_LIMIT="${TIME_LIMIT:-36:00:00}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"

JOB_SUFFIX="${JOB_SUFFIX:-k${NUM_STATES}-y${SAMPLES_Y}-z${GAMES_Z}-seed${BASE_SEED}}"
if [[ "${ABLATION_ORIGINAL_VS_UPDATED}" -eq 1 ]]; then
  JOB_NAME="${JOB_NAME:-passk-${MODEL_TAG}-nothink-ablate-orig-updated-${JOB_SUFFIX}}"
else
  JOB_NAME="${JOB_NAME:-passk-${MODEL_TAG}-nothink-${JOB_SUFFIX}}"
fi
LOG_DIR="${REPO_DIR}/logs"
RUNS_DIR="${REPO_DIR}/single_turn_passk_runs"

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

echo "Preflight:"
test -f "${REPO_DIR}/single_turn_passk_eval.py"
if [[ -n "${FROZEN_LAKE_TRACE}" ]]; then
  test -f "${FROZEN_LAKE_TRACE}"
else
  test -f "${REPO_DIR}/llm_calls_notebook_minimal_frozen_lake.jsonl"
fi
if [[ -n "${SOKOBAN_TRACE}" ]]; then
  test -f "${SOKOBAN_TRACE}"
else
  test -f "${REPO_DIR}/llm_calls_notebook_minimal_sokoban.jsonl"
fi
test -x "${UV}"
test -x "${SGLANG}"

submit_array() {
  local label="$1"
  local array_spec="$2"
  local partition="$3"
  local gpu_request="$4"

  SBATCH_CMD=(
  sbatch
  --job-name="${JOB_NAME}"
  --account="${ACCOUNT}"
  --partition="${partition}"
  --array="${array_spec}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --gpus="${gpu_request}"
  --mem="${MEM}"
  --time="${TIME_LIMIT}"
  --output="${LOG_DIR}/%x-%A_%a.out"
  --error="${LOG_DIR}/%x-%A_%a.err"
  --wrap="
    set -euo pipefail

    cd '${REPO_DIR}'

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export NO_PROXY=localhost,127.0.0.1
    export no_proxy=localhost,127.0.0.1

    if [ -f /etc/profile.d/lmod.sh ]; then
      source /etc/profile.d/lmod.sh
    elif [ -f /usr/share/lmod/lmod/init/bash ]; then
      source /usr/share/lmod/lmod/init/bash
    elif [ -f /sw/lmod/lmod/init/bash ]; then
      source /sw/lmod/lmod/init/bash
    fi

    module load cuda/12.4.1 >/dev/null 2>&1 || true
    module load gcc/13.2.0 >/dev/null 2>&1 || true

    export CUDA_HOME=/sw/cuda/12.4.1
    export CUDA_PATH=/sw/cuda/12.4.1
    export PATH=/sw/gcc/13.2.0/bin:/sw/cuda/12.4.1/bin:/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin:\$PATH
    export LD_LIBRARY_PATH=/sw/gcc/13.2.0/lib64:/sw/cuda/12.4.1/lib64:\${LD_LIBRARY_PATH:-}
    export CC=/sw/gcc/13.2.0/bin/gcc
    export CXX=/sw/gcc/13.2.0/bin/g++
    export CUDAHOSTCXX=/sw/gcc/13.2.0/bin/g++
    export HF_HOME=/gscratch/h2lab/mohanc3/hf-cache
    export TRANSFORMERS_CACHE=/gscratch/h2lab/mohanc3/hf-cache
    export HF_DATASETS_CACHE=/gscratch/h2lab/mohanc3/hf-cache
    export WANDB_DIR=/gscratch/h2lab/mohanc3/wandb
    export XDG_CACHE_HOME=/gscratch/h2lab/mohanc3/xdg-cache
    export TORCH_EXTENSIONS_DIR=/gscratch/h2lab/mohanc3/torch-extensions
    export TVM_FFI_CACHE_DIR=/gscratch/h2lab/mohanc3/tvm-ffi-cache
    export PYTHONUNBUFFERED=1

    mkdir -p \"\$HF_HOME\" \"\$WANDB_DIR\" \"\$XDG_CACHE_HOME\" \"\$TORCH_EXTENSIONS_DIR\" \"\$TVM_FFI_CACHE_DIR\"

    SHARD_INDEX=\${SLURM_ARRAY_TASK_ID}
    PORT=\$((30000 + SHARD_INDEX))
    RUN_NAME=\"\${SLURM_JOB_NAME}-\${SLURM_ARRAY_JOB_ID}-shard\${SHARD_INDEX}\"
    RUN_DIR='${RUNS_DIR}'/\"\${RUN_NAME}\"
    SGLANG_LOG='${LOG_DIR}'/\"\${RUN_NAME}-sglang_server.log\"

    echo \"Job: \$SLURM_JOB_NAME array=\$SLURM_ARRAY_JOB_ID task=\$SHARD_INDEX\"
    echo \"Node: \$(hostname)\"
    echo \"Model: ${MODEL}\"
    echo \"GPU group: ${label}\"
    echo \"Partition/account: ${partition}/${ACCOUNT}\"
    echo \"GPU request: ${gpu_request}\"
    echo \"Shard: \$SHARD_INDEX / ${NUM_SHARDS}\"
    echo \"Ablation original-vs-updated: ${ABLATION_ORIGINAL_VS_UPDATED}\"
    echo \"Updater objective variant: ${UPDATER_OBJECTIVE_VARIANT}\"
    echo \"Base seed: ${BASE_SEED}\"
    echo \"Port: \$PORT\"
    echo \"Run dir: \$RUN_DIR\"

    '${SGLANG}' serve \
      --model-path '${MODEL}' \
      --host 127.0.0.1 \
      --port \"\$PORT\" \
      --mem-fraction-static '${SGLANG_MEM_FRACTION}' \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      --attention-backend triton \
      --sampling-backend pytorch \
      > \"\$SGLANG_LOG\" 2>&1 &

    SERVER_PID=\$!

    cleanup() {
      kill \$SERVER_PID 2>/dev/null || true
    }
    trap cleanup EXIT

    echo \"Waiting for SGLang server...\"
    for i in \$(seq 1 120); do
      if ! kill -0 \$SERVER_PID 2>/dev/null; then
        echo \"ERROR: SGLang server exited before becoming ready\"
        tail -n 120 \"\$SGLANG_LOG\" || true
        exit 1
      fi
      if curl --noproxy \"*\" -s \"http://127.0.0.1:\$PORT/v1/models\" >/dev/null 2>&1; then
        echo \"Server ready after \$((i * 5)) seconds\"
        break
      fi
      sleep 5
    done

    if ! curl --noproxy \"*\" -s \"http://127.0.0.1:\$PORT/v1/models\" >/dev/null 2>&1; then
      echo \"ERROR: SGLang server failed to start\"
      tail -n 120 \"\$SGLANG_LOG\" || true
      exit 1
    fi

    EVAL_ARGS=()
    if [ '${ABLATION_ORIGINAL_VS_UPDATED}' -eq 1 ]; then
      EVAL_ARGS+=(--ablation-original-vs-updated)
      EVAL_ARGS+=(--updater-objective-variant '${UPDATER_OBJECTIVE_VARIANT}')
    fi
    if [ '${ENABLE_PLOTS}' -ne 1 ]; then
      EVAL_ARGS+=(--no-plots)
    fi
    if [ -n '${FROZEN_LAKE_TRACE}' ]; then
      EVAL_ARGS+=(--frozen-lake-trace '${FROZEN_LAKE_TRACE}')
    fi
    if [ -n '${SOKOBAN_TRACE}' ]; then
      EVAL_ARGS+=(--sokoban-trace '${SOKOBAN_TRACE}')
    fi

    '${UV}' run python '${REPO_DIR}/single_turn_passk_eval.py' \
      --env '${ENV_NAME}' \
      --num-states '${NUM_STATES}' \
      --samples-y '${SAMPLES_Y}' \
      --games-z '${GAMES_Z}' \
      --base-seed '${BASE_SEED}' \
      --model '${MODEL}' \
      --server \"http://127.0.0.1:\$PORT/v1\" \
      --disable-thinking \
      --num-shards '${NUM_SHARDS}' \
      --shard-index \"\$SHARD_INDEX\" \
      --outputs-dir '${RUNS_DIR}' \
      --run-name \"\$RUN_NAME\" \
      \"\${EVAL_ARGS[@]}\"

    echo \"Done. RUN_DIR=\$RUN_DIR\"
  "
  )

  echo "Submitting ${JOB_NAME}: group=${label}, array=${array_spec}, account=${ACCOUNT}, partition=${partition}, gpu=${gpu_request}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q ' "${SBATCH_CMD[@]}"
    echo
  else
    "${SBATCH_CMD[@]}"
  fi
}

submit_group() {
  local label="$1"
  local count="$2"
  local max_concurrent="$3"
  local start="$4"
  local partition="$5"
  local gpu_request="$6"

  if [[ "${count}" -le 0 ]]; then
    return 0
  fi
  local end=$((start + count - 1))
  local array_spec="${start}-${end}%${max_concurrent}"
  submit_array "${label}" "${array_spec}" "${partition}" "${gpu_request}"
}

NEXT_SHARD=0
submit_group "a100" "${A100_SHARDS}" "${A100_ARRAY_MAX_CONCURRENT}" "${NEXT_SHARD}" "${A100_PARTITION}" "${A100_GPU_REQUEST}"
NEXT_SHARD=$((NEXT_SHARD + A100_SHARDS))

submit_group "l40" "${L40_SHARDS}" "${L40_ARRAY_MAX_CONCURRENT}" "${NEXT_SHARD}" "${L40_PARTITION}" "${L40_GPU_REQUEST}"
NEXT_SHARD=$((NEXT_SHARD + L40_SHARDS))

submit_group "l40s" "${L40S_SHARDS}" "${L40S_ARRAY_MAX_CONCURRENT}" "${NEXT_SHARD}" "${L40S_PARTITION}" "${L40S_GPU_REQUEST}"
