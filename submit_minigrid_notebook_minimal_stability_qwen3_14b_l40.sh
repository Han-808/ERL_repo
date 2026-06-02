#!/usr/bin/env bash
set -euo pipefail

# MiniGrid notebook_minimal updater stability eval.
#
# Shape:
#   - 4 L40 jobs, one focused notebook_minimal game per GPU
#   - Qwen/Qwen3-14B, disable thinking
#   - For each source notebook_minimal updater prompt:
#       sample the saved updater prompt 4 times
#       apply sampled notebook edit operations to the source notebook
#       evaluate the updated notebook on 10 fixed MiniGrid instances
#       evaluate MemoryS11/MemoryS13 on 5 fixed instances
#
# Output:
#   ${OUTPUTS_ROOT}/${RUN_TAG}/${game}/
#     config.json
#     updater_samples.jsonl
#     rollouts.jsonl
#     sample_summary.csv
#     status.tsv
#
# Usage:
#   bash submit_minigrid_notebook_minimal_stability_qwen3_14b_l40.sh --dry-run
#   bash submit_minigrid_notebook_minimal_stability_qwen3_14b_l40.sh

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
PARTITION="${PARTITION:-gpu-l40}"
GPU_REQUEST="${GPU_REQUEST:-l40:1}"
ARRAY_MAX_CONCURRENT="${ARRAY_MAX_CONCURRENT:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-128G}"
TIME_LIMIT="${TIME_LIMIT:-96:00:00}"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_DIR}/minigrid_notebook_minimal_llm_calls}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/minigrid_notebook_minimal_stability}"

SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
LONG_EVAL_EPISODES="${LONG_EVAL_EPISODES:-5}"
BASE_SEED="${BASE_SEED:-20260601}"
SOURCE_SEED="${SOURCE_SEED:-0}"
SEED_STRIDE="${SEED_STRIDE:-10000}"
UPDATER_MAX_TOKENS="${UPDATER_MAX_TOKENS:-8192}"

SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
PORT_BASE="${PORT_BASE:-32100}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-minigrid-notebook-minimal-stability-${MODEL_TAG}-nothink}"
RUN_TAG="${RUN_TAG:-${JOB_NAME_PREFIX}-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"

DRY_RUN=0
MODE="submit"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --worker) MODE="worker" ;;
    *)
      echo "ERROR: unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

game_for_task() {
  case "$1" in
    0) echo "minigrid_empty_random_5x5" ;;
    1) echo "minigrid_memorys11" ;;
    2) echo "minigrid_memorys13" ;;
    3) echo "minigrid_fourrooms" ;;
    *)
      echo "ERROR: task id must be 0..3, got $1" >&2
      return 1
      ;;
  esac
}

eval_count_for_game() {
  case "$1" in
    minigrid_memorys11|minigrid_memorys13) echo "${LONG_EVAL_EPISODES}" ;;
    *) echo "${EVAL_EPISODES}" ;;
  esac
}

source_file_for_game() {
  local game="$1"
  find "${SOURCE_ROOT}" -type f \
    -name "llm_calls_notebook_minimal_minigrid_${game}.jsonl" \
    2>/dev/null | sort | head -n 1
}

source_episode_count_for_game() {
  local game="$1"
  local file
  file="$(source_file_for_game "${game}")"
  if [[ -z "${file}" ]]; then
    echo 0
    return
  fi
  grep -c "You are a notebook updater for an agent playing a grid puzzle" "${file}" || true
}

setup_hyak_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  unset LLM_TRACE_PATH
  export NO_PROXY=localhost,127.0.0.1
  export no_proxy=localhost,127.0.0.1

  set +u
  if [[ -f /etc/profile.d/lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
  elif [[ -f /sw/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /sw/lmod/lmod/init/bash
  fi

  module load cuda/12.4.1 >/dev/null 2>&1 || true
  module load gcc/13.2.0 >/dev/null 2>&1 || true
  set -u

  export CUDA_HOME=/sw/cuda/12.4.1
  export CUDA_PATH=/sw/cuda/12.4.1
  export PATH=/sw/gcc/13.2.0/bin:/sw/cuda/12.4.1/bin:/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin:${PATH}
  export LD_LIBRARY_PATH=/sw/gcc/13.2.0/lib64:/sw/cuda/12.4.1/lib64:${LD_LIBRARY_PATH:-}
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

  mkdir -p \
    "${HF_HOME}" \
    "${WANDB_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TVM_FFI_CACHE_DIR}"
}

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local sglang_log="$3"

  echo "Waiting for SGLang server..."
  for i in $(seq 1 240); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "ERROR: SGLang server exited before becoming ready"
      tail -n 120 "${sglang_log}" || true
      return 1
    fi
    if curl --noproxy "*" -s "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "Server ready after $((i * 5)) seconds"
      return 0
    fi
    sleep 5
  done

  echo "ERROR: SGLang server failed to start"
  tail -n 120 "${sglang_log}" || true
  return 1
}

run_worker() {
  local task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  local game
  local port
  local run_dir
  local server_pid
  local sglang_log
  local source_file

  game="$(game_for_task "${task_id}")"
  source_file="$(source_file_for_game "${game}")"
  if [[ -z "${source_file}" ]]; then
    echo "ERROR: missing notebook_minimal llm_calls for ${game} under ${SOURCE_ROOT}" >&2
    exit 1
  fi

  port=$((PORT_BASE + task_id))
  run_dir="${OUTPUTS_ROOT}/${RUN_TAG}/${game}"
  sglang_log="${LOG_DIR}/${RUN_TAG}-${game}-${SLURM_ARRAY_JOB_ID}_${task_id}-sglang_server.log"

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${run_dir}"

  echo "Job: ${SLURM_JOB_NAME} array=${SLURM_ARRAY_JOB_ID} task=${task_id}"
  echo "Node: $(hostname)"
  echo "Game: ${game}"
  echo "Model: ${MODEL}"
  echo "Source root: ${SOURCE_ROOT}"
  echo "Source file: ${source_file}"
  echo "Output: ${run_dir}"
  echo "Samples per source episode: ${SAMPLES_PER_EPISODE}"
  echo "Eval episodes for this game: $(eval_count_for_game "${game}")"
  echo "Updater max tokens: ${UPDATER_MAX_TOKENS}"
  echo "Port: ${port}"

  "${SGLANG}" serve \
    --model-path "${MODEL}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --mem-fraction-static "${SGLANG_MEM_FRACTION}" \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph \
    --attention-backend triton \
    --sampling-backend pytorch \
    > "${sglang_log}" 2>&1 &

  server_pid=$!

  cleanup() {
    kill "${server_pid}" 2>/dev/null || true
  }
  trap cleanup EXIT

  wait_for_server "${port}" "${server_pid}" "${sglang_log}"

  "${UV}" run python "${REPO_DIR}/minigrid_notebook_minimal_stability_eval.py" \
    --game "${game}" \
    --source-root "${SOURCE_ROOT}" \
    --outputs-root "${OUTPUTS_ROOT}" \
    --run-name "${RUN_TAG}" \
    --model "${MODEL}" \
    --server "http://127.0.0.1:${port}/v1" \
    --samples-per-episode "${SAMPLES_PER_EPISODE}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --long-eval-episodes "${LONG_EVAL_EPISODES}" \
    --base-seed "${BASE_SEED}" \
    --source-seed "${SOURCE_SEED}" \
    --seed-stride "${SEED_STRIDE}" \
    --updater-max-tokens "${UPDATER_MAX_TOKENS}" \
    --disable-thinking

  echo "Done. OUTPUT_DIR=${run_dir}"
}

print_plan() {
  echo "Run tag: ${RUN_TAG}"
  echo "Model: ${MODEL}"
  echo "Partition: ${PARTITION}"
  echo "GPU request: ${GPU_REQUEST}"
  echo "Output root: ${OUTPUTS_ROOT}"
  echo "Source root: ${SOURCE_ROOT}"
  echo "Samples per source episode: ${SAMPLES_PER_EPISODE}"
  echo "Eval episodes: ${EVAL_EPISODES}; long eval episodes: ${LONG_EVAL_EPISODES}"
  echo "Base seed: ${BASE_SEED}"
  echo "Updater max tokens: ${UPDATER_MAX_TOKENS}"
  echo
  echo "=== Game assignments ==="

  local task_id game source_eps eval_eps samples rollouts source_file
  local total_samples=0
  local total_rollouts=0
  for task_id in 0 1 2 3; do
    game="$(game_for_task "${task_id}")"
    source_file="$(source_file_for_game "${game}")"
    source_eps="$(source_episode_count_for_game "${game}")"
    eval_eps="$(eval_count_for_game "${game}")"
    samples=$((source_eps * SAMPLES_PER_EPISODE))
    rollouts=$((samples * eval_eps))
    total_samples=$((total_samples + samples))
    total_rollouts=$((total_rollouts + rollouts))
    printf "task=%s game=%-34s source_episodes=%-4s updater_samples=%-5s eval_rollouts=%-6s source=%s\n" \
      "${task_id}" "${game}" "${source_eps}" "${samples}" "${rollouts}" "${source_file:-MISSING}"
  done

  echo
  echo "Expected updater samples: ${total_samples}"
  echo "Expected eval rollouts: ${total_rollouts}"
}

submit() {
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}"

  echo "Preflight:"
  test -f "${REPO_DIR}/minigrid_notebook_minimal_stability_eval.py"
  test -f "${REPO_DIR}/methods/notebook_minimal.py"
  test -f "${REPO_DIR}/environments/minigrid_env.py"
  test -d "${SOURCE_ROOT}"
  test -x "${UV}"
  test -x "${SGLANG}"
  for task_id in 0 1 2 3; do
    local game
    local source_file
    game="$(game_for_task "${task_id}")"
    source_file="$(source_file_for_game "${game}")"
    if [[ -z "${source_file}" ]]; then
      echo "ERROR: missing source llm_calls for ${game}" >&2
      exit 1
    fi
  done
  echo "  helper ok"
  echo "  source root: ${SOURCE_ROOT}"

  local sbatch_cmd=(
    sbatch
    --job-name="${RUN_TAG}"
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --array="0-3%${ARRAY_MAX_CONCURRENT}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --gpus="${GPU_REQUEST}"
    --mem="${MEM}"
    --time="${TIME_LIMIT}"
    --output="${LOG_DIR}/%x-%A_%a.out"
    --error="${LOG_DIR}/%x-%A_%a.err"
    --export=ALL,RUN_TAG="${RUN_TAG}"
    "${REPO_DIR}/submit_minigrid_notebook_minimal_stability_qwen3_14b_l40.sh"
    --worker
  )

  echo
  echo "Submitting ${RUN_TAG}: array=0-3%${ARRAY_MAX_CONCURRENT}, partition=${PARTITION}, gpu=${GPU_REQUEST}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q ' "${sbatch_cmd[@]}"
    echo
  else
    "${sbatch_cmd[@]}"
  fi
}

if [[ "${MODE}" == "worker" ]]; then
  run_worker
else
  print_plan
  submit
fi
