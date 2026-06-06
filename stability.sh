#!/usr/bin/env bash
set -euo pipefail

# ACE_ONCE-only MiniGrid stability run with a larger updater budget.
#
# Generator calls keep the default call_lm max_tokens=512.
# ACE_ONCE merged updater calls use ACE_ONCE_UPDATER_MAX_TOKENS=8192 in
# methods/ace_once.py.
#
# Topology defaults use two SGLang agent servers for Qwen3-8B plus one
# Qwen3.5-27B updater server:
#   - one 4xA100 Qwen3-8B agent server with dp=4
#   - one 2xH200 Qwen3-8B agent server with dp=2
#   - one 2xH200 Qwen3.5-27B updater server with dp=2
# The 36 fixed-seed stability workers are split heavy-first across both
# servers so Memory/SimpleCrossing tasks do not create a long single-server tail.
#
# Stability matrix:
#   - same MiniGrid games, fixed seed, max_steps, rewards, and method
#   - Qwen3-8B generator/agent, Qwen3.5-27B updater, no-thinking,
#     temperature=1.0
#   - one fixed seed per game config
#   - each game config is repeated STABILITY_REPEATS times
#   - MemoryS11/S13 use 20 episodes for this stability run
#   - default: 6 configs x 6 repeats = 36 runs
#
# Usage:
#   bash stability.sh --dry-run
#   bash stability.sh
#
# Advanced entrypoints used by Slurm:
#   bash stability.sh --server a100
#   bash stability.sh --server h200
#   bash stability.sh --server updater
#   bash stability.sh --worker a100
#   bash stability.sh --worker h200

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
AGENT_MODEL="${AGENT_MODEL:-${MODEL:-Qwen/Qwen3-8B}}"
AGENT_MODEL_TAG="${AGENT_MODEL_TAG:-${MODEL_TAG:-qwen3-8b}}"
UPDATER_MODEL="${UPDATER_MODEL:-Qwen/Qwen3.5-27B}"
UPDATER_MODEL_TAG="${UPDATER_MODEL_TAG:-qwen35-27b}"
MODEL="${AGENT_MODEL}"
MODEL_TAG="${AGENT_MODEL_TAG}"
METHOD="${METHOD:-ace_once_minigrid}"
SEED_STRIDE="${SEED_STRIDE:-10000}"
UPDATER_MAX_TOKENS_EXPECTED="${UPDATER_MAX_TOKENS_EXPECTED:-8192}"
STABILITY_REPEATS="${STABILITY_REPEATS:-6}"
STABILITY_SEED="${STABILITY_SEED:-0}"
LM_TEMPERATURE="${LM_TEMPERATURE:-1.0}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-minigrid-stability-${AGENT_MODEL_TAG}-agent-${UPDATER_MODEL_TAG}-updater8192}"
RUN_TAG="${RUN_TAG:-${JOB_NAME_PREFIX}-$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/minigrid_stability_updater8192}"
READY_DIR="${READY_DIR:-${LOG_DIR}/minigrid_stability_updater8192_ready/${RUN_TAG}}"

A100_PARTITION="${A100_PARTITION:-gpu-a100}"
A100_GPU_REQUEST="${A100_GPU_REQUEST:-a100:4}"
A100_DP_SIZE="${A100_DP_SIZE:-4}"
A100_PORT="${A100_PORT:-31000}"
A100_SERVER_CPUS="${A100_SERVER_CPUS:-32}"
A100_SERVER_MEM="${A100_SERVER_MEM:-256G}"
A100_SERVER_TIME="${A100_SERVER_TIME:-96:00:00}"
A100_WORKER_ARRAY="${A100_WORKER_ARRAY:-0-15}"
A100_WORKER_MAX_CONCURRENT="${A100_WORKER_MAX_CONCURRENT:-16}"

H200_PARTITION="${H200_PARTITION:-gpu-h200}"
H200_GPU_REQUEST="${H200_GPU_REQUEST:-h200:2}"
H200_DP_SIZE="${H200_DP_SIZE:-2}"
H200_PORT="${H200_PORT:-31100}"
H200_SERVER_CPUS="${H200_SERVER_CPUS:-24}"
H200_SERVER_MEM="${H200_SERVER_MEM:-256G}"
H200_SERVER_TIME="${H200_SERVER_TIME:-96:00:00}"
H200_WORKER_ARRAY="${H200_WORKER_ARRAY:-0-19}"
H200_WORKER_MAX_CONCURRENT="${H200_WORKER_MAX_CONCURRENT:-20}"

UPDATER_PARTITION="${UPDATER_PARTITION:-gpu-h200}"
UPDATER_GPU_REQUEST="${UPDATER_GPU_REQUEST:-h200:2}"
UPDATER_DP_SIZE="${UPDATER_DP_SIZE:-2}"
UPDATER_PORT="${UPDATER_PORT:-31200}"
UPDATER_SERVER_CPUS="${UPDATER_SERVER_CPUS:-24}"
UPDATER_SERVER_MEM="${UPDATER_SERVER_MEM:-256G}"
UPDATER_SERVER_TIME="${UPDATER_SERVER_TIME:-96:00:00}"

WORKER_PARTITION="${WORKER_PARTITION:-gpu-l40}"
WORKER_CPUS_PER_TASK="${WORKER_CPUS_PER_TASK:-1}"
WORKER_MEM="${WORKER_MEM:-4G}"
WORKER_TIME="${WORKER_TIME:-96:00:00}"
CLEANUP_PARTITION="${CLEANUP_PARTITION:-${WORKER_PARTITION}}"

SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
SERVER_READY_WAIT_SECONDS="${SERVER_READY_WAIT_SECONDS:-3600}"
WORKER_READY_WAIT_SECONDS="${WORKER_READY_WAIT_SECONDS:-3600}"
READY_POLL_SECONDS="${READY_POLL_SECONDS:-10}"

DRY_RUN=0
MODE="submit"
POOL=""

setup_hyak_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  export NO_PROXY=localhost,127.0.0.1
  export no_proxy=localhost,127.0.0.1

  # Hyak's lmod init can reference Slurm-only variables on login nodes.
  # Keep nounset for our script, but disable it while initializing modules.
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

ready_file_for_pool() {
  case "$1" in
    a100) echo "${READY_DIR}/a100.url" ;;
    h200) echo "${READY_DIR}/h200.url" ;;
    updater) echo "${READY_DIR}/updater.url" ;;
    *) echo "ERROR: unknown pool '$1'" >&2; return 1 ;;
  esac
}

server_port_for_pool() {
  case "$1" in
    a100) echo "${A100_PORT}" ;;
    h200) echo "${H200_PORT}" ;;
    updater) echo "${UPDATER_PORT}" ;;
    *) echo "ERROR: unknown pool '$1'" >&2; return 1 ;;
  esac
}

server_dp_for_pool() {
  case "$1" in
    a100) echo "${A100_DP_SIZE}" ;;
    h200) echo "${H200_DP_SIZE}" ;;
    updater) echo "${UPDATER_DP_SIZE}" ;;
    *) echo "ERROR: unknown pool '$1'" >&2; return 1 ;;
  esac
}

server_model_for_pool() {
  case "$1" in
    a100|h200) echo "${AGENT_MODEL}" ;;
    updater) echo "${UPDATER_MODEL}" ;;
    *) echo "ERROR: unknown pool '$1'" >&2; return 1 ;;
  esac
}

base_task_count_for_pool() {
  case "$1" in
    a100) echo 16 ;;
    h200) echo 20 ;;
    *) echo "ERROR: unknown pool '$1'" >&2; return 1 ;;
  esac
}

max_task_id_for_pool() {
  local base_count
  base_count="$(base_task_count_for_pool "$1")"
  echo $((base_count - 1))
}

assignment_record() {
  local pool="$1"
  local task_id="$2"
  local base_count

  base_count="$(base_task_count_for_pool "${pool}")"
  if (( task_id < 0 || task_id >= base_count )); then
    echo "ERROR: ${pool} worker task_id must be 0..$((base_count - 1)), got ${task_id}" >&2
    return 1
  fi

  case "${pool}:${task_id}" in
    # A100 receives 16 workers: a mixed set with Memory/SimpleCrossing plus
    # enough short tasks to keep the 4-GPU agent server busy.
    a100:0)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|0" ;;
    a100:1)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|1" ;;
    a100:2)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|2" ;;
    a100:3)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|0" ;;
    a100:4)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|1" ;;
    a100:5)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|2" ;;
    a100:6)  echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|0" ;;
    a100:7)  echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|1" ;;
    a100:8)  echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|0" ;;
    a100:9)  echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|1" ;;
    a100:10) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|2" ;;
    a100:11) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|3" ;;
    a100:12) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|4" ;;
    a100:13) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|5" ;;
    a100:14) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|0" ;;
    a100:15) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|1" ;;

    # H200 receives 20 workers: more request-level parallelism plus the rest of
    # Memory/SimpleCrossing, matching the H200 underfeeding takeaway.
    h200:0)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|3" ;;
    h200:1)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|4" ;;
    h200:2)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|5" ;;
    h200:3)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|3" ;;
    h200:4)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|4" ;;
    h200:5)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|5" ;;
    h200:6)  echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|2" ;;
    h200:7)  echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|3" ;;
    h200:8)  echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|4" ;;
    h200:9)  echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|5" ;;
    h200:10) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|2" ;;
    h200:11) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|3" ;;
    h200:12) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|4" ;;
    h200:13) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|5" ;;
    h200:14) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|0" ;;
    h200:15) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|1" ;;
    h200:16) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|2" ;;
    h200:17) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|3" ;;
    h200:18) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|4" ;;
    h200:19) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|5" ;;
    *) echo "ERROR: unknown ${pool} task_id ${task_id}" >&2; return 1 ;;
  esac
}

load_worker_assignment() {
  local record
  local pool="$1"
  local task_id="$2"

  ENV_ID=""
  EPISODES=""
  MAX_STEPS=""
  ENV_LABEL=""
  SEED=""
  HISTORICAL_GEN_CALLS=""
  BASE_TASK_ID=""
  REPEAT=""

  record="$(assignment_record "${pool}" "${task_id}")"
  IFS='|' read -r ENV_ID EPISODES MAX_STEPS ENV_LABEL HISTORICAL_GEN_CALLS SEED BASE_TASK_ID REPEAT <<< "${record}"
}

print_assignments_for_pool() {
  local pool="$1"
  local max_task task_id seed_offset
  max_task="$(max_task_id_for_pool "${pool}")"

  echo "=== ${pool} worker assignments ==="
  for task_id in $(seq 0 "${max_task}"); do
    load_worker_assignment "${pool}" "${task_id}"
    seed_offset=$((SEED * SEED_STRIDE))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${pool}" \
      "${task_id}" \
      "${METHOD}" \
      "${ENV_ID}" \
      "${EPISODES}" \
      "${MAX_STEPS}" \
      "${SEED}" \
      "${REPEAT}" \
      "${seed_offset}" \
      "${ENV_LABEL}" \
      "${BASE_TASK_ID}" \
      "${HISTORICAL_GEN_CALLS}"
  done
}

print_assignments() {
  print_assignments_for_pool a100
  print_assignments_for_pool h200
  echo
  echo "Base configs: 6"
  echo "Agent model: ${AGENT_MODEL}"
  echo "Updater model: ${UPDATER_MODEL}"
  echo "Server topology: 4 x A100 agent dp=${A100_DP_SIZE} + 2 x H200 agent dp=${H200_DP_SIZE} + 2 x H200 updater dp=${UPDATER_DP_SIZE}"
  echo "A100 worker concurrency: ${A100_WORKER_ARRAY}%${A100_WORKER_MAX_CONCURRENT}"
  echo "H200 worker concurrency: ${H200_WORKER_ARRAY}%${H200_WORKER_MAX_CONCURRENT}"
  echo "Stability seed: ${STABILITY_SEED}"
  echo "Stability repeats per config: ${STABILITY_REPEATS}"
  echo "LM temperature: ${LM_TEMPERATURE}"
  echo "Expected runs: $((6 * STABILITY_REPEATS))"
  echo "Expected episodes/updater calls: $((140 * STABILITY_REPEATS))"
  echo "Expected result files: $((6 * STABILITY_REPEATS))"
  echo "Expected llm_calls files: $((6 * STABILITY_REPEATS))"
  echo "Expected ACE_ONCE updater max_tokens: ${UPDATER_MAX_TOKENS_EXPECTED}"
}

smoke_env() {
  "${UV}" run python -c \
    "import sys; from environments.minigrid_env import MiniGridTextEnv; env=MiniGridTextEnv(sys.argv[1], max_steps=int(sys.argv[2])); obs=env.reset(seed=1); assert f'Step: 0/{sys.argv[2]}' in obs, obs; env.close(); print(f'MiniGrid smoke ok: {sys.argv[1]} max_steps={sys.argv[2]}')" \
    "${ENV_ID}" \
    "${MAX_STEPS}"
}

wait_for_local_server() {
  local port="$1"
  local server_pid="$2"
  local server_log="$3"
  local waited=0

  echo "Waiting for local SGLang server on port ${port}..."
  while (( waited < SERVER_READY_WAIT_SECONDS )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "ERROR: SGLang server exited before becoming ready"
      tail -n 160 "${server_log}" || true
      return 1
    fi
    if curl --noproxy "*" -s "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "Server ready after ${waited} seconds"
      return 0
    fi
    sleep "${READY_POLL_SECONDS}"
    waited=$((waited + READY_POLL_SECONDS))
  done

  echo "ERROR: SGLang server failed to start within ${SERVER_READY_WAIT_SECONDS}s"
  tail -n 160 "${server_log}" || true
  return 1
}

wait_for_remote_server_url() {
  local ready_file="$1"
  local waited=0
  local server_url=""

  echo "Waiting for ready file ${ready_file}..." >&2
  while (( waited < WORKER_READY_WAIT_SECONDS )); do
    if [[ -s "${ready_file}" ]]; then
      server_url="$(<"${ready_file}")"
      if curl --noproxy "*" -s "${server_url}/models" >/dev/null 2>&1; then
        echo "${server_url}"
        return 0
      fi
      echo "Ready file exists, but server is not reachable yet: ${server_url}" >&2
    fi
    sleep "${READY_POLL_SECONDS}"
    waited=$((waited + READY_POLL_SECONDS))
  done

  echo "ERROR: server was not ready within ${WORKER_READY_WAIT_SECONDS}s" >&2
  return 1
}

run_server() {
  local pool="$1"
  local port dp model ready_file server_log server_pid host server_url

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}"

  port="$(server_port_for_pool "${pool}")"
  dp="$(server_dp_for_pool "${pool}")"
  model="$(server_model_for_pool "${pool}")"
  ready_file="$(ready_file_for_pool "${pool}")"
  server_log="${LOG_DIR}/${RUN_TAG}-${pool}-sglang_server.log"
  rm -f "${ready_file}"

  echo "Server pool: ${pool}"
  echo "Node: $(hostname)"
  echo "Model: ${model}"
  echo "Port: ${port}"
  echo "DP size: ${dp}"
  echo "Ready file: ${ready_file}"
  echo "Server log: ${server_log}"

  "${SGLANG}" serve \
    --model-path "${model}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --dp-size "${dp}" \
    --mem-fraction-static "${SGLANG_MEM_FRACTION}" \
    --disable-cuda-graph \
    --disable-piecewise-cuda-graph \
    --attention-backend triton \
    --sampling-backend pytorch \
    > "${server_log}" 2>&1 &

  server_pid=$!

  cleanup() {
    kill "${server_pid}" 2>/dev/null || true
  }
  trap cleanup EXIT

  wait_for_local_server "${port}" "${server_pid}" "${server_log}"

  host="$(hostname -f 2>/dev/null || hostname)"
  server_url="http://${host}:${port}/v1"
  printf "%s\n" "${server_url}" > "${ready_file}"
  echo "Server URL written: ${server_url}"

  wait "${server_pid}"
}

run_worker() {
  local pool="$1"
  local task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  local ready_file updater_ready_file server_url updater_server_url seed_offset status_file run_label outputs_dir

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}"

  load_worker_assignment "${pool}" "${task_id}"
  seed_offset=$((SEED * SEED_STRIDE))
  ready_file="$(ready_file_for_pool "${pool}")"
  updater_ready_file="$(ready_file_for_pool updater)"
  status_file="${OUTPUTS_ROOT}/${RUN_TAG}-${pool}-${task_id}-status.tsv"
  run_label="${RUN_TAG}-${pool}-${task_id}-${METHOD}-${ENV_LABEL}-s${SEED}-r${REPEAT}"
  outputs_dir="${OUTPUTS_ROOT}/${run_label}"

  printf "pool\ttask_id\tbase_task_id\tenv_id\tseed\trepeat\ttemperature\tepisodes\tmax_steps\tseed_offset\thistorical_gen_calls\tserver_url\tstatus\toutputs_dir\tupdater_server_url\n" > "${status_file}"

  echo "Worker pool: ${pool}"
  echo "Task id: ${task_id}"
  echo "Assignment: method=${METHOD} env_id=${ENV_ID} episodes=${EPISODES} max_steps=${MAX_STEPS} seed=${SEED} repeat=${REPEAT} seed_offset=${seed_offset} base_task_id=${BASE_TASK_ID} historical_gen_calls=${HISTORICAL_GEN_CALLS}"
  echo "Outputs: ${outputs_dir}"

  smoke_env
  server_url="$(wait_for_remote_server_url "${ready_file}")"
  updater_server_url="$(wait_for_remote_server_url "${updater_ready_file}")"
  mkdir -p "${outputs_dir}"

  echo "Starting run server=${server_url} updater_server=${updater_server_url}"
  if "${UV}" run python "${REPO_DIR}/run.py" \
    --method "${METHOD}" \
    --env minigrid \
    --minigrid-id "${ENV_ID}" \
    --minigrid-max-steps "${MAX_STEPS}" \
    --seed-offset "${seed_offset}" \
    --episodes "${EPISODES}" \
    --model "${AGENT_MODEL}" \
    --server "${server_url}" \
    --updater-model "${UPDATER_MODEL}" \
    --updater-server "${updater_server_url}" \
    --outputs-dir "${outputs_dir}" \
    --disable-thinking; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tdone\t%s\t%s\n" \
      "${pool}" "${task_id}" "${BASE_TASK_ID}" "${ENV_ID}" "${SEED}" "${REPEAT}" "${LM_TEMPERATURE}" "${EPISODES}" "${MAX_STEPS}" "${seed_offset}" "${HISTORICAL_GEN_CALLS}" "${server_url}" "${outputs_dir}" "${updater_server_url}" >> "${status_file}"
    echo "Done. OUTPUTS_DIR=${outputs_dir}"
  else
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfailed\t%s\t%s\n" \
      "${pool}" "${task_id}" "${BASE_TASK_ID}" "${ENV_ID}" "${SEED}" "${REPEAT}" "${LM_TEMPERATURE}" "${EPISODES}" "${MAX_STEPS}" "${seed_offset}" "${HISTORICAL_GEN_CALLS}" "${server_url}" "${outputs_dir}" "${updater_server_url}" >> "${status_file}"
    echo "ERROR: run failed. OUTPUTS_DIR=${outputs_dir}" >&2
    return 1
  fi
}

script_path() {
  readlink -f "${BASH_SOURCE[0]}"
}

slurm_exports() {
  printf "ALL,REPO_DIR=%s,UV=%s,SGLANG=%s,RUN_TAG=%s,MODEL=%s,MODEL_TAG=%s,AGENT_MODEL=%s,AGENT_MODEL_TAG=%s,UPDATER_MODEL=%s,UPDATER_MODEL_TAG=%s,METHOD=%s,SEED_STRIDE=%s,STABILITY_REPEATS=%s,STABILITY_SEED=%s,LM_TEMPERATURE=%s,LOG_DIR=%s,OUTPUTS_ROOT=%s,READY_DIR=%s,A100_PORT=%s,H200_PORT=%s,UPDATER_PORT=%s,A100_DP_SIZE=%s,H200_DP_SIZE=%s,UPDATER_DP_SIZE=%s,SGLANG_MEM_FRACTION=%s,SERVER_READY_WAIT_SECONDS=%s,WORKER_READY_WAIT_SECONDS=%s,READY_POLL_SECONDS=%s,UPDATER_MAX_TOKENS_EXPECTED=%s" \
    "${REPO_DIR}" \
    "${UV}" \
    "${SGLANG}" \
    "${RUN_TAG}" \
    "${MODEL}" \
    "${MODEL_TAG}" \
    "${AGENT_MODEL}" \
    "${AGENT_MODEL_TAG}" \
    "${UPDATER_MODEL}" \
    "${UPDATER_MODEL_TAG}" \
    "${METHOD}" \
    "${SEED_STRIDE}" \
    "${STABILITY_REPEATS}" \
    "${STABILITY_SEED}" \
    "${LM_TEMPERATURE}" \
    "${LOG_DIR}" \
    "${OUTPUTS_ROOT}" \
    "${READY_DIR}" \
    "${A100_PORT}" \
    "${H200_PORT}" \
    "${UPDATER_PORT}" \
    "${A100_DP_SIZE}" \
    "${H200_DP_SIZE}" \
    "${UPDATER_DP_SIZE}" \
    "${SGLANG_MEM_FRACTION}" \
    "${SERVER_READY_WAIT_SECONDS}" \
    "${WORKER_READY_WAIT_SECONDS}" \
    "${READY_POLL_SECONDS}" \
    "${UPDATER_MAX_TOKENS_EXPECTED}"
}

print_sbatch_command() {
  printf '%q ' "$@"
  echo
}

submit_server_job() {
  local pool="$1"
  local partition gpu_request cpus mem time_limit job_name script
  script="$(script_path)"

  case "${pool}" in
    a100)
      partition="${A100_PARTITION}"
      gpu_request="${A100_GPU_REQUEST}"
      cpus="${A100_SERVER_CPUS}"
      mem="${A100_SERVER_MEM}"
      time_limit="${A100_SERVER_TIME}"
      ;;
    h200)
      partition="${H200_PARTITION}"
      gpu_request="${H200_GPU_REQUEST}"
      cpus="${H200_SERVER_CPUS}"
      mem="${H200_SERVER_MEM}"
      time_limit="${H200_SERVER_TIME}"
      ;;
    updater)
      partition="${UPDATER_PARTITION}"
      gpu_request="${UPDATER_GPU_REQUEST}"
      cpus="${UPDATER_SERVER_CPUS}"
      mem="${UPDATER_SERVER_MEM}"
      time_limit="${UPDATER_SERVER_TIME}"
      ;;
    *) echo "ERROR: unknown server pool '${pool}'" >&2; return 1 ;;
  esac

  job_name="${RUN_TAG}-${pool}-server"
  local cmd=(
    sbatch
    --parsable
    --job-name="${job_name}"
    --account="${ACCOUNT}"
    --partition="${partition}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${cpus}"
    --gpus="${gpu_request}"
    --mem="${mem}"
    --time="${time_limit}"
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.err"
    --export="$(slurm_exports)"
    "${script}"
    --server
    "${pool}"
  )

  echo "Submitting server ${pool}: partition=${partition}, gpu=${gpu_request}, dp=$(server_dp_for_pool "${pool}"), job=${job_name}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_${pool}_SERVER"
  else
    "${cmd[@]}"
  fi
}

submit_worker_job() {
  local pool="$1"
  local array_spec array_max job_name script
  script="$(script_path)"

  case "${pool}" in
    a100)
      array_spec="${A100_WORKER_ARRAY}"
      array_max="${A100_WORKER_MAX_CONCURRENT}"
      ;;
    h200)
      array_spec="${H200_WORKER_ARRAY}"
      array_max="${H200_WORKER_MAX_CONCURRENT}"
      ;;
    *) echo "ERROR: unknown worker pool '${pool}'" >&2; return 1 ;;
  esac

  job_name="${RUN_TAG}-${pool}-workers"
  local cmd=(
    sbatch
    --parsable
    --job-name="${job_name}"
    --account="${ACCOUNT}"
    --partition="${WORKER_PARTITION}"
    --array="${array_spec}%${array_max}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${WORKER_CPUS_PER_TASK}"
    --mem="${WORKER_MEM}"
    --time="${WORKER_TIME}"
    --output="${LOG_DIR}/%x-%A_%a.out"
    --error="${LOG_DIR}/%x-%A_%a.err"
    --export="$(slurm_exports)"
    "${script}"
    --worker
    "${pool}"
  )

  echo "Submitting workers ${pool}: partition=${WORKER_PARTITION}, array=${array_spec}%${array_max}, job=${job_name}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_${pool}_WORKERS"
  else
    "${cmd[@]}"
  fi
}

submit_cleanup_job() {
  local pool="$1"
  local server_job_id="$2"
  local worker_job_id="$3"
  local job_name="${RUN_TAG}-${pool}-cleanup"
  local cmd=(
    sbatch
    --parsable
    --job-name="${job_name}"
    --account="${ACCOUNT}"
    --partition="${CLEANUP_PARTITION}"
    --dependency="afterany:${worker_job_id}"
    --nodes=1
    --ntasks=1
    --cpus-per-task=1
    --mem=1G
    --time=00:10:00
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.err"
    --wrap="scancel ${server_job_id} || true"
  )

  echo "Submitting cleanup ${pool}: partition=${CLEANUP_PARTITION}, after worker=${worker_job_id}, scancel server=${server_job_id}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_${pool}_CLEANUP"
  else
    "${cmd[@]}"
  fi
}

main_submit() {
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}"
  fi

  echo "Run tag: ${RUN_TAG}"
  echo "Agent model: ${AGENT_MODEL}"
  echo "Updater model: ${UPDATER_MODEL}"
  echo "ACE_ONCE updater max_tokens: ${UPDATER_MAX_TOKENS_EXPECTED}"
  echo "Outputs root: ${OUTPUTS_ROOT}"
  echo "Ready dir: ${READY_DIR}"
  echo
  print_assignments
  echo

  echo "Preflight:"
  if [[ "${STABILITY_REPEATS}" -ne 6 ]]; then
    echo "ERROR: stability.sh uses an explicit 6-repeat A100/H200 task split; keep STABILITY_REPEATS=6." >&2
    exit 1
  fi
  test -f "${REPO_DIR}/run.py"
  test -f "${REPO_DIR}/environments/minigrid_env.py"
  test -f "${REPO_DIR}/methods/ace_once.py"
  grep -q "ACE_ONCE_UPDATER_MAX_TOKENS = ${UPDATER_MAX_TOKENS_EXPECTED}" "${REPO_DIR}/methods/ace_once.py"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    test -x "${UV}"
    test -x "${SGLANG}"
  else
    echo "Dry run: not creating directories or requiring UV/SGLang executable checks."
  fi

  local a100_server h200_server updater_server a100_workers h200_workers
  a100_server="$(submit_server_job a100)"
  h200_server="$(submit_server_job h200)"
  updater_server="$(submit_server_job updater)"
  a100_workers="$(submit_worker_job a100)"
  h200_workers="$(submit_worker_job h200)"
  submit_cleanup_job a100 "${a100_server}" "${a100_workers}" >/dev/null
  submit_cleanup_job h200 "${h200_server}" "${h200_workers}" >/dev/null
  submit_cleanup_job updater "${updater_server}" "${a100_workers}:${h200_workers}" >/dev/null

  echo
  echo "Submitted jobs:"
  echo "  a100 server: ${a100_server}"
  echo "  h200 server: ${h200_server}"
  echo "  updater server: ${updater_server}"
  echo "  a100 workers: ${a100_workers}"
  echo "  h200 workers: ${h200_workers}"
  echo
  echo "Monitor:"
  echo "  squeue -u ${USER:-mohanc3}"
  echo "  cat ${OUTPUTS_ROOT}/${RUN_TAG}-*-status.tsv 2>/dev/null | grep -v '^pool'"
}

usage() {
  cat <<EOF
Usage:
  bash $0 [--dry-run]
  bash $0 --server a100|h200|updater
  bash $0 --worker a100|h200
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --server)
      MODE="server"
      POOL="${2:-}"
      shift 2
      ;;
    --worker)
      MODE="worker"
      POOL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${MODE}" in
  submit) main_submit ;;
  server)
    if [[ "${POOL}" != "a100" && "${POOL}" != "h200" && "${POOL}" != "updater" ]]; then
      echo "ERROR: --server requires a100, h200, or updater" >&2
      exit 1
    fi
    run_server "${POOL}"
    ;;
  worker)
    if [[ "${POOL}" != "a100" && "${POOL}" != "h200" ]]; then
      echo "ERROR: --worker requires a100 or h200" >&2
      exit 1
    fi
    run_worker "${POOL}"
    ;;
  *)
    echo "ERROR: bad mode '${MODE}'" >&2
    exit 1
    ;;
esac
