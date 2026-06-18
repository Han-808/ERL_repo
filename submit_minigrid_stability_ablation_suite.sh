#!/usr/bin/env bash
set -euo pipefail

# MiniGrid ACE_ONCE stability ablation suite.
#
# Three fixed-seed stability tasks are launched concurrently:
#   1. Qwen3.5-27B generator, no updater, fixed initial empty playbook.
#   2. Qwen3.5-27B generator, Qwen3-4B updater.
#   3. Qwen3-14B generator, Qwen3-8B updater.
#
# L40/L40S can be used as serving GPUs here:
#   - 2xH200 dp=2 shared 27B generator server for tasks 1 and 2.
#   - 2xA100 dp=2 14B generator server for task 3.
#   - 1xL40 dp=1 4B updater server for task 2.
#   - 2xL40S dp=2 8B updater server for task 3.

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
METHOD="${METHOD:-ace_once_minigrid}"
SEED_STRIDE="${SEED_STRIDE:-10000}"
STABILITY_REPEATS="${STABILITY_REPEATS:-6}"
STABILITY_SEED="${STABILITY_SEED:-0}"
LM_TEMPERATURE="${LM_TEMPERATURE:-1.0}"
UPDATER_MAX_TOKENS_EXPECTED="${UPDATER_MAX_TOKENS_EXPECTED:-8192}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_TAG="${SUITE_TAG:-minigrid-stability-ablation-suite-${RUN_STAMP}}"

NO_UPDATER_RUN_TAG="${NO_UPDATER_RUN_TAG:-minigrid-stability-qwen35-27b-agent-no-updater-initctx-${RUN_STAMP}}"
UPD4_RUN_TAG="${UPD4_RUN_TAG:-minigrid-stability-qwen35-27b-agent-qwen3-4b-updater8192-${RUN_STAMP}}"
UPD8_RUN_TAG="${UPD8_RUN_TAG:-minigrid-stability-qwen3-14b-agent-qwen3-8b-updater8192-${RUN_STAMP}}"

GEN27_MODEL="${GEN27_MODEL:-Qwen/Qwen3.5-27B}"
GEN14_MODEL="${GEN14_MODEL:-Qwen/Qwen3-14B}"
UPD4_MODEL="${UPD4_MODEL:-Qwen/Qwen3-4B}"
UPD8_MODEL="${UPD8_MODEL:-Qwen/Qwen3-8B}"

LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/minigrid_stability_ablation_suite}"
READY_DIR="${READY_DIR:-${LOG_DIR}/minigrid_stability_ablation_ready/${SUITE_TAG}}"

GEN27_PARTITION="${GEN27_PARTITION:-gpu-h200}"
GEN27_GPU_REQUEST="${GEN27_GPU_REQUEST:-h200:2}"
GEN27_DP_SIZE="${GEN27_DP_SIZE:-2}"
GEN27_PORT="${GEN27_PORT:-31000}"
GEN27_SERVER_CPUS="${GEN27_SERVER_CPUS:-24}"
GEN27_SERVER_MEM="${GEN27_SERVER_MEM:-256G}"
GEN27_SERVER_TIME="${GEN27_SERVER_TIME:-96:00:00}"

GEN14_PARTITION="${GEN14_PARTITION:-gpu-a100}"
GEN14_GPU_REQUEST="${GEN14_GPU_REQUEST:-a100:2}"
GEN14_DP_SIZE="${GEN14_DP_SIZE:-2}"
GEN14_PORT="${GEN14_PORT:-31100}"
GEN14_SERVER_CPUS="${GEN14_SERVER_CPUS:-24}"
GEN14_SERVER_MEM="${GEN14_SERVER_MEM:-160G}"
GEN14_SERVER_TIME="${GEN14_SERVER_TIME:-96:00:00}"

UPD4_PARTITION="${UPD4_PARTITION:-gpu-l40}"
UPD4_GPU_REQUEST="${UPD4_GPU_REQUEST:-l40:1}"
UPD4_DP_SIZE="${UPD4_DP_SIZE:-1}"
UPD4_PORT="${UPD4_PORT:-31200}"
UPD4_SERVER_CPUS="${UPD4_SERVER_CPUS:-8}"
UPD4_SERVER_MEM="${UPD4_SERVER_MEM:-64G}"
UPD4_SERVER_TIME="${UPD4_SERVER_TIME:-96:00:00}"

UPD8_PARTITION="${UPD8_PARTITION:-gpu-l40s}"
UPD8_GPU_REQUEST="${UPD8_GPU_REQUEST:-l40s:2}"
UPD8_DP_SIZE="${UPD8_DP_SIZE:-2}"
UPD8_PORT="${UPD8_PORT:-31300}"
UPD8_SERVER_CPUS="${UPD8_SERVER_CPUS:-16}"
UPD8_SERVER_MEM="${UPD8_SERVER_MEM:-128G}"
UPD8_SERVER_TIME="${UPD8_SERVER_TIME:-96:00:00}"

WORKER_PARTITION="${WORKER_PARTITION:-gpu-l40}"
WORKER_CPUS_PER_TASK="${WORKER_CPUS_PER_TASK:-1}"
WORKER_MEM="${WORKER_MEM:-4G}"
WORKER_TIME="${WORKER_TIME:-96:00:00}"
CLEANUP_PARTITION="${CLEANUP_PARTITION:-${WORKER_PARTITION}}"

NO_UPDATER_WORKER_ARRAY="${NO_UPDATER_WORKER_ARRAY:-0-35}"
NO_UPDATER_WORKER_MAX_CONCURRENT="${NO_UPDATER_WORKER_MAX_CONCURRENT:-12}"
UPD4_WORKER_ARRAY="${UPD4_WORKER_ARRAY:-0-35}"
UPD4_WORKER_MAX_CONCURRENT="${UPD4_WORKER_MAX_CONCURRENT:-12}"
UPD8_WORKER_ARRAY="${UPD8_WORKER_ARRAY:-0-35}"
UPD8_WORKER_MAX_CONCURRENT="${UPD8_WORKER_MAX_CONCURRENT:-16}"

SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
SERVER_READY_WAIT_SECONDS="${SERVER_READY_WAIT_SECONDS:-3600}"
WORKER_READY_WAIT_SECONDS="${WORKER_READY_WAIT_SECONDS:-3600}"
READY_POLL_SECONDS="${READY_POLL_SECONDS:-10}"

DRY_RUN=0
MODE="submit"
ROLE=""
TASK_KIND=""

setup_hyak_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
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

ready_file_for_role() {
  case "$1" in
    gen27) echo "${READY_DIR}/gen27.url" ;;
    gen14) echo "${READY_DIR}/gen14.url" ;;
    upd4) echo "${READY_DIR}/upd4.url" ;;
    upd8) echo "${READY_DIR}/upd8.url" ;;
    *) echo "ERROR: unknown role '$1'" >&2; return 1 ;;
  esac
}

server_port_for_role() {
  case "$1" in
    gen27) echo "${GEN27_PORT}" ;;
    gen14) echo "${GEN14_PORT}" ;;
    upd4) echo "${UPD4_PORT}" ;;
    upd8) echo "${UPD8_PORT}" ;;
    *) echo "ERROR: unknown role '$1'" >&2; return 1 ;;
  esac
}

server_dp_for_role() {
  case "$1" in
    gen27) echo "${GEN27_DP_SIZE}" ;;
    gen14) echo "${GEN14_DP_SIZE}" ;;
    upd4) echo "${UPD4_DP_SIZE}" ;;
    upd8) echo "${UPD8_DP_SIZE}" ;;
    *) echo "ERROR: unknown role '$1'" >&2; return 1 ;;
  esac
}

server_model_for_role() {
  case "$1" in
    gen27) echo "${GEN27_MODEL}" ;;
    gen14) echo "${GEN14_MODEL}" ;;
    upd4) echo "${UPD4_MODEL}" ;;
    upd8) echo "${UPD8_MODEL}" ;;
    *) echo "ERROR: unknown role '$1'" >&2; return 1 ;;
  esac
}

run_tag_for_task() {
  case "$1" in
    no_updater) echo "${NO_UPDATER_RUN_TAG}" ;;
    upd4) echo "${UPD4_RUN_TAG}" ;;
    upd8) echo "${UPD8_RUN_TAG}" ;;
    *) echo "ERROR: unknown task '$1'" >&2; return 1 ;;
  esac
}

agent_role_for_task() {
  case "$1" in
    no_updater|upd4) echo "gen27" ;;
    upd8) echo "gen14" ;;
    *) echo "ERROR: unknown task '$1'" >&2; return 1 ;;
  esac
}

updater_role_for_task() {
  case "$1" in
    no_updater) echo "" ;;
    upd4) echo "upd4" ;;
    upd8) echo "upd8" ;;
    *) echo "ERROR: unknown task '$1'" >&2; return 1 ;;
  esac
}

worker_array_for_task() {
  case "$1" in
    no_updater) echo "${NO_UPDATER_WORKER_ARRAY}%${NO_UPDATER_WORKER_MAX_CONCURRENT}" ;;
    upd4) echo "${UPD4_WORKER_ARRAY}%${UPD4_WORKER_MAX_CONCURRENT}" ;;
    upd8) echo "${UPD8_WORKER_ARRAY}%${UPD8_WORKER_MAX_CONCURRENT}" ;;
    *) echo "ERROR: unknown task '$1'" >&2; return 1 ;;
  esac
}

assignment_record() {
  local task_id="$1"
  if (( task_id < 0 || task_id > 35 )); then
    echo "ERROR: worker task_id must be 0..35, got ${task_id}" >&2
    return 1
  fi

  case "${task_id}" in
    0)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|0" ;;
    1)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|1" ;;
    2)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|2" ;;
    3)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|3" ;;
    4)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|4" ;;
    5)  echo "MiniGrid-MemoryS11-v0|20|605|minigrid_memorys11|12021|${STABILITY_SEED}|0|5" ;;
    6)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|0" ;;
    7)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|1" ;;
    8)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|2" ;;
    9)  echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|3" ;;
    10) echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|4" ;;
    11) echo "MiniGrid-MemoryS13-v0|20|845|minigrid_memorys13|3976|${STABILITY_SEED}|1|5" ;;
    12) echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|0" ;;
    13) echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|1" ;;
    14) echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|2" ;;
    15) echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|3" ;;
    16) echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|4" ;;
    17) echo "MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3|6283|${STABILITY_SEED}|2|5" ;;
    18) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|0" ;;
    19) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|1" ;;
    20) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|2" ;;
    21) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|3" ;;
    22) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|4" ;;
    23) echo "MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5|1820|${STABILITY_SEED}|3|5" ;;
    24) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|0" ;;
    25) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|1" ;;
    26) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|2" ;;
    27) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|3" ;;
    28) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|4" ;;
    29) echo "MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms|1740|${STABILITY_SEED}|4|5" ;;
    30) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|0" ;;
    31) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|1" ;;
    32) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|2" ;;
    33) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|3" ;;
    34) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|4" ;;
    35) echo "MiniGrid-DistShift1-v0|20|252|minigrid_distshift1|1551|${STABILITY_SEED}|5|5" ;;
  esac
}

load_worker_assignment() {
  local record task_id="$1"

  ENV_ID=""
  EPISODES=""
  MAX_STEPS=""
  ENV_LABEL=""
  HISTORICAL_GEN_CALLS=""
  SEED=""
  BASE_TASK_ID=""
  REPEAT=""

  record="$(assignment_record "${task_id}")"
  IFS='|' read -r ENV_ID EPISODES MAX_STEPS ENV_LABEL HISTORICAL_GEN_CALLS SEED BASE_TASK_ID REPEAT <<< "${record}"
}

print_assignments() {
  local task_id seed_offset
  echo "=== worker assignments shared by all three tasks ==="
  for task_id in $(seq 0 35); do
    load_worker_assignment "${task_id}"
    seed_offset=$((SEED * SEED_STRIDE))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${task_id}" \
      "${METHOD}" \
      "${ENV_ID}" \
      "${EPISODES}" \
      "${MAX_STEPS}" \
      "${SEED}" \
      "${REPEAT}" \
      "${seed_offset}" \
      "${ENV_LABEL}" \
      "${HISTORICAL_GEN_CALLS}"
  done
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
  local role="$1"
  local port dp model ready_file server_log server_pid host server_url

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}"

  port="$(server_port_for_role "${role}")"
  dp="$(server_dp_for_role "${role}")"
  model="$(server_model_for_role "${role}")"
  ready_file="$(ready_file_for_role "${role}")"
  server_log="${LOG_DIR}/${SUITE_TAG}-${role}-sglang_server.log"
  rm -f "${ready_file}"

  echo "Server role: ${role}"
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
  local task_kind="$1"
  local task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  local run_tag task_root agent_role updater_role agent_model updater_model
  local agent_ready_file updater_ready_file server_url updater_server_url
  local seed_offset status_file run_label outputs_dir updater_arg

  cd "${REPO_DIR}"
  setup_hyak_env
  run_tag="$(run_tag_for_task "${task_kind}")"
  task_root="${OUTPUTS_ROOT}/${run_tag}"
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}" "${task_root}"

  load_worker_assignment "${task_id}"
  seed_offset=$((SEED * SEED_STRIDE))
  agent_role="$(agent_role_for_task "${task_kind}")"
  updater_role="$(updater_role_for_task "${task_kind}")"
  agent_model="$(server_model_for_role "${agent_role}")"
  agent_ready_file="$(ready_file_for_role "${agent_role}")"
  status_file="${task_root}/${run_tag}-${task_kind}-${task_id}-status.tsv"
  run_label="${run_tag}-${task_id}-${METHOD}-${ENV_LABEL}-s${SEED}-r${REPEAT}"
  outputs_dir="${task_root}/${run_label}"

  printf "task\ttask_id\tbase_task_id\tenv_id\tseed\trepeat\ttemperature\tepisodes\tmax_steps\tseed_offset\thistorical_gen_calls\tserver_url\tstatus\toutputs_dir\tupdater_server_url\n" > "${status_file}"

  echo "Task: ${task_kind}"
  echo "Task id: ${task_id}"
  echo "Run tag: ${run_tag}"
  echo "Assignment: method=${METHOD} env_id=${ENV_ID} episodes=${EPISODES} max_steps=${MAX_STEPS} seed=${SEED} repeat=${REPEAT} seed_offset=${seed_offset} base_task_id=${BASE_TASK_ID} historical_gen_calls=${HISTORICAL_GEN_CALLS}"
  echo "Outputs: ${outputs_dir}"

  smoke_env
  server_url="$(wait_for_remote_server_url "${agent_ready_file}")"
  mkdir -p "${outputs_dir}"

  updater_arg=()
  updater_server_url="none"
  if [[ -n "${updater_role}" ]]; then
    updater_model="$(server_model_for_role "${updater_role}")"
    updater_ready_file="$(ready_file_for_role "${updater_role}")"
    updater_server_url="$(wait_for_remote_server_url "${updater_ready_file}")"
    updater_arg=(--updater-model "${updater_model}" --updater-server "${updater_server_url}")
  else
    updater_arg=(--ace-once-disable-updater)
  fi

  echo "Starting run server=${server_url} updater_server=${updater_server_url}"
  if "${UV}" run python "${REPO_DIR}/run.py" \
    --method "${METHOD}" \
    --env minigrid \
    --minigrid-id "${ENV_ID}" \
    --minigrid-max-steps "${MAX_STEPS}" \
    --seed-offset "${seed_offset}" \
    --episodes "${EPISODES}" \
    --model "${agent_model}" \
    --server "${server_url}" \
    "${updater_arg[@]}" \
    --outputs-dir "${outputs_dir}" \
    --disable-thinking; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tdone\t%s\t%s\n" \
      "${task_kind}" "${task_id}" "${BASE_TASK_ID}" "${ENV_ID}" "${SEED}" "${REPEAT}" "${LM_TEMPERATURE}" "${EPISODES}" "${MAX_STEPS}" "${seed_offset}" "${HISTORICAL_GEN_CALLS}" "${server_url}" "${outputs_dir}" "${updater_server_url}" >> "${status_file}"
    echo "Done. OUTPUTS_DIR=${outputs_dir}"
  else
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfailed\t%s\t%s\n" \
      "${task_kind}" "${task_id}" "${BASE_TASK_ID}" "${ENV_ID}" "${SEED}" "${REPEAT}" "${LM_TEMPERATURE}" "${EPISODES}" "${MAX_STEPS}" "${seed_offset}" "${HISTORICAL_GEN_CALLS}" "${server_url}" "${outputs_dir}" "${updater_server_url}" >> "${status_file}"
    echo "ERROR: run failed. OUTPUTS_DIR=${outputs_dir}" >&2
    return 1
  fi
}

script_path() {
  readlink -f "${BASH_SOURCE[0]}"
}

slurm_exports() {
  printf "ALL,REPO_DIR=%s,UV=%s,SGLANG=%s,SUITE_TAG=%s,NO_UPDATER_RUN_TAG=%s,UPD4_RUN_TAG=%s,UPD8_RUN_TAG=%s,GEN27_MODEL=%s,GEN14_MODEL=%s,UPD4_MODEL=%s,UPD8_MODEL=%s,METHOD=%s,SEED_STRIDE=%s,STABILITY_REPEATS=%s,STABILITY_SEED=%s,LM_TEMPERATURE=%s,LOG_DIR=%s,OUTPUTS_ROOT=%s,READY_DIR=%s,GEN27_PORT=%s,GEN14_PORT=%s,UPD4_PORT=%s,UPD8_PORT=%s,GEN27_DP_SIZE=%s,GEN14_DP_SIZE=%s,UPD4_DP_SIZE=%s,UPD8_DP_SIZE=%s,SGLANG_MEM_FRACTION=%s,SERVER_READY_WAIT_SECONDS=%s,WORKER_READY_WAIT_SECONDS=%s,READY_POLL_SECONDS=%s,UPDATER_MAX_TOKENS_EXPECTED=%s" \
    "${REPO_DIR}" \
    "${UV}" \
    "${SGLANG}" \
    "${SUITE_TAG}" \
    "${NO_UPDATER_RUN_TAG}" \
    "${UPD4_RUN_TAG}" \
    "${UPD8_RUN_TAG}" \
    "${GEN27_MODEL}" \
    "${GEN14_MODEL}" \
    "${UPD4_MODEL}" \
    "${UPD8_MODEL}" \
    "${METHOD}" \
    "${SEED_STRIDE}" \
    "${STABILITY_REPEATS}" \
    "${STABILITY_SEED}" \
    "${LM_TEMPERATURE}" \
    "${LOG_DIR}" \
    "${OUTPUTS_ROOT}" \
    "${READY_DIR}" \
    "${GEN27_PORT}" \
    "${GEN14_PORT}" \
    "${UPD4_PORT}" \
    "${UPD8_PORT}" \
    "${GEN27_DP_SIZE}" \
    "${GEN14_DP_SIZE}" \
    "${UPD4_DP_SIZE}" \
    "${UPD8_DP_SIZE}" \
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

server_slurm_config() {
  local role="$1"
  case "${role}" in
    gen27) echo "${GEN27_PARTITION}|${GEN27_GPU_REQUEST}|${GEN27_SERVER_CPUS}|${GEN27_SERVER_MEM}|${GEN27_SERVER_TIME}" ;;
    gen14) echo "${GEN14_PARTITION}|${GEN14_GPU_REQUEST}|${GEN14_SERVER_CPUS}|${GEN14_SERVER_MEM}|${GEN14_SERVER_TIME}" ;;
    upd4) echo "${UPD4_PARTITION}|${UPD4_GPU_REQUEST}|${UPD4_SERVER_CPUS}|${UPD4_SERVER_MEM}|${UPD4_SERVER_TIME}" ;;
    upd8) echo "${UPD8_PARTITION}|${UPD8_GPU_REQUEST}|${UPD8_SERVER_CPUS}|${UPD8_SERVER_MEM}|${UPD8_SERVER_TIME}" ;;
    *) echo "ERROR: unknown role '${role}'" >&2; return 1 ;;
  esac
}

submit_server_job() {
  local role="$1"
  local partition gpu_request cpus mem time_limit job_name script config
  script="$(script_path)"
  config="$(server_slurm_config "${role}")"
  IFS='|' read -r partition gpu_request cpus mem time_limit <<< "${config}"

  job_name="${SUITE_TAG}-${role}-server"
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
    "${role}"
  )

  echo "Submitting server ${role}: partition=${partition}, gpu=${gpu_request}, dp=$(server_dp_for_role "${role}"), model=$(server_model_for_role "${role}"), job=${job_name}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_${role}_SERVER"
  else
    "${cmd[@]}"
  fi
}

submit_worker_job() {
  local task_kind="$1"
  local array_spec job_name script run_tag
  script="$(script_path)"
  run_tag="$(run_tag_for_task "${task_kind}")"
  array_spec="$(worker_array_for_task "${task_kind}")"
  job_name="${run_tag}-workers"
  local cmd=(
    sbatch
    --parsable
    --job-name="${job_name}"
    --account="${ACCOUNT}"
    --partition="${WORKER_PARTITION}"
    --array="${array_spec}"
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
    "${task_kind}"
  )

  echo "Submitting workers ${task_kind}: partition=${WORKER_PARTITION}, array=${array_spec}, job=${job_name}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_${task_kind}_WORKERS"
  else
    "${cmd[@]}"
  fi
}

submit_cleanup_job() {
  local label="$1"
  local server_job_id="$2"
  local worker_dependency="$3"
  local job_name="${SUITE_TAG}-${label}-cleanup"
  local cmd=(
    sbatch
    --parsable
    --job-name="${job_name}"
    --account="${ACCOUNT}"
    --partition="${CLEANUP_PARTITION}"
    --dependency="afterany:${worker_dependency}"
    --nodes=1
    --ntasks=1
    --cpus-per-task=1
    --mem=1G
    --time=00:10:00
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.err"
    --wrap="scancel ${server_job_id} || true"
  )

  echo "Submitting cleanup ${label}: after=${worker_dependency}, scancel server=${server_job_id}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_${label}_CLEANUP"
  else
    "${cmd[@]}"
  fi
}

main_submit() {
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}"
  fi

  echo "Suite tag: ${SUITE_TAG}"
  echo "Outputs root: ${OUTPUTS_ROOT}"
  echo "Ready dir: ${READY_DIR}"
  echo
  echo "Task 1: ${NO_UPDATER_RUN_TAG}"
  echo "  Generator: ${GEN27_MODEL}"
  echo "  Updater: disabled; fixed initial playbook/context"
  echo "Task 2: ${UPD4_RUN_TAG}"
  echo "  Generator: ${GEN27_MODEL}"
  echo "  Updater: ${UPD4_MODEL}"
  echo "Task 3: ${UPD8_RUN_TAG}"
  echo "  Generator: ${GEN14_MODEL}"
  echo "  Updater: ${UPD8_MODEL}"
  echo
  echo "Server topology:"
  echo "  gen27: partition=${GEN27_PARTITION} gpu=${GEN27_GPU_REQUEST} dp=${GEN27_DP_SIZE}"
  echo "  gen14: partition=${GEN14_PARTITION} gpu=${GEN14_GPU_REQUEST} dp=${GEN14_DP_SIZE}"
  echo "  upd4:  partition=${UPD4_PARTITION} gpu=${UPD4_GPU_REQUEST} dp=${UPD4_DP_SIZE}"
  echo "  upd8:  partition=${UPD8_PARTITION} gpu=${UPD8_GPU_REQUEST} dp=${UPD8_DP_SIZE}"
  echo
  echo "Worker arrays:"
  echo "  no_updater: $(worker_array_for_task no_updater)"
  echo "  upd4:       $(worker_array_for_task upd4)"
  echo "  upd8:       $(worker_array_for_task upd8)"
  echo
  print_assignments
  echo

  echo "Preflight:"
  if [[ "${STABILITY_REPEATS}" -ne 6 ]]; then
    echo "ERROR: this suite uses explicit six-repeat assignments; keep STABILITY_REPEATS=6." >&2
    exit 1
  fi
  test -f "${REPO_DIR}/run.py"
  test -f "${REPO_DIR}/environments/minigrid_env.py"
  test -f "${REPO_DIR}/methods/ace_once.py"
  grep -q "ACE_ONCE_UPDATER_MAX_TOKENS = ${UPDATER_MAX_TOKENS_EXPECTED}" "${REPO_DIR}/methods/ace_once.py"
  grep -q "ace_once_disable_updater" "${REPO_DIR}/run.py"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    test -x "${UV}"
    test -x "${SGLANG}"
  else
    echo "Dry run: not creating directories or requiring UV/SGLang executable checks."
  fi

  local gen27_server gen14_server upd4_server upd8_server
  local no_updater_workers upd4_workers upd8_workers

  gen27_server="$(submit_server_job gen27)"
  gen14_server="$(submit_server_job gen14)"
  upd4_server="$(submit_server_job upd4)"
  upd8_server="$(submit_server_job upd8)"

  no_updater_workers="$(submit_worker_job no_updater)"
  upd4_workers="$(submit_worker_job upd4)"
  upd8_workers="$(submit_worker_job upd8)"

  submit_cleanup_job gen27 "${gen27_server}" "${no_updater_workers}:${upd4_workers}" >/dev/null
  submit_cleanup_job gen14 "${gen14_server}" "${upd8_workers}" >/dev/null
  submit_cleanup_job upd4 "${upd4_server}" "${upd4_workers}" >/dev/null
  submit_cleanup_job upd8 "${upd8_server}" "${upd8_workers}" >/dev/null

  echo
  echo "Submitted jobs:"
  echo "  gen27 server: ${gen27_server}"
  echo "  gen14 server: ${gen14_server}"
  echo "  upd4 server:  ${upd4_server}"
  echo "  upd8 server:  ${upd8_server}"
  echo "  no-updater workers: ${no_updater_workers}"
  echo "  upd4 workers:       ${upd4_workers}"
  echo "  upd8 workers:       ${upd8_workers}"
  echo
  echo "Monitor:"
  echo "  squeue -u ${USER:-mohanc3}"
  echo "  cat ${OUTPUTS_ROOT}/*/*-status.tsv 2>/dev/null | grep -v '^task'"
}

usage() {
  cat <<EOF
Usage:
  bash $0 [--dry-run]
  bash $0 --server gen27|gen14|upd4|upd8
  bash $0 --worker no_updater|upd4|upd8
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
      ROLE="${2:-}"
      shift 2
      ;;
    --worker)
      MODE="worker"
      TASK_KIND="${2:-}"
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
  submit)
    main_submit
    ;;
  server)
    case "${ROLE}" in
      gen27|gen14|upd4|upd8) run_server "${ROLE}" ;;
      *) echo "ERROR: --server requires gen27, gen14, upd4, or upd8" >&2; exit 1 ;;
    esac
    ;;
  worker)
    case "${TASK_KIND}" in
      no_updater|upd4|upd8) run_worker "${TASK_KIND}" ;;
      *) echo "ERROR: --worker requires no_updater, upd4, or upd8" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "ERROR: bad mode '${MODE}'" >&2
    exit 1
    ;;
esac
