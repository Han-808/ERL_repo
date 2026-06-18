#!/usr/bin/env bash
set -euo pipefail

# ACE_ONCE-only MiniGrid data generation for updater SFT.
#
# Topology:
#   - one 3xA100 SGLang server with dp=3
#   - one 1xH200 SGLang server with dp=1
#   - CPU worker arrays run MiniGrid game/seed jobs against those servers
#
# Usage:
#   bash submit_minigrid_sft_ace_once_qwen35_27b.sh --dry-run
#   bash submit_minigrid_sft_ace_once_qwen35_27b.sh
#
# Advanced entrypoints used by Slurm:
#   bash submit_minigrid_sft_ace_once_qwen35_27b.sh --server a100
#   bash submit_minigrid_sft_ace_once_qwen35_27b.sh --worker a100

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
MODEL_TAG="${MODEL_TAG:-qwen35-27b}"
METHOD="${METHOD:-ace_once_minigrid}"
SEED_STRIDE="${SEED_STRIDE:-10000}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-minigrid-sft-${MODEL_TAG}-ace-once}"
RUN_TAG="${RUN_TAG:-${JOB_NAME_PREFIX}-$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/minigrid_sft}"
READY_DIR="${READY_DIR:-${LOG_DIR}/minigrid_sft_ready/${RUN_TAG}}"

A100_PARTITION="${A100_PARTITION:-gpu-a100}"
A100_GPU_REQUEST="${A100_GPU_REQUEST:-a100:3}"
A100_DP_SIZE="${A100_DP_SIZE:-3}"
A100_PORT="${A100_PORT:-31000}"
A100_SERVER_CPUS="${A100_SERVER_CPUS:-24}"
A100_SERVER_MEM="${A100_SERVER_MEM:-256G}"
A100_SERVER_TIME="${A100_SERVER_TIME:-96:00:00}"
A100_WORKER_ARRAY="${A100_WORKER_ARRAY:-0-29}"
A100_WORKER_MAX_CONCURRENT="${A100_WORKER_MAX_CONCURRENT:-12}"

H200_PARTITION="${H200_PARTITION:-gpu-h200}"
H200_GPU_REQUEST="${H200_GPU_REQUEST:-h200:1}"
H200_DP_SIZE="${H200_DP_SIZE:-1}"
H200_PORT="${H200_PORT:-31100}"
H200_SERVER_CPUS="${H200_SERVER_CPUS:-16}"
H200_SERVER_MEM="${H200_SERVER_MEM:-256G}"
H200_SERVER_TIME="${H200_SERVER_TIME:-96:00:00}"
H200_WORKER_ARRAY="${H200_WORKER_ARRAY:-0-5}"
H200_WORKER_MAX_CONCURRENT="${H200_WORKER_MAX_CONCURRENT:-6}"

WORKER_PARTITION="${WORKER_PARTITION:-}"
WORKER_CPUS_PER_TASK="${WORKER_CPUS_PER_TASK:-4}"
WORKER_MEM="${WORKER_MEM:-32G}"
WORKER_TIME="${WORKER_TIME:-96:00:00}"

SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
SERVER_READY_WAIT_SECONDS="${SERVER_READY_WAIT_SECONDS:-2400}"
WORKER_READY_WAIT_SECONDS="${WORKER_READY_WAIT_SECONDS:-3600}"
READY_POLL_SECONDS="${READY_POLL_SECONDS:-10}"

DRY_RUN=0
MODE="submit"
POOL=""

setup_hyak_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  export NO_PROXY=localhost,127.0.0.1
  export no_proxy=localhost,127.0.0.1

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
    *) echo "ERROR: unknown pool '$1'" >&2; return 1 ;;
  esac
}

server_port_for_pool() {
  case "$1" in
    a100) echo "${A100_PORT}" ;;
    h200) echo "${H200_PORT}" ;;
    *) echo "ERROR: unknown pool '$1'" >&2; return 1 ;;
  esac
}

server_dp_for_pool() {
  case "$1" in
    a100) echo "${A100_DP_SIZE}" ;;
    h200) echo "${H200_DP_SIZE}" ;;
    *) echo "ERROR: unknown pool '$1'" >&2; return 1 ;;
  esac
}

load_worker_assignment() {
  local pool="$1"
  local task_id="$2"
  local game_index record

  ENV_ID=""
  EPISODES=""
  MAX_STEPS=""
  ENV_LABEL=""
  SEED=""

  if [[ "${pool}" == "a100" ]]; then
    if (( task_id < 0 || task_id > 29 )); then
      echo "ERROR: a100 worker task_id must be 0..29, got ${task_id}" >&2
      return 1
    fi
    game_index=$((task_id / 6))
    SEED=$((task_id % 6))
    case "${game_index}" in
      0) record="MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5" ;;
      1) record="MiniGrid-MemoryS11-v0|40|605|minigrid_memorys11" ;;
      2) record="MiniGrid-FourRooms-v0|20|100|minigrid_fourrooms" ;;
      3) record="MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3" ;;
      4) record="MiniGrid-DistShift1-v0|20|252|minigrid_distshift1" ;;
      *) echo "ERROR: bad a100 game index ${game_index}" >&2; return 1 ;;
    esac
  elif [[ "${pool}" == "h200" ]]; then
    if (( task_id < 0 || task_id > 5 )); then
      echo "ERROR: h200 worker task_id must be 0..5, got ${task_id}" >&2
      return 1
    fi
    SEED="${task_id}"
    record="MiniGrid-MemoryS13-v0|30|845|minigrid_memorys13"
  else
    echo "ERROR: unknown worker pool '${pool}'" >&2
    return 1
  fi

  IFS='|' read -r ENV_ID EPISODES MAX_STEPS ENV_LABEL <<< "${record}"
}

print_assignments_for_pool() {
  local pool="$1"
  local max_task task_id seed_offset
  case "${pool}" in
    a100) max_task=29 ;;
    h200) max_task=5 ;;
    *) echo "ERROR: unknown pool '${pool}'" >&2; return 1 ;;
  esac

  echo "=== ${pool} worker assignments ==="
  for task_id in $(seq 0 "${max_task}"); do
    load_worker_assignment "${pool}" "${task_id}"
    seed_offset=$((SEED * SEED_STRIDE))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${pool}" \
      "${task_id}" \
      "${METHOD}" \
      "${ENV_ID}" \
      "${EPISODES}" \
      "${MAX_STEPS}" \
      "${SEED}" \
      "${seed_offset}" \
      "${ENV_LABEL}"
  done
}

print_assignments() {
  print_assignments_for_pool a100
  print_assignments_for_pool h200
  echo
  echo "Expected runs: 36"
  echo "Expected episodes/updater calls: 1020"
  echo "Expected result files: 36"
  echo "Expected llm_calls files: 36"
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
  local port dp ready_file server_log server_pid host server_url

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}"

  port="$(server_port_for_pool "${pool}")"
  dp="$(server_dp_for_pool "${pool}")"
  ready_file="$(ready_file_for_pool "${pool}")"
  server_log="${LOG_DIR}/${RUN_TAG}-${pool}-sglang_server.log"
  rm -f "${ready_file}"

  echo "Server pool: ${pool}"
  echo "Node: $(hostname)"
  echo "Model: ${MODEL}"
  echo "Port: ${port}"
  echo "DP size: ${dp}"
  echo "Ready file: ${ready_file}"
  echo "Server log: ${server_log}"

  "${SGLANG}" serve \
    --model-path "${MODEL}" \
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
  local ready_file server_url seed_offset status_file run_label outputs_dir

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}"

  load_worker_assignment "${pool}" "${task_id}"
  seed_offset=$((SEED * SEED_STRIDE))
  ready_file="$(ready_file_for_pool "${pool}")"
  status_file="${OUTPUTS_ROOT}/${RUN_TAG}-${pool}-${task_id}-status.tsv"
  run_label="${RUN_TAG}-${pool}-${task_id}-${METHOD}-${ENV_LABEL}-s${SEED}"
  outputs_dir="${OUTPUTS_ROOT}/${run_label}"

  printf "pool\ttask_id\tenv_id\tseed\tepisodes\tmax_steps\tseed_offset\tserver_url\tstatus\toutputs_dir\n" > "${status_file}"

  echo "Worker pool: ${pool}"
  echo "Task id: ${task_id}"
  echo "Assignment: method=${METHOD} env_id=${ENV_ID} episodes=${EPISODES} max_steps=${MAX_STEPS} seed=${SEED} seed_offset=${seed_offset}"
  echo "Outputs: ${outputs_dir}"

  smoke_env
  server_url="$(wait_for_remote_server_url "${ready_file}")"
  mkdir -p "${outputs_dir}"

  echo "Starting run server=${server_url}"
  if "${UV}" run python "${REPO_DIR}/run.py" \
    --method "${METHOD}" \
    --env minigrid \
    --minigrid-id "${ENV_ID}" \
    --minigrid-max-steps "${MAX_STEPS}" \
    --seed-offset "${seed_offset}" \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --server "${server_url}" \
    --outputs-dir "${outputs_dir}" \
    --disable-thinking; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tdone\t%s\n" \
      "${pool}" "${task_id}" "${ENV_ID}" "${SEED}" "${EPISODES}" "${MAX_STEPS}" "${seed_offset}" "${server_url}" "${outputs_dir}" >> "${status_file}"
    echo "Done. OUTPUTS_DIR=${outputs_dir}"
  else
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfailed\t%s\n" \
      "${pool}" "${task_id}" "${ENV_ID}" "${SEED}" "${EPISODES}" "${MAX_STEPS}" "${seed_offset}" "${server_url}" "${outputs_dir}" >> "${status_file}"
    echo "ERROR: run failed. OUTPUTS_DIR=${outputs_dir}" >&2
    return 1
  fi
}

script_path() {
  readlink -f "${BASH_SOURCE[0]}"
}

slurm_exports() {
  printf "ALL,REPO_DIR=%s,UV=%s,SGLANG=%s,RUN_TAG=%s,MODEL=%s,MODEL_TAG=%s,METHOD=%s,SEED_STRIDE=%s,LOG_DIR=%s,OUTPUTS_ROOT=%s,READY_DIR=%s,A100_PORT=%s,H200_PORT=%s,A100_DP_SIZE=%s,H200_DP_SIZE=%s,SGLANG_MEM_FRACTION=%s,SERVER_READY_WAIT_SECONDS=%s,WORKER_READY_WAIT_SECONDS=%s,READY_POLL_SECONDS=%s" \
    "${REPO_DIR}" \
    "${UV}" \
    "${SGLANG}" \
    "${RUN_TAG}" \
    "${MODEL}" \
    "${MODEL_TAG}" \
    "${METHOD}" \
    "${SEED_STRIDE}" \
    "${LOG_DIR}" \
    "${OUTPUTS_ROOT}" \
    "${READY_DIR}" \
    "${A100_PORT}" \
    "${H200_PORT}" \
    "${A100_DP_SIZE}" \
    "${H200_DP_SIZE}" \
    "${SGLANG_MEM_FRACTION}" \
    "${SERVER_READY_WAIT_SECONDS}" \
    "${WORKER_READY_WAIT_SECONDS}" \
    "${READY_POLL_SECONDS}"
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

  echo "Submitting server ${pool}: partition=${partition}, gpu=${gpu_request}, job=${job_name}" >&2
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
    --array="${array_spec}%${array_max}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${WORKER_CPUS_PER_TASK}"
    --mem="${WORKER_MEM}"
    --time="${WORKER_TIME}"
    --output="${LOG_DIR}/%x-%A_%a.out"
    --error="${LOG_DIR}/%x-%A_%a.err"
    --export="$(slurm_exports)"
  )
  if [[ -n "${WORKER_PARTITION}" ]]; then
    cmd+=(--partition="${WORKER_PARTITION}")
  fi
  cmd+=(
    "${script}"
    --worker
    "${pool}"
  )

  echo "Submitting workers ${pool}: array=${array_spec}%${array_max}, job=${job_name}" >&2
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
    --dependency="afterany:${worker_job_id}"
    --nodes=1
    --ntasks=1
    --cpus-per-task=1
    --mem=1G
    --time=00:10:00
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.err"
  )
  if [[ -n "${WORKER_PARTITION}" ]]; then
    cmd+=(--partition="${WORKER_PARTITION}")
  fi
  cmd+=(--wrap "scancel ${server_job_id} || true")

  echo "Submitting cleanup ${pool}: after worker=${worker_job_id}, scancel server=${server_job_id}" >&2
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
  echo "Model: ${MODEL}"
  echo "Outputs root: ${OUTPUTS_ROOT}"
  echo "Ready dir: ${READY_DIR}"
  echo
  print_assignments
  echo

  echo "Preflight:"
  test -f "${REPO_DIR}/run.py"
  test -f "${REPO_DIR}/environments/minigrid_env.py"
  test -f "${REPO_DIR}/methods/ace_once.py"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    test -x "${UV}"
    test -x "${SGLANG}"
  else
    echo "Dry run: not creating directories or requiring UV/SGLang executable checks."
  fi

  local a100_server h200_server a100_workers h200_workers
  a100_server="$(submit_server_job a100)"
  h200_server="$(submit_server_job h200)"
  a100_workers="$(submit_worker_job a100)"
  h200_workers="$(submit_worker_job h200)"
  submit_cleanup_job a100 "${a100_server}" "${a100_workers}" >/dev/null
  submit_cleanup_job h200 "${h200_server}" "${h200_workers}" >/dev/null

  echo
  echo "Submitted jobs:"
  echo "  a100 server: ${a100_server}"
  echo "  h200 server: ${h200_server}"
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
  bash $0 --server a100|h200
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
    if [[ "${POOL}" != "a100" && "${POOL}" != "h200" ]]; then
      echo "ERROR: --server requires a100 or h200" >&2
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
