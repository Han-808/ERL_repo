#!/usr/bin/env bash
set -euo pipefail

# Offline ACE_ONCE updater relabeling for completed MiniGrid SFT runs.
#
# Topology:
#   - one 2xH200 SGLang server with dp=2
#   - one worker array, one episode per task, requesting the server
#
# Usage:
#   bash submit_minigrid_updater_relabel_qwen35_27b.sh --audit-token-range
#   bash submit_minigrid_updater_relabel_qwen35_27b.sh --dry-run
#   bash submit_minigrid_updater_relabel_qwen35_27b.sh
#
# Slurm entrypoints:
#   bash submit_minigrid_updater_relabel_qwen35_27b.sh --server
#   bash submit_minigrid_updater_relabel_qwen35_27b.sh --worker

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
PYTHON_BIN="${PYTHON_BIN:-}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
MODEL_TAG="${MODEL_TAG:-qwen35-27b}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_DIR}/minigrid_sft}"
SOURCE_RUN_TAG="${SOURCE_RUN_TAG:-20260529_131951}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/minigrid_updater_relabel}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-minigrid-updater-relabel-${MODEL_TAG}}"
RUN_TAG="${RUN_TAG:-${JOB_NAME_PREFIX}-$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
READY_DIR="${READY_DIR:-${LOG_DIR}/minigrid_updater_relabel_ready/${RUN_TAG}}"
READY_FILE="${READY_DIR}/h200.url"

H200_PARTITION="${H200_PARTITION:-gpu-h200}"
H200_GPU_REQUEST="${H200_GPU_REQUEST:-h200:2}"
H200_DP_SIZE="${H200_DP_SIZE:-2}"
H200_PORT="${H200_PORT:-31200}"
H200_SERVER_CPUS="${H200_SERVER_CPUS:-24}"
H200_SERVER_MEM="${H200_SERVER_MEM:-512G}"
H200_SERVER_TIME="${H200_SERVER_TIME:-96:00:00}"

WORKER_ARRAY="${WORKER_ARRAY:-0-1019}"
WORKER_MAX_CONCURRENT="${WORKER_MAX_CONCURRENT:-24}"
WORKER_PARTITION="${WORKER_PARTITION:-gpu-l40}"
WORKER_CPUS_PER_TASK="${WORKER_CPUS_PER_TASK:-1}"
WORKER_MEM="${WORKER_MEM:-4G}"
WORKER_TIME="${WORKER_TIME:-24:00:00}"

SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
SERVER_READY_WAIT_SECONDS="${SERVER_READY_WAIT_SECONDS:-3600}"
WORKER_READY_WAIT_SECONDS="${WORKER_READY_WAIT_SECONDS:-3600}"
READY_POLL_SECONDS="${READY_POLL_SECONDS:-10}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-7200}"

DRY_RUN=0
MODE="submit"

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

run_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    "${PYTHON_BIN}" "$@"
  else
    "${UV}" run python "$@"
  fi
}

helper_path() {
  echo "${REPO_DIR}/relabel_minigrid_updater.py"
}

script_path() {
  readlink -f "${BASH_SOURCE[0]}"
}

wait_for_local_server() {
  local server_pid="$1"
  local server_log="$2"
  local waited=0

  echo "Waiting for local SGLang server on port ${H200_PORT}..."
  while (( waited < SERVER_READY_WAIT_SECONDS )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "ERROR: SGLang server exited before becoming ready"
      tail -n 160 "${server_log}" || true
      return 1
    fi
    if curl --noproxy "*" -s "http://127.0.0.1:${H200_PORT}/v1/models" >/dev/null 2>&1; then
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
  local waited=0
  local server_url=""

  echo "Waiting for ready file ${READY_FILE}..." >&2
  while (( waited < WORKER_READY_WAIT_SECONDS )); do
    if [[ -s "${READY_FILE}" ]]; then
      server_url="$(<"${READY_FILE}")"
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
  local server_log server_pid host server_url

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}/${RUN_TAG}" "${READY_DIR}"
  rm -f "${READY_FILE}"

  server_log="${LOG_DIR}/${RUN_TAG}-h200-sglang_server.log"

  echo "Server pool: h200"
  echo "Node: $(hostname)"
  echo "Model: ${MODEL}"
  echo "Port: ${H200_PORT}"
  echo "DP size: ${H200_DP_SIZE}"
  echo "Max tokens used by workers: ${MAX_TOKENS}"
  echo "Ready file: ${READY_FILE}"
  echo "Server log: ${server_log}"

  "${SGLANG}" serve \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${H200_PORT}" \
    --dp-size "${H200_DP_SIZE}" \
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

  wait_for_local_server "${server_pid}" "${server_log}"

  host="$(hostname -f 2>/dev/null || hostname)"
  server_url="http://${host}:${H200_PORT}/v1"
  printf "%s\n" "${server_url}" > "${READY_FILE}"
  echo "Server URL written: ${server_url}"

  wait "${server_pid}"
}

run_worker() {
  local task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  local server_url

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}/${RUN_TAG}" "${READY_DIR}"

  echo "Worker task_id: ${task_id}"
  echo "Run tag: ${RUN_TAG}"
  echo "Source root: ${SOURCE_ROOT}"
  echo "Source run tag: ${SOURCE_RUN_TAG}"
  echo "Output root: ${OUTPUTS_ROOT}/${RUN_TAG}"
  echo "Max tokens: ${MAX_TOKENS}"

  server_url="$(wait_for_remote_server_url)"
  echo "Starting relabel server=${server_url}"

  run_python "$(helper_path)" \
    --task-id "${task_id}" \
    --source-root "${SOURCE_ROOT}" \
    --source-run-tag "${SOURCE_RUN_TAG}" \
    --output-root "${OUTPUTS_ROOT}" \
    --run-tag "${RUN_TAG}" \
    --model "${MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --server-url "${server_url}" \
    --timeout-seconds "${REQUEST_TIMEOUT_SECONDS}"
}

slurm_exports() {
  printf "ALL,REPO_DIR=%s,UV=%s,PYTHON_BIN=%s,SGLANG=%s,RUN_TAG=%s,MODEL=%s,MODEL_TAG=%s,SOURCE_ROOT=%s,SOURCE_RUN_TAG=%s,OUTPUTS_ROOT=%s,MAX_TOKENS=%s,LOG_DIR=%s,READY_DIR=%s,H200_PORT=%s,H200_DP_SIZE=%s,SGLANG_MEM_FRACTION=%s,SERVER_READY_WAIT_SECONDS=%s,WORKER_READY_WAIT_SECONDS=%s,READY_POLL_SECONDS=%s,REQUEST_TIMEOUT_SECONDS=%s" \
    "${REPO_DIR}" \
    "${UV}" \
    "${PYTHON_BIN}" \
    "${SGLANG}" \
    "${RUN_TAG}" \
    "${MODEL}" \
    "${MODEL_TAG}" \
    "${SOURCE_ROOT}" \
    "${SOURCE_RUN_TAG}" \
    "${OUTPUTS_ROOT}" \
    "${MAX_TOKENS}" \
    "${LOG_DIR}" \
    "${READY_DIR}" \
    "${H200_PORT}" \
    "${H200_DP_SIZE}" \
    "${SGLANG_MEM_FRACTION}" \
    "${SERVER_READY_WAIT_SECONDS}" \
    "${WORKER_READY_WAIT_SECONDS}" \
    "${READY_POLL_SECONDS}" \
    "${REQUEST_TIMEOUT_SECONDS}"
}

print_sbatch_command() {
  printf '%q ' "$@"
  echo
}

submit_server_job() {
  local script job_name
  script="$(script_path)"
  job_name="${RUN_TAG}-h200-server"

  local cmd=(
    sbatch
    --parsable
    --job-name="${job_name}"
    --account="${ACCOUNT}"
    --partition="${H200_PARTITION}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${H200_SERVER_CPUS}"
    --gpus="${H200_GPU_REQUEST}"
    --mem="${H200_SERVER_MEM}"
    --time="${H200_SERVER_TIME}"
    --output="${LOG_DIR}/%x-%j.out"
    --error="${LOG_DIR}/%x-%j.err"
    --export="$(slurm_exports)"
    "${script}"
    --server
  )

  echo "Submitting H200 server: partition=${H200_PARTITION}, gpu=${H200_GPU_REQUEST}, dp=${H200_DP_SIZE}, job=${job_name}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_H200_SERVER"
  else
    "${cmd[@]}"
  fi
}

submit_worker_job() {
  local script job_name
  script="$(script_path)"
  job_name="${RUN_TAG}-workers"

  local cmd=(
    sbatch
    --parsable
    --job-name="${job_name}"
    --account="${ACCOUNT}"
    --partition="${WORKER_PARTITION}"
    --array="${WORKER_ARRAY}%${WORKER_MAX_CONCURRENT}"
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
  )

  echo "Submitting workers: partition=${WORKER_PARTITION}, array=${WORKER_ARRAY}%${WORKER_MAX_CONCURRENT}, job=${job_name}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_WORKERS"
  else
    "${cmd[@]}"
  fi
}

submit_cleanup_job() {
  local server_job_id="$1"
  local worker_job_id="$2"
  local job_name="${RUN_TAG}-cleanup"

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
    --wrap="scancel ${server_job_id} || true"
  )

  echo "Submitting cleanup: after worker=${worker_job_id}, scancel server=${server_job_id}" >&2
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    print_sbatch_command "${cmd[@]}" >&2
    echo "DRYRUN_CLEANUP"
  else
    "${cmd[@]}"
  fi
}

print_config() {
  echo "Run tag: ${RUN_TAG}"
  echo "Model: ${MODEL}"
  echo "Max tokens: ${MAX_TOKENS}"
  echo "Source root: ${SOURCE_ROOT}"
  echo "Source run tag: ${SOURCE_RUN_TAG}"
  echo "Outputs root: ${OUTPUTS_ROOT}/${RUN_TAG}"
  echo "Ready file: ${READY_FILE}"
  echo
  echo "Server: partition=${H200_PARTITION} gpu=${H200_GPU_REQUEST} dp=${H200_DP_SIZE} port=${H200_PORT}"
  echo "Workers: partition=${WORKER_PARTITION} array=${WORKER_ARRAY}%${WORKER_MAX_CONCURRENT} cpus=${WORKER_CPUS_PER_TASK} mem=${WORKER_MEM}"
}

preflight() {
  echo "Preflight:"
  test -f "$(helper_path)"
  test -d "${SOURCE_ROOT}"
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}" "${READY_DIR}"
  run_python "$(helper_path)" \
    --source-root "${SOURCE_ROOT}" \
    --source-run-tag "${SOURCE_RUN_TAG}" \
    --list-assignments >/dev/null
  echo "  helper ok"
  echo "  source root exists"
  echo "  expected tasks: 1020"
}

audit_token_range() {
  cd "${REPO_DIR}"
  setup_hyak_env
  run_python "$(helper_path)" \
    --source-root "${SOURCE_ROOT}" \
    --source-run-tag "${SOURCE_RUN_TAG}" \
    --audit-token-range
}

dry_run() {
  cd "${REPO_DIR}"
  setup_hyak_env
  print_config
  echo
  echo "=== Assignments ==="
  run_python "$(helper_path)" \
    --source-root "${SOURCE_ROOT}" \
    --source-run-tag "${SOURCE_RUN_TAG}" \
    --list-assignments
  echo
  DRY_RUN=1
  local server_job worker_job
  server_job="$(submit_server_job)"
  worker_job="$(submit_worker_job)"
  submit_cleanup_job "${server_job}" "${worker_job}" >/dev/null
  echo
  echo "Dry-run jobs:"
  echo "  h200 server: ${server_job}"
  echo "  workers: ${worker_job}"
}

submit_all() {
  cd "${REPO_DIR}"
  print_config
  echo
  setup_hyak_env
  preflight
  echo

  local server_job worker_job cleanup_job
  server_job="$(submit_server_job)"
  worker_job="$(submit_worker_job)"
  cleanup_job="$(submit_cleanup_job "${server_job}" "${worker_job}")"

  echo
  echo "Submitted jobs:"
  echo "  h200 server: ${server_job}"
  echo "  workers: ${worker_job}"
  echo "  cleanup: ${cleanup_job}"
  echo
  echo "Monitor:"
  echo "  squeue -u mohanc3"
  echo "  cat ${OUTPUTS_ROOT}/${RUN_TAG}/status.tsv 2>/dev/null | grep -v '^task_id'"
}

usage() {
  cat <<'EOF'
Usage:
  bash submit_minigrid_updater_relabel_qwen35_27b.sh --audit-token-range
  bash submit_minigrid_updater_relabel_qwen35_27b.sh --dry-run
  bash submit_minigrid_updater_relabel_qwen35_27b.sh
  bash submit_minigrid_updater_relabel_qwen35_27b.sh --server
  bash submit_minigrid_updater_relabel_qwen35_27b.sh --worker
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --audit-token-range)
      MODE="audit-token-range"
      shift
      ;;
    --server)
      MODE="server"
      shift
      ;;
    --worker)
      MODE="worker"
      shift
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
  dry-run)
    dry_run
    ;;
  audit-token-range)
    audit_token_range
    ;;
  server)
    run_server
    ;;
  worker)
    run_worker
    ;;
  submit)
    submit_all
    ;;
  *)
    echo "ERROR: unknown mode ${MODE}" >&2
    exit 1
    ;;
esac
