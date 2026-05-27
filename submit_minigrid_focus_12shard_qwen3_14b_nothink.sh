#!/usr/bin/env bash
set -euo pipefail

# Focused MiniGrid online runs across 12 long-lived SGLang servers:
#   3 A100  shards
#   6 L40   shards
#   3 L40S  shards
#
# Each shard starts one SGLang server, then runs its assigned
# (method, game) groups sequentially. Each group is repeated REPEATS times
# on the same GPU/server with different seed offsets.
#
# Usage:
#   bash submit_minigrid_focus_12shard_qwen3_14b_nothink.sh --dry-run
#   bash submit_minigrid_focus_12shard_qwen3_14b_nothink.sh
#
# Useful overrides:
#   SUBMIT_A100=1 SUBMIT_L40=0 SUBMIT_L40S=0 bash submit_minigrid_focus_12shard_qwen3_14b_nothink.sh
#   REPEATS=1 bash submit_minigrid_focus_12shard_qwen3_14b_nothink.sh --dry-run

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
REPEATS="${REPEATS:-3}"
SEED_STRIDE="${SEED_STRIDE:-10000}"

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-128G}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"

A100_PARTITION="${A100_PARTITION:-gpu-a100}"
A100_GPU_REQUEST="${A100_GPU_REQUEST:-a100:1}"
A100_ARRAY_MAX_CONCURRENT="${A100_ARRAY_MAX_CONCURRENT:-3}"
A100_TIME_LIMIT="${A100_TIME_LIMIT:-72:00:00}"

L40_PARTITION="${L40_PARTITION:-gpu-l40}"
L40_GPU_REQUEST="${L40_GPU_REQUEST:-l40:1}"
L40_ARRAY_MAX_CONCURRENT="${L40_ARRAY_MAX_CONCURRENT:-6}"
L40_TIME_LIMIT="${L40_TIME_LIMIT:-72:00:00}"

L40S_PARTITION="${L40S_PARTITION:-gpu-l40s}"
L40S_GPU_REQUEST="${L40S_GPU_REQUEST:-l40s:1}"
L40S_ARRAY_MAX_CONCURRENT="${L40S_ARRAY_MAX_CONCURRENT:-3}"
L40S_TIME_LIMIT="${L40S_TIME_LIMIT:-72:00:00}"

SUBMIT_A100="${SUBMIT_A100:-1}"
SUBMIT_L40="${SUBMIT_L40:-1}"
SUBMIT_L40S="${SUBMIT_L40S:-1}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-minigrid-focus-${MODEL_TAG}-nothink-3rep}"
LOG_DIR="${REPO_DIR}/logs"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/minigrid}"

DRY_RUN=0

load_groups() {
  local gpu_kind="$1"
  local task_id="$2"
  GROUPS=()

  case "${gpu_kind}:${task_id}" in
    a100:0)
      GROUPS+=("ace_once_minigrid|MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3")
      ;;
    a100:1)
      GROUPS+=("notebook_minimal_mechanism_minigrid|MiniGrid-MemoryS13-v0|30|592|minigrid_memorys13")
      ;;
    a100:2)
      GROUPS+=("ace_once_minigrid|MiniGrid-MemoryS13-v0|30|592|minigrid_memorys13")
      ;;

    l40s:0)
      GROUPS+=("notebook_minimal_minigrid|MiniGrid-MemoryS13-v0|30|592|minigrid_memorys13")
      ;;
    l40s:1)
      GROUPS+=("notebook_minimal_mechanism_minigrid|MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3")
      ;;
    l40s:2)
      GROUPS+=("ace_once_minigrid|MiniGrid-MemoryS11-v0|40|303|minigrid_memorys11")
      ;;

    l40:0)
      GROUPS+=("ace_once_minigrid|MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5")
      ;;
    l40:1)
      GROUPS+=("notebook_minimal_minigrid|MiniGrid-SimpleCrossingS9N3-v0|20|324|minigrid_simplecrossings9n3")
      GROUPS+=("notebook_minimal_minigrid|MiniGrid-DistShift1-v0|20|227|minigrid_distshift1")
      ;;
    l40:2)
      GROUPS+=("notebook_minimal_mechanism_minigrid|MiniGrid-MemoryS11-v0|40|303|minigrid_memorys11")
      GROUPS+=("notebook_minimal_mechanism_minigrid|MiniGrid-FourRooms-v0|20|90|minigrid_fourrooms")
      ;;
    l40:3)
      GROUPS+=("notebook_minimal_minigrid|MiniGrid-MemoryS11-v0|40|303|minigrid_memorys11")
      GROUPS+=("notebook_minimal_minigrid|MiniGrid-FourRooms-v0|20|90|minigrid_fourrooms")
      ;;
    l40:4)
      GROUPS+=("notebook_minimal_mechanism_minigrid|MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5")
      GROUPS+=("notebook_minimal_minigrid|MiniGrid-Empty-Random-5x5-v0|40|100|minigrid_empty_random_5x5")
      GROUPS+=("notebook_minimal_mechanism_minigrid|MiniGrid-DistShift1-v0|20|227|minigrid_distshift1")
      ;;
    l40:5)
      GROUPS+=("ace_once_minigrid|MiniGrid-FourRooms-v0|20|90|minigrid_fourrooms")
      GROUPS+=("ace_once_minigrid|MiniGrid-DistShift1-v0|20|227|minigrid_distshift1")
      ;;
    *)
      echo "ERROR: no group assignment for ${gpu_kind}:${task_id}" >&2
      return 1
      ;;
  esac
}

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

smoke_group_envs() {
  local record method env_id episodes max_steps env_label
  for record in "${GROUPS[@]}"; do
    IFS='|' read -r method env_id episodes max_steps env_label <<< "${record}"
    "${UV}" run python -c \
      "import sys; from environments.minigrid_env import MiniGridTextEnv; env=MiniGridTextEnv(sys.argv[1], max_steps=int(sys.argv[2])); obs=env.reset(seed=1); assert f'Step: 0/{sys.argv[2]}' in obs, obs; env.close(); print(f'MiniGrid smoke ok: {sys.argv[1]} max_steps={sys.argv[2]}')" \
      "${env_id}" \
      "${max_steps}"
  done
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
  local gpu_kind="$1"
  local port_base="$2"
  local task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  local server_pid=""

  cd "${REPO_DIR}"
  setup_hyak_env
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}"

  load_groups "${gpu_kind}" "${task_id}"
  if [[ "${#GROUPS[@]}" -eq 0 ]]; then
    echo "ERROR: empty group assignment for ${gpu_kind}:${task_id}" >&2
    exit 1
  fi

  local port=$((port_base + task_id))
  local server_label="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${task_id}-${gpu_kind}"
  local sglang_log="${LOG_DIR}/${server_label}-sglang_server.log"
  local shard_status="${OUTPUTS_ROOT}/${server_label}-status.tsv"

  echo "Job: ${SLURM_JOB_NAME} array=${SLURM_ARRAY_JOB_ID} task=${task_id}"
  echo "Node: $(hostname)"
  echo "Model: ${MODEL}"
  echo "GPU kind: ${gpu_kind}"
  echo "Repeats per group: ${REPEATS}"
  echo "Seed stride: ${SEED_STRIDE}"
  echo "Port: ${port}"
  echo "Assignments:"
  printf '  %s\n' "${GROUPS[@]}"

  printf "group_index\tmethod\tenv_id\trepeat\tepisodes\tmax_steps\tseed_offset\tstatus\toutputs_dir\n" > "${shard_status}"

  smoke_group_envs

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
    if [[ -n "${server_pid}" ]]; then
      kill "${server_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  wait_for_server "${port}" "${server_pid}" "${sglang_log}"

  local group_index=0
  local record method env_id episodes max_steps env_label repeat seed_offset run_label outputs_dir
  for record in "${GROUPS[@]}"; do
    IFS='|' read -r method env_id episodes max_steps env_label <<< "${record}"
    for repeat in $(seq 1 "${REPEATS}"); do
      seed_offset=$(((repeat - 1) * SEED_STRIDE))
      run_label="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${task_id}-${method}-${env_label}-r${repeat}"
      outputs_dir="${OUTPUTS_ROOT}/${run_label}"
      mkdir -p "${outputs_dir}"

      echo "Starting group=${group_index} repeat=${repeat} method=${method} env_id=${env_id} episodes=${episodes} max_steps=${max_steps} seed_offset=${seed_offset} outputs=${outputs_dir}"

      if "${UV}" run python "${REPO_DIR}/run.py" \
        --method "${method}" \
        --env minigrid \
        --minigrid-id "${env_id}" \
        --minigrid-max-steps "${max_steps}" \
        --seed-offset "${seed_offset}" \
        --episodes "${episodes}" \
        --model "${MODEL}" \
        --server "http://127.0.0.1:${port}/v1" \
        --outputs-dir "${outputs_dir}" \
        --disable-thinking; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\tdone\t%s\n" \
          "${group_index}" "${method}" "${env_id}" "${repeat}" "${episodes}" "${max_steps}" "${seed_offset}" "${outputs_dir}" >> "${shard_status}"
        echo "Done group=${group_index} repeat=${repeat} OUTPUTS_DIR=${outputs_dir}"
      else
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\tfailed\t%s\n" \
          "${group_index}" "${method}" "${env_id}" "${repeat}" "${episodes}" "${max_steps}" "${seed_offset}" "${outputs_dir}" >> "${shard_status}"
        echo "ERROR group=${group_index} repeat=${repeat} method=${method} env_id=${env_id} failed; continuing" >&2
      fi
    done
    group_index=$((group_index + 1))
  done

  echo "Shard done. STATUS=${shard_status}"
}

print_assignments() {
  local gpu_kind task record
  for gpu_kind in a100 l40s l40; do
    local max_task=0
    case "${gpu_kind}" in
      a100) max_task=2 ;;
      l40s) max_task=2 ;;
      l40) max_task=5 ;;
    esac
    for task in $(seq 0 "${max_task}"); do
      load_groups "${gpu_kind}" "${task}"
      echo "=== ${gpu_kind}:${task} ==="
      for record in "${GROUPS[@]}"; do
        echo "${record}"
      done
    done
  done
}

submit_job() {
  local gpu_kind="$1"
  local partition="$2"
  local gpu_request="$3"
  local array_spec="$4"
  local array_max_concurrent="$5"
  local time_limit="$6"
  local port_base="$7"
  local job_suffix="$8"
  local job_name="${JOB_NAME_PREFIX}-${job_suffix}"
  local script_path
  script_path="$(readlink -f "${BASH_SOURCE[0]}")"

  local sbatch_cmd=(
    sbatch
    --job-name="${job_name}"
    --account="${ACCOUNT}"
    --partition="${partition}"
    --array="${array_spec}%${array_max_concurrent}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --gpus="${gpu_request}"
    --mem="${MEM}"
    --time="${time_limit}"
    --output="${LOG_DIR}/%x-%A_%a.out"
    --error="${LOG_DIR}/%x-%A_%a.err"
    "${script_path}"
    --worker
    "${gpu_kind}"
    "${port_base}"
  )

  echo "Submitting ${job_name}: partition=${partition}, gpu=${gpu_request}, array=${array_spec}%${array_max_concurrent}, port_base=${port_base}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q ' "${sbatch_cmd[@]}"
    echo
  else
    "${sbatch_cmd[@]}"
  fi
}

main_submit() {
  mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}"

  echo "Preflight:"
  test -f "${REPO_DIR}/run.py"
  test -f "${REPO_DIR}/environments/minigrid_env.py"
  test -f "${REPO_DIR}/methods/ace_once.py"
  test -f "${REPO_DIR}/methods/notebook_minimal.py"
  test -f "${REPO_DIR}/methods/notebook_minimal_mechanism.py"
  test -x "${UV}"
  test -x "${SGLANG}"

  echo "Assignments:"
  print_assignments

  if [[ "${SUBMIT_A100}" == "1" ]]; then
    submit_job \
      "a100" \
      "${A100_PARTITION}" \
      "${A100_GPU_REQUEST}" \
      "0-2" \
      "${A100_ARRAY_MAX_CONCURRENT}" \
      "${A100_TIME_LIMIT}" \
      30000 \
      "a100"
  fi

  if [[ "${SUBMIT_L40}" == "1" ]]; then
    submit_job \
      "l40" \
      "${L40_PARTITION}" \
      "${L40_GPU_REQUEST}" \
      "0-5" \
      "${L40_ARRAY_MAX_CONCURRENT}" \
      "${L40_TIME_LIMIT}" \
      30100 \
      "l40"
  fi

  if [[ "${SUBMIT_L40S}" == "1" ]]; then
    submit_job \
      "l40s" \
      "${L40S_PARTITION}" \
      "${L40S_GPU_REQUEST}" \
      "0-2" \
      "${L40S_ARRAY_MAX_CONCURRENT}" \
      "${L40S_TIME_LIMIT}" \
      30200 \
      "l40s"
  fi
}

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
  esac
done

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_worker "$@"
else
  main_submit
fi
