#!/usr/bin/env bash
set -euo pipefail

# Full MiniGrid online runs split across 9 long-lived SGLang servers:
#   A100 job: 3 array tasks, all ace_once_minigrid
#   L40 job:  6 array tasks, 3 notebook_minimal_minigrid and
#             3 notebook_minimal_mechanism_minigrid
#
# Each array task starts one SGLang server and runs its env shard sequentially.
# Every (method, env) run gets a fresh output directory under:
#   ${REPO_DIR}/minigrid/<run-label>/
#
# Usage:
#   bash submit_minigrid_online_9shard_qwen3_14b_nothink.sh --dry-run
#   bash submit_minigrid_online_9shard_qwen3_14b_nothink.sh
#
# Useful overrides:
#   EPISODES=80 bash submit_minigrid_online_9shard_qwen3_14b_nothink.sh
#   SUBMIT_A100=1 SUBMIT_L40=0 bash submit_minigrid_online_9shard_qwen3_14b_nothink.sh

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
EPISODES="${EPISODES:-80}"

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-128G}"
A100_TIME_LIMIT="${A100_TIME_LIMIT:-72:00:00}"
L40_TIME_LIMIT="${L40_TIME_LIMIT:-72:00:00}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"

A100_PARTITION="${A100_PARTITION:-gpu-a100}"
A100_GPU_REQUEST="${A100_GPU_REQUEST:-a100:1}"
A100_ARRAY_MAX_CONCURRENT="${A100_ARRAY_MAX_CONCURRENT:-3}"

L40_PARTITION="${L40_PARTITION:-gpu-l40}"
L40_GPU_REQUEST="${L40_GPU_REQUEST:-l40:1}"
L40_ARRAY_MAX_CONCURRENT="${L40_ARRAY_MAX_CONCURRENT:-6}"

SUBMIT_A100="${SUBMIT_A100:-1}"
SUBMIT_L40="${SUBMIT_L40:-1}"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-minigrid-online-${MODEL_TAG}-nothink-k${EPISODES}}"
LOG_DIR="${REPO_DIR}/logs"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/minigrid}"

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}"

echo "Preflight:"
test -f "${REPO_DIR}/run.py"
test -f "${REPO_DIR}/environments/minigrid_env.py"
test -f "${REPO_DIR}/methods/ace_once.py"
test -f "${REPO_DIR}/methods/notebook_minimal.py"
test -f "${REPO_DIR}/methods/notebook_minimal_mechanism.py"
test -x "${UV}"
test -x "${SGLANG}"

submit_sharded_job() {
  local chunk_name="$1"
  local partition="$2"
  local gpu_request="$3"
  local array_spec="$4"
  local array_max_concurrent="$5"
  local time_limit="$6"
  local port_base="$7"
  local method_map="$8"
  local shard_map="$9"
  local job_name="${JOB_NAME_PREFIX}-${chunk_name}"

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

    METHOD_MAP=(${method_map})
    ENV_SHARDS=(${shard_map})
    MINIGRID_IDS=(
      'MiniGrid-BlockedUnlockPickup-v0'
      'MiniGrid-LavaCrossingS9N3-v0'
      'MiniGrid-SimpleCrossingS9N3-v0'
      'MiniGrid-DistShift1-v0'
      'MiniGrid-Dynamic-Obstacles-Random-5x5-v0'
      'MiniGrid-Empty-Random-5x5-v0'
      'MiniGrid-Fetch-5x5-N2-v0'
      'MiniGrid-FourRooms-v0'
      'MiniGrid-LavaGapS5-v0'
      'MiniGrid-MemoryS13-v0'
      'MiniGrid-MemoryS11-v0'
    )

    TASK_ID=\${SLURM_ARRAY_TASK_ID}
    METHOD=\${METHOD_MAP[\$TASK_ID]}
    ENV_INDEX_CSV=\${ENV_SHARDS[\$TASK_ID]}
    IFS=',' read -r -a ENV_INDICES <<< \"\$ENV_INDEX_CSV\"
    PORT=\$((${port_base} + TASK_ID))
    SERVER_LABEL=\"\${SLURM_JOB_NAME}-\${SLURM_ARRAY_JOB_ID}_\${TASK_ID}-\${METHOD}\"
    SGLANG_LOG='${LOG_DIR}'/\"\$SERVER_LABEL-sglang_server.log\"
    SHARD_STATUS='${OUTPUTS_ROOT}'/\"\$SERVER_LABEL-status.tsv\"

    echo \"Job: \$SLURM_JOB_NAME array=\$SLURM_ARRAY_JOB_ID task=\$TASK_ID\"
    echo \"Node: \$(hostname)\"
    echo \"Model: ${MODEL}\"
    echo \"Method: \$METHOD\"
    echo \"Env indices: \$ENV_INDEX_CSV\"
    echo \"Episodes per env: ${EPISODES}\"
    echo \"Partition/account: ${partition}/${ACCOUNT}\"
    echo \"GPU request: ${gpu_request}\"
    echo \"Port: \$PORT\"
    echo -e \"env_index\\tenv_id\\tstatus\\toutputs_dir\" > \"\$SHARD_STATUS\"

    for ENV_IDX in \"\${ENV_INDICES[@]}\"; do
      MINIGRID_ID=\${MINIGRID_IDS[\$ENV_IDX]}
      '${UV}' run python -c \"from environments.minigrid_env import MiniGridTextEnv; env=MiniGridTextEnv('\$MINIGRID_ID'); env.close(); print('MiniGrid smoke ok: \$MINIGRID_ID')\"
    done

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
    for i in \$(seq 1 240); do
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

    for ENV_IDX in \"\${ENV_INDICES[@]}\"; do
      MINIGRID_ID=\${MINIGRID_IDS[\$ENV_IDX]}
      ENV_LABEL=\$('${UV}' run python -c \"from run import minigrid_output_label; print(minigrid_output_label('\$MINIGRID_ID'))\")
      RUN_LABEL=\"\${SLURM_JOB_NAME}-\${SLURM_ARRAY_JOB_ID}_\${TASK_ID}-\${METHOD}-\${ENV_LABEL}\"
      OUTPUTS_DIR='${OUTPUTS_ROOT}'/\"\$RUN_LABEL\"

      mkdir -p \"\$OUTPUTS_DIR\"
      echo \"Starting env_index=\$ENV_IDX env_id=\$MINIGRID_ID outputs=\$OUTPUTS_DIR\"

      if '${UV}' run python '${REPO_DIR}/run.py' \
        --method \"\$METHOD\" \
        --env minigrid \
        --minigrid-id \"\$MINIGRID_ID\" \
        --episodes '${EPISODES}' \
        --model '${MODEL}' \
        --server \"http://127.0.0.1:\$PORT/v1\" \
        --outputs-dir \"\$OUTPUTS_DIR\" \
        --disable-thinking; then
        echo -e \"\$ENV_IDX\\t\$MINIGRID_ID\\tdone\\t\$OUTPUTS_DIR\" >> \"\$SHARD_STATUS\"
        echo \"Done env_index=\$ENV_IDX env_id=\$MINIGRID_ID OUTPUTS_DIR=\$OUTPUTS_DIR\"
      else
        echo -e \"\$ENV_IDX\\t\$MINIGRID_ID\\tfailed\\t\$OUTPUTS_DIR\" >> \"\$SHARD_STATUS\"
        echo \"ERROR env_index=\$ENV_IDX env_id=\$MINIGRID_ID failed; continuing\" >&2
      fi
    done

    echo \"Shard done. STATUS=\$SHARD_STATUS\"
  "
  )

  echo "Submitting ${job_name}: partition=${partition}, gpu=${gpu_request}, array=${array_spec}%${array_max_concurrent}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '%q ' "${sbatch_cmd[@]}"
    echo
  else
    "${sbatch_cmd[@]}"
  fi
}

ACE_METHODS="ace_once_minigrid ace_once_minigrid ace_once_minigrid"
NOTEBOOK_METHODS="notebook_minimal_minigrid notebook_minimal_minigrid notebook_minimal_minigrid notebook_minimal_mechanism_minigrid notebook_minimal_mechanism_minigrid notebook_minimal_mechanism_minigrid"

# Balanced from the 1-episode smoke timings on A100. The slowest environment
# for each method is isolated so the long shard determines less of the run.
ACE_ENV_SHARDS="0 2,6,5,4,7,8 10,9,1,3"
NOTEBOOK_ENV_SHARDS="0 2,7,5,8,3,4 6,10,1,9"
MECHANISM_ENV_SHARDS="9 0,7,5,4,8 2,3,10,6,1"
L40_ENV_SHARDS="${NOTEBOOK_ENV_SHARDS} ${MECHANISM_ENV_SHARDS}"

if [[ "${SUBMIT_A100}" == "1" ]]; then
  submit_sharded_job \
    "a100-ace-once" \
    "${A100_PARTITION}" \
    "${A100_GPU_REQUEST}" \
    "0-2" \
    "${A100_ARRAY_MAX_CONCURRENT}" \
    "${A100_TIME_LIMIT}" \
    30000 \
    "${ACE_METHODS}" \
    "${ACE_ENV_SHARDS}"
fi

if [[ "${SUBMIT_L40}" == "1" ]]; then
  submit_sharded_job \
    "l40-notebooks" \
    "${L40_PARTITION}" \
    "${L40_GPU_REQUEST}" \
    "0-5" \
    "${L40_ARRAY_MAX_CONCURRENT}" \
    "${L40_TIME_LIMIT}" \
    30100 \
    "${NOTEBOOK_METHODS}" \
    "${L40_ENV_SHARDS}"
fi
