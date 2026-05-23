#!/usr/bin/env bash
set -euo pipefail

# Online MiniGrid runs for three context-carrying methods with Qwen3-14B.
#
# Each Slurm array task starts one SGLang server on one GPU and runs exactly
# one (method, MiniGrid env id) pair. Outputs go to a fresh directory under:
#   ${REPO_DIR}/minigrid/<run-label>/
#
# Usage:
#   bash submit_minigrid_online_qwen3_14b_nothink_a100.sh --dry-run
#   bash submit_minigrid_online_qwen3_14b_nothink_a100.sh
#
# Useful overrides:
#   EPISODES=80 A100_ARRAY_MAX_CONCURRENT=4 bash submit_minigrid_online_qwen3_14b_nothink_a100.sh

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
A100_PARTITION="${A100_PARTITION:-gpu-a100}"
A100_GPU_REQUEST="${A100_GPU_REQUEST:-a100:1}"
A100_ARRAY_MAX_CONCURRENT="${A100_ARRAY_MAX_CONCURRENT:-4}"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
EPISODES="${EPISODES:-80}"

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-128G}"
TIME_LIMIT="${TIME_LIMIT:-36:00:00}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"

JOB_NAME="${JOB_NAME:-minigrid-online-${MODEL_TAG}-nothink-k${EPISODES}}"
LOG_DIR="${REPO_DIR}/logs"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/minigrid}"

METHODS=(
  "notebook_minimal_minigrid"
  "notebook_minimal_mechanism_minigrid"
  "ace_once_minigrid"
)

MINIGRID_IDS=(
  "MiniGrid-BlockedUnlockPickup-v0"
  "MiniGrid-LavaCrossingS9N3-v0"
  "MiniGrid-SimpleCrossingS9N3-v0"
  "MiniGrid-DistShift1-v0"
  "MiniGrid-Dynamic-Obstacles-Random-5x5-v0"
  "MiniGrid-Empty-Random-5x5-v0"
  "MiniGrid-Fetch-5x5-N2-v0"
  "MiniGrid-FourRooms-v0"
  "MiniGrid-LavaGapS5-v0"
  "MiniGrid-MemoryS13-v0"
  "MiniGrid-MemoryS11-v0"
)

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

TOTAL_TASKS=$((${#METHODS[@]} * ${#MINIGRID_IDS[@]}))
ARRAY_LAST=$((TOTAL_TASKS - 1))

mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}"

echo "Preflight:"
test -f "${REPO_DIR}/run.py"
test -f "${REPO_DIR}/environments/minigrid_env.py"
test -f "${REPO_DIR}/methods/ace_once.py"
test -f "${REPO_DIR}/methods/notebook_minimal.py"
test -f "${REPO_DIR}/methods/notebook_minimal_mechanism.py"
test -x "${UV}"
test -x "${SGLANG}"

SBATCH_CMD=(
sbatch
--job-name="${JOB_NAME}"
--account="${ACCOUNT}"
--partition="${A100_PARTITION}"
--array="0-${ARRAY_LAST}%${A100_ARRAY_MAX_CONCURRENT}"
--nodes=1
--ntasks=1
--cpus-per-task="${CPUS_PER_TASK}"
--gpus="${A100_GPU_REQUEST}"
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

  METHODS=(
    'notebook_minimal_minigrid'
    'notebook_minimal_mechanism_minigrid'
    'ace_once_minigrid'
  )
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
  NUM_ENVS=\${#MINIGRID_IDS[@]}
  METHOD_IDX=\$((TASK_ID / NUM_ENVS))
  ENV_IDX=\$((TASK_ID % NUM_ENVS))
  METHOD=\${METHODS[\$METHOD_IDX]}
  MINIGRID_ID=\${MINIGRID_IDS[\$ENV_IDX]}

  ENV_LABEL=\$('${UV}' run python -c \"from run import minigrid_output_label; print(minigrid_output_label('\$MINIGRID_ID'))\")
  PORT=\$((30000 + TASK_ID))
  RUN_LABEL=\"\${SLURM_JOB_NAME}-\${SLURM_ARRAY_JOB_ID}_\${TASK_ID}-\${METHOD}-\${ENV_LABEL}\"
  OUTPUTS_DIR='${OUTPUTS_ROOT}'/\"\$RUN_LABEL\"
  SGLANG_LOG='${LOG_DIR}'/\"\$RUN_LABEL-sglang_server.log\"

  mkdir -p \"\$OUTPUTS_DIR\"

  echo \"Job: \$SLURM_JOB_NAME array=\$SLURM_ARRAY_JOB_ID task=\$TASK_ID\"
  echo \"Node: \$(hostname)\"
  echo \"Model: ${MODEL}\"
  echo \"Method: \$METHOD\"
  echo \"MiniGrid id: \$MINIGRID_ID\"
  echo \"Episodes: ${EPISODES}\"
  echo \"Partition/account: ${A100_PARTITION}/${ACCOUNT}\"
  echo \"GPU request: ${A100_GPU_REQUEST}\"
  echo \"Port: \$PORT\"
  echo \"Outputs dir: \$OUTPUTS_DIR\"

  '${UV}' run python -c \"import gymnasium, minigrid; from environments.minigrid_env import MiniGridTextEnv; e=MiniGridTextEnv('\$MINIGRID_ID'); e.close(); print('MiniGrid smoke ok: \$MINIGRID_ID')\"

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

  '${UV}' run python '${REPO_DIR}/run.py' \
    --method \"\$METHOD\" \
    --env minigrid \
    --minigrid-id \"\$MINIGRID_ID\" \
    --episodes '${EPISODES}' \
    --model '${MODEL}' \
    --server \"http://127.0.0.1:\$PORT/v1\" \
    --outputs-dir \"\$OUTPUTS_DIR\" \
    --disable-thinking

  echo \"Done. OUTPUTS_DIR=\$OUTPUTS_DIR\"
"
)

echo "Submitting ${JOB_NAME}: account=${ACCOUNT}, partition=${A100_PARTITION}, gpu=${A100_GPU_REQUEST}, array=0-${ARRAY_LAST}%${A100_ARRAY_MAX_CONCURRENT}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${SBATCH_CMD[@]}"
  echo
else
  "${SBATCH_CMD[@]}"
fi
