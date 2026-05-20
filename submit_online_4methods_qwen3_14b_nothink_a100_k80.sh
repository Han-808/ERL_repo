#!/usr/bin/env bash
set -euo pipefail

# Online 80-episode runs for four methods on four A100 GPUs.
#
# Slurm array mapping:
#   task 0 -> ace_once
#   task 1 -> notebook_minimal
#   task 2 -> notebook_minimal_thinkahead
#   task 3 -> notebook_minimal_mechanism
#
# Each task starts one Qwen3-14B SGLang server on one A100, then runs both:
#   frozen_lake and sokoban
#
# Outputs per task:
#   runs/<run-label>/results_<method>_frozen_lake.json
#   runs/<run-label>/results_<method>_sokoban.json
#   runs/<run-label>/llm_calls_<method>_frozen_lake.jsonl
#   runs/<run-label>/llm_calls_<method>_sokoban.jsonl
#
# Usage:
#   bash submit_online_4methods_qwen3_14b_nothink_a100_k80.sh --dry-run
#   bash submit_online_4methods_qwen3_14b_nothink_a100_k80.sh

REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
UV="${UV:-/gscratch/stf/mohanc3/uv-env/uv-bin/uv}"
SGLANG="${SGLANG:-/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang}"

ACCOUNT="${ACCOUNT:-h2lab}"
A100_PARTITION="${A100_PARTITION:-gpu-a100}"
A100_GPU_REQUEST="${A100_GPU_REQUEST:-a100:1}"
A100_SHARDS="${A100_SHARDS:-4}"
A100_ARRAY_MAX_CONCURRENT="${A100_ARRAY_MAX_CONCURRENT:-4}"

MODEL="${MODEL:-Qwen/Qwen3-14B}"
MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
EPISODES="${EPISODES:-80}"

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-128G}"
TIME_LIMIT="${TIME_LIMIT:-36:00:00}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"

JOB_NAME="${JOB_NAME:-online-${MODEL_TAG}-nothink-4methods-k${EPISODES}}"
LOG_DIR="${REPO_DIR}/logs"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/runs}"

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

mkdir -p "${LOG_DIR}" "${OUTPUTS_ROOT}"

echo "Preflight:"
test -f "${REPO_DIR}/run.py"
test -f "${REPO_DIR}/methods/ace_once.py"
test -f "${REPO_DIR}/methods/notebook_minimal.py"
test -f "${REPO_DIR}/methods/notebook_minimal_thinkahead.py"
test -f "${REPO_DIR}/methods/notebook_minimal_mechanism.py"
test -x "${UV}"
test -x "${SGLANG}"

if [[ "${A100_SHARDS}" -ne 4 ]]; then
  echo "ERROR: this script maps exactly 4 array tasks; leave A100_SHARDS=4." >&2
  exit 1
fi

SBATCH_CMD=(
sbatch
--job-name="${JOB_NAME}"
--account="${ACCOUNT}"
--partition="${A100_PARTITION}"
--array="0-$((A100_SHARDS - 1))%${A100_ARRAY_MAX_CONCURRENT}"
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

  TASK_ID=\${SLURM_ARRAY_TASK_ID}
  case \"\$TASK_ID\" in
    0)
      METHOD='ace_once'
      ;;
    1)
      METHOD='notebook_minimal'
      ;;
    2)
      METHOD='notebook_minimal_thinkahead'
      ;;
    3)
      METHOD='notebook_minimal_mechanism'
      ;;
    *)
      echo \"ERROR: unexpected task id \$TASK_ID\" >&2
      exit 1
      ;;
  esac

  PORT=\$((30000 + TASK_ID))
  RUN_LABEL=\"\${SLURM_JOB_NAME}-\${SLURM_ARRAY_JOB_ID}_\${TASK_ID}-\${METHOD}-both\"
  OUTPUTS_DIR='${OUTPUTS_ROOT}'/\"\$RUN_LABEL\"
  SGLANG_LOG='${LOG_DIR}'/\"\$RUN_LABEL-sglang_server.log\"

  mkdir -p \"\$OUTPUTS_DIR\"

  echo \"Job: \$SLURM_JOB_NAME array=\$SLURM_ARRAY_JOB_ID task=\$TASK_ID\"
  echo \"Node: \$(hostname)\"
  echo \"Model: ${MODEL}\"
  echo \"Method: \$METHOD\"
  echo \"Env: both\"
  echo \"Episodes per env: ${EPISODES}\"
  echo \"Partition/account: ${A100_PARTITION}/${ACCOUNT}\"
  echo \"GPU request: ${A100_GPU_REQUEST}\"
  echo \"Port: \$PORT\"
  echo \"Outputs dir: \$OUTPUTS_DIR\"

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

  '${UV}' run python '${REPO_DIR}/run.py' \
    --method \"\$METHOD\" \
    --env both \
    --episodes '${EPISODES}' \
    --model '${MODEL}' \
    --server \"http://127.0.0.1:\$PORT/v1\" \
    --outputs-dir \"\$OUTPUTS_DIR\" \
    --disable-thinking

  echo \"Done. OUTPUTS_DIR=\$OUTPUTS_DIR\"
"
)

echo "Submitting ${JOB_NAME}: account=${ACCOUNT}, partition=${A100_PARTITION}, gpu=${A100_GPU_REQUEST}, array=0-3%${A100_ARRAY_MAX_CONCURRENT}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '%q ' "${SBATCH_CMD[@]}"
  echo
else
  "${SBATCH_CMD[@]}"
fi
