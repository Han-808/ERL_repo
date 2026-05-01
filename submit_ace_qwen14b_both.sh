#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/gscratch/stf/mohanc3/projects/ERL_repo
UV=/gscratch/stf/mohanc3/uv-env/uv-bin/uv
SGLANG=/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/runs"

sbatch \
  --job-name="ace-qwen3-14b-nothink-k30-both" \
  --account=stf \
  --partition=gpu-l40s \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gpus=l40s:1 \
  --mem=64G \
  --time=6:00:00 \
  --output="${REPO_DIR}/logs/%x-%j.out" \
  --error="${REPO_DIR}/logs/%x-%j.err" \
  --wrap="
    set -euo pipefail

    cd ${REPO_DIR}

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

module load cuda/12.4.1
module load gcc/13.2.0


    export CUDA_HOME=/sw/cuda/12.4.1
    export CUDA_PATH=/sw/cuda/12.4.1
    export PATH=/sw/cuda/12.4.1/bin:/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin:\$PATH
    export LD_LIBRARY_PATH=/sw/cuda/12.4.1/lib64:\${LD_LIBRARY_PATH:-}

    export HF_HOME=/gscratch/stf/mohanc3/hf-cache
    export TRANSFORMERS_CACHE=/gscratch/stf/mohanc3/hf-cache
    export HF_DATASETS_CACHE=/gscratch/stf/mohanc3/hf-cache
    export WANDB_DIR=/gscratch/stf/mohanc3/wandb
    export XDG_CACHE_HOME=/gscratch/stf/mohanc3/xdg-cache
    export TORCH_EXTENSIONS_DIR=/gscratch/stf/mohanc3/torch-extensions
    export TVM_FFI_CACHE_DIR=/gscratch/stf/mohanc3/tvm-ffi-cache
    export PYTHONUNBUFFERED=1

    export RUN_DIR=${REPO_DIR}/runs/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}
    mkdir -p \"\$RUN_DIR\" \"\$RUN_DIR/outputs\"

    echo \"Job: \$SLURM_JOB_NAME \$SLURM_JOB_ID\"
    echo \"Node: \$(hostname)\"
    echo \"Model: Qwen/Qwen3-14B\"
    echo \"Env: both\"
    which nvcc
    nvcc --version

    ${SGLANG} serve \
      --model-path Qwen/Qwen3-14B \
      --host 127.0.0.1 \
      --port 30000 \
      --mem-fraction-static 0.65 \
      --disable-cuda-graph \
      --attention-backend triton \
      --sampling-backend pytorch \
      > \"\$RUN_DIR/sglang_server.log\" 2>&1 &

    SERVER_PID=\$!

    cleanup() {
      kill \$SERVER_PID 2>/dev/null || true
    }
    trap cleanup EXIT

    echo \"Waiting for SGLang server...\"
    for i in \$(seq 1 120); do
      if curl --noproxy \"*\" -s http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
        echo \"Server ready after \$((i * 5)) seconds\"
        break
      fi
      sleep 5
    done

    if ! curl --noproxy \"*\" -s http://127.0.0.1:30000/v1/models >/dev/null 2>&1; then
      echo \"ERROR: SGLang server failed to start\"
      tail -100 \"\$RUN_DIR/sglang_server.log\" || true
      exit 1
    fi

    ${UV} run python ${REPO_DIR}/run.py \
      --method ace \
      --env both \
      --episodes 30 \
      --model Qwen/Qwen3-14B \
      --server http://127.0.0.1:30000/v1 \
      --outputs-dir \"\$RUN_DIR/outputs\" \
      --disable-thinking \
      2>&1 | tee \"\$RUN_DIR/ace_qwen3-14b_nothink_k30_both.log\"

    echo \"Done. RUN_DIR=\$RUN_DIR\"
  "

