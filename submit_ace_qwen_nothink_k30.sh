#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/gscratch/stf/mohanc3/projects/ERL_repo
UV=/gscratch/stf/mohanc3/uv-env/uv-bin/uv
SGLANG=/mmfs1/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/sglang

mkdir -p /gscratch/stf/mohanc3/projects/ERL_repo/logs
mkdir -p /gscratch/stf/mohanc3/projects/ERL_repo/runs

MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

for MODEL in "${MODELS[@]}"; do
  MODEL_TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's|.*/||')
  JOB_NAME="ace-${MODEL_TAG}-nothink-k30"

  sbatch \
    --job-name="$JOB_NAME" \
    --account=stf \
    --partition=gpu-l40s \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gpus=l40s:1 \
    --mem=64G \
    --time=6:00:00 \
    --output=/gscratch/stf/mohanc3/projects/ERL_repo/logs/%x-%j.out \
    --error=/gscratch/stf/mohanc3/projects/ERL_repo/logs/%x-%j.err \
    --wrap="
      set -euo pipefail

      cd /gscratch/stf/mohanc3/projects/ERL_repo

      unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
      export NO_PROXY=localhost,127.0.0.1
      export no_proxy=localhost,127.0.0.1

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

      mkdir -p /gscratch/stf/mohanc3/hf-cache
      mkdir -p /gscratch/stf/mohanc3/wandb
      mkdir -p /gscratch/stf/mohanc3/xdg-cache
      mkdir -p /gscratch/stf/mohanc3/torch-extensions
      mkdir -p /gscratch/stf/mohanc3/tvm-ffi-cache

      export RUN_DIR=/gscratch/stf/mohanc3/projects/ERL_repo/runs/\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}
      mkdir -p \"\$RUN_DIR\" \"\$RUN_DIR/outputs\"

      echo \"Job: \$SLURM_JOB_NAME \$SLURM_JOB_ID\"
      echo \"Node: \$(hostname)\"
      echo \"Model: ${MODEL}\"
      which nvcc
      nvcc --version

      ${SGLANG} serve \
        --model-path '${MODEL}' \
        --host 127.0.0.1 \
        --port 30000 \
        --mem-fraction-static 0.45 \
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

      ${UV} run python /gscratch/stf/mohanc3/projects/ERL_repo/run.py \
        --method ace \
        --env both \
        --episodes 30 \
        --model '${MODEL}' \
        --server http://127.0.0.1:30000/v1 \
        --outputs-dir \"\$RUN_DIR/outputs\" \
        --disable-thinking \
        2>&1 | tee \"\$RUN_DIR/ace_${MODEL_TAG}_nothink_k30.log\"

      echo \"Done. RUN_DIR=\$RUN_DIR\"
    "
done


