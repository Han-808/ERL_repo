#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)

if [ -z "${ID:-}" ]; then
  echo "Error: ID environment variable is not set."
  exit 1
fi

METHOD_LIST=(
  # "ace"               # full ACE: GT + test report + initial playbook
  # "ace_nogt"          # no ground truth code
  # "ace_notest"        # no test report
  # "ace_pb_empty"      # empty playbook
  # "ace_pb_null"     # null playbook
  # "ace_nogt_notest" # minimal: no GT, no test report
  # "hypothesis_v2"
  # "hypothesis_v4_gt"
  # "hypothesis_v3"
  # "ace_once"
  "notebook_minimal"
)

# uv run --python libs/ace/.venv/bin/python 
source "${PROJECT_DIR}/libs/ace/.venv/bin/activate"

# for METHOD in "${METHOD_LIST[@]}"; do
#   python "${SCRIPT_DIR}/run.py" \
#     --method "$METHOD" \
#     --model-name Qwen/Qwen3-8B
# done

# select $ID's method to run
METHOD="${METHOD_LIST[$ID]}"

# CUDA_VISIBLE_DEVICES=$ID \
# python "${SCRIPT_DIR}/run.py" \
#   --method "$METHOD" \
#   --model-name Qwen/Qwen3-8B \
#   --max-steps 10

# CUDA_VISIBLE_DEVICES=$ID \
# python "${SCRIPT_DIR}/run.py" \
#   --method "$METHOD" \
#   --model-name Qwen/Qwen3-4B \
#   --experiment-name "$METHOD-qwen3-4b" \
#   --max-steps 10

# CUDA_VISIBLE_DEVICES=$ID \
# python "${SCRIPT_DIR}/run.py" \
#   --method "$METHOD" \
#   --model-name Qwen/Qwen3-8B \
#   --experiment-name "$METHOD-qwen3-8b" \
#   --max-steps 10

# CUDA_VISIBLE_DEVICES=$ID \
# python "${SCRIPT_DIR}/run.py" \
#   --method "$METHOD" \
#   --model-name Qwen/Qwen3-14B \
#   --experiment-name "$METHOD-qwen3-14b" \
#   --max-steps 10

# CUDA_VISIBLE_DEVICES=$ID \
# python "${SCRIPT_DIR}/run.py" \
#   --method "$METHOD" \
#   --model-name Qwen/Qwen3-32B \
#   --experiment-name "$METHOD-qwen3-32b" \
#   --max-steps 10

CUDA_VISIBLE_DEVICES=$ID \
python "${SCRIPT_DIR}/run.py" \
  --task-limit 2 \
  --method "$METHOD" \
  --model-name Qwen/Qwen3.5-4B \
  --experiment-name "$METHOD-qwen3.5-4b-think" \
  --max-steps 10

CUDA_VISIBLE_DEVICES=$ID \
python "${SCRIPT_DIR}/run.py" \
  --task-limit 2 \
  --method "$METHOD" \
  --model-name Qwen/Qwen3.5-4B \
  --experiment-name "$METHOD-qwen3.5-4b-nothink" \
  --max-steps 10 \
  --disable-thinking
