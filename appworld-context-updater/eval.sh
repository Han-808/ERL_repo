#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)

# uv run --python libs/ace/.venv/bin/python 
source "${PROJECT_DIR}/libs/ace/.venv/bin/activate"

# python evaluate.py ace-qwen3-8b --dataset train

# # Evaluate specific experiments
# python evaluate.py ace-qwen3-8b ace_nogt-qwen3-8b --dataset train

# # Evaluate everything under outputs/
python evaluate.py --all --dataset train


# # Skip re-running if evaluation JSON already exists
# python evaluate.py --all --dataset train --skip-existing

# # Custom outputs dir
# python evaluate.py --all --dataset train --outputs-dir /path/to/outputs


# python evaluate.py --csv
