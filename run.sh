#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

METHOD="${METHOD:-ace}"
ENV="${ENV:-both}"
EPISODES="${EPISODES:-20}"
MODEL="${MODEL:-qwen3-8b}"
SERVER="${SERVER:-http://LOCAL_SERVER/v1}"

python "${SCRIPT_DIR}/run.py" \
  --method "$METHOD" \
  --env "$ENV" \
  --episodes "$EPISODES" \
  --model "$MODEL" \
  --server "$SERVER"
