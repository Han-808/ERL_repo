#!/usr/bin/env bash
set -euo pipefail

# Controlled updater-objective ablation from the original notebook_minimal logs.
#
# This runs both updater variants against the same recovered original
# notebook_minimal states and the same predefined y8z20 evaluation grid:
#   k=40 states, y=8 updater samples, z=20 shared test instances.
#
# GPU shape:
#   2 variants x 4 one-GPU shards each = 8 GPUs total by default.
#   Per variant: 2 L40 + 2 L40S.
#
# Usage:
#   bash submit_passk_original_logs_updater_variants_y8z20_8gpu.sh --dry-run
#   bash submit_passk_original_logs_updater_variants_y8z20_8gpu.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
NUM_STATES="${NUM_STATES:-40}"
SAMPLES_Y="${SAMPLES_Y:-8}"
GAMES_Z="${GAMES_Z:-20}"
BASE_SEED="${BASE_SEED:-20260509}"
PER_VARIANT_L40_SHARDS="${PER_VARIANT_L40_SHARDS:-2}"
PER_VARIANT_L40S_SHARDS="${PER_VARIANT_L40S_SHARDS:-2}"
PER_VARIANT_NUM_SHARDS=$((PER_VARIANT_L40_SHARDS + PER_VARIANT_L40S_SHARDS))

COMMON_ENV=(
  ABLATION_ORIGINAL_VS_UPDATED=1
  ENABLE_PLOTS="${ENABLE_PLOTS:-0}"
  L40_SHARDS="${PER_VARIANT_L40_SHARDS}"
  L40S_SHARDS="${PER_VARIANT_L40S_SHARDS}"
  L40_ARRAY_MAX_CONCURRENT="${PER_VARIANT_L40_SHARDS}"
  L40S_ARRAY_MAX_CONCURRENT="${PER_VARIANT_L40S_SHARDS}"
  NUM_SHARDS="${PER_VARIANT_NUM_SHARDS}"
  NUM_STATES="${NUM_STATES}"
  SAMPLES_Y="${SAMPLES_Y}"
  GAMES_Z="${GAMES_Z}"
  BASE_SEED="${BASE_SEED}"
)

for variant in thinkahead mechanism; do
  job_suffix="k${NUM_STATES}-y${SAMPLES_Y}-z${GAMES_Z}-seed${BASE_SEED}"
  job_name="passk-${MODEL_TAG}-nothink-origlogs-${variant}-${job_suffix}"

  echo "Submitting controlled original-logs passk variant: ${variant}"
  env \
    "${COMMON_ENV[@]}" \
    UPDATER_OBJECTIVE_VARIANT="${variant}" \
    JOB_NAME="${job_name}" \
    bash "${SCRIPT_DIR}/submit_single_turn_passk_qwen3_14b_nothink_array.sh" "$@"
done
