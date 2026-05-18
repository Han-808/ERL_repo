#!/usr/bin/env bash
set -euo pipefail

# Controlled passk eval from each updater variant's own online logs.
#
# This consumes llm_calls produced by:
#   submit_online_notebook_minimal_variants_qwen3_14b_nothink_a100.sh
# and runs the same predefined evaluation shape:
#   k=40 states, y=8 updater samples, z=20 shared test instances.
#
# GPU shape:
#   2 variants x 4 one-GPU shards each = 8 GPUs total by default.
#   Per variant: 2 L40 + 2 L40S.
#
# Usage:
#   bash submit_passk_variant_own_logs_y8z20_8gpu.sh --dry-run
#   bash submit_passk_variant_own_logs_y8z20_8gpu.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="${REPO_DIR:-/gscratch/h2lab/mohanc3/projects/ERL_repo}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_DIR}/runs}"

MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
NUM_STATES="${NUM_STATES:-40}"
SAMPLES_Y="${SAMPLES_Y:-8}"
GAMES_Z="${GAMES_Z:-20}"
BASE_SEED="${BASE_SEED:-20260509}"
PER_VARIANT_L40_SHARDS="${PER_VARIANT_L40_SHARDS:-2}"
PER_VARIANT_L40S_SHARDS="${PER_VARIANT_L40S_SHARDS:-2}"
PER_VARIANT_NUM_SHARDS=$((PER_VARIANT_L40_SHARDS + PER_VARIANT_L40S_SHARDS))

latest_trace() {
  local method="$1"
  local env_name="$2"
  local trace_name="llm_calls_${method}_${env_name}.jsonl"
  find "${OUTPUTS_ROOT}" -type f -name "${trace_name}" -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

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
  method="notebook_minimal_${variant}"
  frozen_lake_trace_var="${variant^^}_FROZEN_LAKE_TRACE"
  sokoban_trace_var="${variant^^}_SOKOBAN_TRACE"
  frozen_lake_trace="$(printenv "${frozen_lake_trace_var}" || true)"
  sokoban_trace="$(printenv "${sokoban_trace_var}" || true)"

  if [[ -z "${frozen_lake_trace}" ]]; then
    frozen_lake_trace="$(latest_trace "${method}" frozen_lake)"
  fi
  if [[ -z "${sokoban_trace}" ]]; then
    sokoban_trace="$(latest_trace "${method}" sokoban)"
  fi

  if [[ -z "${frozen_lake_trace}" || ! -f "${frozen_lake_trace}" ]]; then
    echo "ERROR: missing frozen_lake trace for ${method}" >&2
    echo "Set ${frozen_lake_trace_var}=/path/to/llm_calls_${method}_frozen_lake.jsonl" >&2
    exit 1
  fi
  if [[ -z "${sokoban_trace}" || ! -f "${sokoban_trace}" ]]; then
    echo "ERROR: missing sokoban trace for ${method}" >&2
    echo "Set ${sokoban_trace_var}=/path/to/llm_calls_${method}_sokoban.jsonl" >&2
    exit 1
  fi

  job_suffix="k${NUM_STATES}-y${SAMPLES_Y}-z${GAMES_Z}-seed${BASE_SEED}"
  job_name="passk-${MODEL_TAG}-nothink-ownlogs-${variant}-${job_suffix}"

  echo "Submitting variant-own-logs passk: ${variant}"
  echo "  frozen_lake trace: ${frozen_lake_trace}"
  echo "  sokoban trace:      ${sokoban_trace}"

  env \
    "${COMMON_ENV[@]}" \
    UPDATER_OBJECTIVE_VARIANT="${variant}" \
    FROZEN_LAKE_TRACE="${frozen_lake_trace}" \
    SOKOBAN_TRACE="${sokoban_trace}" \
    JOB_NAME="${job_name}" \
    bash "${SCRIPT_DIR}/submit_single_turn_passk_qwen3_14b_nothink_array.sh" "$@"
done
