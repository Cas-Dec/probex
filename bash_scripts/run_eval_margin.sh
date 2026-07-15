#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-GenCast}"
VAR="${2:-t2m}"
YEAR="${3:-2020}"
TRUTH="${4:-ERA5}"
RECORD_TAG="${5:-1979_2018}"
MARGIN_MAX="${6:-6.0}"
MARGIN_STEP="${7:-0.25}"

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Verify probex environment exists
source "${PROJECT_ROOT}/find_probex_env.sh"

mkdir -p logs "results/${MODEL}/Margin"

echo "Running Margin evaluation"
echo "model       = ${MODEL}"
echo "variable    = ${VAR}"
echo "year        = ${YEAR}"
echo "truth       = ${TRUTH}"
echo "records     = ${RECORD_TAG}"
echo "margin_max  = ${MARGIN_MAX}"
echo "margin_step = ${MARGIN_STEP}"
echo "workdir     = $(pwd)"
date
hostname

"${PROBEX_PYTHON}" -u eval/forecast_evaluation_margin.py \
  --year "${YEAR}" \
  --var "${VAR}" \
  --model "${MODEL}" \
  --truth "${TRUTH}" \
  --record-tag "${RECORD_TAG}" \
  --margin-max "${MARGIN_MAX}" \
  --margin-step "${MARGIN_STEP}"

echo "Finished margin-conditioned evaluation"
date
