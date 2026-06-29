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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate probex

mkdir -p logs "results/${MODEL}/Margin"

# The eval scripts use ./data. If your downloads are in ./Data, make a lowercase link.
if [[ ! -d data && -d Data ]]; then
  ln -s Data data
fi

echo "Running margin-conditioned evaluation"
echo "model       = ${MODEL}"
echo "variable    = ${VAR}"
echo "year        = ${YEAR}"
echo "truth       = ${TRUTH}"
echo "records     = ${RECORD_TAG}"
echo "margin max  = ${MARGIN_MAX}"
echo "margin step = ${MARGIN_STEP}"
echo "workdir     = $(pwd)"
date
hostname

python -u eval/forecast_evaluation_margin.py \
  --year "${YEAR}" \
  --var "${VAR}" \
  --model "${MODEL}" \
  --truth "${TRUTH}" \
  --record-tag "${RECORD_TAG}" \
  --margin-max "${MARGIN_MAX}" \
  --margin-step "${MARGIN_STEP}"

echo "Finished margin-conditioned evaluation"
date
