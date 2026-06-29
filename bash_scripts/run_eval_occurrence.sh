#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-GenCast}"
VAR="${2:-t2m}"
YEAR="${3:-2020}"
TRUTH="${4:-ERA5}"
RECORD_TAG="${5:-1979_2018}"

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate probex

mkdir -p logs "results/${MODEL}/Occurrence"

# The eval scripts use ./data. If your downloads are in ./Data, make a lowercase link.
if [[ ! -d data && -d Data ]]; then
  ln -s Data data
fi

echo "Running occurrence evaluation"
echo "model      = ${MODEL}"
echo "variable   = ${VAR}"
echo "year       = ${YEAR}"
echo "truth      = ${TRUTH}"
echo "records    = ${RECORD_TAG}"
echo "workdir    = $(pwd)"
date
hostname

python -u eval/forecast_evaluation_occurrence.py \
  --year "${YEAR}" \
  --var "${VAR}" \
  --model "${MODEL}" \
  --truth "${TRUTH}" \
  --record-tag "${RECORD_TAG}"

echo "Finished occurrence evaluation"
date
