#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-GenCast}"
VAR="${2:-t2m}"
YEAR="${3:-2020}"
TRUTH="${4:-ERA5}"
RECORD_TAG="${5:-1979_2018}"

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
if [[ ! -f "${PROJECT_ROOT}/find_probex_env.sh" && -f "${PROJECT_ROOT}/../find_probex_env.sh" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Verify probex environment exists
source "./find_probex_env.sh"

mkdir -p logs "results/${MODEL}/Point"

echo "Running Point evaluation"
echo "model      = ${MODEL}"
echo "variable   = ${VAR}"
echo "year       = ${YEAR}"
echo "truth      = ${TRUTH}"
echo "records    = ${RECORD_TAG}"
echo "workdir    = $(pwd)"
date
hostname

"${PROBEX_PYTHON}" -u eval/forecast_evaluation_point.py \
  --year "${YEAR}" \
  --var "${VAR}" \
  --model "${MODEL}" \
  --truth "${TRUTH}" \
  --record-tag "${RECORD_TAG}"

echo "Finished point forecast evaluation"
date
