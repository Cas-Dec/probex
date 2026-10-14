#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-FGN}"
VAR="${2:-t2m}"
YEAR="${3:-2022}"
TRUTH="${4:-ERA5}"
RECORD_TAG="${5:-1979_2021}"
LEADS="${6:-48 120 168 240}"
DEPTHS="${7:-0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.5 3}"
COMPARE_MODELS="${8:-IFS_ENS ArchesWeatherGen uCast FGN}"
MIN_SAMPLE_SIZE="${9:-30}"

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

# Find probex environment
source "./find_probex_env.sh"

mkdir -p logs "results/${MODEL}/Depth"

echo "Running Depth evaluation"
echo "model            = ${MODEL}"
echo "variable         = ${VAR}"
echo "year             = ${YEAR}"
echo "truth            = ${TRUTH}"
echo "records          = ${RECORD_TAG}"
echo "leads            = ${LEADS}"
echo "depths           = ${DEPTHS}"
echo "min-sample-size  = ${MIN_SAMPLE_SIZE}"
echo "compare-models   = ${COMPARE_MODELS}"
echo "workdir          = $(pwd)"
date
hostname

# LEADS/DEPTHS/COMPARE_MODELS are intentionally unquoted below: each holds a
# space-separated list that must word-split into several --leads/--depths/
# --compare-models arguments (argparse nargs="+").
"${PROBEX_PYTHON}" -u eval/forecast_evaluation_depth.py \
  --year "${YEAR}" \
  --var "${VAR}" \
  --model "${MODEL}" \
  --truth "${TRUTH}" \
  --record-tag "${RECORD_TAG}" \
  --leads ${LEADS} \
  --depths ${DEPTHS} \
  --min-sample-size "${MIN_SAMPLE_SIZE}" \
  --compare-models ${COMPARE_MODELS}

echo "Finished depth-conditioned evaluation"
date
