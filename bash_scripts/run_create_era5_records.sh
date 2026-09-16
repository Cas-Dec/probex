#!/usr/bin/env bash
set -euo pipefail

RECORD_TAG="${1:-1979_2021}"
START_YEAR="${2:-1979}"
END_YEAR="${3:-2021}"
HOURS="${4:-0,12}"

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Find probex environment
source "${PROJECT_ROOT}/find_probex_env.sh"

echo "Running ERA5 record climatology creation"
echo "record tag = ${RECORD_TAG}"
echo "start year = ${START_YEAR}"
echo "end year   = ${END_YEAR}"
echo "hours      = ${HOURS}"
echo "workdir    = $(pwd)"
date
hostname

"${PROBEX_PYTHON}" -u download/create_era5_records.py \
  --out-root "${RECORDS_ROOT:-${ERA5_ROOT:-data/wb2}}" \
  --record-tag "${RECORD_TAG}" \
  --start-year "${START_YEAR}" \
  --end-year "${END_YEAR}" \
  --hours "${HOURS}"

echo "Finished ERA5 record climatology creation"
date
