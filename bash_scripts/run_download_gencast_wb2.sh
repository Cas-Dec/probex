#!/usr/bin/env bash
set -euo pipefail

VARIABLE="${1:-2m_temperature}"
LEAD_HOUR="${2:-24}"
# run from directory from which we submitted
PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

OUT_ROOT="${GENCAST_OUT_ROOT:-${FORECAST_ROOT:-${HEAT_DATA:-data}/forecasts}/GenCast}"

# Verify probex environment exists
source "${PROJECT_ROOT}/find_probex_env.sh"

echo "Running GenCast download"
echo "variable    = ${VARIABLE}"
echo "lead hour   = ${LEAD_HOUR}"
echo "year        = 2022"
echo "working dir = $(pwd)"
date
hostname

"${PROBEX_PYTHON}" -u download/download_gencast_wb2.py \
  --year 2020 \
  --variable "${VARIABLE}" \
  --lead-hour "${LEAD_HOUR}" \
  --out-root "${OUT_ROOT}"

echo "Finished single lead GenCast download"
date