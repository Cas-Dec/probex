#!/usr/bin/env bash
set -euo pipefail

VARIABLE="${1:-2m_temperature}"

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

OUT_ROOT="${ERA5_DOWNLOAD_OUT_ROOT:-${FORECAST_ROOT:-${HEAT_DATA:-data}/forecasts}/ERA5}"

# Verify probex environment exists
source "${PROJECT_ROOT}/find_probex_env.sh"

echo "Running ERA5 download"
echo "variable    = ${VARIABLE}"
echo "year        = 2022"
echo "working dir = $(pwd)"
date
hostname

"${PROBEX_PYTHON}" -u download/download_era5_wb2.py \
  --year 2022 \
  --variable "${VARIABLE}" \
  --out-root "${OUT_ROOT}"

echo "Finished ERA5 download"
date