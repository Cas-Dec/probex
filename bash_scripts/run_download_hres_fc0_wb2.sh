#!/usr/bin/env bash
set -euo pipefail

VARIABLE="${1:-2m_temperature}"

# run from directory from which we submitted
cd "${PBS_O_WORKDIR}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

OUT_ROOT="${HRES_FC0_OUT_ROOT:-${FORECAST_ROOT:-${HEAT_DATA:-data}/forecasts}/HRES_fc0}"

# Verify probex environment exists
source "${PROJECT_ROOT}/find_probex_env.sh"

echo "Running HRES_fc0 download"
echo "variable    = ${VARIABLE}"
echo "year        = 2022"
echo "working dir = $(pwd)"
date
hostname

"${PROBEX_PYTHON}" -u download/download_hres_fc0_wb2.py \
  --year 2022 \
  --variable "${VARIABLE}" \
  --out-root "${OUT_ROOT}"

echo "Finished HRES-fc0 download"
date