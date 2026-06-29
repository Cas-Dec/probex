#!/usr/bin/env bash
set -euo pipefail

VARIABLE="${1:-2m_temperature}"
LEAD_HOUR="${2:-24}"
# run from directory from which we submitted
cd "${PBS_O_WORKDIR}"
source "${HOME}/venvs/weatherbench/bin/activate"

echo "Running GenCast download"
echo "variable    = ${VARIABLE}"
echo "lead hour   = ${LEAD_HOUR}"
echo "year        = 2020"
echo "working dir = $(pwd)"
date
hostname

python -u download/download_gencast_wb2.py \
  --year 2020 \
  --variable "${VARIABLE}" \
  --lead-hour "${LEAD_HOUR}" \
  --out-root "data/GenCast"

echo "Finished single lead GenCast download"
date