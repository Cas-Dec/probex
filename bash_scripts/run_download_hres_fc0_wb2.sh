#!/usr/bin/env bash
set -euo pipefail

VARIABLE="${1:-2m_temperature}"

# run from directory from which we submitted
cd "${PBS_O_WORKDIR}"
source "${HOME}/venvs/weatherbench/bin/activate"

echo "Running HRES-fc0 download"
echo "variable    = ${VARIABLE}"
echo "year        = 2020"
echo "working dir = $(pwd)"
date
hostname

python -u download/download_hres_fc0_wb2.py \
  --year 2020 \
  --variable "${VARIABLE}" \
  --out-root "data/HRES_fc0"

echo "Finished HRES-fc0 download"
date