#!/usr/bin/env bash
set -euo pipefail

VARIABLE="${1:-2m_temperature}"

# run from directory from which we submitted
cd "${PBS_O_WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate probex

echo "Running HRES-fc0 download"
echo "variable    = ${VARIABLE}"
echo "year        = 2022"
echo "working dir = $(pwd)"
date
hostname

python -u download/download_hres_fc0_wb2.py \
  --year 2022 \
  --variable "${VARIABLE}" \
  --out-root "/data/gent/vo/000/gvo00090/HEAT/Data/forecasts/HRES_fc0"

echo "Finished HRES-fc0 download"
date