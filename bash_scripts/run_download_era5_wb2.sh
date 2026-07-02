#!/usr/bin/env bash
set -euo pipefail

VARIABLE="${1:-2m_temperature}"

# run from directory from which we submitted
cd "${PBS_O_WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate probex

echo "Running ERA5 download"
echo "variable    = ${VARIABLE}"
echo "year        = 2022"
echo "working dir = $(pwd)"
date
hostname

python -u download/download_era5_wb2.py \
  --year 2022 \
  --variable "${VARIABLE}" \
  --out-root "data/ERA5"

echo "Finished ERA5 download"
date