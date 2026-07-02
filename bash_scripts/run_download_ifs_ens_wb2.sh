#!/usr/bin/env bash
set -euo pipefail

VARIABLE="${1:-2m_temperature}"
LEAD_HOUR="${2:-24}"

# run from directory from which we submitted
cd "${PBS_O_WORKDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate probex

echo "Running IFS ENS download"
echo "variable    = ${VARIABLE}"
echo "lead hour   = ${LEAD_HOUR}"
echo "year        = 2022"
echo "working dir = $(pwd)"
date
hostname

python -u download/download_ifs_ens_wb2.py \
  --year 2022 \
  --variable "${VARIABLE}" \
  --lead-hour "${LEAD_HOUR}" \
  --out-root "/data/gent/vo/000/gvo00090/HEAT/Data/forecasts/IFS_ENS"

echo "Finished single lead IFS ENS download"
date