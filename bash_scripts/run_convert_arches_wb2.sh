#!/usr/bin/env bash
set -euo pipefail

YEAR="${1:-2022}"

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Find probex environment
source "${PROJECT_ROOT}/find_probex_env.sh"

echo "Running ArchesWeatherGen Zarr conversion"
echo "year       = ${YEAR}"
echo "workdir    = $(pwd)"
date
hostname

"${PROBEX_PYTHON}" -u download/convert_arches_wb2.py --year "${YEAR}"

echo "Finished ArchesWeatherGen Zarr conversion"
date
