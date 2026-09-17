#!/bin/bash
# Compare ERA5 vs HRES-fc0 as verification datasets
#
# Usage:
#   bash bash_scripts/run_compare_verification.sh t2m 1979_2021
#   bash bash_scripts/run_compare_verification.sh w10 1979_2021

set -e  # Exit on error

VAR=${1:-t2m}
RECORD_TAG=${2:-1979_2021}

# Locate and activate conda environment
if [ -n "$CONDA_EXE" ]; then
    CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "$VSC_DATA/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$VSC_DATA/miniconda3"
else
    echo "ERROR: Cannot locate conda installation"
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate probex environment
ENV_NAME="probex"
if conda env list | grep -q "^${ENV_NAME} "; then
    conda activate $ENV_NAME
    echo "Activated conda environment: $ENV_NAME"
else
    echo "ERROR: Conda environment '$ENV_NAME' not found"
    exit 1
fi

echo "========================================="
echo "ERA5 vs HRES-fc0 Verification Comparison"
echo "========================================="
echo "Variable: $VAR"
echo "Record tag: $RECORD_TAG"
echo "Started: $(date)"
echo "========================================="

# Run analysis
python -u analysis/compare_era5_hresfc0_verification.py \
    --var $VAR \
    --record-tag $RECORD_TAG

echo "========================================="
echo "Completed: $(date)"
echo "========================================="
