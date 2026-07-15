#!/bin/bash
# Helper script to find and activate the probex conda environment
# This makes bash scripts portable across different conda installations

# Try to find probex environment in common conda locations
PROBEX_ENV=""

# Try conda env list first
if command -v conda &> /dev/null; then
    PROBEX_ENV=$(conda info --envs 2>/dev/null | grep probex | awk '{print $NF}')
fi

# If not found, try searching in default conda locations
if [ -z "$PROBEX_ENV" ]; then
    for base_env in ~/.conda/envs ~/.local/miniconda/envs ~/.local/miniconda3/envs ~/miniconda/envs ~/miniconda3/envs ~/mambaforge/envs; do
        if [ -d "$base_env/probex" ]; then
            PROBEX_ENV="$base_env/probex"
            break
        fi
    done
fi

if [ -z "$PROBEX_ENV" ]; then
    echo "ERROR: probex conda environment not found"
    echo "Please create it from the project root with:"
    echo "  conda env create -f environment.yaml"
    return 1 2>/dev/null || exit 1
fi

# Export the path so scripts can use it
export PROBEX_ENV
export PROBEX_PYTHON="$PROBEX_ENV/bin/python"

# Verify it works
if ! "$PROBEX_PYTHON" -c "import sys" 2>/dev/null; then
    echo "ERROR: Found probex at $PROBEX_ENV but Python is not functional"
    return 1 2>/dev/null || exit 1
fi
