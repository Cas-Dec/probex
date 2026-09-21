#!/usr/bin/env bash
set -euo pipefail

VARIABLE="2m_temperature"
LEAD_HOUR="24"
OUT_ROOT=""

POSITIONAL_ARGS=()
while (($#)); do
  case "$1" in
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --out-root=*)
      OUT_ROOT="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [VARIABLE] [LEAD_HOUR] [--out-root PATH]"
      exit 0
      ;;
    --)
      shift
      POSITIONAL_ARGS+=("$@")
      break
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if (( ${#POSITIONAL_ARGS[@]} > 0 )); then
  VARIABLE="${POSITIONAL_ARGS[0]}"
fi
if (( ${#POSITIONAL_ARGS[@]} > 1 )); then
  LEAD_HOUR="${POSITIONAL_ARGS[1]}"
fi

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

# Load local environment defaults when available.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="${FGN_OUT_ROOT:-${FORECAST_ROOT:-${HEAT_DATA:-data}/forecasts}/FGN}"
fi

# Verify probex environment exists
source "${PROJECT_ROOT}/find_probex_env.sh"

echo "Running FGN download"
echo "variable    = ${VARIABLE}"
echo "lead hour   = ${LEAD_HOUR}"
echo "year        = 2022"
echo "working dir = $(pwd)"
date
hostname

"${PROBEX_PYTHON}" -u download/download_fgn_wb2.py \
  --year 2022 \
  --variable "${VARIABLE}" \
  --lead-hour "${LEAD_HOUR}" \
  --out-root "${OUT_ROOT}"

echo "Finished single lead FGN download"
date
