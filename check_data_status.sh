#!/usr/bin/env bash
# Quick check of data download status (file existence only, no loading)

set -euo pipefail

YEAR="${1:-2022}"
FORECAST_ROOT="/data/gent/vo/000/gvo00090/HEAT/Data/forecasts"

echo "=========================================="
echo "Data Status Check - Year ${YEAR}"
echo "=========================================="

# Function to check if files exist
check_files() {
    local pattern="$1"
    local description="$2"
    local count=$(find ${pattern} -type f 2>/dev/null | wc -l)
    local total="$3"

    if [ $count -eq $total ]; then
        echo "✓ ${description}: ${count}/${total} files"
        return 0
    elif [ $count -gt 0 ]; then
        echo "⚠ ${description}: ${count}/${total} files (INCOMPLETE)"
        return 1
    else
        echo "✗ ${description}: ${count}/${total} files (MISSING)"
        return 1
    fi
}

echo ""
echo "Truth Data (ERA5 - shared store):"
echo "------------------------------------------"
ERA5_STORE="/data/gent/vo/000/gvo00090/HEAT/Data/wb2/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
if [ -d "${ERA5_STORE}" ]; then
    echo "✓ Shared ERA5 store exists (1.5° equiangular, 1959-2023)"
else
    echo "✗ Shared ERA5 store NOT FOUND at ${ERA5_STORE}"
fi

echo ""
echo "Truth Data (HRES_fc0):"
echo "------------------------------------------"
check_files "${FORECAST_ROOT}/HRES_fc0/HRES_fc0_${YEAR}_no_Antarctic_t2m_1p50.zarr" "HRES_fc0 t2m" 1
check_files "${FORECAST_ROOT}/HRES_fc0/HRES_fc0_${YEAR}_no_Antarctic_w10_1p50.zarr" "HRES_fc0 w10" 1

echo ""
echo "Forecasts - IFS_ENS (20 leads × 2 vars = 40 files at 1.5°):"
echo "------------------------------------------"
IFS_COUNT=$(find ${FORECAST_ROOT}/IFS_ENS/IFS_ENS_${YEAR}_no_Antarctic_*_m50_1p50_lead_*.zarr -type d 2>/dev/null | wc -l)
echo "  Found: ${IFS_COUNT}/40 files"
if [ $IFS_COUNT -eq 40 ]; then
    echo "  ✓ Complete"
elif [ $IFS_COUNT -gt 0 ]; then
    echo "  ⚠ Incomplete - still downloading?"
else
    echo "  ✗ Missing - not started?"
fi

echo ""
echo "Forecasts - FGN (5 leads × 2 vars = 10 files):"
echo "------------------------------------------"
FGN_COUNT=$(find ${FORECAST_ROOT}/FGN/FGN_${YEAR}_no_Antarctic_*_m64_1p50_lead_*.zarr -type d 2>/dev/null | wc -l)
echo "  Found: ${FGN_COUNT}/10 files"
if [ $FGN_COUNT -eq 10 ]; then
    echo "  ✓ Complete"
elif [ $FGN_COUNT -gt 0 ]; then
    echo "  ⚠ Incomplete - still downloading?"
    echo "  Files found:"
    ls -1 ${FORECAST_ROOT}/FGN/FGN_${YEAR}_no_Antarctic_*_m64_1p50_lead_*.zarr 2>/dev/null | head -5
else
    echo "  ✗ Missing - not started?"
fi

echo ""
echo "Forecasts - ArchesWeatherGen (4 leads × 2 vars = 8 files):"
echo "------------------------------------------"
ARCHES_COUNT=$(find ${FORECAST_ROOT}/archesweathergen_forecasts_${YEAR}.zarr -type f -name "*.zarray" 2>/dev/null | wc -l)
if [ -d "${FORECAST_ROOT}/archesweathergen_forecasts_${YEAR}.zarr" ]; then
    echo "  ✓ Directory exists (single combined Zarr)"
else
    echo "  ✗ Directory not found"
fi

echo ""
echo "Forecasts - uCast (5 leads × 2 vars = combined Zarr):"
echo "------------------------------------------"
UCAST_COUNT=$(find ${FORECAST_ROOT}/ucast_forecasts_${YEAR}.zarr -type f -name "*.zarray" 2>/dev/null | wc -l)
if [ -d "${FORECAST_ROOT}/ucast_forecasts_${YEAR}.zarr" ]; then
    echo "  ✓ Directory exists (single combined Zarr)"
else
    echo "  ✗ Directory not found"
fi

echo ""
echo "ERA5 Records:"
echo "------------------------------------------"
RECORDS_DIR="data/era5_records"
RECORD_COUNT=$(find ${RECORDS_DIR}/ERA5_record_*_1979_2018_*.nc -type f 2>/dev/null | wc -l)
echo "  Found: ${RECORD_COUNT}/5 files (expected: max/min for t2m, max for w10, plus LSM)"
if [ $RECORD_COUNT -ge 5 ]; then
    echo "  ✓ Complete"
else
    echo "  ⚠ Incomplete"
fi

echo ""
echo "=========================================="
echo "Quick Status Summary"
echo "=========================================="

# Count ready models
READY=0
TOTAL=4

if [ $IFS_COUNT -eq 40 ]; then ((READY++)); fi
if [ $FGN_COUNT -eq 10 ]; then ((READY++)); fi
if [ -d "${FORECAST_ROOT}/archesweathergen_forecasts_${YEAR}.zarr" ]; then ((READY++)); fi
if [ -d "${FORECAST_ROOT}/ucast_forecasts_${YEAR}.zarr" ]; then ((READY++)); fi

echo "Forecast models ready: ${READY}/${TOTAL}"

if [ $READY -eq $TOTAL ]; then
    echo ""
    echo "✓ All forecast data appears present!"
    echo "  Run 'python test_data_readiness.py --quick' to verify loadability"
else
    echo ""
    echo "⚠ Some data still missing/downloading"
fi
