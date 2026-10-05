"""
Conservative regridding utilities for rectilinear lat-lon grids.

Pure NumPy implementation (no JAX dependency) based on WeatherBench2 regridding logic.
Computes overlap-weighted conservative remapping between source and target grids.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _assert_increasing(x: np.ndarray, name: str = "array") -> None:
    """Check that array is strictly increasing."""
    if not (np.diff(x) > 0).all():
        raise ValueError(f"{name} is not strictly increasing: {x}")


def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
    """
    Compute cell boundaries for latitude coordinates.

    Args:
        x: 1D array of latitude cell centers in radians

    Returns:
        1D array of cell boundaries with shape (len(x) + 1,)
        First boundary is -π/2 (South Pole), last is +π/2 (North Pole)
    """
    pi_over_2 = np.pi / 2
    # Cell edges are midpoints between centers, clamped to poles
    midpoints = (x[:-1] + x[1:]) / 2
    return np.concatenate([[-pi_over_2], midpoints, [pi_over_2]])


def _latitude_overlap(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Calculate the area overlap between source and target latitude cells.

    Uses the integral of cos(lat) over each cell (proportional to cell area).

    Args:
        source_points: 1D array of source latitude centers in radians (increasing)
        target_points: 1D array of target latitude centers in radians (increasing)

    Returns:
        2D array with shape (n_target, n_source) containing overlap weights.
        Each element [i, j] is the area overlap between target cell i and source cell j.
    """
    source_bounds = _latitude_cell_bounds(source_points)
    target_bounds = _latitude_cell_bounds(target_points)

    # Upper bounds: min of target upper and source upper
    upper = np.minimum(
        target_bounds[1:, np.newaxis],  # (n_target, 1)
        source_bounds[np.newaxis, 1:]   # (1, n_source)
    )

    # Lower bounds: max of target lower and source lower
    lower = np.maximum(
        target_bounds[:-1, np.newaxis],
        source_bounds[np.newaxis, :-1]
    )

    # Cell area = integral of cos(lat) from lower to upper
    # = sin(upper) - sin(lower), only where upper > lower
    overlap = np.where(upper > lower, np.sin(upper) - np.sin(lower), 0.0)

    return overlap


def _conservative_latitude_weights(
    source_points: np.ndarray,
    target_points: np.ndarray
) -> np.ndarray:
    """
    Create a weight matrix for conservative regridding along latitude.

    Args:
        source_points: 1D latitude coordinates in radians for source cell centers
        target_points: 1D latitude coordinates in radians for target cell centers

    Returns:
        2D array with shape (n_target, n_source). Each row sums to 1.
        To regrid: target_data = weights @ source_data
    """
    _assert_increasing(source_points, "source latitude")
    _assert_increasing(target_points, "target latitude")

    weights = _latitude_overlap(source_points, target_points)

    # Normalize rows to sum to 1
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / row_sums

    assert weights.shape == (len(target_points), len(source_points))
    return weights


def _align_phase_with(x: np.ndarray, target: np.ndarray, period: float) -> np.ndarray:
    """
    Align the phase of periodic coordinates to match a target.

    Shifts x by ±period to minimize distance from target.

    Args:
        x: Values to adjust
        target: Reference values
        period: Periodicity (e.g., 2π for longitude in radians)

    Returns:
        x shifted by 0, +period, or -period
    """
    shift_down = x > target + period / 2
    shift_up = x < target - period / 2
    return x + period * shift_up - period * shift_down


def _periodic_upper_bounds(x: np.ndarray, period: float) -> np.ndarray:
    """Compute upper cell boundaries for periodic coordinates."""
    x_plus = _align_phase_with(np.roll(x, -1), x, period)
    return (x + x_plus) / 2


def _periodic_lower_bounds(x: np.ndarray, period: float) -> np.ndarray:
    """Compute lower cell boundaries for periodic coordinates."""
    x_minus = _align_phase_with(np.roll(x, +1), x, period)
    return (x_minus + x) / 2


def _periodic_overlap(x0, x1, y0, y1, period):
    """Calculate overlap between two intervals on a periodic domain."""
    y0 = _align_phase_with(y0, x0, period)
    y1 = _align_phase_with(y1, x0, period)
    upper = np.minimum(x1, y1)
    lower = np.maximum(x0, y0)
    return np.maximum(upper - lower, 0.0)


def _longitude_overlap(
    first_points: np.ndarray,
    second_points: np.ndarray,
    period: float = 2 * np.pi,
) -> np.ndarray:
    """
    Calculate the area overlap between longitude cells.

    Args:
        first_points: 1D array of longitude cell centers in radians
        second_points: 1D array of longitude cell centers in radians
        period: Periodicity in radians (default 2π)

    Returns:
        2D array with shape (n_first, n_second) containing overlap lengths
    """
    first_points = first_points % period
    first_upper = _periodic_upper_bounds(first_points, period)
    first_lower = _periodic_lower_bounds(first_points, period)

    second_points = second_points % period
    second_upper = _periodic_upper_bounds(second_points, period)
    second_lower = _periodic_lower_bounds(second_points, period)

    # Vectorized pairwise overlap calculation
    overlap = np.zeros((len(first_points), len(second_points)))
    for i in range(len(first_points)):
        for j in range(len(second_points)):
            overlap[i, j] = _periodic_overlap(
                first_lower[i], first_upper[i],
                second_lower[j], second_upper[j],
                period
            )

    return overlap


def _conservative_longitude_weights(
    source_points: np.ndarray,
    target_points: np.ndarray
) -> np.ndarray:
    """
    Create a weight matrix for conservative regridding along longitude.

    Args:
        source_points: 1D longitude coordinates in radians for source cell centers
        target_points: 1D longitude coordinates in radians for target cell centers

    Returns:
        2D array with shape (n_target, n_source). Each row sums to 1.
        To regrid: target_data = weights @ source_data
    """
    _assert_increasing(source_points, "source longitude")
    _assert_increasing(target_points, "target longitude")

    # Note: overlap is computed as (target, source) to get correct dimensions
    weights = _longitude_overlap(target_points, source_points)

    # Normalize rows to sum to 1
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / row_sums

    assert weights.shape == (len(target_points), len(source_points))
    return weights


class ConservativeRegridder:
    """
    Conservative regridder for rectilinear lat-lon grids.

    Computes area-weighted remapping from source to target grid.
    Handles NaN values by computing weighted averages only over valid cells.

    Usage:
        regridder = ConservativeRegridder(source_grid, target_grid)
        target_data = regridder.regrid_array(source_data)
        target_ds = regridder.regrid_dataset(source_ds)
    """

    def __init__(
        self,
        source_lon: np.ndarray,
        source_lat: np.ndarray,
        target_lon: np.ndarray,
        target_lat: np.ndarray
    ):
        """
        Initialize regridder with source and target grids.

        Args:
            source_lon: 1D array of source longitude in degrees
            source_lat: 1D array of source latitude in degrees
            target_lon: 1D array of target longitude in degrees
            target_lat: 1D array of target latitude in degrees

        All arrays must be strictly increasing.
        """
        # Convert degrees to radians
        self.source_lon_rad = np.deg2rad(source_lon)
        self.source_lat_rad = np.deg2rad(source_lat)
        self.target_lon_rad = np.deg2rad(target_lon)
        self.target_lat_rad = np.deg2rad(target_lat)

        # Precompute weight matrices
        print("Computing conservative regridding weights...")
        print(f"  Source grid: {len(source_lon)} lon × {len(source_lat)} lat")
        print(f"  Target grid: {len(target_lon)} lon × {len(target_lat)} lat")

        self.lon_weights = _conservative_longitude_weights(
            self.source_lon_rad, self.target_lon_rad
        )
        self.lat_weights = _conservative_latitude_weights(
            self.source_lat_rad, self.target_lat_rad
        )

        print("  Weight matrices computed.")

    def regrid_array(self, field: np.ndarray, handle_nan: bool = True) -> np.ndarray:
        """
        Regrid a 2D or higher-dimensional array.

        Args:
            field: Array with shape (..., n_source_lat, n_source_lon)
                   Last two dimensions are latitude and longitude
            handle_nan: If True, compute weighted average ignoring NaN values

        Returns:
            Array with shape (..., n_target_lat, n_target_lon)
        """
        if field.ndim < 2:
            raise ValueError(f"Field must have at least 2 dimensions, got {field.ndim}")

        # Check spatial dimensions
        # Standard xarray convention is (..., lat, lon)
        expected_shape = (len(self.source_lat_rad), len(self.source_lon_rad))

        if field.shape[-2:] != expected_shape:
            raise ValueError(
                f"Field spatial shape {field.shape[-2:]} does not match "
                f"source grid (lat, lon) = {expected_shape}. "
                f"Got shape: {field.shape}, expected last 2 dims: {expected_shape}"
            )

        # Transpose to (lon, lat) for regridding
        field = np.swapaxes(field, -2, -1)

        # DEBUG: Check field shape after transpose
        if field.shape[-2:] != (len(self.source_lon_rad), len(self.source_lat_rad)):
            raise ValueError(
                f"After transpose, field shape {field.shape[-2:]} doesn't match "
                f"expected (source_lon, source_lat) = ({len(self.source_lon_rad)}, {len(self.source_lat_rad)})"
            )

        if handle_nan:
            # Compute weighted average ignoring NaNs (like np.nanmean)
            # Total = sum of (values × weights) over non-NaN cells
            # Count = sum of weights over non-NaN cells
            # Result = Total / Count

            nulls = np.isnan(field)
            values_filled = np.where(nulls, 0.0, field)
            mask_filled = np.where(nulls, 0.0, 1.0)

            # Apply regridding: einsum over lon and lat dimensions
            # Weights: lon_weights[i, j] maps source_lon[j] → target_lon[i]
            #          lat_weights[k, l] maps source_lat[l] → target_lat[k]
            # Result[..., i, k] = sum_j sum_l (field[..., j, l] × lon_w[i,j] × lat_w[k,l])

            total = np.einsum(
                'ij,kl,...jl->...ik',
                self.lon_weights,
                self.lat_weights,
                values_filled,
                optimize=True
            )

            count = np.einsum(
                'ij,kl,...jl->...ik',
                self.lon_weights,
                self.lat_weights,
                mask_filled,
                optimize=True
            )

            # Avoid division by zero: result is NaN if no valid source cells
            result = np.where(count > 0, total / count, np.nan)

        else:
            # Simple weighted average (NaNs will propagate)
            result = np.einsum(
                'ij,kl,...jl->...ik',
                self.lon_weights,
                self.lat_weights,
                field,
                optimize=True
            )

        # Transpose back to (lat, lon) convention
        result = np.swapaxes(result, -2, -1)

        return result

    def regrid_dataset(self, ds: xr.Dataset, target_lon: np.ndarray, target_lat: np.ndarray) -> xr.Dataset:
        """
        Regrid all variables in an xarray Dataset.

        Args:
            ds: Dataset with 'longitude' and 'latitude' coordinates
            target_lon: 1D array of target longitude values in degrees
            target_lat: 1D array of target latitude values in degrees

        Returns:
            Regridded dataset with updated longitude/latitude coordinates
        """
        # Ensure latitude is increasing
        if not (ds.latitude.diff('latitude') > 0).all():
            print("  Reversing latitude to ensure increasing order...")
            ds = ds.isel(latitude=slice(None, None, -1))

        assert (ds.latitude.diff('latitude') > 0).all(), "Latitude must be increasing"

        # Apply regridding to all data variables
        regridded_vars = {}

        for var_name in ds.data_vars:
            var = ds[var_name]

            # Check if variable has spatial dimensions
            if 'longitude' not in var.dims or 'latitude' not in var.dims:
                print(f"  Skipping {var_name} (no spatial dims)")
                regridded_vars[var_name] = var
                continue

            print(f"  Regridding {var_name}...")

            # Apply regridding using xr.apply_ufunc
            regridded = xr.apply_ufunc(
                self.regrid_array,
                var,
                input_core_dims=[['longitude', 'latitude']],
                output_core_dims=[['longitude', 'latitude']],
                exclude_dims={'longitude', 'latitude'},
                vectorize=True,  # Loop over time, number, etc.
                dask='parallelized',  # Support dask arrays
                output_dtypes=[var.dtype],
            )

            regridded_vars[var_name] = regridded

        # Create new dataset with target coordinates
        result = xr.Dataset(
            regridded_vars,
            coords={
                'longitude': target_lon,
                'latitude': target_lat,
                **{k: v for k, v in ds.coords.items()
                   if k not in ['longitude', 'latitude']}
            }
        )

        return result
