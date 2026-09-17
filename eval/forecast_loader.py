"""
Helper functions for loading forecast data from different storage formats.

Handles both per-lead-time Zarr files (IFS_ENS, FGN) and combined Zarr
stores (ArchesWeatherGen, uCast).
"""

from pathlib import Path
import numpy as np
import xarray as xr


def load_forecast_for_lead(
    model: str,
    var: str,
    year: int,
    lead_hour: int,
    forecast_root: Path,
    n_members: int,
    resolution_tag: str,
    no_Antarctic: dict,
    variable: str,
) -> xr.DataArray:
    """
    Load forecast data for a specific lead time.

    Parameters
    ----------
    model : str
        Model name (e.g., 'IFS_ENS', 'ArchesWeatherGen', 'uCast', 'FGN')
    var : str
        Variable short name ('t2m' or 'w10')
    year : int
        Forecast year
    lead_hour : int
        Lead time in hours
    forecast_root : Path
        Root directory for forecast data
    n_members : int
        Number of ensemble members
    resolution_tag : str
        Resolution tag (e.g., '1p50')
    no_Antarctic : dict
        Dictionary for spatial subsetting
    variable : str
        Full variable name (e.g., '2m_temperature')

    Returns
    -------
    xr.DataArray
        Forecast data with standardized dimensions
    """

    if model in ["ArchesWeatherGen", "uCast"]:
        return _load_from_combined_zarr(
            model, var, year, lead_hour, forecast_root, no_Antarctic
        )
    else:
        return _load_from_per_lead_file(
            model, var, year, lead_hour, forecast_root,
            n_members, resolution_tag, no_Antarctic, variable
        )


def _load_from_combined_zarr(
    model: str,
    var: str,
    year: int,
    lead_hour: int,
    forecast_root: Path,
    no_Antarctic: dict,
) -> xr.DataArray:
    """Load from combined Zarr store (ArchesWeatherGen, uCast)."""

    # Configure model-specific settings
    if model == "ArchesWeatherGen":
        combined_file = forecast_root / f"archesweathergen_forecasts_{year}.zarr"
        var_name_map = {
            "t2m": "2m_temperature",
            # w10 computed from u/v components (handled separately below)
        }
        member_dim = "member"
    elif model == "uCast":
        combined_file = forecast_root / f"ucast_forecasts_{year}.zarr"
        var_name_map = {
            "t2m": "2m_temperature_predicted",
            # w10 computed from u/v components (handled separately below)
        }
        member_dim = "ensemble_member"
    else:
        raise ValueError(f"Unknown combined Zarr model: {model}")

    # Load combined dataset (only on first call per model)
    if not hasattr(_load_from_combined_zarr, 'cache'):
        _load_from_combined_zarr.cache = {}

    cache_key = (model, year)
    if cache_key not in _load_from_combined_zarr.cache:
        print(f"Loading combined Zarr: {combined_file}")
        ds = xr.open_zarr(combined_file, chunks={})

        # Convert uCast lead_time from timedelta to hours if needed
        if model == "uCast" and np.issubdtype(ds.lead_time.dtype, np.timedelta64):
            lead_time_hours = ds.lead_time.values / np.timedelta64(1, 'h')
            ds = ds.assign_coords(lead_time=lead_time_hours)

        _load_from_combined_zarr.cache[cache_key] = ds

    ds = _load_from_combined_zarr.cache[cache_key]

    # Special handling for wind speed - compute from u/v components
    if var == "w10":
        print(f"Selecting lead_time={lead_hour}h from combined Zarr")
        print("Computing wind speed from u and v components")

        if model == "ArchesWeatherGen":
            u = ds["10m_u_component_of_wind"].sel(lead_time=lead_hour)
            v = ds["10m_v_component_of_wind"].sel(lead_time=lead_hour)
        elif model == "uCast":
            u = ds["10m_u_component_of_wind_predicted"].sel(lead_time=lead_hour)
            v = ds["10m_v_component_of_wind_predicted"].sel(lead_time=lead_hour)

        forecast = np.sqrt(u**2 + v**2)
        forecast.attrs["long_name"] = "10m_wind_speed"
        forecast.attrs["units"] = "m s**-1"
    else:
        variable_name = var_name_map[var]
        # Select lead time
        print(f"Selecting lead_time={lead_hour}h from combined Zarr")
        forecast = ds[variable_name].sel(lead_time=lead_hour)

    # Standardize dimension names
    forecast = forecast.rename({member_dim: "number"})

    if "init_time" in forecast.dims:
        forecast = forecast.rename({"init_time": "time"})

    # Add prediction_timedelta dimension if needed
    # Standard dimension order: ('time', 'number', 'prediction_timedelta', 'longitude', 'latitude')
    if "prediction_timedelta" not in forecast.dims:
        forecast = forecast.expand_dims({"prediction_timedelta": 1}, axis=2)

    # Convert longitude from 0-360 to -180-180 if needed
    if forecast.longitude.max() > 180:
        forecast = forecast.assign_coords(
            longitude=(((forecast.longitude + 180) % 360) - 180)
        ).sortby("longitude")

    # Sort latitude in ascending order
    forecast = forecast.sortby("latitude")

    # Apply spatial subset
    forecast = forecast.sel(no_Antarctic)

    return forecast


def _load_from_per_lead_file(
    model: str,
    var: str,
    year: int,
    lead_hour: int,
    forecast_root: Path,
    n_members: int,
    resolution_tag: str,
    no_Antarctic: dict,
    variable: str,
) -> xr.DataArray:
    """Load from per-lead-time Zarr file (IFS_ENS, FGN, etc.)."""

    forecast_file = (
        forecast_root / model
        / f"{model}_{year}_no_Antarctic_{var}_m{n_members}_{resolution_tag}_lead_{lead_hour:03d}h.zarr"
    )

    print(f"Loading: {forecast_file}")
    forecast = xr.open_zarr(forecast_file, chunks={})[variable]
    forecast = forecast.sel(no_Antarctic)

    return forecast


def clear_combined_cache():
    """Clear the combined Zarr cache (call at end of script to free memory)."""
    if hasattr(_load_from_combined_zarr, 'cache'):
        _load_from_combined_zarr.cache.clear()
