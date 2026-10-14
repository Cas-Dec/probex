"""Check depth_scores against brute-force pairwise CRPS and Brier formulas."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import forecast_evaluation_depth as depth  # noqa: E402


def brute_force(x, y, threshold, sign):
    """Scores per case from the textbook definitions. x: (..., M), y and threshold: (...)."""
    m = x.shape[-1]
    v = lambda z: np.maximum(sign * z, 0)  # threshold weighting max/min(z, r), shifted so r = 0
    vx, vy = v(x - threshold[..., None]), v(y - threshold)
    first = np.abs(vx - vy[..., None]).mean(-1)
    spread = np.abs(vx[..., :, None] - vx[..., None, :]).sum((-1, -2))
    p = (sign * (x - threshold[..., None]) > 0).mean(-1)
    o = sign * (y - threshold) > 0
    return {"rtwcrps": first - spread / (2 * m * (m - 1)),
            "rtwcrps_empirical": first - spread / (2 * m * m),
            "brier": (p - o) ** 2 - p * (1 - p) / (m - 1),
            "brier_empirical": (p - o) ** 2,
            "probability": p,
            "observed_probability": o.astype(float)}


@pytest.mark.parametrize("case", ["max", "min"])
def test_depth_scores_match_brute_force(case):
    rng = np.random.default_rng(1)
    times = np.array(["2022-01-01T00", "2022-02-01T12", "2022-02-02T00"], dtype="datetime64[ns]")
    lat, lon, members, depths = [0.0, 60.0], [0.0, 1.5], 5, [0.0, 0.5]
    coords = {"time": times, "latitude": lat, "longitude": lon}
    truth = xr.DataArray(rng.normal(size=(3, 2, 2)), coords=coords, dims=list(coords))
    forecast = xr.DataArray(rng.normal(size=(3, 2, 2, members)), coords=coords,
                            dims=[*coords, "number"]).transpose("time", "number", "latitude", "longitude")
    hcoords = {"month": np.arange(1, 13), "latitude": lat, "longitude": lon}
    history = xr.Dataset({"record": (list(hcoords), rng.normal(scale=0.3, size=(12, 2, 2))),
                          "scale": (list(hcoords), rng.uniform(0.2, 1.0, size=(12, 2, 2)))}, coords=hcoords)
    surfaces = xr.ones_like(truth.isel(time=0, drop=True), dtype=bool).expand_dims(landsea=["all"])

    result = depth.depth_scores(forecast, truth, history, surfaces, depths, case, base={}).compute()

    sign = 1 if case == "max" else -1
    months = times.astype("datetime64[M]").astype(int) % 12  # month index 0..11
    record, scale = history.record.values[months], history.scale.values[months]
    weight = np.broadcast_to(np.cos(np.deg2rad(lat))[None, :, None], truth.shape)
    x = forecast.transpose("time", "latitude", "longitude", "number").values
    for k in depths:
        expected = brute_force(x, truth.values, record + sign * k * scale, sign)
        for name, field in expected.items():
            got = float(result[name].sel(landsea="all", depth=k))
            assert got == pytest.approx((field * weight).sum() / weight.sum(), abs=1e-6), (name, k)
        assert int(result.record_count.sel(landsea="all", depth=k)) == int(expected["observed_probability"].sum())


def test_fingerprint_mismatch_raises_and_write_is_kept(tmp_path):
    path = tmp_path / "fingerprints.json"
    depth.sync_fingerprints(path, {"a": "1"}, {"model": "m"}, write=True)
    depth.sync_fingerprints(path, {"b": "2"}, {"model": "m"}, write=True)
    depth.sync_fingerprints(path, {"a": "1", "b": "2"}, {"model": "m"})  # both kept, no error
    with pytest.raises(ValueError, match="mismatch"):
        depth.sync_fingerprints(path, {"a": "changed"}, {"model": "m"})
    assert not list(tmp_path.glob("*.partial"))
