import numpy as np
import xarray as xr

from manuscript.scripts.zeroprob_table import render_table


def test_render_table_computes_miss_rates_and_marks_missing_leads(tmp_path) -> None:
    results_root = tmp_path / "results"
    coords = {
        "leadtime": [12, 48, 120],
        "season": ["all"],
        "landsea": ["all"],
        "threshold": [0.0, 0.02],
    }
    recall = np.array([[[[0.75, 0.0]]], [[[0.5, 0.0]]], [[[0.25, 0.0]]]])

    for model in ("IFS_ENS", "FGN", "uCast", "ArchesWeatherGen"):
        truth = "HRES_fc0" if model == "IFS_ENS" else "ERA5"
        for case, var in (("max", "t2m"), ("min", "t2m"), ("max", "w10")):
            dataset = xr.Dataset({"recall": (("leadtime", "season", "landsea", "threshold"), recall)}, coords=coords)
            if model == "ArchesWeatherGen":
                dataset = dataset.sel(leadtime=[48, 120])
            path = results_root / model / "Occurrence" / f"{model}_{case}_occurrence_2022_{var}_no_Antarctic_records_1979_2018_cond_{truth}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_netcdf(path)

    table = render_table(results_root, 2022)

    assert "Heat & IFS-ENS & \\textbf{0.250} & \\textbf{0.500} & \\textbf{0.750}" in table
    assert "Heat & ArchesWeatherGen & -- & \\textbf{0.500} & \\textbf{0.750}" in table
    assert "\\label{tab:supp-zeromiss}" in table
