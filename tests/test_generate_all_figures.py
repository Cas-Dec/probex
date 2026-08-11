from pathlib import Path
import subprocess

import manuscript.scripts.generate_all_figures as generator
import pytest


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_direct_script_execution_displays_help_from_repository_root(
    repository_root: Path,
) -> None:
    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "probex",
            "python",
            "manuscript/scripts/generate_all_figures.py",
            "--help",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: generate_all_figures.py" in result.stdout


def test_generate_all_skips_each_missing_figure_and_prints_manifest(tmp_path: Path, capsys) -> None:
    generated, skipped = generator.generate_all(
        tmp_path / "results", tmp_path / "figures", 2022
    )

    output = capsys.readouterr().out
    assert generated == 0
    assert skipped == 204
    assert "Skipped point_all_all.png:" in output
    assert "Generated: 0" in output
    assert "Skipped: 204" in output
    assert (tmp_path / "figures").is_dir()


def test_generate_all_requests_exact_figure_inventory(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def constructor(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    for name in (
        "plot_point",
        "plot_counts",
        "plot_brier",
        "plot_crps",
        "plot_precision_recall",
        "plot_reliability",
        "plot_margin_intensity",
        "plot_margin_probabilistic",
    ):
        monkeypatch.setattr(generator, name, constructor)

    generated, skipped = generator.generate_all(
        tmp_path / "results", tmp_path / "figures", 2022, record_tag="custom"
    )

    names = [Path(args[1]).name for args, _ in calls]
    assert (generated, skipped) == (204, 0)
    assert len(names) == 204
    assert set(names) == {
        *(f"{family}_{season}_{landsea}.png" for family in ("point", "counts", "brier", "crps") for season in ("all", "DJF", "MAM", "JJA", "SON") for landsea in ("all", "land", "sea")),
        *(f"{family}_{lead}h_{season}_{landsea}.png" for family in ("precision_recall", "reliability") for lead in (48, 120, 168, 240) for season in ("all", "DJF", "MAM", "JJA", "SON") for landsea in ("all", "land", "sea")),
        *(f"{family}_{lead}h_{landsea}.png" for family in ("margin_intensity", "margin_probabilistic") for lead in (48, 120, 168, 240) for landsea in ("all", "land", "sea")),
    }
    assert all(kwargs == {"record_tag": "custom"} for _, kwargs in calls)
