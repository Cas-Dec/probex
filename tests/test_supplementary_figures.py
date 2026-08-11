from pathlib import Path


def test_catalogue_includes_every_generated_figure(tmp_path: Path) -> None:
    from manuscript.scripts.generate_supplementary_figures import write_catalogue

    figures = tmp_path / "fig"
    figures.mkdir()
    for name in ("point_all_all.png", "brier_all_all.png"):
        (figures / name).touch()
    output = tmp_path / "supplementary_figures.tex"

    count = write_catalogue(figures, output)

    text = output.read_text()
    assert count == 2
    assert text.count("\\includegraphics") == 2
    assert "fig/point_all_all.png" in text
    assert "fig/brier_all_all.png" in text
