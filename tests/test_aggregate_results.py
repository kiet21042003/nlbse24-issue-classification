import pandas as pd

from scripts.aggregate_results import _write_pareto_tables


def test_pareto_output_skips_unknown_hardware_by_default(tmp_path) -> None:
    (tmp_path / "pareto_fit.csv").write_text("legacy\n", encoding="utf-8")
    (tmp_path / "pareto_inference.csv").write_text("legacy\n", encoding="utf-8")
    frame = pd.DataFrame(
        [
            {
                "protocol": "cv",
                "model": "demo",
                "hardware_group": "unknown",
                "macro_f1": 0.8,
                "fit_elapsed_seconds": 1.0,
                "inference_elapsed_seconds": 0.1,
            }
        ]
    )

    paths, skipped = _write_pareto_tables(
        frame,
        tmp_path,
        include_unknown_hardware=False,
        assume_hardware_group=None,
    )

    assert paths == {}
    assert skipped == ["cv/unknown"]
    assert not (tmp_path / "pareto_fit.csv").exists()
    assert not (tmp_path / "pareto_inference.csv").exists()


def test_pareto_output_can_use_explicitly_verified_hardware_group(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "protocol": "cv",
                "model": "demo",
                "hardware_group": "unknown",
                "macro_f1": 0.8,
                "fit_elapsed_seconds": 1.0,
                "inference_elapsed_seconds": 0.1,
            }
        ]
    )

    paths, skipped = _write_pareto_tables(
        frame,
        tmp_path,
        include_unknown_hardware=False,
        assume_hardware_group="verified-machine",
    )

    assert skipped == []
    assert set(paths) == {
        "pareto_fit_cv_verified-machine",
        "pareto_inference_cv_verified-machine",
    }
