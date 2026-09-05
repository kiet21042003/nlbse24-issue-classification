"""Tests for the P2 sweep launcher.

The launcher lives in ``scripts/`` rather than in the package, so it is loaded here by
path. Its run-name construction is the invariant that matters most: two grid points that
slugify to the same string would silently share a result directory, and with
``--overwrite`` the second would destroy the first.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nlbse24.evaluation.artifacts import slugify

REPO_ROOT = Path(__file__).resolve().parents[1]
GRIDS = sorted((REPO_ROOT / "configs" / "sweeps").glob("*.toml"))


def load_launcher() -> ModuleType:
    path = REPO_ROOT / "scripts" / "run_p2_experiments.py"
    spec = importlib.util.spec_from_file_location("p2_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["p2_launcher"] = module
    spec.loader.exec_module(module)
    return module


launcher = load_launcher()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "yes"),
        (False, "no"),
        (0.25, "0.25"),
        (2, "2"),
        ([2, 5], "2-5"),
        ("word_char", "word_char"),
    ],
)
def test_format_value_renders_slug_safe_text(value: object, expected: str) -> None:
    assert launcher.format_value(value) == expected


def test_set_path_writes_into_nested_tables() -> None:
    payload: dict = {"classifier": {"linear_svc": {"C": 1.0}}}

    launcher.set_path(payload, "classifier.linear_svc.C", 4.0)
    launcher.set_path(payload, "features.char.enabled", False)

    assert payload["classifier"]["linear_svc"]["C"] == 4.0
    assert payload["features"]["char"]["enabled"] is False


def test_set_path_refuses_to_descend_into_a_scalar() -> None:
    with pytest.raises(ValueError, match="is not a table"):
        launcher.set_path({"classifier": 1}, "classifier.kind", "x")


def test_resolve_family_rejects_an_undeclared_family() -> None:
    with pytest.raises(ValueError, match="must declare model_family"):
        launcher.resolve_family({}, Path("somewhere.toml"))


def test_a_grid_without_axes_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        list(launcher.iter_grid_points({"base": {}}))


def test_an_unknown_preset_is_rejected() -> None:
    grid = {
        "base": {},
        "axes": [{"tag": "feat", "key": "preset", "values": ["missing"]}],
    }

    with pytest.raises(ValueError, match="unknown preset"):
        list(launcher.iter_grid_points(grid))


@pytest.mark.parametrize("path", GRIDS, ids=lambda p: p.stem)
def test_committed_grid_points_are_unique_and_slug_stable(path: Path) -> None:
    grid = launcher.load_toml(path)
    family = launcher.resolve_family(grid, path)
    points = list(launcher.iter_grid_points(grid))
    names = [name for name, _ in points]

    assert names, "a committed grid must expand to at least one point"
    # slugify is what maps a run name onto a results directory.
    assert [slugify(name) for name in names] == names
    assert len(set(names)) == len(names)

    config_cls, _ = launcher.FAMILIES[family]
    for _name, payload in points:
        config_cls.from_mapping(payload)


def test_presets_override_the_base_configuration() -> None:
    grid = launcher.load_toml(REPO_ROOT / "configs" / "sweeps" / "p2_linear_svc_grid.toml")
    points = dict(launcher.iter_grid_points(grid))

    char_only = points["p2_svc__feat-char__c-1.0"]

    assert char_only["features"]["word"]["enabled"] is False
    assert char_only["features"]["char"]["enabled"] is True
    assert char_only["classifier"]["linear_svc"]["C"] == 1.0


def test_sweep_reports_are_named_so_the_shared_ingestion_skips_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep report is a derived summary, not a run artifact.

    P5's result ingestion walks the whole results tree, skips files named
    ``summary-seed-*`` and validates every other JSON against the run-artifact schema.
    A sweep report under any other name therefore makes ``results_dataframe("results")``
    raise on P2's own output, so the name is part of the contract rather than cosmetic.
    """

    monkeypatch.setattr(
        launcher,
        "run_one",
        lambda **_kwargs: {
            "seed": 42,
            "cross_repository_macro_f1": 0.5,
            "repository_macro_f1": {},
            "evaluations": 0,
        },
    )
    args = launcher.build_parser().parse_args(
        [
            "--grid",
            str(REPO_ROOT / "configs" / "sweeps" / "p2_char_ngram_grid.toml"),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert launcher.run_grid(args) == 0

    written = sorted((tmp_path / "sweeps").glob("*.json"))
    assert [path.name for path in written] == [
        "summary-seed-42-p2_char_ngram_grid-cv.json"
    ]
    assert all(path.name.startswith("summary-seed-") for path in tmp_path.rglob("*.json"))
