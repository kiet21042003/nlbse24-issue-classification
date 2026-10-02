"""Launcher for the P2 lightweight-model experiments.

P1's CLI exposes only the Logistic Regression baseline, so this script is the P2
equivalent. It does not reimplement any protocol: splits, metrics, profiling and artifact
writing all come from ``nlbse24.runner.run_classifier_experiment``.

Two modes:

``--config``
    Run one committed configuration under a protocol.

``--grid``
    Run a limited hyperparameter sweep declared in a sweep TOML, one result directory per
    grid point, and print the points ranked by cross-repository macro-F1.

Examples::

    python scripts/run_p2_experiments.py --grid configs/sweeps/p2_linear_svc_grid.toml
    python scripts/run_p2_experiments.py --config configs/tfidf_char_linear_svc.toml \
        --protocol loo
"""

import argparse
import json
import sys
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

# Allow "python scripts/run_p2_experiments.py" from a checkout that was not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.data import CsvIssueRepository  # noqa: E402
from nlbse24.modeling.base import BaseIssueClassifier  # noqa: E402
from nlbse24.modeling.fasttext import (  # noqa: E402
    FastTextClassifier,
    FastTextConfig,
    FastTextStyleConfig,
    FastTextStyleLinearClassifier,
)
from nlbse24.modeling.sparse_models import (  # noqa: E402
    SparseModelConfig,
    TfidfSparseClassifier,
)
from nlbse24.runner import run_classifier_experiment  # noqa: E402

# Every P2 config declares which family it belongs to through a top-level model_family key.
FAMILIES: dict[str, tuple[Any, type[BaseIssueClassifier]]] = {
    "sparse": (SparseModelConfig, TfidfSparseClassifier),
    "fasttext": (FastTextConfig, FastTextClassifier),
    "fasttext_style": (FastTextStyleConfig, FastTextStyleLinearClassifier),
}


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def resolve_family(payload: Mapping[str, Any], path: Path) -> str:
    family = payload.get("model_family")
    if family not in FAMILIES:
        raise ValueError(
            f"{path} must declare model_family as one of {sorted(FAMILIES)}, found {family!r}"
        )
    return str(family)


def make_factory(
    model_cls: type[BaseIssueClassifier], config: Any
) -> Callable[[], BaseIssueClassifier]:
    """Return a named zero-argument factory; the runner builds a fresh model per fold."""

    def model_factory() -> BaseIssueClassifier:
        return model_cls(config)

    return model_factory


def format_value(value: Any) -> str:
    """Render a grid value so that it survives the shared slugify unchanged."""

    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list | tuple):
        return "-".join(format_value(item) for item in value)
    return str(value).replace("/", "-").replace(" ", "")


def set_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    """Assign into a nested mapping using a dotted path such as classifier.linear_svc.C."""

    keys = dotted.split(".")
    cursor = target
    for key in keys[:-1]:
        branch = cursor.setdefault(key, {})
        if not isinstance(branch, dict):
            raise ValueError(f"cannot descend into {dotted!r}: {key!r} is not a table")
        cursor = branch
    cursor[keys[-1]] = value


def deep_copy(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def iter_grid_points(grid: Mapping[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (run_name, config_mapping) for the cartesian product of the declared axes."""

    base = grid.get("base", {})
    presets = grid.get("presets", {})
    prefix = str(grid.get("run_prefix", "p2"))
    axes = grid.get("axes", [])
    if not axes:
        raise ValueError("a sweep grid must declare at least one axis")

    for combination in product(*[axis["values"] for axis in axes]):
        payload = deep_copy(base)
        tags: list[str] = []
        for axis, value in zip(axes, combination, strict=True):
            key = str(axis["key"])
            if key == "preset":
                overrides = presets.get(str(value))
                if overrides is None:
                    raise ValueError(f"unknown preset: {value!r}")
                for dotted, override in overrides.items():
                    set_path(payload, dotted, override)
            else:
                set_path(payload, key, value)
            tags.append(f"{axis['tag']}-{format_value(value)}")
        yield f"{prefix}__" + "__".join(tags), payload


def run_one(
    *,
    family: str,
    payload: Mapping[str, Any],
    run_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config_cls, model_cls = FAMILIES[family]
    if args.seed is not None:
        # Artifacts are keyed by seed, so re-running a frozen config under another seed
        # lands beside the original instead of colliding with it.
        payload = deep_copy(payload)
        set_path(payload, "experiment.seed", args.seed)
    config = config_cls.from_mapping(payload)
    repositories = set(args.repository) if args.repository else None
    summary = run_classifier_experiment(
        repository=CsvIssueRepository(args.data_dir),
        model_factory=make_factory(model_cls, config),
        model_name=run_name,
        output_dir=args.output_dir,
        protocol=args.protocol,
        seed=config.experiment.seed,
        n_splits=config.experiment.n_splits,
        text_fields=config.experiment.text_fields,
        repositories=repositories,
        overwrite=args.overwrite,
        allow_official_test=args.confirm_official_test,
    )
    return summary


def run_single_config(args: argparse.Namespace) -> int:
    path = Path(args.config)
    payload = load_toml(path)
    family = resolve_family(payload, path)
    run_name = args.run_name or path.stem
    summary = run_one(family=family, payload=payload, run_name=run_name, args=args)
    print(json.dumps({"run_name": run_name, **summary}, indent=2, ensure_ascii=False))
    return 0


def run_grid(args: argparse.Namespace) -> int:
    path = Path(args.grid)
    grid = load_toml(path)
    family = resolve_family(grid, path)
    points = list(iter_grid_points(grid))
    if args.only:
        points = [(name, payload) for name, payload in points if args.only in name]
        if not points:
            raise ValueError(f"no grid point matches --only {args.only!r}")
    ranked: list[dict[str, Any]] = []
    seed: int | None = None

    for position, (run_name, payload) in enumerate(points, start=1):
        print(f"[{position}/{len(points)}] {run_name}", flush=True)
        summary = run_one(family=family, payload=payload, run_name=run_name, args=args)
        seed = summary["seed"]
        ranked.append(
            {
                "run_name": run_name,
                "cross_repository_macro_f1": summary["cross_repository_macro_f1"],
                "repository_macro_f1": summary["repository_macro_f1"],
                "evaluations": summary["evaluations"],
            }
        )
        print(f"    cross-repository macro-F1 = {summary['cross_repository_macro_f1']:.4f}")

    ranked.sort(key=lambda row: row["cross_repository_macro_f1"], reverse=True)
    report = {
        "grid": str(path),
        "protocol": args.protocol,
        "model_family": family,
        "seed": seed,
        "points": ranked,
    }
    suffix = f"-{args.only}" if args.only else ""
    # A sweep report is a derived summary rather than a run artifact, but it lands inside
    # the results tree that P5's ingestion walks. That walk skips files named
    # summary-seed-* and validates every other JSON against the run-artifact schema, so a
    # sweep report under any other name makes results_dataframe("results") raise. Reuse
    # the naming convention the per-run summaries already follow.
    destination = (
        Path(args.output_dir)
        / "sweeps"
        / f"summary-seed-{seed}-{path.stem}-{args.protocol}{suffix}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nRanked by cross-repository macro-F1:")
    for row in ranked:
        print(f"  {row['cross_repository_macro_f1']:.4f}  {row['run_name']}")
    print(f"\nSweep report written to {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_p2_experiments", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="run one committed configuration")
    source.add_argument("--grid", type=Path, help="run a sweep declared in a sweep TOML")
    parser.add_argument("--protocol", choices=("cv", "pooled_cv", "loo", "official"), default="cv")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--repository", action="append", help="repeat to select repositories")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--seed", type=int, help="override the configured seed, for seed-robustness runs"
    )
    parser.add_argument(
        "--only", help="run only the grid points whose run name contains this substring"
    )
    parser.add_argument("--run-name", help="artifact directory name; defaults to the config stem")
    parser.add_argument(
        "--confirm-official-test",
        action="store_true",
        help="required guard after configurations have been frozen",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.grid is not None:
        return run_grid(args)
    return run_single_config(args)


if __name__ == "__main__":
    raise SystemExit(main())
