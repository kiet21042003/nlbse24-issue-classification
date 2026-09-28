"""Aggregate validated JSON artifacts into repository/model/resource tables."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.evaluation.analysis import analyze_results, write_analysis_tables  # noqa: E402
from nlbse24.evaluation.tradeoffs import pareto_frontier  # noqa: E402


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def _write_pareto_tables(
    model_table,
    output_dir: Path,
    *,
    include_unknown_hardware: bool,
    assume_hardware_group: str | None,
) -> tuple[dict[str, Path], list[str]]:
    """Write protocol- and hardware-specific Pareto tables.

    Resource measurements are only compared within the same protocol and an
    explicit hardware group. Existing artifacts created before hardware
    metadata was recorded are ``unknown`` and are excluded by default.
    """

    for legacy_name in ("pareto_fit.csv", "pareto_inference.csv"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
    if model_table.empty:
        return {}, []
    frame = model_table.copy()
    if "hardware_group" not in frame.columns:
        frame["hardware_group"] = "unknown"
    if assume_hardware_group:
        frame.loc[frame["hardware_group"] == "unknown", "hardware_group"] = (
            assume_hardware_group
        )

    paths: dict[str, Path] = {}
    skipped: list[str] = []
    for (protocol, hardware_group), group in frame.groupby(
        ["protocol", "hardware_group"], dropna=False, sort=True
    ):
        hardware_group = str(hardware_group)
        if hardware_group == "unknown" and not include_unknown_hardware:
            skipped.append(f"{protocol}/{hardware_group}")
            continue
        suffix = f"{_slug(protocol)}_{_slug(hardware_group)}"
        for name, cost_columns in {
            "pareto_fit": ("fit_elapsed_seconds",),
            "pareto_inference": ("inference_elapsed_seconds",),
        }.items():
            frontier = pareto_frontier(group, cost_columns=cost_columns)
            path = output_dir / f"{name}_{suffix}.csv"
            frontier.to_csv(path, index=False)
            paths[f"{name}_{suffix}"] = path
    return paths, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--include-unknown-hardware",
        action="store_true",
        help="include legacy artifacts without hardware metadata; exploratory only",
    )
    parser.add_argument(
        "--assume-hardware-group",
        help="assign legacy unknown artifacts to an explicitly verified hardware group",
    )
    args = parser.parse_args(argv)

    tables = analyze_results(args.results_dir, validate=not args.skip_validation)
    paths = write_analysis_tables(
        args.results_dir, args.output_dir, validate=not args.skip_validation
    )
    pareto_paths, skipped_groups = _write_pareto_tables(
        tables["models"],
        args.output_dir,
        include_unknown_hardware=args.include_unknown_hardware,
        assume_hardware_group=args.assume_hardware_group,
    )
    paths.update(pareto_paths)
    payload = {
        "results_dir": str(args.results_dir),
        "output_dir": str(args.output_dir),
        "artifact_count": len(tables["evaluations"]),
        "tables": {name: str(path) for name, path in paths.items()},
        "skipped_unknown_hardware_groups": skipped_groups,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
