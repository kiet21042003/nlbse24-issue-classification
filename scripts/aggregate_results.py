"""Aggregate validated JSON artifacts into repository/model/resource tables."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.evaluation.analysis import analyze_results, write_analysis_tables  # noqa: E402
from nlbse24.evaluation.tradeoffs import pareto_frontier  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    tables = analyze_results(args.results_dir, validate=not args.skip_validation)
    paths = write_analysis_tables(
        args.results_dir, args.output_dir, validate=not args.skip_validation
    )
    model_table = tables["models"]
    if not model_table.empty:
        for name, cost_columns in {
            "pareto_fit": ("fit_elapsed_seconds",),
            "pareto_inference": ("inference_elapsed_seconds",),
        }.items():
            frontier = pareto_frontier(
                model_table,
                cost_columns=cost_columns,
            )
            path = args.output_dir / f"{name}.csv"
            frontier.to_csv(path, index=False)
            paths[name] = path
    payload = {
        "results_dir": str(args.results_dir),
        "output_dir": str(args.output_dir),
        "artifact_count": len(tables["evaluations"]),
        "tables": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
