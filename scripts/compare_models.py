"""Run a paired, repository-balanced bootstrap comparison over artifacts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.evaluation.comparisons import (  # noqa: E402
    group_comparable_artifacts,
    paired_bootstrap,
)
from nlbse24.evaluation.results import load_artifacts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--protocol", choices=("cv", "loo", "official", "pooled_cv"), required=True)
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    artifacts = load_artifacts(args.results_dir)
    grouped = group_comparable_artifacts(
        artifacts, protocol=args.protocol, models=args.models, seed=args.seed
    )
    summary, differences = paired_bootstrap(
        grouped, n_resamples=args.resamples, seed=args.seed
    )
    payload = {
        "protocol": args.protocol,
        "summary": summary.to_dict(orient="records"),
        "differences": differences.to_dict(orient="records"),
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
