"""Evaluate a hard or probability-calibrated soft-voting ensemble."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.evaluation.ensemble import evaluate_ensemble  # noqa: E402
from nlbse24.evaluation.results import load_artifacts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--protocol", choices=("cv", "loo", "official", "pooled_cv"), required=True)
    parser.add_argument("--model", action="append", dest="models", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("hard", "soft"), default="hard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    selected = [
        artifact
        for artifact in load_artifacts(args.results_dir)
        if artifact["protocol"] == args.protocol
        and artifact["seed"] == args.seed
        and artifact["model"]["name"] in args.models
    ]
    by_evaluation: dict[tuple[str, str, int, str, str], dict[str, dict]] = defaultdict(dict)
    for artifact in selected:
        key = (
            artifact["protocol"],
            artifact["repository"],
            artifact["seed"],
            artifact["fold"],
            artifact["data"]["test_fingerprint"],
        )
        model = artifact["model"]["name"]
        if model in by_evaluation[key]:
            raise ValueError(f"duplicate artifact for {key} and model {model}")
        by_evaluation[key][model] = artifact
    evaluations = []
    for key, models in sorted(by_evaluation.items()):
        if set(models) != set(args.models):
            raise ValueError(f"incomplete ensemble evaluation {key}: {sorted(models)}")
        evaluations.append([models[model] for model in args.models])
    result = evaluate_ensemble(evaluations, strategy=args.strategy)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
