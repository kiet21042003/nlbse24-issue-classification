"""Paired, repository-balanced analysis of P1's fixed LOO predictions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.constants import LABELS  # noqa: E402
from nlbse24.evaluation.metrics import evaluate_predictions  # noqa: E402
from nlbse24.evaluation.results import load_artifacts  # noqa: E402
from scripts.p2_analysis import (  # noqa: E402
    REPOSITORIES,
    bootstrap_scores,
    confusion,
    cross_repository_f1,
)

MODELS = ("tfidf_logistic_regression", "tfidf_lr_title_only")


def analyze(root: Path, *, resamples: int = 10000, seed: int = 42) -> dict:
    """Resample paired issue rows within each of five fixed repositories."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    artifacts = {}
    for model in MODELS:
        runs = load_artifacts(root / "loo" / model)
        selected = [a for a in runs if a["seed"] == 42]
        by_repo = {a["repository"].replace("/", "__"): a for a in selected}
        if len(selected) != 5 or set(by_repo) != set(REPOSITORIES):
            raise ValueError("expected one seed-42 LOO artifact per repository")
        artifacts[model] = by_repo
    predictions = {model: {} for model in MODELS}
    per_repo = {model: {} for model in MODELS}
    for repo in REPOSITORIES:
        reference = artifacts[MODELS[0]][repo]
        for model in MODELS:
            a = artifacts[model][repo]
            if (a["protocol"] != "loo" or a["fold"] != reference["fold"]
                    or a["data"] != reference["data"]
                    or a["predictions"]["y_true"] != reference["predictions"]["y_true"]
                    or tuple(a["classes"]) != LABELS):
                raise ValueError(f"unaligned LOO artifacts: {model}/{repo}")
            expected_fields = ["title", "body"] if model == MODELS[0] else ["title"]
            if a["model"]["config"]["experiment"]["text_fields"] != expected_fields:
                raise ValueError("unexpected text configuration")
            pred = a["predictions"]
            metrics = evaluate_predictions(pred["y_true"], pred["y_pred"])
            if metrics != a["metrics"]:
                raise ValueError("stored metrics do not match predictions")
            predictions[model][repo] = [tuple(
                np.asarray([LABELS.index(label) for label in pred[key]])
                for key in ("y_true", "y_pred")
            )]
            per_repo[model][repo] = metrics
    boot = bootstrap_scores(predictions, resamples=resamples, seed=seed)
    models = {}
    for model in MODELS:
        per_class = cross_repository_f1(predictions[model])
        pooled = sum(confusion(*predictions[model][repo][0]) for repo in REPOSITORIES)
        models[model] = {
            "macro_f1": float(per_class.mean()),
            "ci95": np.quantile(boot[model], [0.025, 0.975]).tolist(),
            "mean_repository_class_f1": dict(zip(LABELS, per_class.tolist(), strict=True)),
            "pooled_confusion_matrix": pooled.tolist(),
            "per_repository": per_repo[model],
        }
    delta = boot[MODELS[0]] - boot[MODELS[1]]
    return {
        "protocol": "loo", "experiment_seed": 42, "bootstrap_seed": seed,
        "resamples": resamples, "class_order": list(LABELS),
        "method": "paired issue bootstrap within repository; equal repository weight",
        "limitations": "Conditional on fixed predictions and repositories; excludes training "
        "variability and new-repository uncertainty. Issue rows assumed exchangeable; "
        "duplicate/dependent issues may make intervals optimistic.",
        "models": models,
        "title_body_minus_title_only": {
            "point_difference": models[MODELS[0]]["macro_f1"] - models[MODELS[1]]["macro_f1"],
            "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("reports/p1_loo_analysis.json"))
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    result = analyze(args.results_dir, resamples=args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["title_body_minus_title_only"]}))


if __name__ == "__main__":
    main()
