"""Per-class and bootstrap analysis of P2's frozen configurations.

The experiment runner records per-class metrics, a confusion matrix and the raw
predictions for every evaluation, but the summaries it writes reduce all of that to one
cross-repository macro-F1. This script reads the artifacts back and produces the two views
that the macro column hides:

* per-class F1 for each model, plus the pooled confusion matrix of the best one, which is
  where the ``question`` weakness reported in ``reports/p2_lightweight_models.md`` comes
  from;
* a paired bootstrap over the held-out rows, which says which of the differences between
  models are separable from sampling noise.

Aggregation follows ``nlbse24.evaluation.metrics.aggregate_macro_f1`` exactly - score each
evaluation, average the evaluations within a repository, then average the five
repositories with equal weight - so the macro column reproduces the summaries rather than
offering a second opinion on them. Under ``official`` and ``loo`` there is one evaluation
per repository and the distinction does not arise; under ``cv`` there are five, and
pooling their rows instead of averaging their scores would give a different number.

The bootstrap resamples within each evaluation, so the fold and repository structure of
the competition metric is preserved, and scores every model on the *same* resamples so
that differences are paired. Macro-F1 is recomputed from a confusion matrix rather than
through ``sklearn`` on every draw; it is the same quantity - ``zero_division=0`` over a
fixed label set - about three orders of magnitude faster.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from nlbse24.constants import LABELS

FROZEN = (
    "tfidf_char_linear_svc",
    "tfidf_word_linear_svc",
    "fasttext_style_sgd",
    "tfidf_word_complement_nb",
    "fasttext_supervised",
)
REPOSITORIES = (
    "bitcoin__bitcoin",
    "facebook__react",
    "microsoft__vscode",
    "opencv__opencv",
    "tensorflow__tensorflow",
)
N_LABELS = len(LABELS)
Folds = list[tuple[np.ndarray, np.ndarray]]


def load_folds(results_dir: Path, protocol: str, model: str, repository: str, seed: int) -> Folds:
    """Return one (y_true, y_pred) pair per evaluation, label-encoded."""

    directory = results_dir / protocol / model / repository / str(seed)
    if not directory.is_dir():
        raise FileNotFoundError(f"no artifacts under {directory}")
    index = {label: position for position, label in enumerate(LABELS)}
    folds: Folds = []
    for path in sorted(directory.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        folds.append(
            (
                np.array([index[label] for label in artifact["predictions"]["y_true"]]),
                np.array([index[label] for label in artifact["predictions"]["y_pred"]]),
            )
        )
    return folds


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    codes = y_true * N_LABELS + y_pred
    return np.bincount(codes, minlength=N_LABELS**2).reshape(N_LABELS, N_LABELS)


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """Per-label F1 over a stack of confusion matrices shaped (..., n_labels, n_labels)."""

    true_positive = np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = counts.sum(axis=-2) + counts.sum(axis=-1)
    # A label absent from both the truth and the predictions contributes 0, which is what
    # zero_division=0 does in the shared metrics module.
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(true_positive.shape, dtype=float),
        where=denominator > 0,
    )


def cross_repository_f1(model_folds: dict[str, Folds]) -> np.ndarray:
    """Per-label F1, averaged over folds inside a repository and then over repositories."""

    per_repository = [
        np.mean([f1_from_counts(confusion(*fold)) for fold in model_folds[repository]], axis=0)
        for repository in REPOSITORIES
    ]
    return np.mean(per_repository, axis=0)


def bootstrap_scores(
    predictions: dict[str, dict[str, Folds]], *, resamples: int, seed: int
) -> dict[str, np.ndarray]:
    """Cross-repository macro-F1 per resample, paired across models."""

    rng = np.random.default_rng(seed)
    models = list(predictions)
    totals = {model: np.zeros(resamples) for model in models}
    offsets = np.arange(resamples)[:, None] * N_LABELS**2
    for repository in REPOSITORIES:
        n_folds = len(predictions[models[0]][repository])
        per_repository = {model: np.zeros(resamples) for model in models}
        for fold in range(n_folds):
            size = len(predictions[models[0]][repository][fold][0])
            # One index draw per evaluation, reused by every model: that is the pairing.
            draws = rng.integers(0, size, size=(resamples, size))
            for model in models:
                y_true, y_pred = predictions[model][repository][fold]
                codes = (y_true * N_LABELS + y_pred)[draws] + offsets
                counts = np.bincount(codes.ravel(), minlength=resamples * N_LABELS**2).reshape(
                    resamples, N_LABELS, N_LABELS
                )
                per_repository[model] += f1_from_counts(counts).mean(axis=-1)
        for model in models:
            totals[model] += per_repository[model] / n_folds
    return {model: total / len(REPOSITORIES) for model, total in totals.items()}


def report(args: argparse.Namespace) -> int:
    models = tuple(args.models) if args.models else FROZEN
    predictions = {
        model: {
            repository: load_folds(args.results_dir, args.protocol, model, repository, args.seed)
            for repository in REPOSITORIES
        }
        for model in models
    }
    per_label = {model: cross_repository_f1(predictions[model]) for model in models}
    point = {model: float(per_label[model].mean()) for model in models}
    order = sorted(models, key=lambda model: -point[model])

    print(f"## Per-class F1, {args.protocol}, folds averaged within repositories\n")
    labels = " ".join(f"{label:>9s}" for label in LABELS)
    print(f"{'model':26s} {labels}{'macro':>9s}{'spread':>9s}")
    for model in order:
        columns = per_label[model]
        cells = " ".join(f"{value:9.4f}" for value in columns)
        print(f"{model:26s} {cells}{point[model]:9.4f}{columns.max() - columns.min():9.4f}")

    best = order[0]
    pooled = sum(
        confusion(*fold) for repository in REPOSITORIES for fold in predictions[best][repository]
    )
    print(f"\n## Pooled confusion matrix, {args.protocol}, {best}\n")
    print("            " + "".join(f"{label:>10s}" for label in LABELS) + f"{'recall':>9s}")
    for index, label in enumerate(LABELS):
        row = "".join(f"{pooled[index, column]:>10d}" for column in range(N_LABELS))
        print(f"  {label:9s} {row}{pooled[index, index] / pooled[index].sum():9.3f}")
    precision = "".join(
        f"{pooled[column, column] / pooled[:, column].sum():>10.3f}" for column in range(N_LABELS)
    )
    print(f"  precision {precision}")

    boot = bootstrap_scores(predictions, resamples=args.resamples, seed=args.seed)
    print(f"\n## Paired bootstrap, {args.protocol}, {args.resamples} resamples, seed {args.seed}\n")
    print(f"{'model':26s}{'point':>8s}   95% CI")
    for model in order:
        low, high = np.percentile(boot[model], [2.5, 97.5])
        print(f"{model:26s}{point[model]:8.4f}   [{low:.4f}, {high:.4f}]")
    print(f"\n  paired differences from {best}:")
    for model in order[1:]:
        difference = boot[best] - boot[model]
        low, high = np.percentile(difference, [2.5, 97.5])
        verdict = "separable" if low > 0 or high < 0 else "NOT separable from noise"
        print(
            f"    vs {model:26s} {point[best] - point[model]:+.4f}"
            f"  95% CI [{low:+.4f}, {high:+.4f}]"
            f"  P(>0)={float((difference > 0).mean()):.3f}  {verdict}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p2_analysis", description=__doc__)
    parser.add_argument("--protocol", choices=("cv", "loo", "official"), default="official")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument(
        "--models",
        action="append",
        help="repeat to analyse run names other than the five frozen configurations, "
        "for example the arms of the feature ablation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return report(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
