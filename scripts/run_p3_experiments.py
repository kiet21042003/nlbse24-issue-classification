"""P3 RoBERTa experiment script: per-repository (via shared runner) and pooled
(standalone, P3-only ablation) training.

"pooled_cv" is a P3-specific ablation, not one of the project's three official
protocols (see experiment_protocol.md): it trains one model per fold on every
repository pooled together, then evaluates it separately per repository. It
reuses the same artifact schema/writer as the shared runner but does not call
``run_classifier_experiment``, so no changes to runner.py were needed.

Examples::

    python scripts/run_p3_experiments.py --config configs/roberta_full.toml
    python scripts/run_p3_experiments.py --config configs/roberta_lora.toml \
        --protocol pooled_cv
    python scripts/run_p3_experiments.py --config configs/roberta_smoke.toml \
        --protocol cv --repository facebook/react
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Allow "python scripts/run_p3_experiments.py" from a checkout that was not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.data import CsvIssueRepository, DatasetSplit, IssueDatasetService 
from nlbse24.evaluation import ( 
    ResultWriter,
    aggregate_macro_f1,
    evaluate_predictions,
    make_run_artifact,
    profile_call,
)
from nlbse24.modeling.base import BaseIssueClassifier 
from nlbse24.modeling.encoder import RobertaClassifier, RobertaConfig 
from nlbse24.runner import run_classifier_experiment 
from nlbse24.splits import stratified_repository_folds 
from nlbse24.text import compose_texts 

PROTOCOL_CHOICES = ("cv", "loo", "official", "pooled_cv")


def make_factory(config: RobertaConfig) -> Any:
    """Return a named zero-argument factory; the runner builds a fresh model per fold."""

    def model_factory() -> BaseIssueClassifier:
        return RobertaClassifier(config)

    return model_factory


def run_pooled(
    *, config: RobertaConfig, run_name: str, args: argparse.Namespace
) -> dict[str, Any]:
    """P3-only ablation: one model per fold trained on all repos pooled together,
    evaluated separately per repository. Does not call run_classifier_experiment."""

    repositories = set(args.repository) if args.repository else None
    text_fields = config.experiment.text_fields
    service = IssueDatasetService(CsvIssueRepository(args.data_dir))
    writer = ResultWriter(args.output_dir, overwrite=args.overwrite)

    records = service.load(DatasetSplit.TRAIN, repositories)
    result_rows: list[dict[str, Any]] = []
    paths: list[str] = []

    for split in stratified_repository_folds(
        records, n_splits=config.experiment.n_splits, seed=config.experiment.seed
    ):
        train_records = [records[i] for i in split.train_indices]  # pooled: all repos
        model = RobertaClassifier(config)
        train_texts = compose_texts(train_records, fields=text_fields)
        y_train = [record.label for record in train_records]
        _, fit_profile = profile_call(model.fit, train_texts, y_train)

        for repo in sorted({record.repo for record in records}):
            test_indices = [i for i in split.test_indices if records[i].repo == repo]
            if not test_indices:
                continue
            test_records = [records[i] for i in test_indices]
            test_texts = compose_texts(test_records, fields=text_fields)
            y_true = [record.label for record in test_records]

            y_pred_array, inference_profile = profile_call(model.predict, test_texts)
            y_pred = [str(label) for label in y_pred_array]
            scores = model.predict_scores(test_texts)
            metrics = evaluate_predictions(y_true, y_pred)

            artifact = make_run_artifact(
                protocol="pooled_cv",
                fold=split.fold,
                repository=repo,
                seed=config.experiment.seed,
                model_name=run_name,
                model_config=model.get_config(),
                train_records=train_records,
                test_records=test_records,
                classes=model.classes_,
                metrics=metrics,
                resources={
                    "fit": fit_profile.as_dict(),
                    "inference": inference_profile.as_dict(),
                },
                y_true=y_true,
                y_pred=y_pred,
                scores=scores,
            )
            path = writer.write(artifact)
            paths.append(str(path))
            result_rows.append(
                {
                    "repository": repo,
                    "fold": split.fold,
                    "macro_f1": artifact["metrics"]["macro_average"]["f1-score"],
                }
            )

    aggregate = aggregate_macro_f1(result_rows)
    summary = {
        **aggregate,
        "split_strategy": "stratified_pooled_train_evaluated_per_repository",
        "text_fields": list(text_fields),
        "artifacts": paths,
    }
    summary_path = writer.write_summary("pooled_cv", run_name, config.experiment.seed, summary)
    return {**summary, "summary_path": str(summary_path)}


def run_shared(
    *, config: RobertaConfig, run_name: str, args: argparse.Namespace
) -> dict[str, Any]:
    """cv / loo / official: delegate entirely to the shared runner, unmodified."""

    repositories = set(args.repository) if args.repository else None
    return run_classifier_experiment(
        repository=CsvIssueRepository(args.data_dir),
        model_factory=make_factory(config),
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


def run_single_config(args: argparse.Namespace) -> int:
    path = Path(args.config)
    config = RobertaConfig.from_toml(path)
    run_name = args.run_name or RobertaClassifier(config).name

    if args.protocol == "pooled_cv":
        summary = run_pooled(config=config, run_name=run_name, args=args)
    else:
        summary = run_shared(config=config, run_name=run_name, args=args)

    print(json.dumps({"run_name": run_name, **summary}, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_p3_experiments", description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="run one committed configuration"
    )
    parser.add_argument("--protocol", choices=PROTOCOL_CHOICES, default="cv")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--repository", action="append", help="repeat to select repositories")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--run-name", help="artifact directory name; defaults to the model's name"
    )
    parser.add_argument(
        "--confirm-official-test",
        action="store_true",
        help="required guard after configurations have been frozen",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_single_config(args)


if __name__ == "__main__":
    raise SystemExit(main())
