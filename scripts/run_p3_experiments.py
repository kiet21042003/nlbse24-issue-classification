"""P3 RoBERTa experiment script for every shared protocol.

``cv``, ``pooled_cv``, ``loo`` and ``official`` all go through the shared runner
(``run_classifier_experiment``). ``pooled_cv`` was originally a P3-only ablation implemented
in this script; it now lives in the runner so that every model can be run under it.

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

from nlbse24.data import CsvIssueRepository
from nlbse24.modeling.base import BaseIssueClassifier
from nlbse24.modeling.encoder import RobertaClassifier, RobertaConfig
from nlbse24.runner import run_classifier_experiment

PROTOCOL_CHOICES = ("cv", "loo", "official", "pooled_cv")


def make_factory(config: RobertaConfig) -> Any:
    """Return a named zero-argument factory; the runner builds a fresh model per fold."""

    def model_factory() -> BaseIssueClassifier:
        return RobertaClassifier(config)

    return model_factory


def run_shared(*, config: RobertaConfig, run_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Delegate every protocol (cv, pooled_cv, loo, official) to the shared runner."""

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
    parser.add_argument("--run-name", help="artifact directory name; defaults to the model's name")
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
