"""P4 entry point for SetFit experiments through the shared runner."""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

# Allow "python scripts/run_setfit.py" from a checkout that was not pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.data import CsvIssueRepository  # noqa: E402
from nlbse24.modeling.base import BaseIssueClassifier  # noqa: E402
from nlbse24.modeling.setfit_classifier import SetFitClassifier, SetFitConfig  # noqa: E402
from nlbse24.runner import run_classifier_experiment  # noqa: E402


def build_model_factory(
    config_path: str | Path,
) -> tuple[SetFitConfig, Callable[[], BaseIssueClassifier]]:
    config = SetFitConfig.from_toml(config_path)

    def factory() -> BaseIssueClassifier:
        return SetFitClassifier(config)

    return config, factory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SetFit experiment (P4).")
    parser.add_argument("--config", type=Path, required=True, help="Path to a SetFit TOML config.")
    parser.add_argument(
        "--protocol",
        choices=("cv", "loo", "official"),
        default="cv",
        help="cv= tuning; loo=domain transfer; official=frozen-model test.",

    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--repository",
        action="append",
        dest="repository",
        default=None,
        help="repeat to select repositories, e.g. --repository bitcoin/bitcoin",
    )
    parser.add_argument("--run-name", default=None, help="Override the artifact directory name.")
    parser.add_argument(
        "--overwrite", action="store_true", help="allow overwriting existing artifacts"
    )
    parser.add_argument(
        "--confirm-official-test",
        action="store_true",
        help="required guard after configurations have been frozen for official test",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, model_factory = build_model_factory(args.config)
    probe = SetFitClassifier(config)
    repositories = set(args.repository) if args.repository else None
    summary = run_classifier_experiment(
        repository=CsvIssueRepository(args.data_dir),
        model_factory=model_factory,
        model_name=args.run_name or probe.name,
        output_dir=args.output_dir,
        protocol=args.protocol,
        seed=config.experiment.seed,
        n_splits=config.experiment.n_splits,
        text_fields=config.experiment.text_fields,
        repositories=repositories,
        overwrite=args.overwrite,
        allow_official_test=args.confirm_official_test,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
