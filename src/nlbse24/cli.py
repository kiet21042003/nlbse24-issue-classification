"""Command-line entry point for data setup and P1 experiments."""

import argparse
import json
from pathlib import Path
from typing import Any

from nlbse24.data import (
    CsvIssueRepository,
    DatasetSplit,
    IssueDatasetService,
    JsonIssueRepository,
    audit_dataset_splits,
)
from nlbse24.data.download import download_official_dataset
from nlbse24.runner import run_logistic_baseline


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _download(args: argparse.Namespace) -> int:
    paths = download_official_dataset(args.data_dir, force=args.force)
    _json_print({"downloaded": [str(path) for path in paths]})
    return 0


def _validate(args: argparse.Namespace) -> int:
    service = IssueDatasetService(CsvIssueRepository(args.data_dir))
    summaries = {
        split.value: service.validate_official_split(split).as_dict() for split in DatasetSplit
    }
    _json_print(summaries)
    return 0


def _export_json(args: argparse.Namespace) -> int:
    source = CsvIssueRepository(args.data_dir)
    target = JsonIssueRepository(args.output_dir)
    selected = list(DatasetSplit) if args.split == "all" else [DatasetSplit(args.split)]
    output: dict[str, str] = {}
    for split in selected:
        target.save(split, source.load(split))
        output[split.value] = str(target.path_for(split))
    _json_print({"exported": output})
    return 0


def _audit(args: argparse.Namespace) -> int:
    repository = CsvIssueRepository(args.data_dir)
    report = audit_dataset_splits(
        repository.load(DatasetSplit.TRAIN), repository.load(DatasetSplit.TEST)
    )
    _json_print(report)
    return 0


def _run_baseline(args: argparse.Namespace) -> int:
    repositories = set(args.repository) if args.repository else None
    summary = run_logistic_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        protocol=args.protocol,
        repositories=repositories,
        overwrite=args.overwrite,
        allow_official_test=args.confirm_official_test,
        run_name=args.run_name,
    )
    _json_print(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlbse24", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="download and verify official CSVs")
    download.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    download.add_argument("--force", action="store_true")
    download.set_defaults(handler=_download)

    validate = subparsers.add_parser("validate-data", help="validate official dataset shape")
    validate.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    validate.set_defaults(handler=_validate)

    export = subparsers.add_parser("export-json", help="exercise JSON filesystem persistence")
    export.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    export.add_argument("--output-dir", type=Path, default=Path("data/processed/json"))
    export.add_argument("--split", choices=("train", "test", "all"), default="all")
    export.set_defaults(handler=_export_json)

    audit = subparsers.add_parser("audit-data", help="report quality and split overlap")
    audit.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    audit.set_defaults(handler=_audit)

    baseline = subparsers.add_parser("run-baseline", help="run TF-IDF + Logistic Regression")
    baseline.add_argument(
        "--protocol", choices=("cv", "pooled_cv", "loo", "official"), default="cv"
    )
    baseline.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    baseline.add_argument("--output-dir", type=Path, default=Path("results"))
    baseline.add_argument(
        "--config", type=Path, default=Path("configs/logistic_regression.toml")
    )
    baseline.add_argument("--repository", action="append", help="repeat to select repositories")
    baseline.add_argument("--overwrite", action="store_true")
    baseline.add_argument(
        "--run-name",
        default="tfidf_logistic_regression",
        help="unique artifact directory name for a model/config variant",
    )
    baseline.add_argument(
        "--confirm-official-test",
        action="store_true",
        help="required guard after configurations have been frozen",
    )
    baseline.set_defaults(handler=_run_baseline)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
