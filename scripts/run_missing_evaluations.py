"""Run the frozen configurations under every protocol they have not been run under yet.

The experiment matrix is four protocols (``cv``, ``pooled_cv``, ``loo``, ``official``) times the
frozen model configurations. A cell is *complete* when its summary file exists. This script
runs only the cells that are still missing, in an order that puts the cheap ones first, and
never overwrites an existing result.

    python scripts/run_missing_evaluations.py --dry-run
    python scripts/run_missing_evaluations.py --models roberta_base_full --protocols official loo
    python scripts/run_missing_evaluations.py --family setfit

Official-test cells need ``--confirm-official-test``: the guard exists because the official
test labels must never influence configuration choices.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("official", "loo", "pooled_cv", "cv")  # cheapest first


@dataclass(frozen=True)
class Cell:
    run_name: str
    family: str
    config: str
    launcher: str  # script path relative to the repository root


P2 = "scripts/run_p2_experiments.py"
P3 = "scripts/run_p3_experiments.py"
P4 = "scripts/run_setfit.py"

FROZEN = (
    Cell("tfidf_logistic_regression", "sparse", "configs/logistic_regression.toml", "baseline"),
    Cell("tfidf_word_linear_svc", "sparse", "configs/tfidf_word_linear_svc.toml", P2),
    Cell("tfidf_char_linear_svc", "sparse", "configs/tfidf_char_linear_svc.toml", P2),
    Cell("fasttext_style_sgd", "sparse", "configs/fasttext_style_sgd.toml", P2),
    Cell("tfidf_word_complement_nb", "sparse", "configs/tfidf_word_complement_nb.toml", P2),
    Cell("fasttext_supervised", "sparse", "configs/fasttext_supervised.toml", P2),
    Cell("roberta_base_full", "roberta", "configs/roberta_full.toml", P3),
    Cell("roberta_base_adapter", "roberta", "configs/roberta_lora.toml", P3),
    Cell("setfit_mpnet", "setfit", "configs/setfit_mpnet.toml", P4),
    Cell("setfit_minilm", "setfit", "configs/setfit_minilm.toml", P4),
)


def summary_path(results: Path, protocol: str, run_name: str, seed: int) -> Path:
    return results / protocol / run_name / f"summary-seed-{seed}.json"


def command_for(cell: Cell, protocol: str, results: Path) -> list[str]:
    if cell.launcher == "baseline":
        command = [sys.executable, "-m", "nlbse24", "run-baseline", "--config", cell.config]
    else:
        command = [sys.executable, cell.launcher, "--config", cell.config]
    command += ["--protocol", protocol, "--output-dir", str(results), "--run-name", cell.run_name]
    if protocol == "official":
        command.append("--confirm-official-test")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", choices=("sparse", "roberta", "setfit"), action="append")
    parser.add_argument("--models", nargs="+", help="run names to include")
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--dry-run", action="store_true", help="list the missing cells only")
    args = parser.parse_args(argv)

    missing = []
    for protocol in args.protocols:
        for cell in FROZEN:
            if args.family and cell.family not in args.family:
                continue
            if args.models and cell.run_name not in args.models:
                continue
            if not summary_path(args.results_dir, protocol, cell.run_name, args.seed).exists():
                missing.append((cell, protocol))

    print(f"{len(missing)} missing cell(s)")
    for cell, protocol in missing:
        print(f"  {protocol:10s} {cell.run_name}")
    if args.dry_run:
        return 0

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    failures = 0
    for cell, protocol in missing:
        command = command_for(cell, protocol, args.results_dir)
        print(f"\n>>> {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if completed.returncode != 0:
            failures += 1
            print(f"FAILED ({completed.returncode}): {cell.run_name} / {protocol}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
