# NLBSE'24 Issue Classification

Reproducible experiments for classifying GitHub issues as `bug`, `feature`, or
`question` on the official NLBSE'24 dataset.

## Scope

The project compares lightweight sparse models, fine-tuned encoders, Sentence
Transformers, and optional ensembles under one shared data and evaluation
contract. The official task trains one classifier per repository and ranks a
submission by the arithmetic mean of the five repository macro-F1 scores.

P1 owns the common foundation and the TF-IDF + Logistic Regression baseline.
P2-P5 should extend `BaseIssueClassifier` and reuse the same loaders, splitters,
metrics, and result writer.

## Team responsibilities

| Owner | Team member | Student ID | Role | Primary code contribution |
|---|---|---|---|---|
| P1 | Phạm Tuấn Kiệt | 20252751M | Leader, data, protocol, and Logistic Regression | Data loader and EDA; in-memory and file persistence; official, cross-validation, and leave-one-repository-out splitters; TF-IDF word n-grams with a Logistic Regression baseline; shared result schema. |
| P2 | Nguyễn Trung Hiếu | 20261059M | Lightweight models | TF-IDF character n-grams with LinearSVC and ComplementNB; fastText; hyperparameter tuning and feature ablation. |
| P3 | Lê Ngọc Ánh | 20252275M | Fine-tuned encoder | RoBERTa-base full fine-tuning and adapters; per-repository versus pooled training experiments. |
| P4 | Nguyễn Hoàng Nam | 20261081M | Sentence Transformers | Reproduce the SetFit MPNet baseline; evaluate MiniLM; contrastive-sampling ablation. |
| P5 | Bùi Tiến Dũng | 20261204M | Evaluation and integration | Metrics, bootstrap confidence intervals, resource profiling, Pareto analysis, ensemble, result aggregation, and demo notebook; local LLM experiments are optional. |

Each member owns the implementation, tests, experiments, and Method, Results,
and Discussion for their module. P5 integrates shared outputs but does not
replace the scientific analysis of the other members.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
nlbse24 download-data
nlbse24 validate-data
nlbse24 audit-data
nlbse24 run-baseline --protocol cv
```

Linux/macOS users activate the environment with `source .venv/bin/activate`.

Downloaded data is written to `data/raw/` and verified against pinned SHA-256
checksums. Downloaded data stays local. Reviewed JSON runs and summaries under
`results/cv/`, `results/loo/` and `results/pooled_cv/` are versioned;
source ZIPs and model weights stay local. Write exploratory runs outside these
published directories (for example, under `tmp/`) and do not overwrite published runs.

Other supported protocols:

```powershell
# Domain-transfer experiment on training data only
nlbse24 run-baseline --protocol loo

# Final test: run only after the team freezes the chosen configuration
nlbse24 run-baseline --protocol official --confirm-official-test
```

## Repository layout

```text
configs/                 Versioned experiment configuration
data/                    Local raw/processed data (ignored)
docs/                    Architecture and experiment protocol
models/                  Serialized models (ignored)
results/                 Reviewed JSON run artifacts and summaries
src/nlbse24/             Shared Python package
tests/                   Unit and integration tests
```

Read [docs/architecture.md](docs/architecture.md) before adding a model and
[docs/experiment_protocol.md](docs/experiment_protocol.md) before running an
experiment. Model owners should start with
[docs/model_owner_guide.md](docs/model_owner_guide.md).

## Project progress

As of **28 September 2026**, P2 lightweight models, P3 RoBERTa/LoRA, P4
SetFit and P5 evaluation/integration have been merged. A corrected combined
report and initial P5 paired comparison/ensemble analysis are available;
remaining official evaluations and final submission verification are pending.
See [project status and next steps](docs/project_status.md)
and the updated [project plan](docs/NLBSE_Topic_Comparison_and_Project_Plan.docx).
As of **30 September 2026**, a fresh clone includes **1,690 run artifacts**
from P1–P4, including **1,355 original P2 runs** (CV, LOO, official, tuning,
ablations and extra seeds). Original P2 artifacts supersede the matching 150
reproductions, preserved in Git history. See [P2 handoff audit](reports/p2_reproduction_handoff.md).
As of **2 October 2026**, the protocol matrix is nearly complete: every model has
repository-specific CV, and Logistic Regression, the P2 models and RoBERTa also have
pooled CV, LOO and official results (SetFit has CV and, for MPNet, official; see
[protocol coverage](reports/protocol_coverage.md)). Regenerate the table with
`python scripts/protocol_coverage.py` and run missing cells with
`python scripts/run_missing_evaluations.py --dry-run`.
The final report and slides are in [presentation/](presentation/).
Use `python scripts/aggregate_results.py --seed 42` to avoid mixing seeds;
select frozen models explicitly and exclude the React-only `p2_ft_probe` from
five-repository comparisons. Source result ZIPs are not committed. Unverified
hardware identity still prevents a controlled global Pareto comparison.

## P1 baseline status

The first end-to-end training-only 5-fold run completed successfully across all
five repositories (25 evaluations). Its cross-repository macro-F1 is **0.7636**.
This is a pipeline smoke test and preliminary baseline, not the official test
score. Initial ablation and domain-transfer findings are summarized in
[reports/p1_initial_findings.md](reports/p1_initial_findings.md).

## Official data source

Dataset and task definition:
<https://github.com/nlbse2024/issue-report-classification>

The dataset is downloaded from a pinned upstream commit. Do not tune on the
official test set. It is reserved for the final evaluation after configurations
have been frozen using training-only cross-validation.
