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
`results/cv/`, `results/pooled_cv/`, `results/loo/` and `results/official/` are versioned;
source ZIPs, model weights and training checkpoints stay local. Write exploratory runs outside these
published directories (for example, under `tmp/`) and do not overwrite published runs.

Other supported protocols (the same four work for every model through the shared runner):

```powershell
# One model trained on all repositories per fold, evaluated per repository
nlbse24 run-baseline --protocol pooled_cv

# Domain-transfer experiment on training data only
nlbse24 run-baseline --protocol loo

# Final test: run only after the team freezes the chosen configuration
nlbse24 run-baseline --protocol official --confirm-official-test
```

Other models have their own launchers (`scripts/run_p2_experiments.py`,
`scripts/run_p3_experiments.py`, `scripts/run_setfit.py`), all taking
`--config`, `--protocol` and `--run-name`. RoBERTa and SetFit additionally need
`pip install -r requirements-roberta.txt` and `pip install setfit==1.1.1`
(fastText: `requirements-fasttext.txt`) and a CUDA GPU in practice.

To see which model/protocol cells have no result yet, and to run only those:

```powershell
python scripts/protocol_coverage.py                 # table of what exists
python scripts/run_missing_evaluations.py --dry-run # list missing cells
python scripts/run_missing_evaluations.py --protocols pooled_cv
```

## Repository layout

```text
configs/                 Versioned experiment configuration
data/                    Local raw/processed data (ignored)
docs/                    Architecture, experiment protocol and project status
models/                  Serialized models (ignored)
presentation/            Final report (PDF) and slides
reports/                 Findings, protocol coverage and the LaTeX report source
results/                 Reviewed JSON run artifacts and summaries
scripts/                 Experiment launchers, aggregation and coverage tools
src/nlbse24/             Shared Python package
tests/                   Unit and integration tests
```

Read [docs/architecture.md](docs/architecture.md) before adding a model and
[docs/experiment_protocol.md](docs/experiment_protocol.md) before running an
experiment. Model owners should start with
[docs/model_owner_guide.md](docs/model_owner_guide.md).

## Project progress

As of **2 October 2026** every module is merged. Macro-F1 per frozen model and protocol
(seed 42, repository-balanced; a snapshot of
[reports/protocol_coverage.md](reports/protocol_coverage.md), regenerate with
`python scripts/protocol_coverage.py`):

| Model | Repo-specific CV | Pooled CV | LOO | Official |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.7636 | 0.7601 | 0.5876 | 0.7495 |
| Word LinearSVC | 0.7541 | 0.7570 | 0.5898 | 0.7498 |
| Word + Char LinearSVC | 0.7533 | 0.7658 | 0.6003 | 0.7591 |
| Hashed SGD | 0.7030 | 0.7460 | 0.6152 | 0.7322 |
| Supervised fastText | 0.7089 | 0.7126 | 0.5384 | 0.6896 |
| Complement NB | 0.7054 | 0.6757 | 0.4904 | 0.7108 |
| RoBERTa full fine-tuning | 0.7853 | 0.7924 | 0.6867 | 0.8033 |
| RoBERTa LoRA | 0.7800 | 0.7994 | 0.6896 | 0.7937 |
| SetFit MPNet | 0.7953 | – | 0.6854 | 0.8033 |
| SetFit MiniLM | 0.7881 | 0.7795 | 0.6684 | 0.7972 |

Scores are comparable within a column only. The dense models lead under every protocol
(about +0.03 under CV, +0.04 on the official test, +0.07 under LOO over the best sparse
model).

- The cells added on 2 October (Logistic Regression official and pooled CV, pooled CV of
  the P2 models, RoBERTa official and LOO, SetFit official, LOO and MiniLM pooled CV) were run
  with the frozen configurations on one laptop (RTX 3050 6 GB); their artifacts carry the
  hardware group `kiet-laptop-rtx3050-6gb`. They passed schema validation, metric
  recomputation and split-fingerprint checks. Timings from different machines are not
  comparable, and hardware identity is unrecorded for earlier artifacts, so a controlled
  global Pareto comparison is still not possible.
- A fresh clone contains about 1,900 run artifacts, including the **1,355 original P2 runs**
  (CV, LOO, official, tuning, ablations and extra seeds). Original P2 artifacts supersede the
  matching 150 reproductions, preserved in Git history; see the
  [P2 handoff audit](reports/p2_reproduction_handoff.md).
- Use `python scripts/aggregate_results.py --seed 42` to avoid mixing seeds, select frozen
  models explicitly and exclude the React-only `p2_ft_probe` from five-repository
  comparisons. Source result ZIPs are not committed.
- The final report and slides are in [presentation/](presentation/). More detail, ownership
  and next steps: [project status](docs/project_status.md) and the
  [project plan](docs/NLBSE_Topic_Comparison_and_Project_Plan.docx).

## P1 baseline

The TF-IDF + Logistic Regression baseline scores **0.7636** under training-only 5-fold CV,
0.7601 under pooled CV, **0.5876** under leave-one-repository-out and **0.7495** on the
official test (run once, after the configuration was frozen). Ablation and domain-transfer
findings, including the title-only comparison and bootstrap intervals, are in
[reports/p1_initial_findings.md](reports/p1_initial_findings.md).

## Official data source

Dataset and task definition:
<https://github.com/nlbse2024/issue-report-classification>

The dataset is downloaded from a pinned upstream commit. Do not tune on the
official test set. It is reserved for the final evaluation after configurations
have been frozen using training-only cross-validation.
