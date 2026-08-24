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

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
nlbse24 download-data
nlbse24 validate-data
nlbse24 run-baseline --protocol cv
```

Linux/macOS users activate the environment with `source .venv/bin/activate`.

Downloaded data is written to `data/raw/` and verified against pinned SHA-256
checksums. Generated runs are written below `results/`; neither directory's
contents are committed.

## Repository layout

```text
configs/                 Versioned experiment configuration
data/                    Local raw/processed data (ignored)
docs/                    Architecture and experiment protocol
models/                  Serialized models (ignored)
results/                 Machine-readable run artifacts (ignored)
src/nlbse24/             Shared Python package
tests/                   Unit and integration tests
```

Read [docs/architecture.md](docs/architecture.md) before adding a model and
[docs/experiment_protocol.md](docs/experiment_protocol.md) before running an
experiment.

## Official data source

Dataset and task definition:
<https://github.com/nlbse2024/issue-report-classification>

The dataset is downloaded from a pinned upstream commit. Do not tune on the
official test set. It is reserved for the final evaluation after configurations
have been frozen using training-only cross-validation.
