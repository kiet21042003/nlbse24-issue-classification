# Architecture and extension contract

## Design goal

Data storage, experiment protocol, model implementation, and evaluation are
separate layers. Replacing a classifier must not change how issue reports are
loaded, split, evaluated, or serialized.

```text
CLI / experiment runner
        |
        +-- IssueDatasetService -- IssueRepository
        |                            +-- memory
        |                            +-- CSV
        |                            +-- JSON
        |
        +-- split protocol (official / CV / leave-one-repository-out)
        |
        +-- BaseIssueClassifier
        |       +-- P1 Logistic Regression
        |       +-- P2 sparse / fastText
        |       +-- P3 encoders
        |       +-- P4 SetFit
        |
        +-- evaluator + result writer -- JSON artifacts for P5
```

## Stable domain model

An `IssueRecord` contains `repo`, `created_at`, `label`, `title`, and `body`.
Labels are restricted to `bug`, `feature`, and `question`. Missing text is
normalized to an empty string at the repository boundary.

## Repository boundary

`IssueRepository` exposes `load(split)` and `save(split, records)`. Business
logic depends only on this abstraction. The in-memory implementation supports
tests and notebooks; CSV/JSON implementations satisfy filesystem persistence.

## Model boundary

Every model receives already-composed texts and labels. It may not choose its
own data split or calculate its own version of the official metrics. Optional
score output must follow the order returned by `classes_` so P5 can build fair
ensembles.

## Artifact boundary

One JSON file represents one model evaluation on one repository and fold. The
path format is:

```text
results/{protocol}/{model}/{repository}/{seed}/{fold}.json
```

Artifacts include schema version, data identity, model configuration, timing,
memory, per-class metrics, macro metrics, labels, predictions, and optional
scores. This makes aggregation independent of model libraries.

The machine-readable contract is versioned at
[`schemas/result.schema.json`](../schemas/result.schema.json). Any incompatible
change requires a schema-version bump and coordination with P5.
