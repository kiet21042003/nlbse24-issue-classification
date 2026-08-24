# Model owner guide

This is the shortest path for P2-P4 to plug a classifier into the shared P1
infrastructure. P5 consumes the JSON artifacts described by the result schema.

## 1. Implement the classifier contract

Create one module under `src/nlbse24/modeling/` and subclass
`BaseIssueClassifier`.

```python
class MyClassifier(BaseIssueClassifier):
    @property
    def name(self) -> str:
        return "my_model"

    def fit(self, texts, labels):
        # Fit only on the supplied partition.
        return self

    def predict(self, texts):
        # Return a NumPy array of bug/feature/question labels.
        ...

    def predict_scores(self, texts):
        # Optional: shape (n_examples, n_classes), aligned to classes_.
        ...

    @property
    def classes_(self):
        ...

    def get_config(self):
        # Return every setting required to reproduce the run.
        ...
```

The model must not read CSV files, choose folds, or calculate metrics internally.

## 2. Use the generic runner

```python
from nlbse24.data import CsvIssueRepository
from nlbse24.runner import run_classifier_experiment

summary = run_classifier_experiment(
    repository=CsvIssueRepository("data/raw"),
    model_factory=lambda: MyClassifier(...),
    model_name="my_model",
    protocol="cv",
    seed=42,
    n_splits=5,
    text_fields=("title", "body"),
    output_dir="results",
)
```

Use a named factory function instead of a lambda in committed code so linting
passes. The example is abbreviated for readability.

## 3. Ownership boundaries

- **P2:** `sparse_models.py`, `fasttext.py`, and their configs/tests.
- **P3:** `encoder.py`, adapter/full fine-tuning configs, and their tests.
- **P4:** `setfit_classifier.py`, sampling configs, and their tests.
- **P5:** result ingestion, bootstrap confidence intervals, Pareto plots,
  ensemble logic, and demo notebook.
- **P1:** shared contracts, dataset protocol, Logistic Regression, CI, and
  cross-module changes.

Avoid editing another owner's model module. Changes to `domain.py`, `splits.py`,
`runner.py`, or the result schema require a short team review because they affect
all experiments.

## 4. Definition of done for a model

- Unit test for fit/predict and score order.
- Training-only smoke run succeeds on one repository.
- Full five-fold run uses seed 42 and the committed preprocessing.
- JSON artifacts pass schema/version checks.
- Config, runtime, RAM, and limitations are documented.
- No official test run until the team freezes the configuration.
