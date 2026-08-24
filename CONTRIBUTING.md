# Contributing

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
pytest
```

## Model integration contract

1. Add a module under `src/nlbse24/modeling/`.
2. Implement `BaseIssueClassifier` without reading files inside the model.
3. Accept plain texts and labels in `fit`; return labels in `predict`.
4. Return stable model metadata through `get_config`.
5. Add a focused unit test and a smoke test using the shared runner.
6. Write outputs only through the shared result writer.

Do not change official splits, label names, metric definitions, or result schema
inside a model module. Propose contract changes separately so all model owners
can review them.

## Experiment discipline

- Fix random seeds and record all hyperparameters.
- Tune only on the official training set.
- Use identical text fields and splits for comparisons unless the experiment is
  explicitly an ablation.
- Never overwrite an existing run silently.
- Each owner writes the method, results, and limitations for their own module.
