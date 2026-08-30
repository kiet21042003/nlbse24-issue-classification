# P2 lightweight models

Date: 2026-08-31

Owner: P2 (Nguyễn Trung Hiếu, 20261059M). Module scope: TF-IDF sparse classifiers,
fastText, and the tuning and feature ablation behind them.

All cross-validation and leave-one-repository-out numbers use the official **training set
only**. Official-test numbers were produced after the configurations in `configs/` were
frozen and committed, and no test label influenced any configuration choice. Unless a row
says otherwise, results use seed 42 and the shared `title` + `body` text composition.

Environment: Python 3.11.16, scikit-learn 1.7.2, numpy 1.26.4, pandas 2.2.0,
`fasttext-wheel` 0.9.2, Windows 11, matching the pinned stack in `requirements.txt` and
the CI job. Reference points from P1's Logistic Regression: **0.7636** CV, **0.5876** LOO.

## Method

Four models, all behind the shared `BaseIssueClassifier` contract and all run through
P1's `run_classifier_experiment`, so splits, metrics, profiling and artifact writing are
identical to every other workstream.

| Model | Run name | Description |
|---|---|---|
| Sparse linear | `tfidf_char_linear_svc` | TF-IDF **word ⊕ character** union into LinearSVC |
| Sparse linear | `tfidf_word_linear_svc` | TF-IDF word n-grams only into LinearSVC |
| Sparse Bayes | `tfidf_word_complement_nb` | TF-IDF word n-grams only into ComplementNB |
| fastText | `fasttext_supervised` | Facebook's supervised fastText, optional dependency |
| Hashed linear | `fasttext_style_sgd` | `HashingVectorizer` + `SGDClassifier(log_loss)` |

Both TF-IDF branches live inside the pipeline, so they are refitted on each fold's
training rows only. `tfidf_char_linear_svc` is named for the branch that distinguishes it
from P1's word-only baseline; it enables **both** branches.

The hashed linear model is a dependency-free bag-of-n-grams model with a linear softmax
head. It shares fastText's hashing trick and linear objective but has no shared embedding
layer, so it is a lightweight baseline in the same family rather than a reimplementation
of fastText.

## Seed robustness

The tuning grid was scored on the same folds the CV column reports, so a single-seed
argmax is optimistic. Both frozen sparse configurations were therefore re-run on four
seeds.

| Model | seed 42 | seed 1 | seed 2 | seed 3 | Mean | Spread |
|---|---:|---:|---:|---:|---:|---:|
| `tfidf_char_linear_svc` | 0.7533 | 0.7494 | 0.7519 | 0.7479 | **0.7506** | 0.0054 |
| `tfidf_word_linear_svc` | 0.7541 | 0.7434 | 0.7457 | 0.7467 | **0.7475** | 0.0107 |

This matters: on seed 42 alone the word-only model looks better, and on the four-seed mean
the union is ahead and half as variable. Seed 42 is the top of the word-only range. Any
claim separating these two configurations on within-repository cross-validation is
unsupported; the differences that do hold up appear under domain transfer and on the
official test.

## Domain transfer (leave-one-repository-out)

Each held-out repository is scored by a model pooled over the other four, on training data
only.

| Model | bitcoin | react | vscode | opencv | tensorflow | **Cross-repo** |
|---|---:|---:|---:|---:|---:|---:|
| `tfidf_char_linear_svc` | 0.6230 | 0.6017 | 0.5462 | 0.5949 | 0.6358 | **0.6003** |
| `tfidf_word_linear_svc` | 0.5670 | 0.6128 | 0.5327 | 0.6081 | 0.6285 | **0.5898** |
| P1 Logistic Regression | 0.5604 | 0.6134 | 0.5648 | 0.5895 | 0.6097 | **0.5876** |

Here the character branch pays for itself: +0.0105 over word-only features and +0.0127
over P1's baseline, driven almost entirely by bitcoin (+0.0560 over word-only, +0.0626
over P1). Character n-grams capture sub-word regularities that survive a change of
repository, whereas word features are more tied to a project's own vocabulary and issue
template. This is the clearest positive result for character features in this workstream.

The usual caveat from P1's analysis applies: leave-one-repository-out changes both the
domain and the training-set size, so it is not a controlled estimate of domain shift
alone.

## Official test

Run once per model after the configurations were frozen and committed, using the
`--confirm-official-test` guard.

| Model | bitcoin | react | vscode | opencv | tensorflow | **Cross-repo** |
|---|---:|---:|---:|---:|---:|---:|
| `tfidf_char_linear_svc` | 0.7084 | 0.8109 | 0.7471 | 0.7760 | 0.7532 | **0.7591** |
| `tfidf_word_linear_svc` | 0.7094 | 0.7878 | 0.7238 | 0.7509 | 0.7769 | **0.7498** |

The union leads by 0.0093, consistent with the transfer result and opposite to the
single-seed CV ordering. For external context, the NLBSE'24 competition report puts the
SetFit baseline near 0.827 and a published fastText entry at 0.718; a tuned TF-IDF plus
LinearSVC therefore lands between them while training in about four seconds per
repository.

## Notes for P5 (integration and ensembling)

- Every model reports `score_semantics` inside `model.config`. `tfidf_*_linear_svc`
  returns **`decision_margin`** - unbounded one-vs-rest margins, *not* probabilities.
  `tfidf_word_complement_nb`, `fasttext_supervised` and `fasttext_style_sgd` return
  **`probability`**. Soft-voting across these without rescaling will let the LinearSVC
  margins dominate by magnitude alone.
- Score columns follow `classes_`, which is always `("bug", "feature", "question")`.
- Sparse models also report the fitted vocabulary size under
  `model.config.fitted.vocabulary_size`, which the data audit asked for and which is the
  natural x-axis for a size/accuracy plot.

## Limitations

- **Timings are not wall-clock.** The shared profiler wraps every fit in `tracemalloc`,
  whose overhead scales with allocation count; character n-gram analysis allocates one
  string per n-gram and is penalised more than word analysis. Figures are comparable
  across P1-P5 because everyone uses the same profiler, not across machines.
- **`python_peak_mb` is blind to native memory.** On one `facebook/react` fold,
  `fasttext_supervised` reported 4.61 MB of Python peak against 198.50 MB of actual RSS
  growth. Any Pareto plot including fastText must use `rss_delta_mb`.
- **Selection bias.** Configurations were chosen by argmax over a grid scored on the same
  CV folds. The seed table above is the correction; treat single-seed CV gaps below about
  0.01 macro-F1 as noise.
- **Inherited data issues.** Three exact-text keys are shared between train and test, one
  of them with contradictory labels, so official scores carry a small optimistic
  memorisation component. `predict_scores` is called outside the shared profiler and after
  the timed `predict`, so LinearSVC pays for inference twice in wall-clock terms but only
  once in the recorded figure.

## Reproduction

```powershell
conda create -y -n nlbse24 python=3.11
conda activate nlbse24
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pip install -r requirements-fasttext.txt   # optional fastText backend
nlbse24 download-data

# tuning, training data only
python scripts/run_p2_experiments.py --grid configs/sweeps/p2_linear_svc_grid.toml
python scripts/run_p2_experiments.py --grid configs/sweeps/p2_complement_nb_grid.toml
python scripts/run_p2_experiments.py --grid configs/sweeps/p2_fasttext_grid.toml
python scripts/run_p2_experiments.py --grid configs/sweeps/p2_fasttext_style_grid.toml

# frozen configurations
python scripts/run_p2_experiments.py --config configs/tfidf_char_linear_svc.toml --protocol cv
python scripts/run_p2_experiments.py --config configs/tfidf_char_linear_svc.toml --protocol cv --seed 1
python scripts/run_p2_experiments.py --config configs/tfidf_char_linear_svc.toml --protocol loo
python scripts/run_p2_experiments.py --config configs/tfidf_char_linear_svc.toml `
    --protocol official --confirm-official-test
```

Run one experiment at a time when quoting timings: the recorded `elapsed_seconds` is wall
clock, so concurrent runs inflate it badly.

## For whoever merges this branch

This branch adds only new files and modifies none, so it should merge without conflicts.
Two follow-ups belong to the integrator rather than to P2:

- `README.md` has no pointer to these reports. Add one next to the P2 row, and mention
  `requirements-fasttext.txt` as an optional extra.
- `src/nlbse24/evaluation/artifacts.py` records versions for a fixed package list that
  cannot include fastText, so the backend version is carried in `model.config.backend`
  instead. If the team wants it in `environment.packages`, that is a shared-code change
  needing P1's review.
