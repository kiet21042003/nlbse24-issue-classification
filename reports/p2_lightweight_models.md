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

Five configurations across three model families, all behind the shared
`BaseIssueClassifier` contract and all run through P1's `run_classifier_experiment`, so
splits, metrics, profiling and artifact writing are identical to every other workstream.

| Model | Run name | Description |
|---|---|---|
| Sparse linear | `tfidf_char_linear_svc` | TF-IDF **word ⊕ character** union into LinearSVC |
| Sparse linear | `tfidf_word_linear_svc` | TF-IDF word n-grams only into LinearSVC |
| Sparse Bayes | `tfidf_word_complement_nb` | TF-IDF word n-grams only into ComplementNB |
| fastText | `fasttext_supervised` | Supervised fastText, subwords off, optional dep |
| Hashed linear | `fasttext_style_sgd` | `HashingVectorizer` + `SGDClassifier(log_loss)` |

Both TF-IDF branches live inside the pipeline, so they are refitted on each fold's
training rows only. `tfidf_char_linear_svc` is named for the branch that distinguishes it
from P1's word-only baseline; it enables **both** branches.

The hashed linear model is a dependency-free bag-of-n-grams model with a linear softmax
head. It shares fastText's hashing trick and linear objective but has no shared embedding
layer, so it is a lightweight baseline in the same family rather than a reimplementation
of fastText.

## fastText is sensitive to the order of its training file

This one is worth writing down because it is invisible from the outside and it changed a
conclusion.

fastText reads its training file sequentially, never shuffles it, and decays the learning
rate across the run. The shared dataset is sorted by label - every repository is exactly
100 `bug` rows, then 100 `feature`, then 100 `question` - and the runner hands partitions
over in that order, so the first version of the adapter wrote a label-sorted file. Under
the shared protocol, with everything else held fixed, that single detail costs:

| Training file order | bitcoin | react | vscode | opencv | tensorflow | **Cross-repo** |
|---|---:|---:|---:|---:|---:|---:|
| as handed over (label-sorted) | 0.6642 | 0.7939 | 0.6982 | 0.6327 | 0.2970 | **0.6172** |
| shuffled from the seed | 0.6515 | 0.7950 | 0.7254 | 0.6825 | 0.6901 | **0.7089** |

On `tensorflow/tensorflow` the sorted file collapses the model. In the fold inspected,
all 60 held-out issues were predicted `question`, the last label in the file, for a
macro-F1 of 0.167; the 0.2970 five-fold mean says the other folds were only marginally
better. `facebook/react` is untouched. Of the 0.0917 the fix is worth overall, 0.0786
comes from tensorflow alone - one near-degenerate repository was making fastText look far
weaker than it is.

Every other model in this project is order-invariant: LinearSVC and ComplementNB solve an
order-independent objective, and `SGDClassifier` shuffles internally each epoch. So
reporting the sorted-file number would not have been a property of fastText, it would have
been a property of the adapter. `FastTextClassifier.fit` now permutes the lines with
`random.Random(config.experiment.seed)` before writing them, which also gives the run a
reproducible order that fastText 0.9.2's own API cannot express - it exposes no seed
parameter at all. Every fastText number in this report comes from the shuffled version.

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

## Model family comparison

Cross-repository macro-F1 for every frozen configuration, on all three protocols. CV and
LOO use the training set only; the official column was produced afterwards.

| Model | CV | LOO | Official test |
|---|---:|---:|---:|
| `tfidf_word_linear_svc` | **0.7541** | 0.5898 | 0.7498 |
| `tfidf_char_linear_svc` | 0.7533 | 0.6003 | **0.7591** |
| `fasttext_supervised` | 0.7089 | 0.5384 | 0.6896 |
| `tfidf_word_complement_nb` | 0.7054 | 0.4904 | 0.7108 |
| `fasttext_style_sgd` | 0.7030 | **0.6152** | 0.7322 |
| P1 Logistic Regression | 0.7636 | 0.5876 | not run |

Two things stand out.

**The protocols do not agree on a winner.** The LinearSVC pair leads on within-repository
CV and on the official test, but the best transfer model is the hashed linear one, which
is simultaneously the *worst* of the five on CV. Ranking lightweight models on CV alone
would have picked the wrong model for a cross-project deployment.

**Transfer separates the families far more than CV does.** The CV column spans 0.0511
between best and worst; the LOO column spans 0.1248. In-domain cross-validation compresses
the differences between these model families, which is worth remembering when RQ1 is
written up: five models that look 5 points apart in-domain are 12 points apart the moment
the repository changes.

No lightweight model beats P1's Logistic Regression on CV. Two of them beat it on transfer.

## Domain transfer (leave-one-repository-out)

Each held-out repository is scored by a model pooled over the other four, on training data
only.

| Model | bitcoin | react | vscode | opencv | tensorflow | **Cross-repo** |
|---|---:|---:|---:|---:|---:|---:|
| `fasttext_style_sgd` | 0.6203 | 0.6078 | 0.6449 | 0.5853 | 0.6179 | **0.6152** |
| `tfidf_char_linear_svc` | 0.6230 | 0.6017 | 0.5462 | 0.5949 | 0.6358 | **0.6003** |
| `tfidf_word_linear_svc` | 0.5670 | 0.6128 | 0.5327 | 0.6081 | 0.6285 | **0.5898** |
| P1 Logistic Regression | 0.5604 | 0.6134 | 0.5648 | 0.5895 | 0.6097 | **0.5876** |
| `fasttext_supervised` | 0.5837 | 0.6149 | 0.4317 | 0.5142 | 0.5477 | **0.5384** |
| `tfidf_word_complement_nb` | 0.5274 | 0.3445 | 0.4710 | 0.5018 | 0.6071 | **0.4904** |

Within the TF-IDF family the character branch pays for itself: +0.0105 over word-only
features and +0.0127 over P1's baseline, driven almost entirely by bitcoin (+0.0560 over
word-only, +0.0626 over P1). Character n-grams capture sub-word regularities that survive
a change of repository, whereas word features are more tied to a project's own vocabulary
and issue template.

The overall winner, `fasttext_style_sgd`, also uses character n-grams, and it is the only
model that stays above 0.585 on every held-out repository - it has no repository it is
bad at. That is consistent with the character-feature story, though it is not proof of it:
the hashed model differs from the TF-IDF models in more than one way at once (hashing
instead of a fitted vocabulary, no idf weighting, and an SGD-fitted log-loss objective).

ComplementNB transfers worst by a wide margin, and its react column (0.3445) is a partial
collapse rather than a uniform decline. A naive Bayes model estimates per-class word
distributions directly, so when the held-out repository's vocabulary and issue template
are unlike the pooled four, those estimates transfer poorly; a discriminative model only
has to keep a decision boundary roughly in place.

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
| `fasttext_style_sgd` | 0.7199 | 0.7877 | 0.7177 | 0.7575 | 0.6784 | **0.7322** |
| `tfidf_word_complement_nb` | 0.6693 | 0.7464 | 0.7097 | 0.6904 | 0.7384 | **0.7108** |
| `fasttext_supervised` | 0.6602 | 0.7942 | 0.7373 | 0.7185 | 0.5378 | **0.6896** |

The union leads by 0.0093 over word-only features, consistent with the transfer result and
opposite to the single-seed CV ordering. The order of the five models matches CV except
that `fasttext_style_sgd` moves up two places and `fasttext_supervised` drops to last.

fastText's official score is dragged down by one repository: 0.5378 on
`tensorflow/tensorflow`, against 0.6901 for the same configuration under CV. It is the
same repository that exposed the training-order defect above, and it stays the weakest
column for this model even after the fix, so the remaining gap is a property of fastText
on this repository rather than of the adapter.

For external context, the NLBSE'24 competition report puts the SetFit baseline near 0.827
and a published fastText entry at 0.718. A tuned TF-IDF plus LinearSVC lands between them
at 0.7591, for a fit that takes about four seconds on a fold of 240 issues.

## Cost

Mean per evaluation over the 25 cross-validation evaluations of each frozen
configuration: a fit on roughly 240 issues and inference on 60. These figures come from a
dedicated pass that ran **one configuration at a time with nothing else on the machine**,
because `elapsed_seconds` is wall clock and the tuning sweeps were run concurrently.
Accuracy was identical to the concurrent runs in every cell, which also confirms that all
five models - fastText included - are now reproducible run to run.

| Model | fit s | inference s | python peak MB | RSS delta MB | vocabulary | CV | Official |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tfidf_word_complement_nb` | 0.429 | 0.114 | 8.36 | 2.78 | 6,107 | 0.7054 | 0.7108 |
| `tfidf_word_linear_svc` | 0.460 | 0.121 | 8.36 | 2.97 | 6,107 | 0.7541 | 0.7498 |
| `fasttext_supervised` | 0.576 | **0.009** | 5.55 | 196.04 | hashed | 0.7089 | 0.6896 |
| `fasttext_style_sgd` | 3.879 | 0.932 | 54.37 | 47.19 | hashed | 0.7030 | 0.7322 |
| `tfidf_char_linear_svc` | 3.932 | 1.394 | 43.02 | 11.14 | 32,566 | 0.7533 | 0.7591 |

**`tfidf_word_linear_svc` dominates on CV**: nothing here is both cheaper and more
accurate. It is also the only model in the project so far that fits a repository in under
half a second. `tfidf_word_complement_nb` is the clearest dominated point - the same cost
to within 7%, for 0.049 less macro-F1.

**fastText owns the latency axis.** Its 0.009 s per fold is 13x faster than the cheapest
TF-IDF model and 155x faster than the union, about 0.15 ms per issue: the model is a
single averaged embedding lookup plus one matrix multiply, with no vocabulary hashing to
redo at predict time. It pays for that in memory, and the payment is invisible to the
shared profiler - 5.55 MB of Python peak against **196 MB of real RSS growth**, because
the embedding matrix is allocated in C++.

**The character branch is the expensive way to buy accuracy.** Against word-only features
it costs 8.5x the fit time, 11.5x the inference time and 5.1x the peak memory for +0.0093
on the official test and -0.0008 on CV. It is worth it if transfer or official accuracy is
the target, and not otherwise; both operating points are frozen so the integrator can pick.

For a Pareto plot, the frontier on (official macro-F1, inference seconds) is three points:
`fasttext_supervised` (0.6896, 0.009), `tfidf_word_linear_svc` (0.7498, 0.121) and
`tfidf_char_linear_svc` (0.7591, 1.394). Use `rss_delta_mb`, not `python_peak_mb`, if
memory is an axis.

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

- **Timings carry profiler overhead.** The shared profiler wraps every fit in
  `tracemalloc`, whose overhead scales with allocation count; character n-gram analysis
  allocates one string per n-gram and is penalised more than word analysis. Figures are
  comparable across P1-P5 because everyone uses the same profiler, but they are not a
  hardware benchmark and they do not transfer across machines.
- **Absolute timings drift; ratios do not.** Two configurations were measured twice, each
  time as the only process on the machine. Both came back about 45% slower on the second
  pass (word-only fit 0.460 s then 0.674 s, char-only 3.036 s then 4.358 s), while the
  char-to-word ratio held at 6.6x then 6.5x and the memory and vocabulary figures were
  identical to the last digit. Read the cost table as ratios between models, not as
  absolute seconds, and do not read the third decimal at all.
- **`python_peak_mb` is blind to native memory.** `fasttext_supervised` reports 5.55 MB of
  Python peak against 196 MB of real RSS growth, because its embedding matrix is allocated
  in C++. Any Pareto plot including fastText must use `rss_delta_mb`.
- **fastText's official score rests on one bad repository.** Its `tensorflow/tensorflow`
  column is 0.5378 against 0.6901 under CV; the other four columns are in line with the
  other models. Treat its cross-repository mean as a mean over an outlier, not as an even
  weakness.
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

# the five frozen configurations, on all three protocols
$configs = @(
    "configs/tfidf_char_linear_svc.toml",
    "configs/tfidf_word_linear_svc.toml",
    "configs/tfidf_word_complement_nb.toml",
    "configs/fasttext_supervised.toml",
    "configs/fasttext_style_sgd.toml"
)
foreach ($c in $configs) {
    python scripts/run_p2_experiments.py --config $c --protocol cv
    python scripts/run_p2_experiments.py --config $c --protocol loo
    python scripts/run_p2_experiments.py --config $c --protocol official --confirm-official-test
}

# seed robustness for the two sparse LinearSVC configurations
foreach ($s in 1, 2, 3) {
    python scripts/run_p2_experiments.py --config configs/tfidf_char_linear_svc.toml `
        --protocol cv --seed $s
    python scripts/run_p2_experiments.py --config configs/tfidf_word_linear_svc.toml `
        --protocol cv --seed $s
}

# feature ablations
python scripts/run_p2_experiments.py --config configs/ablations/p2_char_only.toml --protocol cv
python scripts/run_p2_experiments.py --config configs/ablations/p2_title_only.toml --protocol cv
```

The loop above is written serially on purpose. `elapsed_seconds` is wall clock, so running
experiments concurrently inflates it badly - concurrent runs on this machine reported fits
of over an hour for a model that takes four seconds. Accuracy is unaffected by
concurrency, so only the cost table needs the serial treatment.

## For whoever merges this branch

This branch adds only new files and modifies none, so it should merge without conflicts.
Two follow-ups belong to the integrator rather than to P2:

- `README.md` has no pointer to these reports. Add one next to the P2 row, and mention
  `requirements-fasttext.txt` as an optional extra.
- `src/nlbse24/evaluation/artifacts.py` records versions for a fixed package list that
  cannot include fastText, so the backend version is carried in `model.config.backend`
  instead. If the team wants it in `environment.packages`, that is a shared-code change
  needing P1's review.
