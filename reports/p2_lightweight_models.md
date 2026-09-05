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
unsupported.

The same turns out to be true of the other two protocols once they are given intervals
rather than point estimates. A paired bootstrap over the held-out rows puts the union
ahead by +0.0008 on CV, +0.0105 under transfer and +0.0093 on the official test, and every
one of those intervals crosses zero:

| Protocol | union - word only | 95% CI | P(union ahead) |
|---|---:|---|---:|
| CV | +0.0008 | [-0.0107, +0.0126] | 0.554 |
| LOO | +0.0105 | [-0.0056, +0.0267] | 0.899 |
| Official | +0.0093 | [-0.0020, +0.0210] | 0.945 |

So no single protocol separates these two feature sets, and neither does the set of them
together. What is left is weaker than a significance claim and should be stated as such:
the union is ahead on the four-seed CV mean and on all three protocol means, but only on
9 of the 15 per-repository columns those protocols produce - against 7.5 expected by
chance. The aggregate sign is consistent; the per-repository evidence behind it is close
to a coin flip, which is exactly what the per-class decomposition in
`reports/p2_feature_ablation.md` predicts, since the union wins `bug` and loses `question`
and which effect dominates depends on the repository.

That is enough to justify a default and not enough to report a better model. Both
operating points are frozen for this reason, and the choice between them should be made on
cost and on which class matters, not on these five macro-F1 columns.

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
would have picked the wrong model for a cross-project deployment. A paired bootstrap
confirms the part of this that matters - `fasttext_style_sgd` really is ahead of
`tfidf_word_linear_svc` under transfer, with an interval clear of zero - while leaving the
top two transfer models statistically tied. Both protocol sections below carry the
intervals.

**Transfer separates the families far more than CV does.** The CV column spans 0.0511
between best and worst; the LOO column spans 0.1248. In-domain cross-validation compresses
the differences between these model families, which is worth remembering when RQ1 is
written up: five models that look 5 points apart in-domain are 12 points apart the moment
the repository changes.

No lightweight model beats P1's Logistic Regression on CV. Three beat it on transfer,
though only two by a margin worth stating: `fasttext_style_sgd` by 0.0276 and
`tfidf_char_linear_svc` by 0.0127, with `tfidf_word_linear_svc` ahead by 0.0022, which is
a tie.

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

Within the TF-IDF family the character branch is ahead by +0.0105 over word-only features
and +0.0127 over P1's baseline - a gap of the same size as the official one and, like it,
inside the noise band (95% CI [-0.0056, +0.0267]) - driven almost entirely by bitcoin
(+0.0560 over word-only, +0.0626 over P1). Character n-grams capture sub-word regularities that survive
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

### How much of that ordering is real

The same paired bootstrap as the official-test section, applied to the leave-one-repository-out
predictions.

| Model | LOO | 95% CI |
|---|---:|---|
| `fasttext_style_sgd` | 0.6152 | [0.5900, 0.6382] |
| `tfidf_char_linear_svc` | 0.6003 | [0.5747, 0.6232] |
| `tfidf_word_linear_svc` | 0.5898 | [0.5642, 0.6124] |
| `fasttext_supervised` | 0.5384 | [0.5124, 0.5619] |
| `tfidf_word_complement_nb` | 0.4904 | [0.4652, 0.5134] |

| Paired difference from `fasttext_style_sgd` | delta | 95% CI | P(delta > 0) |
|---|---:|---|---:|
| `tfidf_char_linear_svc` | +0.0149 | [-0.0062, +0.0359] | 0.919 |
| `tfidf_word_linear_svc` | +0.0254 | [+0.0030, +0.0479] | 0.987 |
| `fasttext_supervised` | +0.0768 | [+0.0525, +0.1019] | 1.000 |
| `tfidf_word_complement_nb` | +0.1249 | [+0.0993, +0.1497] | 1.000 |

Calling `fasttext_style_sgd` the transfer winner outright is more than the data supports:
its lead over the character union is +0.0149 with an interval that crosses zero. Its lead
over the *word-only* model does clear zero, and that is the comparison the
protocol-disagreement claim actually needs - the model that is worst on CV is
significantly better than `tfidf_word_linear_svc` under domain shift. So the disagreement
between protocols is real; the specific identity of the transfer winner is not settled
between the two character-feature models.

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
that `fasttext_style_sgd` moves up two places and `fasttext_supervised` drops to last. How
much of that ordering is separable from sampling noise is quantified at the end of this
section, and the answer is not all of it.

fastText's official score is dragged down by one repository: 0.5378 on
`tensorflow/tensorflow`, against 0.6901 for the same configuration under CV. It is the
same repository that exposed the training-order defect above, and it stays the weakest
column for this model even after the fix, so the remaining gap is a property of fastText
on this repository rather than of the adapter.

For external context, the NLBSE'24 competition report puts the SetFit baseline near 0.827
and a published fastText entry at 0.718. A tuned TF-IDF plus LinearSVC lands between them
at 0.7591, for a fit that takes about four seconds on a fold of 240 issues.

### How much of that ordering is real

The cross-repository score is a mean over 1,500 test issues, so it carries sampling error
that a table of point estimates hides. Each model was bootstrapped over its own test rows,
10,000 resamples drawn within each repository so that the five-repository structure of the
metric is preserved. Pairs were compared on the *same* resamples, which is why the
difference intervals are much narrower than the two score intervals they are built from.
Both tables come from `scripts/p2_analysis.py --protocol official`; Monte-Carlo error puts
the interval edges themselves at about the fourth decimal.

| Model | Official | 95% CI |
|---|---:|---|
| `tfidf_char_linear_svc` | 0.7591 | [0.7361, 0.7795] |
| `tfidf_word_linear_svc` | 0.7498 | [0.7267, 0.7702] |
| `fasttext_style_sgd` | 0.7322 | [0.7092, 0.7532] |
| `tfidf_word_complement_nb` | 0.7108 | [0.6864, 0.7325] |
| `fasttext_supervised` | 0.6896 | [0.6655, 0.7111] |

| Paired difference from `tfidf_char_linear_svc` | delta | 95% CI | P(delta > 0) |
|---|---:|---|---:|
| `tfidf_word_linear_svc` | +0.0093 | [-0.0020, +0.0210] | 0.945 |
| `fasttext_style_sgd` | +0.0269 | [+0.0089, +0.0447] | 0.999 |
| `tfidf_word_complement_nb` | +0.0483 | [+0.0303, +0.0664] | 1.000 |
| `fasttext_supervised` | +0.0695 | [+0.0468, +0.0925] | 1.000 |

**The one gap that does not survive is the headline one.** The union's 0.0093 lead over
word-only features has an interval that crosses zero, and 0.0093 sits below the 0.01
threshold this report already applies to its CV numbers. Applying that threshold
consistently, the two feature sets are not separable on the official test either - the
same verdict the four-seed table reached for cross-validation. What the official test does
establish is the distance to the other three families: the margins over
`fasttext_style_sgd`, ComplementNB and `fasttext_supervised` all keep their intervals
clear of zero.

The honest cross-protocol summary of the character branch is therefore weaker than the
point estimates alone suggest. It is never behind, it leads on every protocol, and on no
single protocol is that lead distinguishable from noise. It is a defensible default
because the sign is consistent across three protocols and four seeds, not because any one
of them proves it.

## Per-class behaviour

Every artifact carries per-class precision, recall and a confusion matrix, and every
macro-F1 column above averages all of that away. `scripts/p2_analysis.py` reads them back
without refitting anything; two facts only appear per class.

| Model | bug | feature | question | Macro | Spread |
|---|---:|---:|---:|---:|---:|
| `tfidf_char_linear_svc` | 0.7770 | **0.7845** | 0.7159 | 0.7591 | 0.0686 |
| `tfidf_word_linear_svc` | 0.7623 | 0.7749 | 0.7121 | 0.7498 | 0.0628 |
| `fasttext_style_sgd` | 0.7727 | 0.7378 | 0.6861 | 0.7322 | 0.0866 |
| `tfidf_word_complement_nb` | 0.7089 | 0.7567 | 0.6669 | 0.7108 | 0.0897 |
| `fasttext_supervised` | 0.7404 | 0.6905 | 0.6379 | 0.6896 | 0.1025 |

Official test, per-class F1 averaged over the five repositories, ordered by macro-F1.

**`question` is the hardest class for all five models, with no exception.** It is last on
every row, by between 0.03 and 0.09 F1. The class is not rarer than the others - the
dataset is exactly balanced at 100 issues per class per repository - so this is a property
of the category rather than of its support.

**Weaker models do not degrade evenly, they degrade on `question`.** Read the spread
column against the macro column and the two run opposite: the strongest model has the
narrowest per-class spread (0.0686) and the weakest has the widest (0.1025). Macro-F1 on
this task is to a large extent a statement about how much of `question` a model recovers.
That is worth knowing before ensembling, because soft-voting five models that all fail on
the same class will not repair that class - the errors are correlated by construction.

The pooled confusion matrix for the best model, over all 1,500 official test issues, shows
where the mass goes:

| true \ predicted | bug | feature | question | Recall |
|---|---:|---:|---:|---:|
| **bug** | 379 | 43 | 78 | 0.758 |
| **feature** | 34 | 406 | 60 | 0.812 |
| **question** | 58 | 86 | 356 | 0.712 |
| **Precision** | 0.805 | 0.759 | 0.721 | |

`question` fails symmetrically: 144 of its own issues go to the other two classes, and 138
issues from those classes are wrongly called `question`, so neither a precision fix nor a
recall fix alone would help much. The single largest off-diagonal cell is `question`
predicted as `feature` (86), and `bug` against `question` is the largest confusion once
both directions are added (78 + 58 = 136).

Both pairings are plausible in the text rather than surprising. An issue that describes
something not working, phrased as a request for help, carries surface features of all
three categories at once, and a feature request phrased as "is there a way to..." is
lexically a question. The NLBSE'24 labels come from the repositories' own issue templates
and maintainer triage rather than from an adjudicated annotation protocol, so some of this
confusion is likely present in the labels themselves and sets a ceiling that no amount of
feature engineering on this dataset will lift.

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

**The character branch is an expensive way to buy accuracy that cannot be demonstrated.**
Against word-only features it costs 8.5x the fit time, 11.5x the inference time and 5.1x
the peak memory for +0.0093 on the official test, +0.0105 under transfer and -0.0008 on
CV - three gaps whose bootstrap intervals all cross zero. On macro-F1 alone the honest
recommendation is the cheap configuration. The union earns its cost only where the
per-class and per-repository structure matters: `bug` classification, and repositories
whose issues have no template. Both operating points are frozen so the integrator can
pick, and this is the paragraph to read before picking.

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
- **All five models are weakest on the same class.** `question` is last for every one of
  them, and the models that score worst overall are the ones that lose the most there. An
  ensemble of these five should be expected to inherit that weakness rather than to
  average it away; the per-class section has the numbers. If the ensemble is meant to
  improve macro-F1, the useful diversity has to come from a family that fails somewhere
  else - P3's encoder or P4's Sentence Transformers - not from a second sparse model.

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
- **The bootstrap covers test sampling only.** Resampling the test rows estimates how much
  the score would move on a different sample of issues from these five repositories. It
  holds the trained model fixed, so it says nothing about training-set variation, and it
  cannot correct the selection bias above - a configuration chosen by argmax on CV keeps
  that optimism no matter how its official score is bootstrapped. The intervals also
  assume issues are drawn independently, which the shared-text duplicates noted below
  mildly violate. Read them as the *narrowest* credible intervals, not the widest.
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

# per-class tables, pooled confusion matrix and the paired bootstrap, read back from the
# artifacts above - no model is refitted, so this is seconds rather than minutes
python scripts/p2_analysis.py --protocol official
python scripts/p2_analysis.py --protocol loo
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
