# P1 initial experimental findings

Date: 2026-08-24
Updated: 2026-09-28 (title-only LOO ablation)
All results use seed 42 and the official training set only.

## RQ: Does the issue body help a lightweight baseline?

| Repository | Title + body | Title only | Difference |
|---|---:|---:|---:|
| bitcoin/bitcoin | 0.6892 | 0.4690 | +0.2202 |
| facebook/react | 0.8179 | 0.7570 | +0.0609 |
| microsoft/vscode | 0.7362 | 0.5728 | +0.1634 |
| opencv/opencv | 0.7650 | 0.6424 | +0.1226 |
| tensorflow/tensorflow | 0.8098 | 0.5355 | +0.2742 |
| **Cross-repository macro-F1** | **0.7636** | **0.5953** | **+0.1683** |

The body is essential for this baseline, especially on TensorFlow and Bitcoin.
React loses much less, suggesting its issue titles are more informative or more
template-consistent.

The efficiency trade-off is substantial. Averaged over 25 evaluations on the
same machine, title-only took 0.046 seconds to fit and 0.006 seconds to infer,
with 0.68 MB peak Python allocation. Title + body took 0.718 seconds to fit and
0.159 seconds to infer, with 8.46 MB peak Python allocation. These measurements
are suitable for within-machine comparison, not universal hardware claims.

## RQ: Does the model transfer to an unseen repository?

The title + body baseline obtained **0.5876** cross-repository macro-F1 under
leave-one-repository-out evaluation, compared with **0.7636** under within-repo
CV. The 0.1761 difference indicates strong repository-specific language and
templates.

| Held-out repository | LOO macro-F1 |
|---|---:|
| bitcoin/bitcoin | 0.5604 |
| facebook/react | 0.6134 |
| microsoft/vscode | 0.5648 |
| opencv/opencv | 0.5895 |
| tensorflow/tensorflow | 0.6097 |

LOO changes both the training domain and training-set size, so the difference is
evidence of transfer difficulty rather than a controlled causal estimate. It is
still operationally important: a classifier trained on other projects should
not be assumed to work on a new repository without adaptation.

### Title-only LOO ablation (28 September 2026)

The existing title-only configuration was evaluated without retuning, using seed
42 and the same five held-out repositories as the title + body baseline.
Each fit uses 1,200 official training issues from four repositories and evaluates
on the remaining 300 training issues. No official test data is used.

| Held-out repository | Title + body LOO | Title-only LOO |
|---|---:|---:|
| bitcoin/bitcoin | 0.5604 | 0.5116 |
| facebook/react | 0.6134 | 0.5649 |
| microsoft/vscode | 0.5648 | 0.4540 |
| opencv/opencv | 0.5895 | 0.5374 |
| tensorflow/tensorflow | 0.6097 | 0.4693 |
| **Cross-repository macro-F1** | **0.5876** | **0.5075** |

Title + body performs better on all five repositories, with an average advantage
of approximately **0.0801** macro-F1. Title-only drops from **0.5953** in CV to
**0.5075** in LOO. These single-seed descriptive results support retaining the
body for this baseline under domain transfer; they are not a statistical
significance claim by themselves; the conditional bootstrap analysis below
quantifies sampling uncertainty. The runs were collected on different dates, so their timings
are not treated as a controlled efficiency comparison.

All five artifacts passed prediction-to-metric and probability checks. Training
and evaluation fingerprints and ordered ground-truth labels match the existing
title + body LOO artifacts. Results are versioned under
`results/loo/tfidf_lr_title_only/`, including `summary-seed-42.json`.

### Paired bootstrap and class-level analysis

We used 10,000 bootstrap resamples (RNG seed 42). For each repository, sample
issue rows with replacement and apply the same indices to both models, recompute
macro-F1 over all three fixed classes, then average the five repository scores
with equal weight. Intervals are percentile 95% CIs; the observed point estimate
is not replaced by the mean bootstrap score.

| Quantity | Observed macro-F1 / difference | 95% CI |
|---|---:|---|
| Title + body | 0.5876 | [0.5621, 0.6108] |
| Title only | 0.5075 | [0.4814, 0.5312] |
| Title + body minus title only | +0.0801 | [+0.0484, +0.1120] |

The paired difference interval excludes zero, supporting an advantage for
title + body under this resampling model. These intervals are conditional on
the fixed trained models and the five observed repositories: they do not measure
training variability or uncertainty for new repositories. Issue rows are assumed
exchangeable within a repository; duplicate or dependent issues may make the
intervals optimistic. This is a post-hoc analysis, not an official-test result.

Per-class F1 below is averaged equally across repositories (not calculated from
the pooled confusion matrix).

| Class | Title + body F1 | Title-only F1 | Difference |
|---|---:|---:|---:|
| bug | 0.5463 | 0.4402 | +0.1061 |
| feature | 0.6693 | 0.5996 | +0.0697 |
| question | 0.5471 | 0.4826 | +0.0645 |

The largest descriptive gain is on `bug`. `feature` remains the strongest class
for both models. This identifies where the observed gains occur, not a causal
explanation of which body content helps.

Pooled confusion counts across all 1,500 held-out predictions are shown for error
inspection only. Rows are true labels; columns are predicted labels in the order
`bug`, `feature`, `question`. Each true class contains 500 issues.

| Model | True class | Predicted bug | Predicted feature | Predicted question |
|---|---|---:|---:|---:|
| Title + body | bug | 234 | 84 | 182 |
| Title + body | feature | 29 | 360 | 111 |
| Title + body | question | 79 | 119 | 302 |
| Title only | bug | 196 | 97 | 207 |
| Title only | feature | 74 | 308 | 118 |
| Title only | question | 113 | 124 | 263 |

Both models frequently predict `question` for true `bug` issues; the count drops
from 207 to 182 when using body text. Confusing `feature` with `bug` drops from
74 to 29. These are aggregate counts, not a claim that the same individual errors
were all corrected.

Machine-readable results, including metrics per repository, are in
[`p1_loo_analysis.json`](p1_loo_analysis.json). The analysis script validates
alignment, text configurations and stored metrics before reusing the existing
paired-bootstrap implementation from `scripts/p2_analysis.py`.

## Reproduction

```powershell
nlbse24 run-baseline --protocol cv
nlbse24 run-baseline --protocol loo
nlbse24 run-baseline --protocol cv `
  --config configs/ablations/logistic_title_only.toml `
  --run-name tfidf_lr_title_only
nlbse24 run-baseline --protocol loo `
  --config configs/ablations/logistic_title_only.toml `
  --run-name tfidf_lr_title_only
```

The official test set remains unused by these experiments.

Reproduce the analysis without training (output kept outside `results/` so it
cannot be mistaken for a run artifact):

```powershell
python scripts/p1_loo_analysis.py --resamples 10000
```
