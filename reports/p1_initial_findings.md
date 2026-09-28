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
significance claim. The runs were collected on different dates, so their timings
are not treated as a controlled efficiency comparison.

All five artifacts passed prediction-to-metric and probability checks. Training
and evaluation fingerprints and ordered ground-truth labels match the existing
title + body LOO artifacts. Results are versioned under
`results/loo/tfidf_lr_title_only/`, including `summary-seed-42.json`.

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
