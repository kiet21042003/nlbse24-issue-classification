# P2 tuning and feature ablation

Date: 2026-08-31

All numbers below use seed 42 and the official **training set only**, under the shared
stratified, exact-text-grouped five-fold cross-validation protocol (25 evaluations per
configuration). No official test data was read while selecting any configuration. Text is
the shared `title` + `body` composition unless a row says otherwise.

Environment: Python 3.11.16, scikit-learn 1.7.2, numpy 1.26.4, pandas 2.2.0,
`fasttext-wheel` 0.9.2, Windows 11. This matches the pinned stack in `requirements.txt`
and the CI job, so these numbers are directly comparable with P1's.

Reference points from P1's Logistic Regression baseline: cross-repository macro-F1
**0.7636** (CV), **0.5953** (title only), **0.5876** (leave-one-repository-out).

## RQ: do character n-grams help issue-report classification?

Feature sets are declared in configuration, so `word`, `char` and `word_char` share one
code path. The character branch is `char_wb` 2-5 grams; the word branch is 1-2 grams.

| Feature set | C | bitcoin | react | vscode | opencv | tensorflow | **Cross-repo** |
|---|---:|---:|---:|---:|---:|---:|---:|
| word | 1.0 | 0.6755 | 0.8274 | 0.7257 | 0.7508 | 0.7909 | **0.7541** |
| word + char | 1.0 | 0.7082 | 0.7865 | 0.7386 | 0.7525 | 0.7807 | **0.7533** |
| word | 4.0 | 0.6669 | 0.8100 | 0.7317 | 0.7416 | 0.7912 | **0.7483** |
| word + char | 4.0 | 0.6958 | 0.7865 | 0.7316 | 0.7451 | 0.7812 | **0.7480** |
| word | 0.25 | 0.6783 | 0.8076 | 0.7190 | 0.7155 | 0.8040 | **0.7449** |
| char | 1.0 | 0.6977 | 0.7797 | 0.7530 | 0.7445 | 0.7476 | **0.7445** |
| word + char | 0.25 | 0.6788 | 0.7930 | 0.7291 | 0.7369 | 0.7816 | **0.7439** |
| char | 4.0 | 0.6985 | 0.7824 | 0.7285 | 0.7485 | 0.7444 | **0.7405** |
| char | 0.25 | 0.6694 | 0.7575 | 0.7293 | 0.6958 | 0.7311 | **0.7166** |

**Not on this single seed - but the single seed is misleading.** On seed 42 the union
(0.7533) sits 0.0008 below word-only (0.7541), and character features alone (0.7445) are
clearly worse than word features alone. Repeating both frozen configurations over four
seeds reverses the ordering: the union averages **0.7506** against **0.7475** for
word-only, with roughly half the spread (0.0054 against 0.0107). The seed-42 row above is
simply the top of the word-only range. The honest reading is that the two feature sets are
within noise of each other on within-repository cross-validation, and that a
single-seed argmax is not enough to separate them. Full seed table in
`reports/p2_lightweight_models.md`.

**Per repository the union is the more useful model.** On seed 42 it gains +0.0327 on
bitcoin and +0.0129 on vscode against word-only, and loses 0.0409 on react. Bitcoin is the
repository P1's Logistic Regression handled worst (0.6892) and the only one of the five
whose issues carry no GitHub issue template - its text is free-form, so word features are
sparser and subword evidence is worth more. Character features are therefore not a uniform
improvement but a targeted one, and the five-repository mean hides that.

`C = 1.0` wins in every feature set, so regularisation strength and feature choice do not
interact here.

## RQ: what does the character branch cost?

Mean per evaluation over the 25 CV evaluations of each configuration.

| Feature set | C | fit s | inference s | peak MB | vocabulary | Cross-repo |
|---|---:|---:|---:|---:|---:|---:|
| word | 1.0 | 0.441 | 0.116 | 8.35 | 6,107 | 0.7541 |
| char | 1.0 | 3.037 | 1.335 | 40.94 | 26,459 | 0.7445 |
| word + char | 1.0 | 4.097 | 1.452 | 43.02 | 32,566 | 0.7533 |

Relative to word-only features, the union costs **9.3x the fit time, 12.5x the inference
time, 5.2x the peak memory and 5.3x the vocabulary** for an accuracy difference that is
within seed noise on this protocol. That is why both operating points are frozen:
`configs/tfidf_word_linear_svc.toml` when cost dominates, and
`configs/tfidf_char_linear_svc.toml` when accuracy and transfer do - the union wins both
the leave-one-repository-out and the official-test comparisons, where the extra cost does
buy something.

The picture changes under domain transfer, where the character branch does earn its cost -
see `reports/p2_lightweight_models.md`.

## RQ: how much does the issue body contribute?

| Text fields | bitcoin | react | vscode | opencv | tensorflow | **Cross-repo** |
|---|---:|---:|---:|---:|---:|---:|
| title + body | 0.7082 | 0.7865 | 0.7386 | 0.7525 | 0.7807 | **0.7533** |
| title only | 0.4741 | 0.7463 | 0.5798 | 0.6556 | 0.5488 | **0.6009** |

Dropping the body costs 0.1524 macro-F1, closely matching the 0.1683 P1 measured for
Logistic Regression. The body carries most of the signal for every model family tested
here, and the character branch does not compensate for its absence.

## Threats to validity

- **Selection bias.** The configuration was chosen by taking the argmax over this grid on
  the same folds the table reports, so the winning row is an optimistic estimate of its
  own score. Seed-robustness results in `reports/p2_lightweight_models.md` quantify the
  spread.
- **Timings are inflated and not wall-clock.** The shared profiler wraps every fit in
  `tracemalloc`, whose overhead grows with the number of Python allocations. Character
  n-gram analysis allocates one string per n-gram, so character configurations are
  penalised more than word configurations. The numbers are internally comparable across
  P1-P5 because everyone uses the same profiler; they are not a hardware benchmark.
- **One machine, one seed per row.** Differences below roughly 0.01 macro-F1 should not be
  read as real.

## Reproduction

```powershell
python scripts/run_p2_experiments.py --grid configs/sweeps/p2_linear_svc_grid.toml --protocol cv
python scripts/run_p2_experiments.py --grid configs/sweeps/p2_complement_nb_grid.toml --protocol cv
python scripts/run_p2_experiments.py --config configs/ablations/p2_title_only.toml --protocol cv
```
