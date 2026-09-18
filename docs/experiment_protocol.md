# Experiment protocol

## Dataset identity

- Upstream repository: `nlbse2024/issue-report-classification`
- Pinned commit: `2927bc67eb42db8affd16eaf3e5a6d74f3063961`
- Train: 1,500 issues
- Test: 1,500 issues
- Repositories: React, TensorFlow, VS Code, Bitcoin, OpenCV
- Labels: `bug`, `feature`, `question`
- Each official split contains 100 examples per repository and label.

The downloader verifies both CSV files by SHA-256 before accepting them.

## Protocols

### Training-only 5-fold CV

Use stratified, group-aware five-fold cross-validation independently within each
repository. Rows with identical normalized title + body are first grouped, then
groups are stratified across folds. This prevents exact duplicates from
appearing on both sides of a CV split.
This is the default protocol for tuning. All models compared in an experiment
must reuse the exact same fold assignment and random seed.

### Pooled training-only CV (P3 ablation)

`pooled_cv` uses the same repository-stratified, exact-text-grouped folds
as `cv`. Train one model on all repositories' training partitions per fold,
then evaluate separately on each repository's held-out partition. This gives
five fits and 25 evaluation artifacts. It is not leave-one-repository-out:
each evaluated repository also contributes training rows.

Keep `pooled_cv` separate from `cv` in comparisons and artifact directories.
Fit resource measurements are repeated across the five repository artifacts
of a fold; count each fit once when computing total training cost. Schema 1.0
accepts this additional protocol; older readers must be updated before ingestion.

### Official per-repository evaluation

Train five independent classifiers: for each repository, train on its official
training rows and evaluate on its official test rows. Run this only after the
configuration has been frozen.

### Leave-one-repository-out

For each held-out repository, train one pooled model on the other four
repositories from the official training set and evaluate on the held-out
repository's training rows. This protocol measures domain transfer without
touching the official test set.

## Text construction

The default input is:

```text
{title}\n\n{body}
```

Whitespace is normalized, but code, URLs, stack traces, and issue-template text
are retained. `title`-only is an explicit ablation, not an undocumented change.

## Metrics

For each repository report precision, recall, and F1 for all three classes plus
macro averages. Cross-repository performance is the arithmetic mean of the five
repository macro-F1 values, matching the competition ranking rule.

## Leakage controls

- Do not fit a vectorizer or sampler before splitting.
- Do not use the official test labels for tuning, thresholding, or model choice.
- Fit preprocessing inside each model pipeline using only that fold's training
  rows.
- Group exact normalized-text duplicates into the same CV fold.
- Persist the seed, indices/data hash, configuration, and library versions.
