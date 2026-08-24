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

Use stratified five-fold cross-validation independently within each repository.
This is the default protocol for tuning. All models compared in an experiment
must reuse the exact same fold assignment and random seed.

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
- Persist the seed, indices/data hash, configuration, and library versions.
