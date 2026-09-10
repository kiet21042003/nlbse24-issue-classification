# P3 — RoBERTa-base full fine-tuning: per-repository vs. pooled

Date: 2026-09-10
Model: `roberta-base`, full fine-tuning (no adapter)
Protocol: stratified 5-fold CV, 5 repositories
Seed: 42
Text: title + body
Config: `configs/roberta_full.toml` (5 epochs, max_seq_length=512)
Hardware: Colab GPU (T4)

## Results

| Repository | Per-repository | Pooled | Δ (pooled − per-repo) |
|---|---:|---:|---:|
| bitcoin/bitcoin | 0.6422 | 0.7509 | +0.1087 |
| facebook/react | 0.7254 | 0.8381 | +0.1127 |
| microsoft/vscode | 0.6779 | 0.7660 | +0.0880 |
| opencv/opencv | 0.6419 | 0.7629 | +0.1210 |
| tensorflow/tensorflow | 0.6352 | 0.8579 | +0.2227 |
| **Cross-repository mean** | **0.6645** | **0.7951** | **+0.1306** |

## Runtime

| Protocol | Fit calls | Total train time | Mean per fit |
|---|---:|---:|---:|
| Per-repository | 25 (5 folds × 5 repos) | 1489.8s (~24.8 min) | ~59.6s |
| Pooled | 5 (5 folds, all repos combined) | 1474.4s (~24.6 min) | ~294.9s |

Total wall-clock time is similar between protocols, but pooled reaches it with
5x fewer fit calls — each pooled fit trains on ~5x more data (all repositories
combined) and produces predictions for all 5 repositories, while per-repository
fits and evaluates one repository at a time.

## Findings

- Pooled training outperforms per-repository training on every single
  repository, by 0.09–0.22 macro-F1. The gain is largest on
  tensorflow/tensorflow (+0.22) and smallest on microsoft/vscode (+0.09).
- This is consistent with RoBERTa-base being data-hungry: full fine-tuning on
  a single repository's fold (a few hundred issues) likely underfits, while
  pooling all 5 repositories gives the model roughly 5x more training
  examples per fit, at comparable total wall-clock cost.
- Unlike the smoke-test result (2 folds, 1 epoch, 1 repository, 150 examples),
  this full run is on real data volume and shows a large, consistent gap in
  the same direction — the smoke result direction held, but the earlier
  smoke-test magnitude was not representative.

## NOTES

- LoRA adapter results (both protocols) are not yet included in this report.
- No hyperparameter tuning was done for either protocol; per-repository and
  pooled used identical config (epochs, learning rate, batch size). Whether a
  separate, protocol-specific config would close or widen this gap is an open
  question — see team discussion on this before further runs.
- Title-only vs. title+body ablation (as done for the P1 baseline) has not
  been run for RoBERTa.

## Reproduction

Each command below runs all 5 repositories in one call. Run on T4 GPU.

```bash
# Full fine-tuning, per-repository
python scripts/run_roberta_experiments.py --config configs/roberta_full.toml --protocol cv

# Full fine-tuning, pooled
python scripts/run_roberta_experiments.py --config configs/roberta_full.toml --protocol pooled_cv

# LoRA adapter, per-repository
python scripts/run_roberta_experiments.py --config configs/roberta_lora.toml --protocol cv

# LoRA adapter, pooled
python scripts/run_roberta_experiments.py --config configs/roberta_lora.toml --protocol pooled_cv
```

To run a single repository instead of all 5, add `--repository <org/repo>`,
e.g. `--repository bitcoin/bitcoin`. Valid repository names:
`bitcoin/bitcoin`, `facebook/react`, `microsoft/vscode`, `opencv/opencv`,
`tensorflow/tensorflow`.

