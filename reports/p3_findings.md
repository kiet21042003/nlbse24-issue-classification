# P3 — RoBERTa-base: Full Fine-tuning vs. LoRA

Seed 42, text = title + body, `max_seq_length=160`, hardware: Colab GPU (T4 x2)

Full fine-tuning: `configs/roberta_full.toml` (20 epochs, lr=2e-5, wd=0.1)
LoRA: `configs/roberta_lora.toml` (final config version: r=48, lora_alpha=96, target=[query,key,value,attention.output.dense], 20 epochs, lr=3e-4, wd=0.0)

## Per-repository — cv (per-repository protocol)

| Repository | Full FT | LoRA | Difference |
|---|---:|---:|---:|
| bitcoin/bitcoin | 0.7351 | 0.7345 | -0.0006 |
| facebook/react | 0.8076 | 0.8316 | +0.0240 |
| microsoft/vscode | 0.7407 | 0.7562 | +0.0155 |
| opencv/opencv | 0.7632 | 0.7533 | -0.0099 |
| tensorflow/tensorflow | 0.8797 | 0.8247 | -0.0550 |
| **Cross-repository mean** | **0.7853** | **0.7800** | **-0.0052** |

## Per-repository — pooled_cv (pooled protocol)

| Repository | Full FT | LoRA | Difference |
|---|---:|---:|---:|
| bitcoin/bitcoin | 0.7581 | 0.7489 | -0.0092 |
| facebook/react | 0.8348 | 0.8618 | +0.0271 |
| microsoft/vscode | 0.7213 | 0.7554 | +0.0342 |
| opencv/opencv | 0.7584 | 0.7590 | +0.0006 |
| tensorflow/tensorflow | 0.8893 | 0.8721 | -0.0172 |
| **Cross-repository mean** | **0.7924** | **0.7994** | **+0.0071** |

## Insights

**LoRA reaches parity with full fine-tuning at this scale.** Across both
protocols, LoRA is within ±0.007 macro-F1 of full fine-tuning on average —
slightly behind on cv (-0.0052), slightly ahead on pooled (+0.0071). LoRA
updates only the low-rank adapters on 4 attention sub-modules (r=48), a small
fraction of RoBERTa-base's ~125M parameters, yet matches full fine-tuning's
cross-repository performance.

**Per-repository variance is larger than the aggregate gap suggests.**
tensorflow/tensorflow consistently favors full fine-tuning (-0.055 cv, -0.017
pooled), while microsoft/vscode and facebook/react consistently favor LoRA
(+0.015 to +0.034). The cross-repository mean masks this — a model chosen
purely on the aggregate score could underperform on specific repositories.

**Tuning history.** Several LoRA configurations were tried before final config version:
- Matched to full FT's hyperparameters exactly (same lr, epochs) — badly
  underfit (cv macro-F1 ~0.54, training loss near the ln(3)≈1.10 random
  baseline).
- Increasing epochs alone, increasing rank alone, and adding FFN target
  modules alone each gave partial improvements but none matched final config version.
- final config version's combination (r=48, lr=3e-4, wd=0.0, 20 epochs) gave the best and most
  stable result. A more aggressive variant (r=64, +`intermediate.dense`,
  lr=3e-4, wd=0.0, 25 epochs) reached a similar aggregate score but was
  noticeably less stable — opencv/opencv dropped to 0.64 on cv, a much larger
  swing than any single repository showed under final config version. final config version was kept for its
  stability and lower parameter count.
- Full fine-tuning was also re-run at 20 epochs (matching LoRA's epoch
  budget) specifically to keep this final comparison fair on training
  exposure, not just on hyperparameter values.

**Why learning rate and weight decay differ between the two methods.** This
is standard practice, not an inconsistency: LoRA's low-rank matrices are
randomly initialized (not pretrained), so they need a substantially higher
learning rate to converge within a comparable epoch budget than full
fine-tuning of already-pretrained weights, which uses a small LR to avoid
catastrophic forgetting. LoRA's much smaller number of trainable parameters
also makes it less prone to overfitting, so a lower weight decay (0.0 vs 0.1)
is appropriate. What was kept identical across both methods: backbone,
dataset, folds, seed, `max_seq_length`, and epoch budget.

## How to run

```bash
# Full fine-tuning
python scripts/run_roberta_experiments.py --config configs/roberta_full_epoch20.toml --protocol cv --overwrite
python scripts/run_roberta_experiments.py --config configs/roberta_full_epoch20.toml --protocol pooled_cv --overwrite

# LoRA (final config version)
python scripts/run_roberta_experiments.py --config configs/roberta_lora.toml --protocol cv --overwrite
python scripts/run_roberta_experiments.py --config configs/roberta_lora.toml --protocol pooled_cv --overwrite
```

## Artifacts
- All folders in `artifacts/roberta_full/` and `artifacts/roberta_lora/` are stored at [here](https://husteduvn-my.sharepoint.com/:f:/g/personal/anh_ln252275m_sis_hust_edu_vn/IgBPo-B9kMiXSbBPz0IOW0hQAZG_jkSOp-ELoIHUbe1h7vM?e=ZPGg8P)