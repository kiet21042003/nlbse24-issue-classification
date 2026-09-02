# P4 SetFit findings

Date: TODO (fill after experiments, seed 42)
Protocol: training-only 5-fold CV per repository (stratified, exact-text-grouped) and optional LOO; official test locked until team freezes config.

## 1. Method

- **Backbones:** `sentence-transformers/all-mpnet-base-v2` (reproduction) vs `sentence-transformers/all-MiniLM-L6-v2` (cheaper variant). Both use SetFit contrastive fine-tuning + logistic-regression head (library defaults: CosineSimilarityLoss).
- **Reproduction recipe (`configs/setfit_mpnet.toml`):** full 300 training rows per repository (no few-shot subsampling), `num_iterations=20`, `num_epochs=1`, `batch_size=(16, 2)`, `max_seq_length=128` tail truncation, seed 42. Matches NLBSE'24 `2-Template-SetFit.ipynb` except two documented deviations: team text contract `"{title}\n\n{body}"` (template uses single space) and 128-token truncation for the 21k-word audit outlier.
- **MiniLM variant (`configs/setfit_minilm.toml`):** identical recipe, backbone swapped.
- **Contrastive-sampling ablations:**
  - Few-shot `k` per class: `configs/ablations/setfit_mpnet_k8.toml` (k=8), `k16` (k=16), `k32` (k=32) via `max_examples_per_class`, stratified seeded subsample.
  - Pair-generation `num_iterations`: `setfit_mpnet_it5.toml` (5, paper HP lower bound) and `it80` (80, upper bound).
- **Runner:** `scripts/run_setfit.py` → `nlbse24.runner.run_classifier_experiment` (same splits/metrics/artifacts as P1). Per-repository models, cross-repository score = arithmetic mean of 5 repo macro-F1 (competition ranking rule).
- **Baselines for comparison:** P1 TF-IDF+LR CV macro-F1 **0.7636** (`reports/p1_cv_baseline.md`); official NLBSE'24 SetFit test macro-F1 **0.8270** (train 1500 → test 1500, per-repo models).

## 2. Results

> Fill after running. Each row = one `scripts/run_setfit.py --config ... --overwrite` summary JSON. Use `results/cv/<model>/<seed>/fold-*.json` resources fields for runtime/RAM.

| Config | Cross-repo macro-F1 (mean of 5 repos) | bitcoin | react | vscode | opencv | tensorflow | Fit s (mean) | Infer s (mean) | Peak RAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `setfit_mpnet` (full, 20it) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `setfit_minilm` (full, 20it) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `setfit_mpnet_k8` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `setfit_mpnet_k16` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `setfit_mpnet_k32` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `setfit_mpnet_it5` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `setfit_mpnet_it80` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Optional LOO (domain transfer, training data only):

| Held-out repo | LOO macro-F1 (setfit_mpnet) |
|---|---:|
| bitcoin/bitcoin | TODO |
| facebook/react | TODO |
| microsoft/vscode | TODO |
| opencv/opencv | TODO |
| tensorflow/tensorflow | TODO |
| **Cross-repo LOO mean** | TODO |

Per-class breakdown (precision/recall/F1) per repo: TODO — copy from artifact `metrics.per_label`.

## 3. Discussion

- **Reproduction vs official 0.8270:** TODO — compare CV (training-only) vs official test; note protocol mismatch (5-fold CV mean vs single train/test split). Do not claim test score without ` --protocol official --confirm-official-test` after freeze.
- **MPNet vs MiniLM:** TODO — accuracy delta vs inference cost (MiniLM ~3x smaller). Discuss Pareto trade-off for P5.
- **Few-shot k:** TODO — does k=8 already beat P1 0.7636? Diminishing returns k=8→16→32 vs full 100/class. Relate to SetFit paper few-shot regime.
- **num_iterations:** TODO — 5 vs 20 vs 80 effect on embedding loss; cost vs gain.
- **Limitations:** TODO — single-space vs `\n\n` text concat deviation; 128-token truncation may cut stack traces/templates; per-repo models vs pooled; seed 42 only (no bootstrap CI yet — see P5 `evaluation/bootstrap.py`).
- **Threats:** TODO — duplicate title+body grouping, repository-specific templates, GPU vs CPU runtime comparability.

## 4. Reproduction

```powershell
# Reproduction (5-fold CV, 25 evaluations, seed 42)
python scripts/run_setfit.py --config configs/setfit_mpnet.toml --overwrite
# MiniLM
python scripts/run_setfit.py --config configs/setfit_minilm.toml --overwrite
# Ablations (one variable vs reproduction)
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k8.toml --overwrite
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k16.toml --overwrite
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k32.toml --overwrite
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_it5.toml --overwrite
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_it80.toml --overwrite
# Domain transfer (optional)
python scripts/run_setfit.py --config configs/setfit_mpnet.toml --protocol loo --run-name setfit_mpnet_loo --overwrite
# Official test only after freeze:
# python scripts/run_setfit.py --config configs/setfit_mpnet.toml --protocol official --confirm-official-test --overwrite
```

Local smoke (subset, tiny hyperparams, `all-mpnet-base-v2` but fast):

```powershell
python tmp/local_setfit_smoke.py
```

Artifacts: `results/cv/<model>/42/<repo>/fold-*.json` validated against `schemas/result.schema.json`.

## 5. Checklist (Definition of Done, `docs/model_owner_guide.md`)

- [ ] Unit tests pass: `python -m pytest tests/test_setfit.py tests/test_setfit_mock_pipeline.py -v`
- [ ] `ruff check src tests scripts` clean
- [ ] Smoke run artifacts pass schema/version checks
- [ ] Full 5-fold CV for MPNet and MiniLM at seed 42 completed
- [ ] Ablations k ∈ {8,16,32} and it ∈ {5,80} completed
- [ ] Runtime and peak RAM recorded from artifact `resources`
- [ ] This report filled: Method / Results / Discussion
- [ ] No official-test run until team freezes configurations
