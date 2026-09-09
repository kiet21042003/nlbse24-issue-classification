# P4 SetFit findings

Date: 2026-09-08 (seed 42, GPU NVIDIA RTX PRO 6000 Blackwell 96GB)
Protocol: training-only 5-fold CV per repository (stratified, exact-text-grouped); official test locked until team freezes config. LOO not run (out of scope, see §2).
Environment: `setfit==1.1.1`, `sentence-transformers==3.4.1` (downgraded from 5.2.1, incompatible with setfit 1.1.1), `datasets==2.21.0` (downgraded from 4.0.0, incompatible with setfit 1.1.1), `torch==2.7.0+cu128`, `device="cuda"` in all 7 TOML configs.

## 1. Method

- **Backbones:** `sentence-transformers/all-mpnet-base-v2` (reproduction) vs `sentence-transformers/all-MiniLM-L6-v2` (cheaper variant). Both use SetFit contrastive fine-tuning + logistic-regression head (library defaults: CosineSimilarityLoss).
- **Reproduction recipe (`configs/setfit_mpnet.toml`):** full 300 training rows per repository (no few-shot subsampling), `num_iterations=20`, `num_epochs=1`, `batch_size=(16, 2)`, `max_seq_length=128` tail truncation, seed 42. Matches NLBSE'24 `2-Template-SetFit.ipynb` except two documented deviations: team text contract `"{title}\n\n{body}"` (template uses single space) and 128-token truncation for the 21k-word audit outlier.
- **MiniLM variant (`configs/setfit_minilm.toml`):** identical recipe, backbone swapped.
- **Contrastive-sampling ablations:**
  - Few-shot `k` per class: `configs/ablations/setfit_mpnet_k8.toml` (k=8), `k16` (k=16), `k32` (k=32) via `max_examples_per_class`, stratified seeded subsample.
  - Pair-generation `num_iterations`: `setfit_mpnet_it5.toml` (5, paper HP lower bound) and `it80` (80, upper bound).
- **Runner:** `scripts/run_setfit.py` → `nlbse24.runner.run_classifier_experiment` (same splits/metrics/artifacts as P1). Per-repository models, cross-repository score = arithmetic mean of 5 repo macro-F1 (competition ranking rule). Each config uses a distinct `--run-name` (`setfit_mpnet`, `setfit_minilm`, `setfit_mpnet_k8/k16/k32/it5/it80`) so result directories never collide.
- **Baselines for comparison:** P1 TF-IDF+LR CV macro-F1 **0.7636** (`reports/p1_cv_baseline.md`); official NLBSE'24 SetFit test macro-F1 **0.8270** (train 1500 → test 1500, per-repo models).

## 2. Results

Each row = one `scripts/run_setfit.py --config ... --run-name ... --overwrite` summary JSON (`results/cv/<run-name>/summary-seed-42.json`, 25 evaluations each). Runtime/RAM = mean over the 25 fold artifacts' `resources.fit/inference` fields (`elapsed_seconds`; peak = max of `python_peak_mb`, `rss_after_mb`).

| Config | Cross-repo macro-F1 (mean of 5 repos) | bitcoin | react | vscode | opencv | tensorflow | Fit s (mean) | Infer s (mean) | Peak RAM MB (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `setfit_mpnet` (full, 20it) | 0.7953 | 0.7198 | 0.8554 | 0.7778 | 0.7505 | 0.8731 | 145.4 | 0.23 | 2594 |
| `setfit_minilm` (full, 20it) | 0.7881 | 0.7460 | 0.8396 | 0.7720 | 0.7314 | 0.8516 | 55.1 | 0.07 | 2655 |
| `setfit_mpnet_k8` | 0.6785 | 0.5994 | 0.7164 | 0.7639 | 0.5673 | 0.7457 | 22.9 | 0.19 | 2199 |
| `setfit_mpnet_k16` | 0.7375 | 0.6647 | 0.7730 | 0.7739 | 0.6730 | 0.8027 | 35.0 | 0.20 | 2318 |
| `setfit_mpnet_k32` | 0.7833 | 0.6994 | 0.8413 | 0.7877 | 0.7392 | 0.8492 | 60.9 | 0.20 | 2476 |
| `setfit_mpnet_it5` | 0.7976 | 0.7043 | 0.8431 | 0.7907 | 0.7803 | 0.8696 | 44.2 | 0.23 | 2521 |
| `setfit_mpnet_it80` | 0.7970 | 0.7397 | 0.8520 | 0.7774 | 0.7545 | 0.8616 | 533.6 | 0.21 | 2505 |

LOO (domain transfer, training data only): **not run — out of scope for P4**. Training-only CV answers all P4 research questions (reproduction, backbone cost, sampling ablations); LOO can be added later by P5 if domain-transfer analysis is needed.

Per-class F1 averaged over all folds (precision/recall per repo below):

| Config | bug F1 | feature F1 | question F1 |
|---|---:|---:|---:|
| `setfit_mpnet` | 0.8051 | 0.8071 | 0.7738 |
| `setfit_minilm` | 0.7963 | 0.7920 | 0.7761 |
| `setfit_mpnet_k8` | 0.7068 | 0.7190 | 0.6098 |
| `setfit_mpnet_k16` | 0.7514 | 0.7551 | 0.7058 |
| `setfit_mpnet_k32` | 0.7818 | 0.7986 | 0.7696 |
| `setfit_mpnet_it5` | 0.8005 | 0.8116 | 0.7807 |
| `setfit_mpnet_it80` | 0.8083 | 0.8105 | 0.7723 |

Per-class breakdown (P/R/F1, mean of 5 folds) per repo for the two main configs:

`setfit_mpnet`:

| Repo | bug | feature | question |
|---|---|---|---|
| bitcoin/bitcoin | 0.677/0.700/0.683 | 0.817/0.790/0.802 | 0.691/0.670/0.674 |
| facebook/react | 0.933/0.950/0.941 | 0.807/0.870/0.836 | 0.835/0.750/0.789 |
| microsoft/vscode | 0.716/0.800/0.751 | 0.791/0.770/0.769 | 0.878/0.760/0.813 |
| opencv/opencv | 0.712/0.720/0.713 | 0.832/0.750/0.786 | 0.732/0.780/0.752 |
| tensorflow/tensorflow | 0.928/0.950/0.937 | 0.880/0.810/0.842 | 0.823/0.860/0.840 |

`setfit_minilm`:

| Repo | bug | feature | question |
|---|---|---|---|
| bitcoin/bitcoin | 0.725/0.690/0.706 | 0.829/0.821/0.824 | 0.691/0.730/0.708 |
| facebook/react | 0.922/0.930/0.925 | 0.822/0.830/0.823 | 0.786/0.760/0.771 |
| microsoft/vscode | 0.708/0.790/0.744 | 0.762/0.760/0.758 | 0.882/0.760/0.814 |
| opencv/opencv | 0.686/0.690/0.687 | 0.806/0.680/0.734 | 0.733/0.830/0.774 |
| tensorflow/tensorflow | 0.884/0.960/0.920 | 0.887/0.770/0.822 | 0.805/0.830/0.813 |

## 3. Discussion

- **Reproduction vs official 0.8270:** our MPNet CV cross-repo mean is **0.7953**, +0.0317 over P1 (0.7636). The gap to the official 0.8270 is expected and not a failure: different protocol (training-only 5-fold CV mean vs single train-1500→test-1500 split) and two documented deviations (team `\n\n` text concat vs template single space; 128-token truncation). No test score is claimed; no `--protocol official` run was made before freeze.
- **MPNet vs MiniLM:** MiniLM reaches **0.7881**, only −0.0072 behind MPNet, at **2.6x faster fit** (55.1s vs 145.4s per fold) and 3x faster inference (0.07s vs 0.23s) with ~22M vs ~109M params. MiniLM beats P1 on every repo and is the clear Pareto-efficient choice for P5 (ensemble/demo under budget). Interestingly MiniLM beats MPNet on bitcoin (0.7460 vs 0.7198), the hardest repo for both sparse and dense models.
- **Few-shot k:** steep diminishing-returns curve — k=8 (0.6785) < P1, k=16 (0.7375) still < P1, k=32 (0.7833) ≈ full-data regime but still −0.012 below full (0.7953, ~100/class). `question` is the few-shot bottleneck (k=8 question F1 only 0.6098). Consistent with the SetFit paper: contrastive pair generation compensates for few labels but cannot fully replace data volume here.
- **num_iterations:** flat — it5 (0.7976) ≈ it80 (0.7970) ≈ base 20it (0.7953) within noise, while fit cost scales ~12x (44s → 145s → 534s per fold). Pair-generation count is not the lever on this full-data task; keep the paper default 20 (or even 5) and spend budget on data/backbone instead.
- **Repo pattern:** tensorflow and react are easy for all configs (≥0.84 for full-data); bitcoin and opencv are hard (≤0.75). Same ranking as P1, so difficulty is data-intrinsic (bitcoin template/duplicate lightness), not model-specific.
- **Limitations:** single-space vs `\n\n` concat deviation; 128-token truncation may cut stack traces/templates; per-repo models (no pooled training); seed 42 only, no bootstrap CI yet (see P5 `evaluation/bootstrap.py`).
- **Threats:** duplicate title+body grouping handled by group-aware folds; repository-specific templates may inflate within-repo scores; GPU vs CPU runtime comparability (all timings here are Blackwell CUDA).

## 4. Reproduction

```powershell
# Reproduction (5-fold CV, 25 evaluations, seed 42) — distinct --run-name per variant
python scripts/run_setfit.py --config configs/setfit_mpnet.toml --run-name setfit_mpnet --overwrite
# MiniLM
python scripts/run_setfit.py --config configs/setfit_minilm.toml --run-name setfit_minilm --overwrite
# Ablations (one variable vs reproduction)
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k8.toml --run-name setfit_mpnet_k8 --overwrite
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k16.toml --run-name setfit_mpnet_k16 --overwrite
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k32.toml --run-name setfit_mpnet_k32 --overwrite
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_it5.toml --run-name setfit_mpnet_it5 --overwrite
python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_it80.toml --run-name setfit_mpnet_it80 --overwrite
# LOO skipped (out of scope for P4)
# Official test only after freeze:
# python scripts/run_setfit.py --config configs/setfit_mpnet.toml --protocol official --confirm-official-test --overwrite
```

Environment pins required for `setfit==1.1.1` (see `reports/p4_runbook.md`):

```powershell
pip install "sentence-transformers>=3,<4"  # 3.4.1; 5.x breaks setfit 1.1.1 Trainer callbacks
pip install "datasets==2.21.0"             # 4.x breaks setfit 1.1.1 model-card widget code
```

Local smoke (subset, tiny hyperparams, `all-mpnet-base-v2` but fast):

```powershell
python tmp/local_setfit_smoke.py
```

Artifacts: `results/cv/<run-name>/<repo>/42/fold-*.json` + `results/cv/<run-name>/summary-seed-42.json`, validated against `schemas/result.schema.json`.

## 5. Checklist (Definition of Done, `docs/model_owner_guide.md`)

- [x] Unit tests pass: `python -m pytest tests/test_setfit.py tests/test_setfit_mock_pipeline.py -v` (20 passed)
- [x] `ruff check src tests scripts` clean
- [x] Smoke run artifacts pass schema/version checks (bitcoin-only MPNet CV, macro-F1 0.7198)
- [x] Full 5-fold CV for MPNet and MiniLM at seed 42 completed (25 evals each)
- [x] Ablations k ∈ {8,16,32} and it ∈ {5,80} completed (25 evals each)
- [x] Runtime and peak RAM recorded from artifact `resources`
- [x] This report filled: Method / Results / Discussion
- [x] No official-test run until team freezes configurations
