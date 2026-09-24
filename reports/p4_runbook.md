# P4 Runbook — SetFit experiments step-by-step (Nam)

Owner: Nguyễn Hoàng Nam (P4). Machine: NVIDIA RTX PRO 6000 Blackwell 96GB VRAM, CUDA.
Goal: reproduce SetFit MPNet baseline, evaluate MiniLM, run contrastive-sampling ablations — all training-only 5-fold CV, seed 42 — and fill `reports/p4_setfit_findings.md`.
Result of this run (2026-09-08): 7 configs × 25 evals = 175 fits, all `EXIT=0`.

## Environment (actual experiment machine, 2026-09-08/09)

GPU (`nvidia-smi`):

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM total | 97887 MiB (~102 GB per `torch.cuda`) |
| Compute capability | 12.0 (sm_120), 188 SMs |
| Driver | 580.95.05, CUDA driver API 13.0 |
| Power cap | 600W (idle ~36W during polling) |
| MIG / ECC | Disabled / N/A |

Host: Linux `6.8.0-124-generic` x86_64, Python `3.12.10`, 128 CPUs, ~1007 GB RAM.
Observed GPU util during SetFit contrastive phase: ~5 it/s per fold at `it80` (2400 steps/fold), ~5.2 GB VRAM used (single run, batch 16, `max_seq_length=128`) — VRAM was never a bottleneck; do NOT run parallel SetFit jobs anyway (process-level contention, not memory).

Software (exact versions at experiment time):

| Package | Version | Note |
|---|---|---|
| torch | 2.7.0+cu128 (built CUDA 12.8) | `torch.cuda.is_available() == True` |
| setfit | 1.1.1 | pinned by P4 |
| sentence-transformers | 3.4.1 | downgraded from preinstalled 5.2.1 (see Step 2) |
| transformers | 4.57.3 | preinstalled, kept |
| datasets | 2.21.0 | downgraded from preinstalled 4.0.0 (see Step 2) |
| accelerate | 1.14.0 | preinstalled, kept |
| huggingface-hub | 0.35.0 | preinstalled, kept |
| evaluate | 0.4.6 | preinstalled, kept |
| tokenizers / safetensors | 0.22.0 / 0.7.0 | preinstalled, kept |
| scikit-learn / scipy | 1.7.2 / 1.15.2 | repo-pinned |
| numpy / pandas | 1.26.4 / 2.2.0 | repo-pinned |
| joblib / psutil | 1.5.2 / 7.1.2 | repo-pinned |
| pytest / ruff | 9.0.1 / 0.12.12 | dev deps |

Backbone weights (downloaded once from Hugging Face Hub on first smoke run, then cached): `sentence-transformers/all-mpnet-base-v2` (~420MB), `sentence-transformers/all-MiniLM-L6-v2` (~90MB).
Data: `data/raw/issues_train.csv` + `issues_test.csv` via `nlbse24 download-data` (SHA-256 verified), train 1500 / test 1500.
All runs used `device="cuda"` (Step 4); timings in §Results are Blackwell-CUDA numbers and are NOT comparable to CPU or other-GPU runs.

## Step 0 — Install

```bash
pip install -e .
pip install -q "setfit==1.1.1"
pip install -qr requirements-dev.txt
```

## Step 1 — Verify GPU and base env

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
python -c "import torch, setfit; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: `RTX PRO 6000 Blackwell`, `torch 2.7.0+cu128 True`, setfit `1.1.1`.

## Step 2 — Fix dependency incompatibilities (REQUIRED, otherwise Step 4 crashes)

`setfit==1.1.1` was tested against the 2024 stack (`compat-tests` pins) and breaks with the 2025 preinstalled versions:

| Failure | Cause | Fix |
|---|---|---|
| `SentenceTransformerModelCardCallback.on_init_end() missing ... 'trainer'` | `sentence-transformers==5.2.1` new callback API | `pip install -q "sentence-transformers>=3,<4"` → 3.4.1 |
| `AttributeError: 'Column' object has no attribute 'sort'` (setfit model-card widget code) | `datasets==4.0.0` returns `Column` instead of `list` | `pip install -q "datasets==2.21.0"` |

```bash
pip install -q "sentence-transformers>=3,<4"
pip install -q "datasets==2.21.0"
python -c "import sentence_transformers, datasets; print(sentence_transformers.__version__, datasets.__version__)"
```

Expected: `3.4.1 2.21.0`. Cosmetic only (model-card widget examples) — zero effect on metrics.

## Step 3 — Data + contract checks

```bash
nlbse24 download-data
nlbse24 validate-data        # expect train 1500 / test 1500, 5 repos x 3 labels x 100
python -m pytest tests/test_setfit.py tests/test_setfit_mock_pipeline.py -v   # expect 20 passed
ruff check src tests scripts # expect "All checks passed!"
```

## Step 4 — Enable CUDA in configs

Uncomment `device = "cuda"` under `[model]` in all 7 TOMLs (or at least the two mains):

```bash
sed -i 's/^# device = "cuda"$/device = "cuda"/' \
  configs/setfit_mpnet.toml configs/setfit_minilm.toml \
  configs/ablations/setfit_mpnet_k8.toml configs/ablations/setfit_mpnet_k16.toml \
  configs/ablations/setfit_mpnet_k32.toml configs/ablations/setfit_mpnet_it5.toml \
  configs/ablations/setfit_mpnet_it80.toml
grep -rn "device" configs/setfit_*.toml configs/ablations/setfit_*.toml
```

## Step 5 — Smoke test, 1 repo (~10 min, downloads ~420MB MPNet weights once)

```bash
python scripts/run_setfit.py --config configs/setfit_mpnet.toml --repository bitcoin/bitcoin --overwrite
```

Expected: `evaluations: 5`, artifacts in `results/cv/setfit_mpnet/bitcoin__bitcoin/42/fold-{1..5}.json`.
This run: bitcoin macro-F1 `0.7198`.

## Step 6 — Full 7 CV configs (each = 5 repos x 5 folds = 25 fits)

> IMPORTANT: always pass a distinct `--run-name`. All ablation TOMLs share the MPNet
> backbone, so `probe.name` is `setfit_mpnet` for all of them — omitting `--run-name`
> overwrites the reproduction results (happened once with k8; recovered, see Step 7).
> Correct CLI flag is `--repository` (repeatable), NOT `--repositories`.

```bash
mkdir -p logs
nohup python scripts/run_setfit.py --config configs/setfit_mpnet.toml --run-name setfit_mpnet --overwrite > logs/mpnet.log 2>&1 &
nohup python scripts/run_setfit.py --config configs/setfit_minilm.toml --run-name setfit_minilm --overwrite > logs/minilm.log 2>&1 &
nohup python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k8.toml --run-name setfit_mpnet_k8 --overwrite > logs/k8.log 2>&1 &
nohup python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k16.toml --run-name setfit_mpnet_k16 --overwrite > logs/k16.log 2>&1 &
nohup python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_k32.toml --run-name setfit_mpnet_k32 --overwrite > logs/k32.log 2>&1 &
nohup python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_it5.toml --run-name setfit_mpnet_it5 --overwrite > logs/it5.log 2>&1 &
nohup python scripts/run_setfit.py --config configs/ablations/setfit_mpnet_it80.toml --run-name setfit_mpnet_it80 --overwrite > logs/it80.log 2>&1 &
```

Run strictly sequentially (one at a time) to avoid GPU contention. Monitor with:

```bash
find results/cv/<run-name> -name "fold-*.json" | wc -l   # expect 25 at finish
tail -c 300 logs/<run-name>.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Wall time on Blackwell (this run): mpnet ~45-60min, minilm ~20min, k8 ~15min, k16 ~20min, k32 ~30min, it5 ~20min, it80 ~2.5h.

Results (cross-repo macro-F1, `results/cv/<run-name>/summary-seed-42.json`):

| run-name | cross | bitcoin | react | vscode | opencv | tensorflow |
|---|---:|---:|---:|---:|---:|---:|
| setfit_mpnet | 0.7953 | 0.7198 | 0.8554 | 0.7778 | 0.7505 | 0.8731 |
| setfit_minilm | 0.7881 | 0.7460 | 0.8396 | 0.7720 | 0.7314 | 0.8516 |
| setfit_mpnet_k8 | 0.6785 | 0.5994 | 0.7164 | 0.7639 | 0.5673 | 0.7457 |
| setfit_mpnet_k16 | 0.7375 | 0.6647 | 0.7730 | 0.7739 | 0.6730 | 0.8027 |
| setfit_mpnet_k32 | 0.7833 | 0.6994 | 0.8413 | 0.7877 | 0.7392 | 0.8492 |
| setfit_mpnet_it5 | 0.7976 | 0.7043 | 0.8431 | 0.7907 | 0.7803 | 0.8696 |
| setfit_mpnet_it80 | 0.7970 | 0.7397 | 0.8520 | 0.7774 | 0.7545 | 0.8616 |

## Step 7 — Recovery recipe (only if a run is killed before writing its summary)

Symptoms: 25 (or 24) `fold-*.json` exist but `summary-seed-42.json` is stale/missing.
In this run: k8 without `--run-name` overwrote `setfit_mpnet/`; then the mpnet re-run was killed at 24/25 folds (stale file: `tensorflow__tensorflow/42/fold-5.json` still had `max_examples_per_class=8`).

```bash
# 1. Find stale folds (expect all null for full-data configs)
python -c "
import json, glob
for f in sorted(glob.glob('results/cv/<run-name>/*/*/fold-*.json')):
    if json.load(open(f))['model']['config']['sampling']['max_examples_per_class'] is not None:
        print('STALE:', f)"
# 2. Re-run ONLY the affected repo (5 folds, ~10 min)
python scripts/run_setfit.py --config <same-toml> --repository <owner/repo> --run-name <run-name> --overwrite
#    (this also rewrites <run-name>/summary-seed-42.json with repo-only stats — replaced in step 3)
# 3. Rebuild the 25-eval summary deterministically with the shared library functions
python -c "
import json
from nlbse24.evaluation import aggregate_macro_f1, ResultWriter
repos = ['bitcoin/bitcoin','facebook/react','microsoft/vscode','opencv/opencv','tensorflow/tensorflow']
rows, paths = [], []
for i in range(1, 6):
    for repo in sorted(repos):
        p = f'results/cv/<run-name>/{repo.replace(\"/\",\"__\")}/42/fold-{i}.json'
        a = json.load(open(p))
        rows.append({'repository': repo, 'fold': f'fold-{i}', 'macro_f1': a['metrics']['macro_average']['f1-score']})
        paths.append(p)
summary = {**aggregate_macro_f1(rows), 'split_strategy': 'stratified_text_group_folds_per_repository',
           'text_fields': ['title','body'], 'artifacts': paths}
print(ResultWriter('results', overwrite=True).write_summary('cv', '<run-name>', 42, summary))"
```

Verified: rebuilt `setfit_mpnet` summary byte-identical in scores to the original successful run (cross `0.7953`).

## Step 8 — Fill the report

Copy per-config cross-repo + per-repo macro-F1 from each `summary-seed-42.json`, runtime/RAM means from the fold artifacts (`resources.fit/inference.elapsed_seconds`, peak = max of `python_peak_mb`/`rss_after_mb`), per-class tables from `metrics.per_class` → `reports/p4_setfit_findings.md` (§2 Results, §3 Discussion). LOO skipped (out of scope, agreed). Never run `--protocol official` before team freeze.

## Step 9 — Final verify

```bash
ruff check src tests scripts
python -m pytest tests/test_setfit.py tests/test_setfit_mock_pipeline.py -q   # expect 20 passed
python -m pytest tests -q   # full suite
```

## Forbidden

- `--protocol official` / `--confirm-official-test` before freeze (test leakage).
- Parallel SetFit runs on the same GPU.
- Omitting `--run-name` for ablations (directory collision).
- Editing `runner.py`, `splits.py`, result schema, or other owners' modules without team review.
