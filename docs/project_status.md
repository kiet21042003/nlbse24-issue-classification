# Project status — 28 September 2026

Model-integration status was verified on 18 September; artifact publication was
updated on 22 September. P1 title-only LOO was added on 28 September (five runs,
macro-F1 0.5075; title + body LOO 0.5876). See
[P1 findings](../reports/p1_initial_findings.md) for verification and reproduction.
The paired 10,000-resample bootstrap gives a title + body advantage of 0.0801
(95% CI [0.0484, 0.1120]); class-level F1 and confusion counts are also available.
This analysis is conditional on fixed predictions, not training or new-repository uncertainty.
This is a progress snapshot, not a new experimental protocol. The original
21-day timeline remains a planning baseline, not a record of actual completion dates.

## Integrated work

| Owner | Completed and reviewed | Remaining handoff / evaluation |
|---|---|---|
| P1 — Kiệt | Shared data, persistence, splits, schema and Logistic Regression; CV, title ablation and LOO findings. | Coordinate final configuration freeze, artifact handoff and end-to-end verification; final LR official evaluation. |
| P2 — Hiếu | Lightweight models and findings merged; five frozen configs independently reproduced on CV/LOO, with 150 verified run artifacts. | Supply original official-test, sweep, extra-seed and feature-ablation artifacts; preserve existing official-test history. |
| P3 — Ánh | RoBERTa full/LoRA merged; 100 verified per-repository and pooled CV artifacts imported; pooled schema and reproduction commands fixed. | Final official evaluation after configuration freeze; contribute Method/Results/Discussion to the combined report. |
| P4 — Nam | SetFit merged; 175 verified CV artifacts covering seven configurations, with findings and a runbook. | Freeze the selected configuration and complete final official evaluation; contribute report sections. LOO has not been run. |
| P5 — Dũng | Evaluation/integration branch merged; validated reader, aggregation, paired bootstrap, resource/ensemble tools and demo notebook. LR–MPNet CV comparison and illustrative hard vote executed. | Final selected-model comparisons, controlled resource benchmarks and final demo/submission verification. |

P2, P4 and P3 were merged in `4599fa8`, `9220948` and `77bf724`, respectively.
P5 revision `09fe724` was merged in `1a8bbae` on 28 September. Local checks
reported **178 passed, 2 skipped**, **86.66%** coverage and clean Ruff checks.
The two skipped tests require the real encoder
backend; the review did not rerun GPU training. The
[main CI run](https://github.com/kiet21042003/nlbse24-issue-classification/actions/runs/36453032812)
also succeeded after the pytest import-path fix in `b98f4c3`.

## Artifact handoff

- P3: `results/cv/roberta_base_{full,adapter}/` and
  `results/pooled_cv/roberta_base_{full,adapter}/`; 25 evaluations per configuration/protocol.
- P4: seven model/configuration directories under `results/cv/`, 25 evaluations each.
- P2: five frozen configurations under `results/cv/` and `results/loo/`,
  25 CV and five LOO evaluations each. These are new training-only reproductions,
  not recovered original logs. See [P2 reproduction handoff](../reports/p2_reproduction_handoff.md).
- The shared result reader accepts **485 artifacts**: P1 60, P2 150, P3 100 and P4 175.
- As of 28 September, the reviewed **485 run artifacts and 25 JSON summaries**
  are versioned, together with `results/p3_import_manifest.txt`. A fresh clone
  contains these results. Source ZIPs and model weights remain local; preserve
  original archives for audit. P2 original official-test/tuning artifacts are
  still outstanding. New P2 timings were collected under uncontrolled concurrent
  load and must not be used as a controlled cost/Pareto benchmark.
- Import P3 with `python scripts/import_p3_results.py results/p3_results.zip --output-dir results`
  only when rebuilding into an output tree without existing P3 runs (a fresh
  clone already contains them). The importer normalizes names and
  paths, preserves predictions/metrics and writes an import manifest.

P3 metrics were recalculated from predictions, scores checked against predicted
labels, and test/train fingerprints checked against the shared folds, including
pooled training. All four groups contain 25 unique repository/fold pairs.
These packaging/schema fixes do **not** require retraining.

## Next steps, in order

1. **P1 + model owners:** record final configurations selected using training-only
   CV; collect remaining original P2 artifacts. P5 ingestion of the 485 available
   run artifacts has been verified.
2. **P5 + P1:** build separate per-repository CV, pooled CV and LOO tables.
   P3 pooled LoRA macro-F1 **0.7994** is not an official score or evidence of
   superiority over per-repository models. Keep published official baselines separate.
3. **P5 + P2–P4:** complete the planned ensemble evaluation without tuning on
   its evaluation labels or the official test. Use independent validation/nested
   folds if learning ensemble weights. Report resource comparisons on matched
   hardware, and do not count pooled fit time five times across repository artifacts.
4. **P1–P4 + P5:** run remaining official evaluations only after freeze, then
   aggregate CIs and final tables. P2 already reports official results: do not
   present the entire project as having an untouched official test, or retune on it.
5. **All members:** integrate module write-ups, document uncompleted scope,
   verify a fresh clone, finish the demo and rehearse contribution explanations.

Merged code and audited CV artifacts complete the model-integration milestone;
they do not complete the final evaluation or submission milestone.
