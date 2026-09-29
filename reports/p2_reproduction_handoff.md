# P2 artifact reproduction handoff

The original P2 branch contains model code and findings but only `.gitkeep`
under `results/`. This handoff regenerates the five frozen configurations on
training-only CV and LOO, using seed 42, without tuning or official evaluation.
New artifacts are reproductions, not recovered original experiment logs.

Code baseline: `1a8bbae` (P5 integration). Configuration files:

- `configs/tfidf_word_linear_svc.toml`
- `configs/tfidf_char_linear_svc.toml` (word + character union)
- `configs/tfidf_word_complement_nb.toml`
- `configs/fasttext_style_sgd.toml`
- `configs/fasttext_supervised.toml`

For each configuration, run:

```powershell
python scripts/run_p2_experiments.py --config configs/<name>.toml --protocol cv
python scripts/run_p2_experiments.py --config configs/<name>.toml --protocol loo
```

Use a separate output directory if results already exist; do not overwrite
published runs. Supervised fastText requires the optional `fasttext-wheel==0.9.2`
backend. It was installed into an ignored temporary directory for this run,
without changing the shared environment or requirements. The implementation uses
one fastText thread and deterministic seeded input ordering.

Hashed SGD emitted a convergence warning at the frozen `max_iter=30` budget.
The configuration was not changed to suppress this warning; these runs reproduce
the selected operating point, not a claim that every optimizer converged.

## Verified results

| Frozen configuration | CV macro-F1 | LOO macro-F1 |
|---|---:|---:|
| Word LinearSVC | 0.754074 | 0.589808 |
| Word + character LinearSVC | 0.753318 | 0.600307 |
| Word ComplementNB | 0.705404 | 0.490366 |
| Hashed SGD | 0.703045 | 0.615224 |
| Supervised fastText | 0.708907 | 0.538442 |

All ten aggregates match the existing P2 report at its displayed four-decimal
precision. Each configuration has 25 CV and five LOO evaluations: **150 unique
run IDs**, plus ten summaries. For every run, metrics were recalculated from
`y_true/y_pred`, score argmax checked against predicted labels, train/test
fingerprints and label order checked against P1, and saved parameters checked
against the frozen TOML. Summary scores were recalculated repository-first.
The shared P5 reader accepts **485 total run artifacts** after this addition.

## Resource caveat

Runs were collected locally while other review/build work was occurring; fastText
also overlapped with part of the sparse-model execution. Recorded timings and
memory are diagnostic measurements, not a controlled benchmark. Their hardware
group labels identify this reproduction session; they do not certify comparable
load conditions. Do not use these runs to assert a new cost/Pareto ranking.

## Scope still requiring original handoff

Original official-test predictions, hyperparameter sweeps, extra seeds and
feature-ablation artifacts are not reconstructed by these commands. Previously
reported official scores retain their original provenance and must not be
presented as independently verified through this training-only reproduction.
