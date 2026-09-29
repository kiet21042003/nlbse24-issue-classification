# Report source and reproduction

Editable source imported from Dung's `Seminar_2.zip`, revision `09fe724`.
The original source is preserved in that Git revision. This directory is the
maintained source; the root ZIP is a convenience export of these source files.
Review corrections: P1 LOO/CI, protocol definitions, memory measurement labels,
EDA findings, ownership/configuration details, and executed P5 analysis.

From this directory (MiKTeX/TeX Live with the required packages):

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Published PDF: `../../Seminar_2_Report.pdf`. Build intermediates are ignored.
The cover date is September 25, 2026, as requested; it is not the revision date.

From the repository root, reproduce P1 analysis:

```powershell
python scripts/p1_loo_analysis.py --resamples 10000
```

P5 analysis was first executed from reviewed revision `09fe724`, now merged
into main in `1a8bbae`. Reproduce with the integrated scripts:

```powershell
python scripts/compare_models.py --results-dir results --protocol cv --model tfidf_logistic_regression --model setfit_mpnet --resamples 10000 --output reports/p5_cv_comparison.json
python scripts/run_ensemble.py --results-dir results --protocol cv --model tfidf_logistic_regression --model setfit_mpnet --output reports/p5_cv_ensemble.json
```

These commands use fixed predictions, not GPU training or official test data.
P2 CV/LOO values for all five frozen configurations were independently reproduced
on 28 September (150 new run artifacts); see `reports/p2_reproduction_handoff.md`.
They match the published table at its displayed precision. Original P2 official-test,
sweep and extra-seed artifacts remain outstanding: those claims are sourced from
P2 findings, not independently recomputed here. New reproduction timings are not
a controlled hardware benchmark.
