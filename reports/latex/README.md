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
The existing cover date is retained as supplied, not asserted as the revision date.

From the repository root, reproduce P1 analysis:

```powershell
python scripts/p1_loo_analysis.py --resamples 10000
```

P5 analysis uses its reviewed source without requiring a merge into main:

```powershell
git worktree add --detach tmp/p5-report-tools 09fe724
python tmp/p5-report-tools/scripts/compare_models.py --results-dir results --protocol cv --model tfidf_logistic_regression --model setfit_mpnet --resamples 10000 --output reports/p5_cv_comparison.json
python tmp/p5-report-tools/scripts/run_ensemble.py --results-dir results --protocol cv --model tfidf_logistic_regression --model setfit_mpnet --output reports/p5_cv_ensemble.json
```

Use a new temporary worktree path if the suggested one already exists. These
commands use fixed predictions, not GPU training or official test data. P2 table
values are sourced from `reports/p2_lightweight_models.md`; its full raw artifact
handoff is still outstanding. Do not represent those values as independently
recomputed here.
