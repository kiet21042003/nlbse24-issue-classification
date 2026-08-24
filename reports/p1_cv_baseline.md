# P1 preliminary CV baseline

Date: 2026-08-24  
Protocol: training-only stratified, exact-text-grouped 5-fold CV per repository
Model: word TF-IDF (1-2 grams) + Logistic Regression  
Seed: 42  
Text: title + body

| Repository | Mean macro-F1 |
|---|---:|
| bitcoin/bitcoin | 0.6892 |
| facebook/react | 0.8179 |
| microsoft/vscode | 0.7362 |
| opencv/opencv | 0.7650 |
| tensorflow/tensorflow | 0.8098 |
| **Cross-repository mean** | **0.7636** |

This run verifies that downloading, persistence, preprocessing, fold generation,
model fitting, metrics, resource profiling, and JSON artifact writing work
end-to-end. It is not comparable directly with the published NLBSE'24 SetFit
test score because this table is training-only cross-validation.

The current weakest repository is Bitcoin. P1's next model-analysis work should
inspect its confusion matrices and high-confidence errors before changing
hyperparameters. The official test remains locked until configurations are
frozen by the team.
