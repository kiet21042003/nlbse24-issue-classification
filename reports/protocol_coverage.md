# Protocol coverage (seed 42)

Repository-balanced macro-F1 per protocol, computed from the versioned run artifacts (`python scripts/protocol_coverage.py`). `not run` means no artifacts exist; nothing is imputed. Scores from different protocols answer different questions and must not be ranked against each other.

| Model | Owner | Repo-specific CV | Pooled CV | LOO | Official |
|---|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | P1 | 0.7636 | 0.7601 | 0.5876 | 0.7495 |
| Word LinearSVC | P2 | 0.7541 | 0.7570 | 0.5898 | 0.7498 |
| Word + Char LinearSVC | P2 | 0.7533 | 0.7658 | 0.6003 | 0.7591 |
| Hashed SGD | P2 | 0.7030 | 0.7460 | 0.6152 | 0.7322 |
| Supervised fastText | P2 | 0.7089 | 0.7126 | 0.5384 | 0.6896 |
| Complement NB | P2 | 0.7054 | 0.6757 | 0.4904 | 0.7108 |
| RoBERTa full fine-tuning | P3 | 0.7853 | 0.7924 | 0.6867 | 0.8033 |
| RoBERTa LoRA | P3 | 0.7800 | 0.7994 | 0.6896 | 0.7937 |
| SetFit MPNet | P4 | 0.7953 | not run | 0.6854 | 0.8033 |
| SetFit MiniLM | P4 | 0.7881 | 0.7795 | 0.6684 | 0.7972 |
