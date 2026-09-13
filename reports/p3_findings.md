# P3 — RoBERTa-base: full fine-tuning vs. LoRA, per-repository vs. pooled

Date: 2026-09-11
Seed: 42
Text: title + body
Hardware: Kagle GPU (T4x2)

## Results so far

# P3 — RoBERTa results

Seed 42, text = title + body, hardware: Colab T4.

| # | Config | Protocol | Cross-repo macro-F1 | Loss |
|---|---|---|---:|---|
| 1 | Full FT | cv (per-repo) | 0.6917 | ~0.8 |
| 2 | Full FT | pooled_cv | **0.7986** | <0.6 |
| 3 | LoRA (old config) | cv (per-repo) | ~0.59 (partial) | ~1.01 |
| 4 | LoRA (old config) | pooled_cv | 0.7645 | <0.7 |

Artifacts can be found at link [here](https://husteduvn-my.sharepoint.com/:f:/g/personal/anh_ln252275m_sis_hust_edu_vn/IgBPo-B9kMiXSbBPz0IOW0hQAZG_jkSOp-ELoIHUbe1h7vM?e=cOydyp).

**TODO:** run with new `configs/roberta_lora.toml` — lattest committed file doesn't
pass `ruff`/tests (`target_modules` and other keys don't match what
`tests/test_encoder.py::test_committed_lora_config_loads` expects). Re-run
rows 3-4 with the corrected, full-FT-matched config once fixed.
