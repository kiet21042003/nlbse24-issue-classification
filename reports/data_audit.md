# Official dataset audit

Date: 2026-08-24  
Command: `nlbse24 audit-data`

## Confirmed structure

Both official splits contain 1,500 rows. Every combination of five repositories
and three labels contains exactly 100 rows per split. Train has no missing body;
test has two missing bodies, which the repository adapter converts to empty
strings.

## Duplicate and leakage findings

| Check | Train | Test | Across train/test |
|---|---:|---:|---:|
| Exact normalized title + body duplicates | 1 | 1 | 3 shared keys |
| Repeated title within the same repository | 20 | 10 | 16 shared keys |
| Conflicting labels for exact text | 0 | 0 | 1 |
| Conflicting labels for the same title | — | — | 3 |

The three exact cross-split overlaps are all real rows in the official files:

- Bitcoin title `#F`: labeled `feature` in both splits; train also contains a
  same-label duplicate.
- Bitcoin title `.`: `bug` in train but `feature` in test.
- TensorFlow CUDA DLL title: labeled `bug` in both splits.

Therefore, the official test score may receive a small optimistic contribution
from memorization, while the contradictory Bitcoin item can also penalize a
consistent text-only classifier. The team should report the official protocol
unchanged for comparability, then add a secondary sensitivity result with these
three shared exact-text keys removed. Test labels must not be used for model
selection.

## Text length

| Split | Mean words | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Train | 236.6 | 150 | 536.1 | 21,595 |
| Test | 224.2 | 151 | 562.3 | 5,050 |

The extreme train outlier matters for Transformer memory and truncation. P3/P4
must record maximum sequence length and truncation policy; sparse models should
record vocabulary size and peak memory.

## Protocol action already applied

Training-only CV now stratifies normalized-text groups independently per
repository, assigning every identical title + body group to one fold. This
removes within-training exact-duplicate leakage without altering or deleting
official rows.
