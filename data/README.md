# Data directory

Run `nlbse24 download-data` from the repository root. The command downloads the
two official CSVs into `data/raw/`, verifies pinned SHA-256 checksums, and writes
a local provenance manifest.

Expected local files:

```text
data/raw/issues_train.csv
data/raw/issues_test.csv
data/raw/manifest.json
```

The CSVs and manifest are intentionally ignored by Git. They can always be
recreated from the pinned upstream commit. To demonstrate the second filesystem
persistence implementation, run `nlbse24 export-json`.
