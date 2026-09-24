"""Import a flattened P3 ZIP without changing experimental measurements."""

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nlbse24.evaluation import ResultWriter, aggregate_macro_f1  # noqa: E402
from nlbse24.evaluation.results import validate_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    writer = ResultWriter(args.output_dir)
    planned = {}
    groups = defaultdict(list)
    mapping = []
    names = {
        "roberta_roberta_base_full": "roberta_base_full",
        "roberta_roberta_base_adapter": "roberta_base_adapter",
    }
    with zipfile.ZipFile(args.archive) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith(".json"):
                continue
            artifact = json.loads(archive.read(member))
            if "predictions" not in artifact:
                continue
            validate_artifact(artifact)
            original = artifact["model"]["name"]
            artifact["model"]["name"] = names.get(original, original)
            path = writer.artifact_path(artifact)
            if path in planned or path.exists():
                raise ValueError(f"Refusing duplicate or existing artifact: {path}")
            planned[path] = artifact
            key = (artifact["protocol"], artifact["model"]["name"], artifact["seed"])
            groups[key].append((path, artifact))
            mapping.append(
                {
                    "source": member.filename,
                    "target": str(path),
                    "original_model": original,
                    "run_id": artifact["run_id"],
                }
            )
    for protocol, model, seed in groups:
        if (args.output_dir / protocol / model / f"summary-seed-{seed}.json").exists():
            raise ValueError("Refusing to overwrite an existing summary")
    manifest = args.output_dir / "p3_import_manifest.txt"
    if manifest.exists():
        raise ValueError(f"Refusing to overwrite {manifest}")
    for artifact in planned.values():
        writer.write(artifact)
    for (protocol, model, seed), entries in groups.items():
        rows = [
            {"repository": a["repository"], "macro_f1": a["metrics"]["macro_average"]["f1-score"]}
            for _, a in entries
        ]
        summary = aggregate_macro_f1(rows)
        summary.update(
            artifacts=[str(p) for p, _ in entries],
            text_fields=entries[0][1]["model"]["config"]["experiment"]["text_fields"],
            split_strategy=(
                "stratified_pooled_train_evaluated_per_repository"
                if protocol == "pooled_cv"
                else "stratified_text_group_folds_per_repository"
            ),
        )
        writer.write_summary(protocol, model, seed, summary)
    manifest.write_text(
        json.dumps(
            {
                "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                "mapping": mapping,
                "changes": "Only model names/paths normalized; summaries rebuilt. ZIP unchanged.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Imported {len(planned)} artifacts and {len(groups)} summaries")


if __name__ == "__main__":
    main()
