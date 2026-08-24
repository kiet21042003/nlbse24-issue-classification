"""Download the pinned official dataset with integrity verification."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from nlbse24.constants import UPSTREAM_COMMIT, UPSTREAM_REPOSITORY


@dataclass(frozen=True, slots=True)
class DatasetFile:
    name: str
    sha256: str

    @property
    def url(self) -> str:
        return (
            "https://raw.githubusercontent.com/nlbse2024/issue-report-classification/"
            f"{UPSTREAM_COMMIT}/data/{self.name}"
        )


OFFICIAL_FILES = (
    DatasetFile(
        "issues_train.csv",
        "18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5",
    ),
    DatasetFile(
        "issues_test.csv",
        "4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_dataset(destination: str | Path, force: bool = False) -> list[Path]:
    """Download both official splits atomically and reject checksum mismatches."""

    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for item in OFFICIAL_FILES:
        target = directory / item.name
        if target.exists() and sha256_file(target) == item.sha256 and not force:
            downloaded.append(target)
            continue
        if target.exists() and not force:
            raise ValueError(
                f"checksum mismatch for existing file: {target}; use force to replace it"
            )

        temporary = target.with_suffix(target.suffix + ".download")
        request = Request(item.url, headers={"User-Agent": "nlbse24-research-project/0.1"})
        try:
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            actual = sha256_file(temporary)
            if actual != item.sha256:
                raise ValueError(
                    f"checksum mismatch for {item.name}: expected {item.sha256}, found {actual}"
                )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded.append(target)

    manifest = {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "files": [
            {"name": item.name, "sha256": item.sha256, "url": item.url}
            for item in OFFICIAL_FILES
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return downloaded
