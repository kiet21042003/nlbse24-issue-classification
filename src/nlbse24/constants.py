"""Dataset constants shared by all experiment modules."""

from typing import Final

LABELS: Final[tuple[str, ...]] = ("bug", "feature", "question")
REPOSITORIES: Final[tuple[str, ...]] = (
    "bitcoin/bitcoin",
    "facebook/react",
    "microsoft/vscode",
    "opencv/opencv",
    "tensorflow/tensorflow",
)

UPSTREAM_COMMIT: Final[str] = "2927bc67eb42db8affd16eaf3e5a6d74f3063961"
UPSTREAM_REPOSITORY: Final[str] = "https://github.com/nlbse2024/issue-report-classification"
