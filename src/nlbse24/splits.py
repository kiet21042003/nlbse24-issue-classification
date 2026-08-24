"""Deterministic, leakage-safe experiment split protocols."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from nlbse24.domain import IssueRecord
from nlbse24.text import compose_text, normalize_whitespace


@dataclass(frozen=True, slots=True)
class SplitFold:
    protocol: str
    fold: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    held_out_repository: str | None = None

    def validate(self) -> None:
        train = set(self.train_indices)
        test = set(self.test_indices)
        if not train or not test:
            raise ValueError(f"{self.protocol}/{self.fold} contains an empty partition")
        overlap = train.intersection(test)
        if overlap:
            raise ValueError(f"{self.protocol}/{self.fold} leaks {len(overlap)} rows")


def stratified_repository_folds(
    records: list[IssueRecord], n_splits: int = 5, seed: int = 42
) -> list[SplitFold]:
    """Create matching stratified folds independently inside every repository."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    indices_by_repo: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_repo[record.repo].append(index)
    if not indices_by_repo:
        raise ValueError("cannot split an empty dataset")

    fold_train: list[list[int]] = [[] for _ in range(n_splits)]
    fold_test: list[list[int]] = [[] for _ in range(n_splits)]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for repo in sorted(indices_by_repo):
        repo_indices = np.asarray(indices_by_repo[repo], dtype=int)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index in repo_indices:
            key = normalize_whitespace(compose_text(records[index])).casefold()
            grouped_indices[key].append(int(index))
        groups = list(grouped_indices.values())
        group_labels: list[str] = []
        for group in groups:
            labels = {records[index].label for index in group}
            if len(labels) != 1:
                raise ValueError(f"exact duplicate text has conflicting labels in {repo}")
            group_labels.append(labels.pop())
        for fold_index, (local_train, local_test) in enumerate(
            splitter.split(np.arange(len(groups)), np.asarray(group_labels))
        ):
            fold_train[fold_index].extend(
                index for group_index in local_train for index in groups[group_index]
            )
            fold_test[fold_index].extend(
                index for group_index in local_test for index in groups[group_index]
            )

    folds = [
        SplitFold(
            protocol="cv",
            fold=f"fold-{fold_index + 1}",
            train_indices=tuple(sorted(fold_train[fold_index])),
            test_indices=tuple(sorted(fold_test[fold_index])),
        )
        for fold_index in range(n_splits)
    ]
    for fold in folds:
        fold.validate()
    return folds


def leave_one_repository_out_folds(records: list[IssueRecord]) -> list[SplitFold]:
    """Use each repository as a held-out domain exactly once."""

    repositories = sorted({record.repo for record in records})
    if len(repositories) < 2:
        raise ValueError("leave-one-repository-out requires at least two repositories")
    folds: list[SplitFold] = []
    for repo in repositories:
        test_indices = tuple(index for index, record in enumerate(records) if record.repo == repo)
        train_indices = tuple(index for index, record in enumerate(records) if record.repo != repo)
        fold = SplitFold(
            protocol="loo",
            fold=repo.replace("/", "__"),
            train_indices=train_indices,
            test_indices=test_indices,
            held_out_repository=repo,
        )
        fold.validate()
        folds.append(fold)
    return folds
