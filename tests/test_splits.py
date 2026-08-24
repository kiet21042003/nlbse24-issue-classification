from collections import Counter

from nlbse24.domain import IssueRecord
from nlbse24.splits import leave_one_repository_out_folds, stratified_repository_folds


def test_stratified_repository_folds_are_balanced_and_disjoint(
    sample_records: list[IssueRecord],
) -> None:
    folds = stratified_repository_folds(sample_records, n_splits=5, seed=42)
    assert len(folds) == 5
    seen_test_indices: list[int] = []
    for fold in folds:
        fold.validate()
        seen_test_indices.extend(fold.test_indices)
        counts = Counter(
            (sample_records[index].repo, sample_records[index].label)
            for index in fold.test_indices
        )
        assert set(counts.values()) == {2}
    assert sorted(seen_test_indices) == list(range(len(sample_records)))


def test_leave_one_repository_out_holds_out_only_target(
    sample_records: list[IssueRecord],
) -> None:
    folds = leave_one_repository_out_folds(sample_records)
    assert len(folds) == 2
    for fold in folds:
        train_repos = {sample_records[index].repo for index in fold.train_indices}
        test_repos = {sample_records[index].repo for index in fold.test_indices}
        assert test_repos == {fold.held_out_repository}
        assert fold.held_out_repository not in train_repos
