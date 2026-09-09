import json
from pathlib import Path

import numpy as np
import pytest

from nlbse24.modeling.setfit_classifier import (
    SetFitClassifier,
    SetFitConfig,
    subsample_indices_per_class,
)


def test_setfit_config_defaults_validate() -> None:
    config = SetFitConfig()
    config.validate()
    assert config.experiment.seed == 42
    assert config.model.max_seq_length == 128
    assert config.sampling.max_examples_per_class is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"experiment": {"n_splits": 1}},
        {"experiment": {"text_fields": []}},
        {"model": {"backbone": ""}},
        {"model": {"max_seq_length": 0}},
        {"sampling": {"num_iterations": 0}},
        {"sampling": {"num_epochs": 0}},
        {"sampling": {"batch_size": 0}},
        {"sampling": {"head_batch_size": 0}},
        {"sampling": {"max_examples_per_class": 0}},
    ],
)
def test_setfit_config_rejects_invalid_mappings(overrides: dict) -> None:
    with pytest.raises(ValueError):
        SetFitConfig.from_mapping(overrides)


def test_committed_setfit_configs_load() -> None:
    mpnet = SetFitConfig.from_toml("configs/setfit_mpnet.toml")
    assert mpnet.model.backbone.endswith("all-mpnet-base-v2")
    assert mpnet.sampling.num_iterations == 20
    assert mpnet.sampling.num_epochs == 1
    assert (mpnet.sampling.batch_size, mpnet.sampling.head_batch_size) == (16, 2)
    assert mpnet.sampling.max_examples_per_class is None
    minilm = SetFitConfig.from_toml("configs/setfit_minilm.toml")
    assert minilm.model.backbone.endswith("all-MiniLM-L6-v2")
    for name in ("setfit_mpnet_k8", "setfit_mpnet_k16", "setfit_mpnet_k32"):
        ablation = SetFitConfig.from_toml(f"configs/ablations/{name}.toml")
        shots = int(name.rsplit("_k", 1)[1])
        assert ablation.sampling.max_examples_per_class == shots
        assert ablation.sampling.num_iterations == 20
    for name in ("setfit_mpnet_it5", "setfit_mpnet_it80"):
        ablation = SetFitConfig.from_toml(f"configs/ablations/{name}.toml")
        iterations = int(name.rsplit("_it", 1)[1])
        assert ablation.sampling.num_iterations == iterations
        assert ablation.sampling.max_examples_per_class is None


def test_model_names_are_filesystem_safe_and_distinct() -> None:
    mpnet = SetFitClassifier(SetFitConfig.from_toml("configs/setfit_mpnet.toml"))
    minilm = SetFitClassifier(SetFitConfig.from_toml("configs/setfit_minilm.toml"))
    assert mpnet.name == "setfit_mpnet"
    assert minilm.name == "setfit_minilm"
    assert "/" not in mpnet.name and "/" not in minilm.name


def test_get_config_is_json_serializable() -> None:
    payload = json.dumps(SetFitClassifier().get_config())
    assert "backbone" in payload


def test_subsample_indices_per_class_is_stratified_and_seeded() -> None:
    labels = ["a", "b"] * 10
    first = subsample_indices_per_class(labels, max_examples=3, seed=42)
    second = subsample_indices_per_class(labels, max_examples=3, seed=42)
    assert first == second
    assert len(first) == 6
    kept_labels = [labels[index] for index in first]
    assert kept_labels.count("a") == 3
    assert kept_labels.count("b") == 3


def test_subsample_none_keeps_everything() -> None:
    labels = ["a", "b", "c"]
    assert subsample_indices_per_class(labels, max_examples=None, seed=42) == [0, 1, 2]


def test_contract_before_fit(tmp_path: Path) -> None:
    model = SetFitClassifier()
    assert isinstance(json.dumps(model.get_config()), str)
    with pytest.raises(ValueError):
        model.fit(["text"], ["bug", "feature"])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(["text"])
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = model.classes_
    scores: np.ndarray | None = None
    try:
        scores = model.predict_scores(["text"])
    except RuntimeError as error:
        assert "not fitted" in str(error)
    assert scores is None or scores.shape[0] == 1
