"""Tests for the P3 RoBERTa classifier contract."""

import pytest

from nlbse24.modeling.encoder import (
    AdapterSettings,
    ExperimentSettings,
    ModelSettings,
    RobertaClassifier,
    RobertaConfig,
    TrainingSettings,
)

# --- Config validation (no heavy deps needed) -------------------------------


def test_default_config_is_valid() -> None:
    RobertaConfig().validate()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"experiment": ExperimentSettings(n_splits=1)}, "n_splits"),
        ({"experiment": ExperimentSettings(text_fields=())}, "text_fields"),
        ({"model": ModelSettings(backbone="")}, "backbone"),
        ({"model": ModelSettings(max_seq_length=0)}, "max_seq_length"),
        ({"model": ModelSettings(num_labels=1)}, "num_labels"),
        ({"adapter": AdapterSettings(enabled=True, r=0)}, "adapter.r"),
        (
            {"adapter": AdapterSettings(enabled=True, target_modules=())},
            "adapter.target_modules",
        ),
        ({"training": TrainingSettings(num_train_epochs=0)}, "num_train_epochs"),
        ({"training": TrainingSettings(per_device_train_batch_size=0)}, "train_batch_size"),
        ({"training": TrainingSettings(learning_rate=0.0)}, "learning_rate"),
    ],
)
def test_invalid_config_raises(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RobertaConfig(**overrides)


def test_config_from_mapping_round_trips_tuples() -> None:
    config = RobertaConfig.from_mapping(
        {
            "experiment": {"text_fields": ["title"]},
            "adapter": {"enabled": True, "target_modules": ["query"]},
        }
    )
    assert config.experiment.text_fields == ("title",)
    assert config.adapter.target_modules == ("query",)


def test_committed_full_config_loads() -> None:
    config = RobertaConfig.from_toml("configs/roberta_full.toml")
    assert config.adapter.enabled is False
    assert config.experiment.n_splits == 5


def test_committed_lora_config_loads() -> None:
    config = RobertaConfig.from_toml("configs/roberta_lora.toml")
    assert config.adapter.enabled is True
    assert config.adapter.target_modules == ("query", "value")


# --- Classifier contract guards (no heavy deps needed) ----------------------


def test_name_reflects_backbone_and_adapter_flag() -> None:
    
    full = RobertaClassifier(RobertaConfig())
    assert full.name == "roberta_base_full"

    lora = RobertaClassifier(RobertaConfig(adapter=AdapterSettings(enabled=True)))
    assert lora.name == "roberta_base_adapter"


def test_predict_before_fit_raises() -> None:
    model = RobertaClassifier(RobertaConfig())
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(["some text"])


def test_classes_before_fit_raises() -> None:
    model = RobertaClassifier(RobertaConfig())
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = model.classes_


def test_fit_rejects_mismatched_lengths() -> None:
    model = RobertaClassifier(RobertaConfig())
    with pytest.raises(ValueError, match="equal length"):
        model.fit(["a", "b"], ["bug"])


def test_fit_rejects_empty_dataset() -> None:
    model = RobertaClassifier(RobertaConfig())
    with pytest.raises(ValueError, match="empty dataset"):
        model.fit([], [])


def test_get_config_is_json_serializable_shape() -> None:
    model = RobertaClassifier(RobertaConfig())
    config = model.get_config()
    assert config["model"]["backbone"] == "roberta-base"
    assert config["adapter"]["enabled"] is False


def test_fit_without_transformers_raises_helpful_import_error(monkeypatch) -> None:
    """Confirms the environment's actual current state: without
    `requirements-roberta.txt` installed, fit() must fail with guidance
    rather than a bare ImportError/traceback."""
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    model = RobertaClassifier(RobertaConfig())
    with pytest.raises(ImportError, match="requirements-roberta.txt"):
        model.fit(["bug report here"], ["bug"])


# --- Integration: real fit/predict/LoRA (skipped unless deps + network) ----
# No @pytest.mark.slow here: pytest.importorskip below already skips these
# tests cleanly when transformers/torch/peft are absent (e.g. on CI, which
# only installs requirements-dev.txt), without needing a registered marker.

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")


def test_fit_predict_round_trip_full() -> None:
    """Requires network access to download `roberta-base` on first run.
    Kept tiny (2 examples/class, 1 epoch) to stay runnable on a weak CPU."""
    config = RobertaConfig(
        experiment=ExperimentSettings(seed=42),
        model=ModelSettings(max_seq_length=32),
        training=TrainingSettings(num_train_epochs=1, per_device_train_batch_size=2),
    )
    texts = [
        "crash error failure",
        "broken exception thrown",
        "please add an option",
        "new capability request",
    ]
    labels = ["bug", "bug", "feature", "feature"]

    model = RobertaClassifier(config).fit(texts, labels)
    predictions = model.predict(texts)
    scores = model.predict_scores(texts)

    assert predictions.shape == (4,)
    assert scores.shape == (4, 2)


def test_fit_predict_round_trip_lora() -> None:
    pytest.importorskip("peft")
    config = RobertaConfig(
        adapter=AdapterSettings(enabled=True, r=4),
        model=ModelSettings(max_seq_length=32),
        training=TrainingSettings(num_train_epochs=1, per_device_train_batch_size=2),
    )
    texts = ["crash error failure", "please add an option"]
    labels = ["bug", "feature"]

    model = RobertaClassifier(config).fit(texts, labels)
    predictions = model.predict(texts)
    assert predictions.shape == (2,)