"""P3 RoBERTa-base sequence classifiers (full fine-tuning and LoRA adapters)."""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from nlbse24.modeling.base import BaseIssueClassifier


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    seed: int = 42
    n_splits: int = 5
    text_fields: tuple[str, ...] = ("title", "body")


@dataclass(frozen=True, slots=True)
class ModelSettings:
    backbone: str = "roberta-base"
    max_seq_length: int = 512
    num_labels: int = 3
    device: str | None = None


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.0


@dataclass(frozen=True, slots=True)
class RobertaConfig:
    experiment: ExperimentSettings = ExperimentSettings()
    model: ModelSettings = ModelSettings()
    adapter: AdapterSettings = AdapterSettings()
    training: TrainingSettings = TrainingSettings()

    @classmethod
    def from_toml(cls, path: str | Path) -> "RobertaConfig":
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RobertaConfig":
        experiment = dict(payload.get("experiment", {}))
        model = dict(payload.get("model", {}))
        adapter = dict(payload.get("adapter", {}))
        training = dict(payload.get("training", {}))
        if "text_fields" in experiment:
            experiment["text_fields"] = tuple(experiment["text_fields"])
        if "target_modules" in adapter:
            adapter["target_modules"] = tuple(adapter["target_modules"])
        config = cls(
            experiment=ExperimentSettings(**experiment),
            model=ModelSettings(**model),
            adapter=AdapterSettings(**adapter),
            training=TrainingSettings(**training),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.experiment.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not self.experiment.text_fields:
            raise ValueError("text_fields must not be empty")
        if not self.model.backbone:
            raise ValueError("backbone must not be empty")
        if self.model.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.model.num_labels < 2:
            raise ValueError("num_labels must be at least 2")
        if self.adapter.enabled and self.adapter.r < 1:
            raise ValueError("adapter.r must be at least 1 when adapters are enabled")
        if self.adapter.enabled and not self.adapter.target_modules:
            raise ValueError("adapter.target_modules must not be empty when adapters are enabled")
        if self.training.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be at least 1")
        if self.training.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be at least 1")
        if self.training.per_device_eval_batch_size < 1:
            raise ValueError("per_device_eval_batch_size must be at least 1")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class RobertaClassifier(BaseIssueClassifier):
    """RoBERTa-base sequence classification, either full fine-tuning or a LoRA adapter.

    Truncation policy: inputs longer than ``max_seq_length`` tokens are
    tail-truncated by the tokenizer with ``padding="max_length"``, matching
    the P1 baseline's truncation behaviour.
    """

    def __init__(self, config: RobertaConfig | None = None) -> None:
        self.config = config or RobertaConfig()
        self.config.validate()
        self._classes: tuple[str, ...] | None = None
        self._label2id: dict[str, int] | None = None
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        suffix = "adapter" if self.config.adapter.enabled else "full"
        short = self.config.model.backbone.rsplit("/", 1)[-1].lower().replace("-", "_")
        return f"roberta_{short}_{suffix}"

    @property
    def classes_(self) -> tuple[str, ...]:
        self._check_fitted()
        return self._classes

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Self:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        if not texts:
            raise ValueError("cannot fit on an empty dataset")
        try:
            from datasets import Dataset as HFDataset
            from transformers import DataCollatorWithPadding
            from transformers import RobertaConfig as HFRobertaConfig
            from transformers import (
                RobertaForSequenceClassification,
                RobertaTokenizerFast,
                Trainer,
                TrainingArguments,
            )
        except ImportError as error:
            raise ImportError(
                "Transformers extras are missing; install them with "
                "`pip install -r requirements-roberta.txt`"
            ) from error

        self._classes = tuple(sorted(set(labels)))
        self._label2id = {label: index for index, label in enumerate(self._classes)}
        id2label = {index: label for label, index in self._label2id.items()}

        tokenizer = RobertaTokenizerFast.from_pretrained(self.config.model.backbone)
        hf_config = HFRobertaConfig.from_pretrained(
            self.config.model.backbone,
            num_labels=len(self._classes),
            label2id=self._label2id,
            id2label=id2label,
        )
        model = RobertaForSequenceClassification.from_pretrained(
            self.config.model.backbone, config=hf_config
        )
        if self.config.adapter.enabled:
            model = self._wrap_with_adapter(model)

        max_length = self.config.model.max_seq_length

        def preprocess_function(examples: Mapping[str, Any]) -> Mapping[str, Any]:
            return tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=max_length
            )

        dataset = HFDataset.from_dict(
            {
                "text": [str(text) for text in texts],
                "label": [self._label2id[str(label)] for label in labels],
            }
        )
        dataset = dataset.map(preprocess_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir=f"output/{self.name}",
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            seed=self.config.experiment.seed,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()

        self._model = model
        self._tokenizer = tokenizer
        return self

    def _wrap_with_adapter(self, model: Any) -> Any:
        """Attach a LoRA adapter and freeze the backbone; only the adapter trains."""

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:
            raise ImportError(
                "PEFT extras are missing; install them with "
                "`pip install -r requirements-roberta.txt`"
            ) from error
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.adapter.r,
            lora_alpha=self.config.adapter.lora_alpha,
            lora_dropout=self.config.adapter.lora_dropout,
            target_modules=list(self.config.adapter.target_modules),
        )
        return get_peft_model(model, lora_config)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        scores = self.predict_scores(texts)
        predicted_ids = scores.argmax(axis=1)
        return np.asarray([self._classes[int(index)] for index in predicted_ids], dtype=object)

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        self._check_fitted()
        import torch

        max_length = self.config.model.max_seq_length
        device = self.config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()

        encoded = self._tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = self._model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        scores = np.asarray(probabilities, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != len(self._classes):
            raise RuntimeError(
                f"unexpected score shape {scores.shape}; expected (n, {len(self._classes)})"
            )
        return scores

    def get_config(self) -> dict[str, Any]:
        return self.config.as_dict()

    def _check_fitted(self) -> None:
        if self._model is None or self._classes is None:
            raise RuntimeError("RobertaClassifier is not fitted yet; call fit() first")