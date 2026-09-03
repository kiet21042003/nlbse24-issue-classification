"""P3 RoBERTa experiment script: điều phối per-repository và pooled training.
 
"""

from pathlib import Path
from typing import Any, Literal

from nlbse24.data import CsvIssueRepository
from nlbse24.modeling import BaseIssueClassifier
from nlbse24.modeling.roberta_classifier import AdapterSettings, RobertaClassifier, RobertaConfig
from nlbse24.runner import run_classifier_experiment


def run_roberta_experiment(
    *,
    protocol: Literal["cv", "pooled_cv"],
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
    config: RobertaConfig | None = None,
    config_path: str | Path | None = None,
    repositories: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """protocol='cv' -> per-repository (train và eval riêng trên từng repo).
    protocol='pooled_cv' -> pooled (train gộp mọi repo, eval riêng theo repo)."""
    if config is not None and config_path is not None:
        raise ValueError("chỉ truyền config hoặc config_path, không dùng cả hai")

    resolved_config = config or (
        RobertaConfig.from_toml(config_path) if config_path else RobertaConfig()
    )

    def model_factory() -> BaseIssueClassifier:
        return RobertaClassifier(resolved_config)

    model_name = RobertaClassifier(resolved_config).name

    return run_classifier_experiment(
        repository=CsvIssueRepository(data_dir),
        model_factory=model_factory,
        model_name=model_name,
        output_dir=output_dir,
        protocol=protocol,
        seed=resolved_config.experiment.seed,
        n_splits=resolved_config.experiment.n_splits,
        text_fields=resolved_config.experiment.text_fields,
        repositories=repositories,
        overwrite=overwrite,
    )


def main() -> None:
    output_dir = Path("results/roberta")
    for adapter_enabled in (False, True):
        config = RobertaConfig(adapter=AdapterSettings(enabled=adapter_enabled))
        suffix = "adapter" if adapter_enabled else "full"

        per_repo_summary = run_roberta_experiment(
            protocol="cv", config=config, output_dir=output_dir
        )
        print(f"per-repository ({suffix}):", per_repo_summary)

        pooled_summary = run_roberta_experiment(
            protocol="pooled_cv", config=config, output_dir=output_dir
        )
        print(f"pooled ({suffix}):", pooled_summary)


if __name__ == "__main__":
    main()