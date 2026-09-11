"""Historical experiment API; definitions live in config and execution in train."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from pubmedqa.config.experiments import (
    DEFAULT_BASELINE_OUTPUT_DIR,
    DEFAULT_LOW_DATA_MAX_TRAIN_EXAMPLES,
    DEFAULT_LOW_DATA_POLICY,
    DEFAULT_LOW_DATA_SEED,
    DEFAULT_PATHS,
    DEFAULT_PQA_ARTIFICIAL_TRAIN_PATH,
    DEFAULT_PQA_ARTIFICIAL_VALIDATION_PATH,
    DEFAULT_PQA_LABELED_CV_PATH,
    DEFAULT_PQA_LABELED_TEST_PATH,
    DEFAULT_SHARED_DEFAULTS,
    DEFAULT_TRAIN_OUTPUT_DIR,
    RUN_SPECS,
    ExperimentPaths,
    LowDataPolicy,
    RunSpec,
    SharedTrainDefaults,
    list_run_tags,
    resolve_data_fraction,
    resolve_run_spec,
)

if TYPE_CHECKING:
    from pubmedqa.config import EnvironmentConfig
    from pubmedqa.train.study import (
        build_baseline_runner,
        build_full_ft_config,
        build_lora_config,
    )


def build_runner(
    *,
    run_id: str,
    run_tag: str,
    paths: ExperimentPaths = DEFAULT_PATHS,
    baseline_output_dir: Path = DEFAULT_BASELINE_OUTPUT_DIR,
    train_output_dir: Path = DEFAULT_TRAIN_OUTPUT_DIR,
    defaults: SharedTrainDefaults = DEFAULT_SHARED_DEFAULTS,
    environment: EnvironmentConfig | None = None,
    target_layer_overrides: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[str, Any]:
    from pubmedqa.config import EnvironmentConfig
    from pubmedqa.train.study import build_baseline_runner, build_training_config

    spec = resolve_run_spec(run_tag)
    environment = environment or EnvironmentConfig.from_env()

    if spec.method == "baseline":
        return spec.method, build_baseline_runner(
            run_id=run_id,
            spec=spec,
            output_dir=baseline_output_dir,
            defaults=defaults,
            environment=environment,
        )
    config = build_training_config(
        run_id=run_id,
        run_tag=run_tag,
        paths=paths,
        output_dir=train_output_dir,
        defaults=defaults,
        target_layer_overrides=target_layer_overrides,
    )
    # Only historical callers construct a class; the study calls run_training.
    from pubmedqa.full_finetune import PubMedQAFullFineTuner
    from pubmedqa.lora_finetune import PubMedQALoRAFineTuner

    adapter = (
        PubMedQAFullFineTuner if spec.method == "full-ft" else PubMedQALoRAFineTuner
    )
    return spec.method, adapter(config, environment)


def __getattr__(name):
    if name in {"build_baseline_runner", "build_full_ft_config", "build_lora_config"}:
        return getattr(import_module("pubmedqa.train.study"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_BASELINE_OUTPUT_DIR",
    "DEFAULT_LOW_DATA_MAX_TRAIN_EXAMPLES",
    "DEFAULT_LOW_DATA_POLICY",
    "DEFAULT_LOW_DATA_SEED",
    "DEFAULT_PATHS",
    "DEFAULT_PQA_ARTIFICIAL_TRAIN_PATH",
    "DEFAULT_PQA_ARTIFICIAL_VALIDATION_PATH",
    "DEFAULT_PQA_LABELED_CV_PATH",
    "DEFAULT_PQA_LABELED_TEST_PATH",
    "DEFAULT_SHARED_DEFAULTS",
    "DEFAULT_TRAIN_OUTPUT_DIR",
    "RUN_SPECS",
    "ExperimentPaths",
    "LowDataPolicy",
    "RunSpec",
    "SharedTrainDefaults",
    "build_baseline_runner",
    "build_full_ft_config",
    "build_lora_config",
    "build_runner",
    "list_run_tags",
    "resolve_data_fraction",
    "resolve_run_spec",
]
