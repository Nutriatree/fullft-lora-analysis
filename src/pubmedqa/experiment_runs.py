"""Compatibility facade for the experiment registry and runner factory.

New code should import data-only definitions from :mod:`pubmedqa.experiments.specs`
and concrete construction functions from :mod:`pubmedqa.experiments.factory`.
"""

from pubmedqa.experiments.factory import (
    build_baseline_runner,
    build_full_ft_config,
    build_lora_config,
    build_runner,
)
from pubmedqa.experiments.specs import (
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
