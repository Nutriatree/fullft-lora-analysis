"""Data-only run recipes and conversion into one training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pubmedqa.config.train import (
    TRAIN_CONFIG,
    TRAIN_LAYER_CONFIG,
    TRAIN_LORA_CONFIG,
    LoRAOptions,
    TrainingConfig,
)

DEFAULT_BASELINE_OUTPUT_DIR = Path("outputs/pubmedqa_eval")


DEFAULT_TRAIN_OUTPUT_DIR = Path("outputs/pubmedqa_train")


DEFAULT_PQA_ARTIFICIAL_TRAIN_PATH = Path(
    "data/processed/posttrain_v1/pqa_artificial/train.jsonl"
)


DEFAULT_PQA_ARTIFICIAL_VALIDATION_PATH = Path(
    "data/processed/posttrain_v1/pqa_artificial/validation.jsonl"
)


DEFAULT_PQA_LABELED_CV_PATH = Path("data/processed/pqa_labeled/cv.jsonl")


DEFAULT_PQA_LABELED_TEST_PATH = Path("data/processed/pqa_labeled/test.jsonl")


DEFAULT_LOW_DATA_MAX_TRAIN_EXAMPLES = 2048


DEFAULT_LOW_DATA_SEED = TRAIN_CONFIG.default_seed


@dataclass(frozen=True)
class ExperimentPaths:
    train_path: Path
    validation_path: Path
    test_path: Path
    baseline_eval_path: Path


@dataclass(frozen=True)
class SharedTrainDefaults:
    model_name: str = TRAIN_CONFIG.default_model_name
    num_epochs: int = TRAIN_CONFIG.default_num_epochs
    train_batch_size: int = TRAIN_CONFIG.default_train_batch_size
    eval_batch_size: int = TRAIN_CONFIG.default_eval_batch_size
    gradient_accumulation_steps: int = TRAIN_CONFIG.default_grad_accum_steps
    full_ft_learning_rate: float = TRAIN_CONFIG.default_learning_rate
    lora_learning_rate: float = TRAIN_LORA_CONFIG.default_learning_rate
    weight_decay: float = TRAIN_CONFIG.default_weight_decay
    warmup_ratio: float = TRAIN_CONFIG.default_warmup_ratio
    max_grad_norm: float = TRAIN_CONFIG.default_max_grad_norm
    max_input_tokens: int | None = None
    max_new_tokens: int = TRAIN_CONFIG.default_max_new_tokens
    max_train_examples: int | None = None
    max_validation_examples: int | None = None
    max_test_examples: int | None = None
    device: str = TRAIN_CONFIG.default_device
    dtype: str = TRAIN_CONFIG.default_dtype
    attn_implementation: str | None = TRAIN_CONFIG.default_attn_implementation
    trust_remote_code: bool = False
    cpu_threads: int = TRAIN_CONFIG.default_cpu_threads
    log_every_steps: int = TRAIN_CONFIG.default_log_every_steps
    num_workers: int = TRAIN_CONFIG.default_num_workers
    full_ft_gradient_checkpointing: bool = TRAIN_CONFIG.default_gradient_checkpointing
    lora_gradient_checkpointing: bool = TRAIN_LORA_CONFIG.default_gradient_checkpointing
    save_optimizer_state: bool = TRAIN_CONFIG.default_save_optimizer_state
    strict_parser: bool = False
    seed: int = TRAIN_CONFIG.default_seed
    checkpoint_percents: tuple[int, ...] = (
        TRAIN_LAYER_CONFIG.default_checkpoint_percents
    )
    distributed_mode: str = TRAIN_CONFIG.default_distributed_mode
    fsdp_cpu_offload: bool = TRAIN_CONFIG.default_fsdp_cpu_offload


@dataclass(frozen=True)
class LowDataPolicy:
    max_train_examples: int
    seed: int


@dataclass(frozen=True)
class RunSpec:
    run_tag: str
    method: str  # baseline | full-ft | lora
    condition: str
    data_regime: str
    data_fraction: float
    max_train_examples: int | None = None
    target_modules: tuple[str, ...] = ()
    target_layers: tuple[int, ...] = ()
    layer_scope: str = "all"
    lora_rank: int | None = None
    lora_alpha: float | None = None
    lora_dropout: float | None = None
    lora_bias: str | None = None
    lora_task_type: str | None = None
    modules_to_save: tuple[str, ...] = ()
    merge_for_eval: bool = False
    notes: str | None = None


DEFAULT_PATHS = ExperimentPaths(
    train_path=DEFAULT_PQA_ARTIFICIAL_TRAIN_PATH,
    validation_path=DEFAULT_PQA_ARTIFICIAL_VALIDATION_PATH,
    test_path=DEFAULT_PQA_LABELED_TEST_PATH,
    baseline_eval_path=DEFAULT_PQA_LABELED_TEST_PATH,
)


DEFAULT_SHARED_DEFAULTS = SharedTrainDefaults()


DEFAULT_LOW_DATA_POLICY = LowDataPolicy(
    max_train_examples=DEFAULT_LOW_DATA_MAX_TRAIN_EXAMPLES,
    seed=DEFAULT_LOW_DATA_SEED,
)


RUN_SPECS: dict[str, RunSpec] = {
    "B0": RunSpec(
        run_tag="B0",
        method="baseline",
        condition="baseline",
        data_regime="labeled-test",
        data_fraction=0.0,
        notes="Pretrained baseline evaluation on PQA-L test.",
    ),
    "F1": RunSpec(
        run_tag="F1",
        method="full-ft",
        condition="full-ft",
        data_regime="full-data",
        data_fraction=1.0,
        notes="Full FT reference run.",
    ),
    "L1": RunSpec(
        run_tag="L1",
        method="lora",
        condition="lora",
        data_regime="full-data",
        data_fraction=1.0,
        target_modules=("q_proj", "v_proj"),
        layer_scope="all",
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_bias="none",
        lora_task_type="CAUSAL_LM",
        notes="LoRA reference run.",
    ),
    "L2": RunSpec(
        run_tag="L2",
        method="lora",
        condition="lora-qkvo",
        data_regime="full-data",
        data_fraction=1.0,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        layer_scope="all",
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_bias="none",
        lora_task_type="CAUSAL_LM",
        notes="Target module comparison run.",
    ),
    "L3": RunSpec(
        run_tag="L3",
        method="lora",
        condition="lora-r4",
        data_regime="full-data",
        data_fraction=1.0,
        target_modules=("q_proj", "v_proj"),
        layer_scope="all",
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
        lora_bias="none",
        lora_task_type="CAUSAL_LM",
        notes="Low-rank LoRA run.",
    ),
    "L4": RunSpec(
        run_tag="L4",
        method="lora",
        condition="lora-r16",
        data_regime="full-data",
        data_fraction=1.0,
        target_modules=("q_proj", "v_proj"),
        layer_scope="all",
        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.0,
        lora_bias="none",
        lora_task_type="CAUSAL_LM",
        notes="High-rank LoRA run.",
    ),
    "LL1": RunSpec(
        run_tag="LL1",
        method="lora",
        condition="selective-lora-high-update",
        data_regime="full-data",
        data_fraction=1.0,
        target_modules=("q_proj", "v_proj"),
        layer_scope="high-update",
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_bias="none",
        lora_task_type="CAUSAL_LM",
        notes="Selective LoRA with high-update layers injected externally.",
    ),
    "LL2": RunSpec(
        run_tag="LL2",
        method="lora",
        condition="selective-lora-low-update",
        data_regime="full-data",
        data_fraction=1.0,
        target_modules=("q_proj", "v_proj"),
        layer_scope="low-update",
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_bias="none",
        lora_task_type="CAUSAL_LM",
        notes="Selective LoRA low-update control run.",
    ),
    "F2": RunSpec(
        run_tag="F2",
        method="full-ft",
        condition="full-ft-low-data",
        data_regime="low-data",
        data_fraction=0.0,
        max_train_examples=DEFAULT_LOW_DATA_POLICY.max_train_examples,
        notes="Low-data Full FT run.",
    ),
    "L5": RunSpec(
        run_tag="L5",
        method="lora",
        condition="lora-low-data",
        data_regime="low-data",
        data_fraction=0.0,
        max_train_examples=DEFAULT_LOW_DATA_POLICY.max_train_examples,
        target_modules=("q_proj", "v_proj"),
        layer_scope="all",
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_bias="none",
        lora_task_type="CAUSAL_LM",
        notes="Low-data LoRA run.",
    ),
}


def list_run_tags() -> tuple[str, ...]:
    return tuple(RUN_SPECS.keys())


def resolve_run_spec(run_tag: str) -> RunSpec:
    if run_tag not in RUN_SPECS:
        available = ", ".join(list_run_tags())
        raise KeyError(f"Unknown run tag {run_tag!r}. Available: {available}")
    return RUN_SPECS[run_tag]


def resolve_data_fraction(spec: RunSpec, paths: ExperimentPaths) -> float:
    if spec.method == "baseline":
        return spec.data_fraction
    if spec.data_fraction > 0.0:
        return spec.data_fraction
    try:
        with paths.train_path.open("r", encoding="utf-8") as file:
            total_lines = sum(1 for _ in file)
    except FileNotFoundError:
        return 0.0
    if total_lines <= 0 or spec.max_train_examples is None:
        return 0.0
    return min(1.0, spec.max_train_examples / total_lines)


def _shared_train_kwargs(
    run_id: str,
    spec: RunSpec,
    paths: ExperimentPaths,
    output_dir: Path,
    defaults: SharedTrainDefaults,
) -> dict[str, Any]:
    """Resolve common run settings without constructing training resources."""
    # Keep torch out of module import so listing run recipes stays lightweight.
    from pubmedqa.model.device import resolve_dtype

    return {
        "run_id": run_id,
        "run_tag": spec.run_tag,
        "model_name": defaults.model_name,
        "condition": spec.condition,
        "data_regime": spec.data_regime,
        "data_fraction": resolve_data_fraction(spec, paths),
        "train_path": paths.train_path,
        "validation_path": paths.validation_path,
        "test_path": paths.test_path,
        "output_dir": output_dir,
        "num_epochs": defaults.num_epochs,
        "train_batch_size": defaults.train_batch_size,
        "eval_batch_size": defaults.eval_batch_size,
        "gradient_accumulation_steps": defaults.gradient_accumulation_steps,
        "weight_decay": defaults.weight_decay,
        "warmup_ratio": defaults.warmup_ratio,
        "max_grad_norm": defaults.max_grad_norm,
        "max_input_tokens": defaults.max_input_tokens,
        "max_new_tokens": defaults.max_new_tokens,
        "device": defaults.device,
        "dtype": resolve_dtype(defaults.dtype),
        "attn_implementation": defaults.attn_implementation,
        "trust_remote_code": defaults.trust_remote_code,
        "cpu_threads": defaults.cpu_threads,
        "log_every_steps": defaults.log_every_steps,
        "save_every_epoch": True,
        "eval_every_epoch": True,
        "max_train_examples": defaults.max_train_examples
        if defaults.max_train_examples is not None
        else spec.max_train_examples,
        "max_validation_examples": defaults.max_validation_examples,
        "max_test_examples": defaults.max_test_examples,
        "num_workers": defaults.num_workers,
        "save_optimizer_state": defaults.save_optimizer_state,
        "strict_parser": defaults.strict_parser,
        "seed": defaults.seed,
        "notes": spec.notes,
        "track_layerwise_updates": True,
        "checkpoint_percents": defaults.checkpoint_percents,
        "distributed_mode": defaults.distributed_mode,
        "fsdp_cpu_offload": defaults.fsdp_cpu_offload,
    }


def build_full_ft_config(
    *,
    run_id: str,
    spec: RunSpec,
    paths: ExperimentPaths,
    output_dir: Path,
    defaults: SharedTrainDefaults,
) -> TrainingConfig:
    return TrainingConfig(
        **_shared_train_kwargs(run_id, spec, paths, output_dir, defaults),
        method_name="full-ft",
        learning_rate=defaults.full_ft_learning_rate,
        gradient_checkpointing=defaults.full_ft_gradient_checkpointing,
        adapter=None,
    )


def build_lora_config(
    *,
    run_id: str,
    spec: RunSpec,
    paths: ExperimentPaths,
    output_dir: Path,
    defaults: SharedTrainDefaults,
    target_layers_override: tuple[int, ...] | None = None,
) -> TrainingConfig:
    target_layers = (
        spec.target_layers if target_layers_override is None else target_layers_override
    )
    if spec.layer_scope != "all" and not target_layers:
        raise ValueError(
            f"{spec.run_tag} is a selective LoRA run and requires explicit target layers. "
            "Provide a run-specific override after selecting layers from the preceding analysis."
        )
    return TrainingConfig(
        **_shared_train_kwargs(run_id, spec, paths, output_dir, defaults),
        method_name="lora",
        learning_rate=defaults.lora_learning_rate,
        gradient_checkpointing=defaults.lora_gradient_checkpointing,
        adapter=LoRAOptions(
            rank=spec.lora_rank or 8,
            alpha=spec.lora_alpha or 16.0,
            dropout=spec.lora_dropout or 0.0,
            target_modules=spec.target_modules,
            target_layers=target_layers,
            layer_scope=spec.layer_scope,
            bias=spec.lora_bias or "none",
            task_type=spec.lora_task_type or "CAUSAL_LM",
            modules_to_save=spec.modules_to_save,
            merge_for_eval=spec.merge_for_eval,
        ),
    )


def build_training_config(
    *,
    run_id: str,
    run_tag: str,
    paths: ExperimentPaths,
    output_dir: Path,
    defaults: SharedTrainDefaults,
    target_layer_overrides: Mapping[str, tuple[int, ...]] | None = None,
) -> TrainingConfig:
    """Convert one registry recipe into the configuration consumed by training."""
    spec = resolve_run_spec(run_tag)
    if spec.method == "full-ft":
        return build_full_ft_config(
            run_id=run_id,
            spec=spec,
            paths=paths,
            output_dir=output_dir,
            defaults=defaults,
        )
    if spec.method == "lora":
        override = (target_layer_overrides or {}).get(run_tag)
        if override and spec.layer_scope == "all":
            raise ValueError(
                f"{run_tag} is not a selective LoRA run and cannot receive a layer override."
            )
        return build_lora_config(
            run_id=run_id,
            spec=spec,
            paths=paths,
            output_dir=output_dir,
            defaults=defaults,
            target_layers_override=override,
        )
    raise ValueError(f"Unsupported training method {spec.method!r}")
