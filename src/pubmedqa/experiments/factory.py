"""Construct concrete runners from data-only experiment specifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pubmedqa.config import EnvironmentConfig
from pubmedqa.experiments.specs import (
    DEFAULT_BASELINE_OUTPUT_DIR,
    DEFAULT_PATHS,
    DEFAULT_SHARED_DEFAULTS,
    DEFAULT_TRAIN_OUTPUT_DIR,
    ExperimentPaths,
    RunSpec,
    SharedTrainDefaults,
    resolve_data_fraction,
    resolve_run_spec,
)
from pubmedqa.inference.contracts import ModelRuntimeConfig
from pubmedqa.inference.runner import PubMedQAEvaluationRunner
from pubmedqa.runtime.torch_runtime import resolve_dtype
from pubmedqa.training.engine import FullFineTuneConfig
from pubmedqa.training.strategies.full import PubMedQAFullFineTuner
from pubmedqa.training.strategies.lora import LoRAFineTuneConfig, PubMedQALoRAFineTuner

def build_baseline_runner(
    *,
    run_id: str,
    spec: RunSpec,
    output_dir: Path,
    defaults: SharedTrainDefaults,
    environment: EnvironmentConfig,
) -> PubMedQAEvaluationRunner:
    runtime = ModelRuntimeConfig(
        model_name=defaults.model_name,
        backend="torch",
        device=defaults.device,
        dtype="auto" if defaults.dtype == "auto" else defaults.dtype,
        max_new_tokens=defaults.max_new_tokens,
        batch_size=defaults.eval_batch_size,
        max_input_tokens=defaults.max_input_tokens,
        attn_implementation=defaults.attn_implementation,
        trust_remote_code=defaults.trust_remote_code,
        cpu_threads=defaults.cpu_threads,
        strict_parser=defaults.strict_parser,
    )
    return PubMedQAEvaluationRunner(
        run_id=run_id,
        model_name=defaults.model_name,
        condition=spec.condition,
        output_dir=output_dir,
        runtime=runtime,
        environment=environment,
    )


def build_full_ft_config(
    *,
    run_id: str,
    spec: RunSpec,
    paths: ExperimentPaths,
    output_dir: Path,
    defaults: SharedTrainDefaults,
) -> FullFineTuneConfig:
    return FullFineTuneConfig(
        run_id=run_id,
        run_tag=spec.run_tag,
        method_name="full-ft",
        model_name=defaults.model_name,
        condition=spec.condition,
        data_regime=spec.data_regime,
        data_fraction=resolve_data_fraction(spec, paths),
        train_path=paths.train_path,
        validation_path=paths.validation_path,
        test_path=paths.test_path,
        output_dir=output_dir,
        num_epochs=defaults.num_epochs,
        train_batch_size=defaults.train_batch_size,
        eval_batch_size=defaults.eval_batch_size,
        gradient_accumulation_steps=defaults.gradient_accumulation_steps,
        learning_rate=defaults.full_ft_learning_rate,
        weight_decay=defaults.weight_decay,
        warmup_ratio=defaults.warmup_ratio,
        max_grad_norm=defaults.max_grad_norm,
        max_input_tokens=defaults.max_input_tokens,
        max_new_tokens=defaults.max_new_tokens,
        device=defaults.device,
        dtype=resolve_dtype(defaults.dtype),
        attn_implementation=defaults.attn_implementation,
        trust_remote_code=defaults.trust_remote_code,
        cpu_threads=defaults.cpu_threads,
        log_every_steps=defaults.log_every_steps,
        save_every_epoch=True,
        eval_every_epoch=True,
        max_train_examples=(
            defaults.max_train_examples
            if defaults.max_train_examples is not None
            else spec.max_train_examples
        ),
        max_validation_examples=defaults.max_validation_examples,
        max_test_examples=defaults.max_test_examples,
        num_workers=defaults.num_workers,
        gradient_checkpointing=defaults.full_ft_gradient_checkpointing,
        save_optimizer_state=defaults.save_optimizer_state,
        strict_parser=defaults.strict_parser,
        seed=defaults.seed,
        target_modules=(),
        target_layers=(),
        layer_scope="all",
        lora_rank=None,
        lora_alpha=None,
        lora_dropout=None,
        notes=spec.notes,
        track_layerwise_updates=True,
        checkpoint_percents=defaults.checkpoint_percents,
        distributed_mode=defaults.distributed_mode,
        fsdp_cpu_offload=defaults.fsdp_cpu_offload,
    )


def build_lora_config(
    *,
    run_id: str,
    spec: RunSpec,
    paths: ExperimentPaths,
    output_dir: Path,
    defaults: SharedTrainDefaults,
    target_layers_override: tuple[int, ...] | None = None,
) -> LoRAFineTuneConfig:
    target_layers = spec.target_layers if target_layers_override is None else target_layers_override
    if spec.layer_scope != "all" and not target_layers:
        raise ValueError(
            f"{spec.run_tag} is a selective LoRA run and requires explicit target layers. "
            "Provide a run-specific override after selecting layers from the preceding analysis."
        )
    return LoRAFineTuneConfig(
        run_id=run_id,
        run_tag=spec.run_tag,
        method_name="lora",
        model_name=defaults.model_name,
        condition=spec.condition,
        data_regime=spec.data_regime,
        data_fraction=resolve_data_fraction(spec, paths),
        train_path=paths.train_path,
        validation_path=paths.validation_path,
        test_path=paths.test_path,
        output_dir=output_dir,
        num_epochs=defaults.num_epochs,
        train_batch_size=defaults.train_batch_size,
        eval_batch_size=defaults.eval_batch_size,
        gradient_accumulation_steps=defaults.gradient_accumulation_steps,
        learning_rate=defaults.lora_learning_rate,
        weight_decay=defaults.weight_decay,
        warmup_ratio=defaults.warmup_ratio,
        max_grad_norm=defaults.max_grad_norm,
        max_input_tokens=defaults.max_input_tokens,
        max_new_tokens=defaults.max_new_tokens,
        device=defaults.device,
        dtype=resolve_dtype(defaults.dtype),
        attn_implementation=defaults.attn_implementation,
        trust_remote_code=defaults.trust_remote_code,
        cpu_threads=defaults.cpu_threads,
        log_every_steps=defaults.log_every_steps,
        save_every_epoch=True,
        eval_every_epoch=True,
        max_train_examples=(
            defaults.max_train_examples
            if defaults.max_train_examples is not None
            else spec.max_train_examples
        ),
        max_validation_examples=defaults.max_validation_examples,
        max_test_examples=defaults.max_test_examples,
        num_workers=defaults.num_workers,
        gradient_checkpointing=defaults.lora_gradient_checkpointing,
        save_optimizer_state=defaults.save_optimizer_state,
        strict_parser=defaults.strict_parser,
        seed=defaults.seed,
        target_modules=spec.target_modules,
        target_layers=target_layers,
        layer_scope=spec.layer_scope,
        lora_rank=spec.lora_rank or 8,
        lora_alpha=spec.lora_alpha or 16.0,
        lora_dropout=spec.lora_dropout or 0.0,
        lora_bias=spec.lora_bias or "none",
        lora_task_type=spec.lora_task_type or "CAUSAL_LM",
        modules_to_save=spec.modules_to_save,
        merge_for_eval=spec.merge_for_eval,
        notes=spec.notes,
        track_layerwise_updates=True,
        checkpoint_percents=defaults.checkpoint_percents,
        distributed_mode=defaults.distributed_mode,
        fsdp_cpu_offload=defaults.fsdp_cpu_offload,
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
    if spec.method == "full-ft":
        config = build_full_ft_config(
            run_id=run_id,
            spec=spec,
            paths=paths,
            output_dir=train_output_dir,
            defaults=defaults,
        )
        return spec.method, PubMedQAFullFineTuner(config, environment)
    if spec.method == "lora":
        target_layers_override = None
        if target_layer_overrides is not None:
            target_layers_override = target_layer_overrides.get(run_tag)
        if target_layers_override and spec.layer_scope == "all":
            raise ValueError(f"{run_tag} is not a selective LoRA run and cannot receive a layer override.")
        config = build_lora_config(
            run_id=run_id,
            spec=spec,
            paths=paths,
            output_dir=train_output_dir,
            defaults=defaults,
            target_layers_override=target_layers_override,
        )
        return spec.method, PubMedQALoRAFineTuner(config, environment)
    raise ValueError(f"Unsupported method {spec.method!r}")
