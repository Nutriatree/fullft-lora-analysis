"""Resolve run inputs and execute a sequential ML study in one visible program."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pubmedqa.config import EnvironmentConfig
from pubmedqa.config.eval import ModelRuntimeConfig
from pubmedqa.config.experiments import (
    ExperimentPaths,
    RunSpec,
    SharedTrainDefaults,
    list_run_tags,
    resolve_data_fraction,
    resolve_run_spec,
)
from pubmedqa.config.full_ft import FullFineTuneConfig
from pubmedqa.config.lora import LoRAFineTuneConfig
from pubmedqa.data.records import load_local_jsonl, write_json
from pubmedqa.eval.inference import PubMedQAEvaluationRunner
from pubmedqa.model.device import resolve_dtype
from pubmedqa.train.distributed import (
    SynchronizedOperationError,
    TrainingSession,
    distributed_control_group,
    run_on_rank_zero,
    run_rank_local,
)
from pubmedqa.train.pipeline import run_training


@dataclass(frozen=True)
class ExperimentStudy:
    """All inputs required to execute or inspect one multi-run study."""

    run_id: str
    run_tags: tuple[str, ...]
    paths: ExperimentPaths
    defaults: SharedTrainDefaults
    baseline_output_dir: Path
    train_output_dir: Path
    target_layer_overrides: Mapping[str, tuple[int, ...]]
    environment: EnvironmentConfig
    continue_on_error: bool = False


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


def _shared_train_kwargs(
    run_id: str,
    spec: RunSpec,
    paths: ExperimentPaths,
    output_dir: Path,
    defaults: SharedTrainDefaults,
) -> dict[str, Any]:
    """Resolve the settings shared by Full FT and LoRA exactly once."""
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
) -> FullFineTuneConfig:
    return FullFineTuneConfig(
        **_shared_train_kwargs(run_id, spec, paths, output_dir, defaults),
        method_name="full-ft",
        learning_rate=defaults.full_ft_learning_rate,
        gradient_checkpointing=defaults.full_ft_gradient_checkpointing,
        target_modules=(),
        target_layers=(),
        layer_scope="all",
        lora_rank=None,
        lora_alpha=None,
        lora_dropout=None,
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
    target_layers = (
        spec.target_layers if target_layers_override is None else target_layers_override
    )
    if spec.layer_scope != "all" and not target_layers:
        raise ValueError(
            f"{spec.run_tag} is a selective LoRA run and requires explicit target layers. "
            "Provide a run-specific override after selecting layers from the preceding analysis."
        )
    return LoRAFineTuneConfig(
        **_shared_train_kwargs(run_id, spec, paths, output_dir, defaults),
        method_name="lora",
        learning_rate=defaults.lora_learning_rate,
        gradient_checkpointing=defaults.lora_gradient_checkpointing,
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
    )


def build_training_config(
    *,
    run_id: str,
    run_tag: str,
    paths: ExperimentPaths,
    output_dir: Path,
    defaults: SharedTrainDefaults,
    target_layer_overrides: Mapping[str, tuple[int, ...]] | None = None,
) -> FullFineTuneConfig:
    """Resolve a training variant without constructing a runner or any ML resources."""
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


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def format_run(spec_tag: str) -> str:
    spec = resolve_run_spec(spec_tag)
    return (
        f"{spec_tag}\tmethod={spec.method}\tcondition={spec.condition}\t"
        f"data_regime={spec.data_regime}\t"
        f"target_modules={','.join(spec.target_modules) or '-'}\t"
        f"rank={spec.lora_rank if spec.lora_rank is not None else '-'}\t"
        f"layer_scope={spec.layer_scope}"
    )


def print_run_table(run_tags: Sequence[str] | None = None) -> None:
    for run_tag in run_tags or list_run_tags():
        print(format_run(run_tag))


def validate_target_layer_overrides(
    run_tags: Sequence[str],
    overrides: Mapping[str, tuple[int, ...]],
) -> None:
    selected_runs = set(run_tags)
    unused_overrides = sorted(set(overrides) - selected_runs)
    if unused_overrides:
        raise ValueError(
            "Target-layer overrides were supplied for runs not selected in --runs: "
            + ", ".join(unused_overrides)
        )

    for run_tag in run_tags:
        spec = resolve_run_spec(run_tag)
        override = overrides.get(run_tag)
        if override and spec.layer_scope == "all":
            raise ValueError(
                f"{run_tag} is not a selective LoRA run and cannot receive a layer override."
            )
        if spec.layer_scope != "all" and not spec.target_layers and not override:
            raise ValueError(
                f"{run_tag} requires --target-layer-override {run_tag}=LAYER[,LAYER...] "
                "after layer-wise analysis."
            )


def save_manifest(study: ExperimentStudy) -> Path:
    manifest_path = study.train_output_dir / study.run_id / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "run_id": study.run_id,
            "requested_runs": list(study.run_tags),
            "paths": to_jsonable(asdict(study.paths)),
            "shared_defaults": to_jsonable(asdict(study.defaults)),
            "baseline_output_dir": str(study.baseline_output_dir),
            "train_output_dir": str(study.train_output_dir),
            "run_specs": {
                run_tag: to_jsonable(asdict(resolve_run_spec(run_tag)))
                for run_tag in study.run_tags
            },
            "target_layer_overrides": to_jsonable(dict(study.target_layer_overrides)),
        },
    )
    return manifest_path


def execute_study(study: ExperimentStudy, *, dry_run: bool = False) -> None:
    """Validate, record, and execute all runs in a study in registry order."""

    if not study.run_tags:
        raise RuntimeError("No run tags were provided.")
    validate_target_layer_overrides(study.run_tags, study.target_layer_overrides)

    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0
    if dry_run:
        if is_main_process:
            print(f"[manifest] {save_manifest(study)}")
            print_run_table(study.run_tags)
        return

    failures: list[dict[str, str]] = []
    with distributed_control_group(study.defaults.distributed_mode) as control_group:
        run_on_rank_zero(
            lambda: save_manifest(study),
            is_main_process=is_main_process,
            control_group=control_group,
            operation_name="study manifest",
        )
        for run_tag in study.run_tags:
            spec = resolve_run_spec(run_tag)
            if is_main_process:
                print(
                    f"\n[run] {run_tag} method={spec.method} condition={spec.condition}"
                )
            try:
                if spec.method == "baseline":
                    runner = run_rank_local(
                        lambda: build_baseline_runner(
                            run_id=study.run_id,
                            spec=spec,
                            output_dir=study.baseline_output_dir,
                            defaults=study.defaults,
                            environment=study.environment,
                        ),
                        control_group=control_group,
                        operation_name=f"{run_tag} build baseline",
                    )

                    def run_baseline() -> str:
                        examples = load_local_jsonl(study.paths.baseline_eval_path)
                        items, summary = runner.evaluate(examples)
                        run_dir = runner.save_results(items, summary)
                        return (
                            f"[done] {run_tag} acc={summary.accuracy:.4f} "
                            f"macro_f1={summary.macro_f1:.4f} saved={run_dir}"
                        )

                    message = run_on_rank_zero(
                        run_baseline,
                        is_main_process=is_main_process,
                        control_group=control_group,
                        operation_name=f"{run_tag} baseline evaluation",
                    )
                    if is_main_process:
                        print(message)
                else:
                    config = run_rank_local(
                        lambda: build_training_config(
                            run_id=study.run_id,
                            run_tag=run_tag,
                            paths=study.paths,
                            output_dir=study.train_output_dir,
                            defaults=study.defaults,
                            target_layer_overrides=study.target_layer_overrides,
                        ),
                        control_group=control_group,
                        operation_name=f"{run_tag} resolve training config",
                    )
                    session = run_rank_local(
                        lambda: TrainingSession.from_config(config),
                        control_group=control_group,
                        operation_name=f"{run_tag} prepare session",
                    )
                    # A run borrows the study's group. run_training closes only
                    # run-owned resources; the outer context owns study teardown.
                    session.control_group = control_group
                    summary = run_training(config, study.environment, session=session)
                    if is_main_process:
                        print(
                            f"[done] {run_tag} "
                            f"best_acc={summary.best_validation_accuracy:.4f} "
                            f"best_macro_f1={summary.best_validation_macro_f1:.4f} "
                            f"saved={summary.best_checkpoint_dir}"
                        )
            except Exception as exc:
                if control_group is not None and not isinstance(
                    exc, SynchronizedOperationError
                ):
                    # A crash/collective error has no safe rank-wide recovery point.
                    # Do not enter another collective or reuse the group for a run.
                    raise
                failures.append({"run_tag": run_tag, "error": str(exc)})
                print(f"[failed][rank={rank}] {run_tag}: {exc}")
                keep_going = run_on_rank_zero(
                    lambda: study.continue_on_error,
                    is_main_process=is_main_process,
                    control_group=control_group,
                    operation_name="continue-on-error decision",
                )
                if not keep_going:
                    break

    if failures:
        raise RuntimeError(f"Experiment execution failed: {failures}")
