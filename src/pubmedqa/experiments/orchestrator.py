"""Application service for executing a sequence of registered experiments."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pubmedqa.config import EnvironmentConfig
from pubmedqa.experiments.factory import build_runner
from pubmedqa.experiments.specs import (
    ExperimentPaths,
    SharedTrainDefaults,
    list_run_tags,
    resolve_run_spec,
)
from pubmedqa.inference.runner import load_local_jsonl
from pubmedqa.runtime.distributed import distributed_control_group, run_on_rank_zero
from pubmedqa.runtime.io import write_json


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
    if is_main_process:
        print(f"[manifest] {save_manifest(study)}")
        print_run_table(study.run_tags)
    if dry_run:
        return

    failures: list[dict[str, str]] = []
    with distributed_control_group(study.defaults.distributed_mode) as control_group:
        for run_tag in study.run_tags:
            spec = resolve_run_spec(run_tag)
            if is_main_process:
                print(f"\n[run] {run_tag} method={spec.method} condition={spec.condition}")
            try:
                method, runner = build_runner(
                    run_id=study.run_id,
                    run_tag=run_tag,
                    paths=study.paths,
                    baseline_output_dir=study.baseline_output_dir,
                    train_output_dir=study.train_output_dir,
                    defaults=study.defaults,
                    environment=study.environment,
                    target_layer_overrides=study.target_layer_overrides,
                )
                if method == "baseline":
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
                    try:
                        summary = runner.run()
                    finally:
                        runner.close()
                    if is_main_process:
                        print(
                            f"[done] {run_tag} "
                            f"best_acc={summary.best_validation_accuracy:.4f} "
                            f"best_macro_f1={summary.best_validation_macro_f1:.4f} "
                            f"saved={summary.best_checkpoint_dir}"
                        )
            except Exception as exc:
                failures.append({"run_tag": run_tag, "error": str(exc)})
                print(f"[failed][rank={rank}] {run_tag}: {exc}")
                if not study.continue_on_error:
                    break

    if failures:
        raise RuntimeError(f"Experiment execution failed: {failures}")
