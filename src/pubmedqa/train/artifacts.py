"""Persist run configuration and summaries without changing experiment schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from pubmedqa.config.full_ft import FullFineTuneConfig
from pubmedqa.data.records import current_time_iso, safe_name, write_json


@dataclass(frozen=True)
class TrainingSummary:
    run_id: str
    run_tag: str
    method_name: str
    model_name: str
    condition: str
    data_regime: str
    data_fraction: float
    target_modules: tuple[str, ...]
    target_layers: tuple[int, ...]
    layer_scope: str
    lora_rank: int | None
    lora_alpha: float | None
    lora_dropout: float | None
    notes: str | None
    title: str
    start_time: str
    end_time: str
    total_training_time_seconds: float
    epochs: int
    optimizer_steps: int
    steps_per_epoch: int
    train_examples: int
    validation_examples: int
    test_examples: int
    total_params: int
    trainable_params: int
    trainable_ratio: float
    idle_allocated_gb: float | None
    idle_reserved_gb: float | None
    model_loaded_allocated_gb: float | None
    model_loaded_reserved_gb: float | None
    peak_train_allocated_gb: float | None
    peak_train_reserved_gb: float | None
    final_train_loss: float
    best_checkpoint_dir: str
    best_checkpoint_kind: str
    best_checkpoint_percent: float
    best_epoch: int
    best_validation_accuracy: float
    best_validation_macro_f1: float
    best_validation_class_f1: dict[str, float]
    best_validation_loss: float
    final_validation_accuracy: float
    final_validation_macro_f1: float
    final_validation_class_f1: dict[str, float]
    final_validation_loss: float
    test_accuracy: float | None
    test_macro_f1: float | None
    test_class_f1: dict[str, float] | None
    test_loss: float | None
    test_invalid_rate: float | None
    train_val_gap_loss: float | None
    train_val_gap_accuracy: float | None
    train_val_gap_macro_f1: float | None
    total_checkpoint_size_bytes: int
    best_checkpoint_size_bytes: int
    final_checkpoint_size_bytes: int
    training_samples_per_second: float
    training_tokens_per_second: float
    training_seconds_per_step: float
    inference_examples_per_second: float
    inference_avg_latency_seconds: float
    inference_peak_allocated_gb: float | None
    inference_peak_reserved_gb: float | None


@dataclass(frozen=True)
class RunFiles:
    """Artifact identity and paths; holds no execution or numerical state."""

    output_root: Path
    run_id: str
    run_tag: str
    model_name: str
    condition: str

    @classmethod
    def from_config(cls, config):
        root = (
            config.output_dir
            / config.run_id
            / safe_name(config.model_name)
            / safe_name(config.condition)
        )
        return cls(
            root, config.run_id, config.run_tag, config.model_name, config.condition
        )

    @property
    def title(self):
        return (
            f"{self.run_id}__{safe_name(self.model_name)}__{safe_name(self.condition)}"
        )

    @property
    def checkpoints_dir(self):
        return self.output_root / "checkpoints"

    @property
    def logs_dir(self):
        return self.output_root / "logs"

    @property
    def evaluations_dir(self):
        return self.output_root / "evaluations"

    @property
    def layerwise_dir(self):
        return self.output_root / "layerwise_updates"

    @property
    def transitions_dir(self):
        return self.output_root / "prediction_transitions"

    def prepare(self):
        for directory in (
            self.checkpoints_dir,
            self.logs_dir,
            self.evaluations_dir,
            self.layerwise_dir,
            self.transitions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def write_run_config(files, config, environment, distributed) -> None:
    # LoRA's extra settings retain their historical metadata/analysis files;
    # config.json keeps exactly the shared FullFineTuneConfig field set.
    shared_fields = {item.name for item in fields(FullFineTuneConfig)}
    write_json(
        files.output_root / "config.json",
        {
            **{
                key: value
                for key, value in asdict(config).items()
                if key in shared_fields
            },
            "dtype": str(config.dtype).replace("torch.", ""),
            "environment": {"hf_token_set": bool(environment.hf_token)},
            "distributed": distributed,
        },
    )


def write_run_results(
    files,
    config,
    summary,
    distributed_runtime,
    distributed,
) -> None:
    write_json(files.output_root / "summary.json", asdict(summary))
    write_json(files.output_root / "distributed_runtime.json", distributed_runtime)
    write_json(
        files.output_root / "run_metadata.json",
        {
            "run_id": config.run_id,
            "run_tag": config.run_tag,
            "method_name": config.method_name,
            "condition": config.condition,
            "model_name": config.model_name,
            "data_regime": config.data_regime,
            "data_fraction": config.data_fraction,
            "target_modules": list(config.target_modules),
            "target_layers": list(config.target_layers),
            "layer_scope": config.layer_scope,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "notes": config.notes,
            "track_layerwise_updates": config.track_layerwise_updates,
            "analysis_schema_version": 2,
            "validation_loss_reduction": "token_mean",
            "training_loss_reduction": "microbatch_mean",
            "created_at": current_time_iso(),
            "distributed": distributed,
        },
    )


def write_checkpoint_index(
    files, checkpoint_records, best_checkpoint, test_metrics
) -> None:
    """Write the final artifact index only after analysis persistence succeeds."""
    write_json(
        files.output_root / "artifacts.json",
        {
            "checkpoints": [asdict(record) for record in checkpoint_records],
            "best_checkpoint": asdict(best_checkpoint),
        },
    )
    if test_metrics is not None:
        write_json(files.evaluations_dir / "test_summary.json", asdict(test_metrics))


def write_lora_results(files, adapter, summary) -> None:
    output_summary = {
        **asdict(summary),
        "adapter_params": summary.trainable_params,
        "adapter_checkpoint_size_bytes": summary.best_checkpoint_size_bytes,
        "merged_checkpoint_size_bytes": None,
        "lora_bias": adapter.lora_bias,
        "lora_task_type": adapter.lora_task_type,
        "modules_to_save": list(adapter.modules_to_save),
        "merge_for_eval": adapter.merge_for_eval,
    }
    write_json(files.output_root / "summary.json", output_summary)
    run_metadata_path = files.output_root / "run_metadata.json"
    if run_metadata_path.is_file():
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {}
    metadata.update(
        {
            "lora_bias": adapter.lora_bias,
            "lora_task_type": adapter.lora_task_type,
            "modules_to_save": list(adapter.modules_to_save),
            "merge_for_eval": adapter.merge_for_eval,
        }
    )
    write_json(run_metadata_path, metadata)
