"""Public training records shared by Full FT and LoRA."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainStepLog:
    global_step: int
    epoch: int
    step_in_epoch: int
    learning_rate: float
    train_loss: float
    gradient_norm: float
    batch_size: int
    input_tokens: int
    target_tokens: int
    step_time_seconds: float
    samples_per_second: float
    tokens_per_second: float


@dataclass(frozen=True)
class EvalPrediction:
    pubid: str
    gold_label: str
    predicted_label: str | None
    response_text: str
    parse_method: str | None
    parse_error: str | None


@dataclass(frozen=True)
class EvalMetrics:
    split: str
    loss: float
    accuracy: float
    macro_f1: float
    class_f1: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    label_order: tuple[str, ...]
    invalid_rate: float
    num_examples: int
    num_parsed: int
    elapsed_seconds: float
    avg_latency_seconds: float
    examples_per_second: float
    tokens_per_second: float
    peak_allocated_gb: float | None
    peak_reserved_gb: float | None


@dataclass(frozen=True)
class EvalResult:
    metrics: EvalMetrics
    predictions: list[EvalPrediction]


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_kind: str
    checkpoint_percent: float
    epoch: int
    step_in_epoch: int
    global_step: int
    elapsed_seconds: float
    checkpoint_dir: str
    checkpoint_size_bytes: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float
    validation_class_f1: dict[str, float]
    validation_invalid_rate: float
    learning_rate: float
    gradient_norm: float
    peak_allocated_gb: float | None
    peak_reserved_gb: float | None


@dataclass(frozen=True)
class LayerwiseReference:
    parameter_name: str
    layer_index: int | None
    module_name: str
    component_name: str
    shape: tuple[int, ...]
    num_parameters: int
    base_weight_norm: float
    base_tensor: torch.Tensor


@dataclass(frozen=True)
class LayerwiseUpdateRecord:
    checkpoint_kind: str
    checkpoint_percent: float
    epoch: int
    global_step: int
    layer_index: int | None
    module_name: str
    component_name: str
    parameter_name: str
    shape: tuple[int, ...]
    num_parameters: int
    base_weight_norm: float
    current_weight_norm: float
    update_norm: float
    relative_update_norm: float
    incremental_update_norm: float
    relative_incremental_update_norm: float
    incremental_update_cosine_similarity: float | None
    cumulative_update_share: float
    incremental_update_share: float


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
