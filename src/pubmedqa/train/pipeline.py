"""The training program: prepare -> update -> validate -> persist -> test.

Full fine-tuning and LoRA differ at model/analysis preparation, not in loop
ownership. Runtime resources and small records have their own bounded state;
the model, optimizer, schedule and measurements remain local to this run.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict
from functools import partial

import torch
from transformers import get_linear_schedule_with_warmup

import pubmedqa.train.checkpoints as checkpoint_ops
import pubmedqa.train.full_ft as analysis_ops
import pubmedqa.train.lora as lora_analysis_ops
from pubmedqa.config import EnvironmentConfig
from pubmedqa.config.full_ft import FullFineTuneConfig, validate_training_config
from pubmedqa.config.lora import LoRAFineTuneConfig
from pubmedqa.data.records import current_time_iso, load_local_jsonl, write_jsonl
from pubmedqa.data.supervised import (
    PubMedQASupervisedDataset,
    SupervisedDataCollator,
    build_eval_dataloader,
    build_train_dataloader,
    validate_supervised_dataset,
)
from pubmedqa.eval.validation import (
    EvalMetrics,
    EvalPrediction,
    EvalResult,
    EvaluationSettings,
    evaluate_checkpoint_on_main,
    evaluate_split,
)
from pubmedqa.model.device import (
    count_parameters as _count_parameters,
)
from pubmedqa.model.device import (
    memory_snapshot as _memory_snapshot,
)
from pubmedqa.model.full_ft import load_full_model
from pubmedqa.model.loading import ModelLoadOptions
from pubmedqa.model.lora import AdapterOptions, load_lora_model
from pubmedqa.train.artifacts import (
    RunFiles,
    TrainingSummary,
    write_checkpoint_index,
    write_lora_results,
    write_run_config,
    write_run_results,
)
from pubmedqa.train.checkpoints import CheckpointRecord
from pubmedqa.train.checkpoints import (
    build_checkpoint_schedule as _build_checkpoint_schedule,
)
from pubmedqa.train.distributed import TrainingSession
from pubmedqa.train.loop import TrainingMemory, TrainStepLog, train_epoch


def run_training(
    config: FullFineTuneConfig,
    environment: EnvironmentConfig,
    *,
    session: TrainingSession | None = None,
) -> TrainingSummary:
    """Run one experiment; close owned resources on success and every failure."""
    session = session or TrainingSession.from_config(config)
    try:
        session.initialize()
        return _run_training(config, environment, session)
    finally:
        session.close()


def _run_training(config, environment, session):
    files = RunFiles.from_config(config)
    options = ModelLoadOptions(
        device=session.device,
        dtype=config.dtype,
        distributed_mode=config.distributed_mode,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
        hf_token=environment.hf_token,
    )
    adapter = (
        AdapterOptions.from_config(config)
        if isinstance(config, LoRAFineTuneConfig)
        else None
    )
    history = lora_analysis_ops.AdapterHistory()
    # Bind fixed, narrow inputs locally; these are not global strategy hooks.
    load_model = (
        partial(load_full_model, options=options)
        if adapter is None
        else partial(load_lora_model, options=options, adapter=adapter)
    )
    analysis = analysis_ops if adapter is None else lora_analysis_ops
    capture_references = partial(
        analysis.capture_layerwise_references,
        files=files,
        session=session,
        enabled=config.track_layerwise_updates,
    )
    write_updates = partial(
        analysis.write_layerwise_update_artifacts,
        files=files,
        session=session,
        enabled=config.track_layerwise_updates,
        **({} if adapter is None else {"history": history}),
    )
    save_checkpoint = partial(
        checkpoint_ops.save_checkpoint
        if adapter is None
        else checkpoint_ops.save_lora_checkpoint,
        files=files,
        session=session,
        save_optimizer_state=config.save_optimizer_state,
        **({} if adapter is None else {"adapter": adapter}),
    )
    record_checkpoint = partial(
        checkpoint_ops.record_checkpoint,
        session=session,
        save_checkpoint=save_checkpoint,
        write_updates=write_updates,
    )
    evaluation_settings = EvaluationSettings.from_config(config, session.device)
    evaluate = partial(
        evaluate_split,
        settings=evaluation_settings,
        evaluations_dir=files.evaluations_dir,
    )
    evaluate_checkpoint = partial(
        evaluate_checkpoint_on_main,
        session=session,
        settings=evaluation_settings,
        evaluations_dir=files.evaluations_dir,
        load_model=load_model,
    )
    create_snapshot = partial(
        checkpoint_ops.create_evaluation_snapshot, files=files, session=session
    )
    remove_snapshot = partial(
        checkpoint_ops.remove_evaluation_snapshot, session=session
    )
    build_train_loader = partial(
        build_train_dataloader,
        batch_size=config.train_batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        max_input_tokens=config.max_input_tokens,
        world_size=session.world_size,
        rank=session.rank,
    )
    build_eval_loader = partial(
        build_eval_dataloader,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        max_input_tokens=config.max_input_tokens,
    )

    def load_examples():
        return tuple(
            load_local_jsonl(path)[:limit] if path else []
            for path, limit in (
                (config.train_path, config.max_train_examples),
                (config.validation_path, config.max_validation_examples),
                (config.test_path, config.max_test_examples),
            )
        )

    def write_log(filename, records):
        session.run_on_main_process(
            lambda: write_jsonl(
                files.logs_dir / filename, (asdict(row) for row in records)
            ),
            operation_name=f"write {filename}",
        )

    def write_transitions(**predictions):
        session.run_on_main_process(
            lambda: analysis_ops.write_prediction_transition_artifacts(
                files=files, **predictions
            ),
            operation_name="write_prediction_transition_artifacts",
        )

    session.run_rank_local(
        lambda: validate_training_config(config),
        operation_name="config preflight",
    )
    session.run_on_main_process(
        files.prepare, operation_name="prepare output directories"
    )
    session.run_rank_local(session.set_runtime, operation_name="runtime setup")

    start_time = current_time_iso()
    run_started = time.perf_counter()
    idle_memory = _memory_snapshot(session.device)

    tokenizer, model = session.run_rank_local(
        lambda: load_model(config.model_name),
        operation_name="model load",
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    base_references = capture_references(model)
    total_params, trainable_params, trainable_ratio = _count_parameters(model)

    train_examples, validation_examples, test_examples = session.run_rank_local(
        load_examples,
        operation_name="dataset load",
    )

    def build_loaders():
        train_dataset = PubMedQASupervisedDataset(train_examples, tokenizer)
        validation_dataset = PubMedQASupervisedDataset(validation_examples, tokenizer)
        collator = SupervisedDataCollator(tokenizer, config.max_input_tokens)
        for dataset in (train_dataset, validation_dataset):
            validate_supervised_dataset(dataset, collator, config.train_batch_size)
        return (
            build_train_loader(train_dataset, tokenizer),
            build_eval_loader(validation_dataset, tokenizer),
        )

    (train_loader, train_sampler), validation_loader = session.run_rank_local(
        build_loaders, operation_name="data preflight"
    )

    reference_validation_result = session.run_on_main_process(
        lambda: evaluate(
            model=model,  # noqa: F821 - synchronous callback runs before the later del model.
            tokenizer=tokenizer,
            supervised_loader=validation_loader,
            examples=validation_examples,
            split_name="validation_reference",
        ),
        operation_name="reference validation",
    )

    model = session.wrap_model(model)

    loaded_memory = _memory_snapshot(session.device)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_optimizer_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    checkpoint_schedule = _build_checkpoint_schedule(
        total_optimizer_steps,
        config.checkpoint_percents,
    )
    next_schedule_index = 0

    session.run_on_main_process(
        lambda: write_run_config(files, config, environment, session.metadata()),
        operation_name="write_run_config",
    )

    if session.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(session.device)

    step_logs: list[TrainStepLog] = []
    checkpoint_records: list[CheckpointRecord] = []
    best_checkpoint: CheckpointRecord | None = None
    last_validation_metrics: EvalMetrics | None = None
    global_step = 0
    total_seen_samples = 0
    total_seen_tokens = 0
    train_loss = 0.0
    cumulative_train_loss_total = 0.0
    cumulative_train_loss_count = 0
    saved_checkpoint_steps: set[int] = set()
    previous_snapshots = {
        reference.parameter_name: reference.base_tensor.clone()
        for reference in base_references
    }
    previous_incremental_updates: dict[str, torch.Tensor] = {}
    training_memory = TrainingMemory(session.device)

    previous_validation_predictions: list[EvalPrediction] | None = None
    previous_validation_split_name: str | None = None

    checkpoint_records.append(
        record_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            step_in_epoch=0,
            global_step=0,
            checkpoint_kind="reference",
            checkpoint_percent=0.0,
            elapsed_seconds=0.0,
            train_loss=None,
            gradient_norm=None,
            validation_metrics=reference_validation_result.metrics,
            references=base_references,
            previous_snapshots=previous_snapshots,
            previous_incremental_updates={},
            save_model_files=False,
        )
    )
    last_validation_metrics = reference_validation_result.metrics
    previous_validation_predictions = reference_validation_result.predictions
    previous_validation_split_name = "validation_reference"

    for epoch in range(1, config.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss_total = 0.0
        epoch_loss_count = 0
        for step in train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            device=session.device,
            autocast_context=session.autocast_context,
            ddp_enabled=session.ddp_enabled,
            fsdp_enabled=session.fsdp_enabled,
            memory_tracker=training_memory,
            control_group=session.control_group,
        ):
            step_log = step.log
            global_step = step_log.global_step
            batch_index = step_log.step_in_epoch
            gradient_norm = step_log.gradient_norm
            epoch_loss_total += step.loss_total
            epoch_loss_count += step.micro_batches
            cumulative_train_loss_total += step.loss_total
            cumulative_train_loss_count += step.micro_batches
            total_seen_samples += step_log.batch_size
            total_seen_tokens += step_log.input_tokens
            step_logs.append(step_log)

            if config.log_every_steps > 0 and global_step % config.log_every_steps == 0:
                write_log("train_steps.jsonl", step_logs)

            while (
                next_schedule_index < len(checkpoint_schedule)
                and global_step >= checkpoint_schedule[next_schedule_index][1]
            ):
                checkpoint_percent, _ = checkpoint_schedule[next_schedule_index]
                split_name = f"validation_pct_{checkpoint_percent:03d}"
                checkpoint_elapsed = time.perf_counter() - run_started
                checkpoint_dir = save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step_in_epoch=batch_index,
                    global_step=global_step,
                    checkpoint_kind="scheduled",
                    checkpoint_percent=float(checkpoint_percent),
                    elapsed_seconds=checkpoint_elapsed,
                    validation_metrics=last_validation_metrics,
                    save_model_files=True,
                )
                validation_result = evaluate_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    examples=validation_examples,
                    split_name=split_name,
                )
                validation_metrics = validation_result.metrics
                last_validation_metrics = validation_metrics
                if (
                    previous_validation_predictions is not None
                    and previous_validation_split_name is not None
                ):
                    write_transitions(
                        from_split=previous_validation_split_name,
                        to_split=split_name,
                        previous_predictions=previous_validation_predictions,
                        current_predictions=validation_result.predictions,
                    )
                previous_validation_predictions = validation_result.predictions
                previous_validation_split_name = split_name
                checkpoint_record = record_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step_in_epoch=batch_index,
                    global_step=global_step,
                    checkpoint_kind="scheduled",
                    checkpoint_percent=float(checkpoint_percent),
                    elapsed_seconds=checkpoint_elapsed,
                    train_loss=(
                        cumulative_train_loss_total / cumulative_train_loss_count
                        if cumulative_train_loss_count > 0
                        else None
                    ),
                    gradient_norm=gradient_norm,
                    validation_metrics=validation_metrics,
                    references=base_references,
                    previous_snapshots=previous_snapshots,
                    previous_incremental_updates=previous_incremental_updates,
                    # Model files were saved before standalone validation.
                    save_model_files=False,
                )
                checkpoint_records.append(checkpoint_record)
                saved_checkpoint_steps.add(global_step)
                if best_checkpoint is None or checkpoint_ops._is_better_checkpoint(
                    checkpoint_record, best_checkpoint
                ):
                    best_checkpoint = checkpoint_record
                next_schedule_index += 1

        train_loss = epoch_loss_total / max(1, epoch_loss_count)
        validation_metrics: EvalMetrics | None = None
        epoch_checkpoint_saved = False
        if config.eval_every_epoch and (
            not checkpoint_records or checkpoint_records[-1].global_step != global_step
        ):
            split_name = f"validation_epoch_{epoch:03d}"
            checkpoint_percent = 100.0 * global_step / total_optimizer_steps
            checkpoint_elapsed = time.perf_counter() - run_started
            if config.save_every_epoch:
                evaluation_checkpoint_dir = save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step_in_epoch=len(train_loader),
                    global_step=global_step,
                    checkpoint_kind="epoch_end",
                    checkpoint_percent=checkpoint_percent,
                    elapsed_seconds=checkpoint_elapsed,
                    validation_metrics=last_validation_metrics,
                    save_model_files=True,
                )
                epoch_checkpoint_saved = True
            else:
                evaluation_checkpoint_dir = create_snapshot(
                    model=model,
                    tokenizer=tokenizer,
                    split_name=split_name,
                )
            try:
                validation_result = evaluate_checkpoint(
                    checkpoint_dir=evaluation_checkpoint_dir,
                    examples=validation_examples,
                    split_name=split_name,
                )
            finally:
                if not epoch_checkpoint_saved:
                    remove_snapshot(evaluation_checkpoint_dir)
            validation_metrics = validation_result.metrics
            last_validation_metrics = validation_metrics
            if (
                previous_validation_predictions is not None
                and previous_validation_split_name is not None
            ):
                write_transitions(
                    from_split=previous_validation_split_name,
                    to_split=split_name,
                    previous_predictions=previous_validation_predictions,
                    current_predictions=validation_result.predictions,
                )
            previous_validation_predictions = validation_result.predictions
            previous_validation_split_name = split_name

        if config.save_every_epoch and global_step not in saved_checkpoint_steps:
            if validation_metrics is None:
                raise RuntimeError("Checkpoint saving requires validation metrics.")
            checkpoint_record = record_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step_in_epoch=len(train_loader),
                global_step=global_step,
                checkpoint_kind="epoch_end",
                checkpoint_percent=100.0 * global_step / total_optimizer_steps,
                elapsed_seconds=time.perf_counter() - run_started,
                train_loss=train_loss,
                gradient_norm=step_logs[-1].gradient_norm if step_logs else None,
                validation_metrics=validation_metrics,
                references=base_references,
                previous_snapshots=previous_snapshots,
                previous_incremental_updates=previous_incremental_updates,
                # A persisted snapshot was created before validation.
                save_model_files=not epoch_checkpoint_saved,
            )
            checkpoint_records.append(checkpoint_record)
            if best_checkpoint is None or checkpoint_ops._is_better_checkpoint(
                checkpoint_record, best_checkpoint
            ):
                best_checkpoint = checkpoint_record
            saved_checkpoint_steps.add(global_step)

        write_log("train_steps.jsonl", step_logs)
        if checkpoint_records:
            write_log("checkpoints.jsonl", checkpoint_records)

    if last_validation_metrics is None:
        final_snapshot_dir = create_snapshot(
            model=model,
            tokenizer=tokenizer,
            split_name="validation",
        )
        try:
            final_validation_result = evaluate_checkpoint(
                checkpoint_dir=final_snapshot_dir,
                examples=validation_examples,
                split_name="validation",
            )
        finally:
            remove_snapshot(final_snapshot_dir)
        last_validation_metrics = final_validation_result.metrics
        if (
            previous_validation_predictions is not None
            and previous_validation_split_name is not None
        ):
            write_transitions(
                from_split=previous_validation_split_name,
                to_split="validation",
                previous_predictions=previous_validation_predictions,
                current_predictions=final_validation_result.predictions,
            )

    if best_checkpoint is None:
        checkpoint_record = record_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config.num_epochs,
            step_in_epoch=len(train_loader),
            global_step=global_step,
            checkpoint_kind="final",
            checkpoint_percent=100.0,
            elapsed_seconds=time.perf_counter() - run_started,
            train_loss=train_loss,
            gradient_norm=step_logs[-1].gradient_norm if step_logs else None,
            validation_metrics=last_validation_metrics,
            references=base_references,
            previous_snapshots=previous_snapshots,
            previous_incremental_updates=previous_incremental_updates,
            save_model_files=True,
        )
        checkpoint_records.append(checkpoint_record)
        best_checkpoint = checkpoint_record
        write_log("checkpoints.jsonl", checkpoint_records)

    final_validation_metrics = last_validation_metrics
    del optimizer
    del scheduler
    del model
    gc.collect()
    if session.device.type == "cuda":
        torch.cuda.empty_cache()

    test_metrics: EvalMetrics | None = None
    if test_examples:

        def evaluate_best_checkpoint() -> EvalResult:
            best_tokenizer, best_model = load_model(best_checkpoint.checkpoint_dir)
            try:
                best_test_dataset = PubMedQASupervisedDataset(
                    test_examples, best_tokenizer
                )
                best_test_loader = build_eval_loader(best_test_dataset, best_tokenizer)
                return evaluate(
                    model=best_model,
                    tokenizer=best_tokenizer,
                    supervised_loader=best_test_loader,
                    examples=test_examples,
                    split_name="test",
                )
            finally:
                del best_model
                gc.collect()
                if session.device.type == "cuda":
                    torch.cuda.empty_cache()

        test_result = session.run_on_main_process(
            evaluate_best_checkpoint,
            operation_name="best-checkpoint test evaluation",
        )
        test_metrics = test_result.metrics

    end_time = current_time_iso()
    total_elapsed = time.perf_counter() - run_started
    total_seen_samples = session.all_reduce_int(total_seen_samples)
    total_seen_tokens = session.all_reduce_int(total_seen_tokens)
    total_checkpoint_size = sum(
        record.checkpoint_size_bytes for record in checkpoint_records
    )
    final_checkpoint_size = checkpoint_records[-1].checkpoint_size_bytes
    best_checkpoint_size = best_checkpoint.checkpoint_size_bytes
    peak_train_memory = training_memory.snapshot()
    distributed_runtime = session.collect_memory(
        idle_memory=idle_memory,
        loaded_memory=loaded_memory,
        peak_train_memory=peak_train_memory,
    )

    summary = TrainingSummary(
        run_id=config.run_id,
        run_tag=config.run_tag,
        method_name=config.method_name,
        model_name=config.model_name,
        condition=config.condition,
        data_regime=config.data_regime,
        data_fraction=config.data_fraction,
        target_modules=config.target_modules,
        target_layers=config.target_layers,
        layer_scope=config.layer_scope,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        notes=config.notes,
        title=files.title,
        start_time=start_time,
        end_time=end_time,
        total_training_time_seconds=total_elapsed,
        epochs=config.num_epochs,
        optimizer_steps=global_step,
        steps_per_epoch=steps_per_epoch,
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        test_examples=len(test_examples),
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_ratio=trainable_ratio,
        idle_allocated_gb=idle_memory["allocated_gb"],
        idle_reserved_gb=idle_memory["reserved_gb"],
        model_loaded_allocated_gb=loaded_memory["allocated_gb"],
        model_loaded_reserved_gb=loaded_memory["reserved_gb"],
        peak_train_allocated_gb=peak_train_memory["max_allocated_gb"],
        peak_train_reserved_gb=peak_train_memory["max_reserved_gb"],
        final_train_loss=train_loss,
        best_checkpoint_dir=best_checkpoint.checkpoint_dir,
        best_checkpoint_kind=best_checkpoint.checkpoint_kind,
        best_checkpoint_percent=best_checkpoint.checkpoint_percent,
        best_epoch=best_checkpoint.epoch,
        best_validation_accuracy=best_checkpoint.validation_accuracy,
        best_validation_macro_f1=best_checkpoint.validation_macro_f1,
        best_validation_class_f1=best_checkpoint.validation_class_f1,
        best_validation_loss=best_checkpoint.validation_loss,
        final_validation_accuracy=final_validation_metrics.accuracy,
        final_validation_macro_f1=final_validation_metrics.macro_f1,
        final_validation_class_f1=final_validation_metrics.class_f1,
        final_validation_loss=final_validation_metrics.loss,
        test_accuracy=None if test_metrics is None else test_metrics.accuracy,
        test_macro_f1=None if test_metrics is None else test_metrics.macro_f1,
        test_class_f1=None if test_metrics is None else test_metrics.class_f1,
        test_loss=None if test_metrics is None else test_metrics.loss,
        test_invalid_rate=None if test_metrics is None else test_metrics.invalid_rate,
        train_val_gap_loss=train_loss - final_validation_metrics.loss,
        train_val_gap_accuracy=None,
        train_val_gap_macro_f1=None,
        total_checkpoint_size_bytes=total_checkpoint_size,
        best_checkpoint_size_bytes=best_checkpoint_size,
        final_checkpoint_size_bytes=final_checkpoint_size,
        training_samples_per_second=(total_seen_samples / total_elapsed)
        if total_elapsed > 0
        else 0.0,
        training_tokens_per_second=(total_seen_tokens / total_elapsed)
        if total_elapsed > 0
        else 0.0,
        training_seconds_per_step=(total_elapsed / global_step)
        if global_step > 0
        else 0.0,
        inference_examples_per_second=final_validation_metrics.examples_per_second,
        inference_avg_latency_seconds=final_validation_metrics.avg_latency_seconds,
        inference_peak_allocated_gb=final_validation_metrics.peak_allocated_gb,
        inference_peak_reserved_gb=final_validation_metrics.peak_reserved_gb,
    )

    # Only persistence is rank-zero work; tensor collectives have already finished.
    def write_results():
        write_run_results(
            files,
            config,
            summary,
            distributed_runtime,
            session.metadata(),
        )
        if adapter is None:
            analysis_ops.write_analysis_groups(
                files,
                summary,
                checkpoint_records,
                step_logs,
                checkpoint_percents=config.checkpoint_percents,
            )
        else:
            lora_analysis_ops.write_analysis_groups(
                files,
                summary,
                checkpoint_records,
                step_logs,
                checkpoint_percents=config.checkpoint_percents,
                adapter=adapter,
                history=history,
                enabled=config.track_layerwise_updates,
            )
        write_checkpoint_index(files, checkpoint_records, best_checkpoint, test_metrics)
        if adapter is not None:
            write_lora_results(files, adapter, summary)

    session.run_on_main_process(write_results, operation_name="write_run_results")
    return summary
