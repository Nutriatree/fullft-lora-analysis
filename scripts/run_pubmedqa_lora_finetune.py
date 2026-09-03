#!/usr/bin/env python3
"""Run torch-based LoRA fine-tuning for PubMedQA."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubmedqa.evaluation import EnvironmentConfig
from pubmedqa.lora_finetune import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_CHECKPOINT_PERCENTS,
    DEFAULT_CONDITION,
    DEFAULT_CPU_THREADS,
    DEFAULT_DATA_FRACTION,
    DEFAULT_DATA_REGIME,
    DEFAULT_DEVICE,
    DEFAULT_DISTRIBUTED_MODE,
    DEFAULT_DTYPE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FSDP_CPU_OFFLOAD,
    DEFAULT_GRAD_ACCUM_STEPS,
    DEFAULT_LAYER_SCOPE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY_STEPS,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_BIAS,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_MERGE_FOR_EVAL,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_LORA_TASK_TYPE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_METHOD_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_NOTES,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_TAG,
    DEFAULT_SAVE_OPTIMIZER_STATE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    LoRAFineTuneConfig,
    PubMedQALoRAFineTuner,
    _load_target_layers,
    _normalize_lora_target_modules,
    _resolve_dtype,
)

CLI_DEFAULT_DTYPE = "bfloat16" if DEFAULT_DTYPE == "bf16" else DEFAULT_DTYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-regime", default=DEFAULT_DATA_REGIME)
    parser.add_argument("--data-fraction", type=float, default=DEFAULT_DATA_FRACTION)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--distributed-mode", choices=("single", "fsdp"), default=DEFAULT_DISTRIBUTED_MODE)
    parser.add_argument("--fsdp-cpu-offload", action="store_true", default=DEFAULT_FSDP_CPU_OFFLOAD)
    parser.add_argument("--dtype", default=CLI_DEFAULT_DTYPE, choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--attn-implementation", default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--log-every-steps", type=int, default=DEFAULT_LOG_EVERY_STEPS)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-save-optimizer-state", action="store_true")
    parser.add_argument("--strict-parser", action="store_true")
    parser.add_argument("--target-modules", default=",".join(DEFAULT_LORA_TARGET_MODULES))
    parser.add_argument("--target-layers", default="")
    parser.add_argument("--target-layers-file", default=None)
    parser.add_argument("--layer-scope", default=DEFAULT_LAYER_SCOPE)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--lora-bias", default=DEFAULT_LORA_BIAS)
    parser.add_argument("--lora-task-type", default=DEFAULT_LORA_TASK_TYPE)
    parser.add_argument("--modules-to-save", default="")
    parser.add_argument("--merge-for-eval", action="store_true", default=DEFAULT_LORA_MERGE_FOR_EVAL)
    parser.add_argument("--notes", default=DEFAULT_NOTES)
    parser.add_argument("--no-track-layerwise-updates", action="store_true")
    parser.add_argument("--checkpoint-percents", default="25,50,75,100")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LoRAFineTuneConfig(
        run_id=args.run_id,
        run_tag=args.run_tag,
        method_name=args.method_name,
        model_name=args.model_name,
        condition=args.condition,
        data_regime=args.data_regime,
        data_fraction=args.data_fraction,
        train_path=args.train_file,
        validation_path=args.validation_file,
        test_path=args.test_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=_resolve_dtype(args.dtype),
        attn_implementation=None if args.attn_implementation in {"", "none", "auto"} else args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        cpu_threads=args.cpu_threads,
        log_every_steps=args.log_every_steps,
        save_every_epoch=True,
        eval_every_epoch=True,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
        max_test_examples=args.max_test_examples,
        num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        save_optimizer_state=DEFAULT_SAVE_OPTIMIZER_STATE and not args.no_save_optimizer_state,
        strict_parser=args.strict_parser,
        seed=args.seed,
        target_modules=_normalize_lora_target_modules(
            tuple(part.strip() for part in args.target_modules.split(",") if part.strip())
        ),
        target_layers=_load_target_layers(args.target_layers, args.target_layers_file),
        layer_scope=args.layer_scope,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        lora_task_type=args.lora_task_type,
        modules_to_save=tuple(part.strip() for part in args.modules_to_save.split(",") if part.strip()),
        merge_for_eval=args.merge_for_eval,
        notes=args.notes,
        track_layerwise_updates=not args.no_track_layerwise_updates,
        checkpoint_percents=tuple(
            int(part.strip()) for part in args.checkpoint_percents.split(",") if part.strip()
        ) or DEFAULT_CHECKPOINT_PERCENTS,
        distributed_mode=args.distributed_mode,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
    )
    trainer = PubMedQALoRAFineTuner(config, EnvironmentConfig(hf_token=args.hf_token))
    summary = trainer.run()
    print(summary.title)
    print(f"Best checkpoint: {summary.best_checkpoint_dir}")
    print(f"Best validation ACC: {summary.best_validation_accuracy:.4f}")
    print(f"Best validation Macro F1: {summary.best_validation_macro_f1:.4f}")
    print(f"Final train loss: {summary.final_train_loss:.4f}")
    print(f"Final validation loss: {summary.final_validation_loss:.4f}")
    if summary.test_accuracy is not None:
        print(f"Test ACC: {summary.test_accuracy:.4f}")
        print(f"Test Macro F1: {summary.test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
