#!/usr/bin/env python3
"""Run torch-based full fine-tuning for PubMedQA."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubmedqa.config import EnvironmentConfig
from pubmedqa.config.full_ft import TRAIN_FULL_FINE_TUNE_CONFIG, TRAIN_LAYER_CONFIG, parse_checkpoint_percents
from pubmedqa.model.device import resolve_dtype as _resolve_dtype
from pubmedqa.config.full_ft import FullFineTuneConfig
from pubmedqa.train.pipeline import run_training

CLI_DEFAULT_DTYPE = (
    "bfloat16"
    if TRAIN_FULL_FINE_TUNE_CONFIG.default_dtype == "bf16"
    else TRAIN_FULL_FINE_TUNE_CONFIG.default_dtype
)
DEFAULT_CHECKPOINT_PERCENTS = TRAIN_LAYER_CONFIG.default_checkpoint_percents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_output_dir)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-tag", default=TRAIN_FULL_FINE_TUNE_CONFIG.default_run_tag)
    parser.add_argument("--method-name", default=TRAIN_FULL_FINE_TUNE_CONFIG.default_method_name)
    parser.add_argument("--condition", default=TRAIN_FULL_FINE_TUNE_CONFIG.default_condition)
    parser.add_argument("--model-name", default=TRAIN_FULL_FINE_TUNE_CONFIG.default_model_name)
    parser.add_argument("--data-regime", default=TRAIN_FULL_FINE_TUNE_CONFIG.default_data_regime)
    parser.add_argument("--data-fraction", type=float, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_data_fraction)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--num-epochs", type=int, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_num_epochs)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_eval_batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=TRAIN_FULL_FINE_TUNE_CONFIG.default_grad_accum_steps,
    )
    parser.add_argument("--learning-rate", type=float, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_warmup_ratio)
    parser.add_argument("--max-grad-norm", type=float, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_max_grad_norm)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_max_new_tokens)
    parser.add_argument("--device", default=TRAIN_FULL_FINE_TUNE_CONFIG.default_device)
    parser.add_argument(
        "--distributed-mode",
        choices=("single", "ddp", "fsdp"),
        default=TRAIN_FULL_FINE_TUNE_CONFIG.default_distributed_mode,
    )
    parser.add_argument(
        "--fsdp-cpu-offload",
        action="store_true",
        default=TRAIN_FULL_FINE_TUNE_CONFIG.default_fsdp_cpu_offload,
    )
    parser.add_argument("--dtype", default=CLI_DEFAULT_DTYPE, choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--attn-implementation", default=TRAIN_FULL_FINE_TUNE_CONFIG.default_attn_implementation)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_cpu_threads)
    parser.add_argument("--log-every-steps", type=int, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_log_every_steps)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_num_workers)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-save-optimizer-state", action="store_true")
    parser.add_argument("--strict-parser", action="store_true")
    parser.add_argument("--target-modules", default="")
    parser.add_argument("--target-layers", default="")
    parser.add_argument("--layer-scope", default=TRAIN_LAYER_CONFIG.default_layer_scope)
    parser.add_argument("--lora-rank", type=int, default=TRAIN_LAYER_CONFIG.default_lora_rank)
    parser.add_argument("--lora-alpha", type=float, default=TRAIN_LAYER_CONFIG.default_lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=TRAIN_LAYER_CONFIG.default_lora_dropout)
    parser.add_argument("--notes", default=TRAIN_LAYER_CONFIG.default_notes)
    parser.add_argument("--no-track-layerwise-updates", action="store_true")
    parser.add_argument(
        "--checkpoint-percents",
        type=lambda raw: parse_checkpoint_percents(raw, DEFAULT_CHECKPOINT_PERCENTS),
        default=DEFAULT_CHECKPOINT_PERCENTS,
    )
    parser.add_argument("--seed", type=int, default=TRAIN_FULL_FINE_TUNE_CONFIG.default_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FullFineTuneConfig(
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
        save_optimizer_state=TRAIN_FULL_FINE_TUNE_CONFIG.default_save_optimizer_state and not args.no_save_optimizer_state,
        strict_parser=args.strict_parser,
        seed=args.seed,
        target_modules=tuple(part.strip() for part in args.target_modules.split(",") if part.strip()),
        target_layers=tuple(int(part.strip()) for part in args.target_layers.split(",") if part.strip()),
        layer_scope=args.layer_scope,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        notes=args.notes,
        track_layerwise_updates=not args.no_track_layerwise_updates,
        checkpoint_percents=args.checkpoint_percents,
        distributed_mode=args.distributed_mode,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
    )
    summary = run_training(config, EnvironmentConfig(hf_token=args.hf_token))

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
