#!/usr/bin/env python3
"""Run PubMedQA baseline, Full FT, and LoRA experiments from a central run registry."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from pubmedqa.config import EnvironmentConfig
from pubmedqa.experiments.orchestrator import (
    ExperimentStudy,
    execute_study,
    print_run_table as _print_run_table,
    to_jsonable as _jsonable,
    validate_target_layer_overrides as _validate_target_layer_overrides,
)
from pubmedqa.experiments.specs import (
    DEFAULT_BASELINE_OUTPUT_DIR,
    DEFAULT_PATHS,
    DEFAULT_SHARED_DEFAULTS,
    ExperimentPaths,
    SharedTrainDefaults,
    list_run_tags,
)
from pubmedqa.runtime.distributed import (
    destroy_process_groups as _destroy_distributed_process_groups,
    initialize_control_group as _initialize_distributed_control_group,
    run_on_rank_zero as _run_on_rank_zero,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Optional run identifier. Defaults to a timestamp-based value.")
    parser.add_argument("--runs", default="B0,F1,L1,L2,L3,L4,LL1")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--baseline-output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR)
    parser.add_argument("--train-output-dir", type=Path, default=Path("outputs/pubmedqa_train"))
    parser.add_argument("--train-file", type=Path, default=DEFAULT_PATHS.train_path)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_PATHS.validation_path)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_PATHS.test_path)
    parser.add_argument("--baseline-eval-file", type=Path, default=DEFAULT_PATHS.baseline_eval_path)
    parser.add_argument("--model-name", default=DEFAULT_SHARED_DEFAULTS.model_name)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_SHARED_DEFAULTS.num_epochs)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_SHARED_DEFAULTS.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_SHARED_DEFAULTS.eval_batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_SHARED_DEFAULTS.gradient_accumulation_steps,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Legacy override that sets both Full FT and LoRA learning rates.",
    )
    parser.add_argument(
        "--full-ft-learning-rate",
        type=float,
        default=DEFAULT_SHARED_DEFAULTS.full_ft_learning_rate,
    )
    parser.add_argument(
        "--lora-learning-rate",
        type=float,
        default=DEFAULT_SHARED_DEFAULTS.lora_learning_rate,
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_SHARED_DEFAULTS.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_SHARED_DEFAULTS.warmup_ratio)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_SHARED_DEFAULTS.max_grad_norm)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_SHARED_DEFAULTS.max_input_tokens)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_SHARED_DEFAULTS.max_new_tokens)
    parser.add_argument("--device", default=DEFAULT_SHARED_DEFAULTS.device)
    parser.add_argument(
        "--distributed-mode",
        choices=("single", "ddp", "fsdp"),
        default=DEFAULT_SHARED_DEFAULTS.distributed_mode,
    )
    parser.add_argument("--fsdp-cpu-offload", action="store_true", default=DEFAULT_SHARED_DEFAULTS.fsdp_cpu_offload)
    parser.add_argument("--dtype", default=DEFAULT_SHARED_DEFAULTS.dtype)
    parser.add_argument("--attn-implementation", default=DEFAULT_SHARED_DEFAULTS.attn_implementation)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_SHARED_DEFAULTS.cpu_threads)
    parser.add_argument("--log-every-steps", type=int, default=DEFAULT_SHARED_DEFAULTS.log_every_steps)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_SHARED_DEFAULTS.num_workers)
    parser.add_argument("--max-train-examples", type=int, default=DEFAULT_SHARED_DEFAULTS.max_train_examples)
    parser.add_argument(
        "--max-validation-examples",
        type=int,
        default=DEFAULT_SHARED_DEFAULTS.max_validation_examples,
    )
    parser.add_argument("--max-test-examples", type=int, default=DEFAULT_SHARED_DEFAULTS.max_test_examples)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=None,
        help="Legacy override that enables gradient checkpointing for both methods.",
    )
    parser.add_argument(
        "--full-ft-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHARED_DEFAULTS.full_ft_gradient_checkpointing,
    )
    parser.add_argument(
        "--lora-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHARED_DEFAULTS.lora_gradient_checkpointing,
    )
    parser.add_argument("--no-save-optimizer-state", action="store_true")
    parser.add_argument("--strict-parser", action="store_true")
    parser.add_argument("--checkpoint-percents", default="25,50,75,100")
    parser.add_argument(
        "--target-layer-override",
        action="append",
        default=[],
        metavar="RUN_TAG=LAYER[,LAYER...]",
        help="Inject selected LoRA layers for a selective run, for example LL1=20,21,22.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SHARED_DEFAULTS.seed)
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def _parse_run_tags(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_checkpoint_percents(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_target_layer_overrides(values: list[str]) -> dict[str, tuple[int, ...]]:
    overrides: dict[str, tuple[int, ...]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--target-layer-override must use RUN_TAG=LAYER[,LAYER...] format."
            )
        run_tag, raw_layers = value.split("=", maxsplit=1)
        normalized_tag = run_tag.strip()
        layers = tuple(sorted({int(item.strip()) for item in raw_layers.split(",") if item.strip()}))
        if not normalized_tag or not layers:
            raise ValueError(
                "--target-layer-override requires a run tag and at least one layer index."
            )
        overrides[normalized_tag] = layers
    return overrides


def _generate_run_id() -> str:
    return f"exp_{time.strftime('%Y%m%d_%H%M%S')}"


def _build_paths(args: argparse.Namespace) -> ExperimentPaths:
    return ExperimentPaths(
        train_path=args.train_file,
        validation_path=args.validation_file,
        test_path=args.test_file,
        baseline_eval_path=args.baseline_eval_file,
    )


def _build_defaults(args: argparse.Namespace) -> SharedTrainDefaults:
    return replace(
        DEFAULT_SHARED_DEFAULTS,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        full_ft_learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else args.full_ft_learning_rate
        ),
        lora_learning_rate=(
            args.learning_rate
            if args.learning_rate is not None
            else args.lora_learning_rate
        ),
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
        max_test_examples=args.max_test_examples,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=None if args.attn_implementation in {"", "none", "auto"} else args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        cpu_threads=args.cpu_threads,
        log_every_steps=args.log_every_steps,
        num_workers=args.num_workers,
        full_ft_gradient_checkpointing=(
            args.gradient_checkpointing
            if args.gradient_checkpointing is not None
            else args.full_ft_gradient_checkpointing
        ),
        lora_gradient_checkpointing=(
            args.gradient_checkpointing
            if args.gradient_checkpointing is not None
            else args.lora_gradient_checkpointing
        ),
        save_optimizer_state=not args.no_save_optimizer_state,
        strict_parser=args.strict_parser,
        checkpoint_percents=_parse_checkpoint_percents(args.checkpoint_percents),
        distributed_mode=args.distributed_mode,
        fsdp_cpu_offload=args.fsdp_cpu_offload,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    run_id = args.run_id or _generate_run_id()
    run_tags = _parse_run_tags(args.runs)

    if args.list_runs:
        _print_run_table(list_run_tags())
        return

    paths = _build_paths(args)
    defaults = _build_defaults(args)
    target_layer_overrides = _parse_target_layer_overrides(args.target_layer_override)
    environment = EnvironmentConfig(hf_token=args.hf_token) if args.hf_token else EnvironmentConfig.from_env()
    execute_study(
        ExperimentStudy(
            run_id=run_id,
            run_tags=tuple(run_tags),
            paths=paths,
            defaults=defaults,
            baseline_output_dir=args.baseline_output_dir,
            train_output_dir=args.train_output_dir,
            target_layer_overrides=target_layer_overrides,
            environment=environment,
            continue_on_error=args.continue_on_error,
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
