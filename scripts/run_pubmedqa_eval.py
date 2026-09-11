#!/usr/bin/env python3
"""Run torch-based PubMedQA evaluation for a selected base model."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import torch

from pubmedqa.config.env import env_bool, env_int, env_optional_int
from pubmedqa.config.env import EnvironmentConfig
from pubmedqa.config.eval import EVAL_CONFIG, ModelRuntimeConfig
from pubmedqa.data.records import load_local_jsonl, write_json
from pubmedqa.eval.backends import (
    DEFAULT_MLX_MODEL_MAP,
    is_apple_silicon,
    resolve_attention,
    resolve_backend,
    resolve_batch_size,
    resolve_torch_device,
    resolve_torch_dtype,
)
from pubmedqa.eval.inference import PubMedQAEvaluationRunner
from pubmedqa.model.device import configure_parallelism


@dataclass(frozen=True)
class EnvironmentBatchConfig:
    """Legacy environment inputs for the multi-model evaluation command."""

    test_path: Path
    output_dir: Path
    models: tuple[str, ...]
    expected_test_size: int | None
    condition: str
    run_id: str
    runtime_defaults: ModelRuntimeConfig
    environment: EnvironmentConfig

    @classmethod
    def from_env(cls) -> "EnvironmentBatchConfig":
        test_path_raw = os.getenv(EVAL_CONFIG.env_test_path)
        if not test_path_raw:
            raise RuntimeError(
                f"{EVAL_CONFIG.env_test_path} is required for --from-env evaluation."
            )

        expected_raw = (
            os.getenv(
                EVAL_CONFIG.env_expected_test_size,
                str(EVAL_CONFIG.default_expected_test_size),
            )
            .strip()
            .lower()
        )
        expected_size = (
            None if expected_raw in {"", "none", "off"} else int(expected_raw)
        )
        raw_models = os.getenv(EVAL_CONFIG.env_models)
        models = (
            tuple(part.strip() for part in raw_models.split(",") if part.strip())
            if raw_models
            else EVAL_CONFIG.default_base_models
        )
        if not models:
            raise ValueError(f"{EVAL_CONFIG.env_models} contains no model names.")

        attention = (
            os.getenv(
                EVAL_CONFIG.env_attn_implementation,
                EVAL_CONFIG.default_attn_implementation,
            )
            .strip()
            .lower()
        )
        batch_raw = os.getenv(EVAL_CONFIG.env_batch_size)
        runtime = ModelRuntimeConfig(
            model_name=models[0],
            backend=os.getenv(EVAL_CONFIG.env_backend, EVAL_CONFIG.default_backend)
            .strip()
            .lower(),
            device=os.getenv(EVAL_CONFIG.env_device, EVAL_CONFIG.default_device)
            .strip()
            .lower(),
            dtype=os.getenv(EVAL_CONFIG.env_dtype, EVAL_CONFIG.default_dtype)
            .strip()
            .lower(),
            max_new_tokens=env_int(
                EVAL_CONFIG.env_max_new_tokens, EVAL_CONFIG.default_max_new_tokens
            ),
            batch_size=int(batch_raw) if batch_raw else None,
            max_input_tokens=env_optional_int(EVAL_CONFIG.env_max_input_tokens),
            attn_implementation=None if attention in {"", "none"} else attention,
            trust_remote_code=env_bool(EVAL_CONFIG.env_trust_remote_code, False),
            cpu_threads=env_int(
                EVAL_CONFIG.env_cpu_threads, EVAL_CONFIG.default_cpu_threads
            ),
            strict_parser=env_bool(
                EVAL_CONFIG.env_strict_parser, EVAL_CONFIG.default_strict_parser
            ),
        )
        return cls(
            test_path=Path(test_path_raw),
            output_dir=Path(
                os.getenv(
                    EVAL_CONFIG.env_output_dir, str(EVAL_CONFIG.default_output_dir)
                )
            ),
            models=models,
            expected_test_size=expected_size,
            condition=os.getenv(
                EVAL_CONFIG.env_condition, EVAL_CONFIG.default_condition
            ),
            run_id=os.getenv(EVAL_CONFIG.env_run_id)
            or datetime.now().strftime("%Y%m%d_%H%M%S"),
            runtime_defaults=runtime,
            environment=EnvironmentConfig.from_env(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-env",
        action="store_true",
        help="Evaluate PUBMEDQA_MODELS sequentially using PUBMEDQA_* variables.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        required=False,
        help="Local JSONL split file, for example data/processed/pqa_labeled/test.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--run-id", help="Run identifier used in output paths and titles."
    )
    parser.add_argument(
        "--model-name",
        required=False,
        help="Base model to evaluate.",
    )
    parser.add_argument(
        "--condition", help="Condition label such as baseline, full-ft, or lora."
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token injected at runtime if required.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Deprecated alias for --device, retained for command compatibility.",
    )
    parser.add_argument(
        "--torch-dtype", default="bfloat16", choices=("float16", "bfloat16", "float32")
    )
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--strict-parser", action="store_true")
    parser.add_argument("--expected-size", type=int, default=None)
    return parser.parse_args()


def build_runtime(args: argparse.Namespace) -> ModelRuntimeConfig:
    """Translate CLI values into the public inference runtime contract."""

    return ModelRuntimeConfig(
        model_name=args.model_name,
        backend="torch",
        device=args.device_map or args.device,
        dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        attn_implementation=args.attn_implementation,
        cpu_threads=args.cpu_threads,
        trust_remote_code=args.trust_remote_code,
        strict_parser=args.strict_parser,
    )


def print_runtime(runtime: ModelRuntimeConfig) -> None:
    """Print the resolved backend immediately before an evaluation run."""
    backend = resolve_backend(runtime.backend)
    print(f"[runtime] backend={backend}")
    print(f"[runtime] apple_silicon={is_apple_silicon()}")
    print(f"[runtime] cuda_available={torch.cuda.is_available()}")
    print(f"[runtime] mps_available={torch.backends.mps.is_available()}")
    if backend == "torch":
        device = resolve_torch_device(runtime.device)
        dtype = resolve_torch_dtype(device, runtime.dtype)
        attention = resolve_attention(device, runtime.attn_implementation)
        print(f"[runtime] device={device}")
        if device.type == "cuda":
            print(f"[runtime] gpu={torch.cuda.get_device_name(0)}")
        print(f"[runtime] dtype={dtype}")
        print(f"[runtime] attention={attention or 'model-default'}")
    else:
        print("[runtime] device=mlx")
        print("[runtime] dtype=model-native")
    print(f"[runtime] cpu_threads={runtime.cpu_threads}")


def run_environment_batch(config: EnvironmentBatchConfig) -> None:
    """Evaluate models sequentially so each backend releases device memory first."""
    configure_parallelism(config.runtime_defaults.cpu_threads)
    print_runtime(config.runtime_defaults)
    print(f"[runtime] hf_token_set={bool(config.environment.hf_token)}")
    examples = load_local_jsonl(config.test_path, config.expected_test_size)
    print(f"[dataset] path={config.test_path}")
    print(f"[dataset] num_examples={len(examples)}")

    summaries = []
    for model_name in config.models:
        runtime = replace(config.runtime_defaults, model_name=model_name)
        backend = resolve_backend(runtime.backend)
        if backend == "torch":
            device = resolve_torch_device(runtime.device)
            batch_size = resolve_batch_size(
                model_name, "torch", device.type, runtime.batch_size
            )
        else:
            batch_size = 1
        print(f"\n[model] evaluating {model_name}")
        print(f"[model] backend={backend} batch_size={batch_size}")
        if backend == "mlx":
            print(
                f"[model] resolved_model={DEFAULT_MLX_MODEL_MAP.get(model_name, model_name)}"
            )

        runner = PubMedQAEvaluationRunner(
            run_id=config.run_id,
            model_name=model_name,
            condition=config.condition,
            output_dir=config.output_dir,
            runtime=runtime,
            environment=config.environment,
        )
        items, summary = runner.evaluate(examples)
        run_dir = runner.save_results(items, summary)
        summaries.append(asdict(summary))
        print(
            f"[result] acc={summary.accuracy:.4f} "
            f"macro_f1={summary.macro_f1:.4f} "
            f"invalid={summary.invalid_rate:.4f} "
            f"time={summary.elapsed_seconds:.1f}s "
            f"examples/s={summary.examples_per_second:.2f} "
            f"saved={run_dir}"
        )
    write_json(config.output_dir / config.run_id / "all_models_summary.json", summaries)


def main() -> None:
    args = parse_args()
    if args.from_env:
        run_environment_batch(EnvironmentBatchConfig.from_env())
        return
    missing = [
        name
        for name in ("data_file", "run_id", "model_name", "condition")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Single-model evaluation requires: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    configure_parallelism(args.cpu_threads)

    runtime = build_runtime(args)
    runner = PubMedQAEvaluationRunner(
        run_id=args.run_id,
        model_name=args.model_name,
        condition=args.condition,
        output_dir=args.output_dir,
        runtime=runtime,
        environment=EnvironmentConfig(hf_token=args.hf_token),
    )
    examples = load_local_jsonl(args.data_file, expected_size=args.expected_size)
    items, summary = runner.evaluate(examples)
    run_dir = runner.save_results(items, summary)

    print(summary.title)
    print(f"Saved to: {run_dir}")
    print(f"ACC: {summary.accuracy:.4f}")
    print(f"Macro F1: {summary.macro_f1:.4f}")
    print(f"Invalid rate: {summary.invalid_rate:.4f}")


if __name__ == "__main__":
    main()
