#!/usr/bin/env python3
"""Run torch-based PubMedQA evaluation for a selected base model."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubmedqa.config import EnvironmentConfig
from pubmedqa.eval import ModelRuntimeConfig
from pubmedqa.eval.inference import (
    DEFAULT_BASE_MODELS,
    PubMedQAEvaluationRunner,
    load_local_jsonl,
)
from pubmedqa.model.device import configure_parallelism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Local JSONL split file, for example data/processed/pqa_labeled/test.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id", required=True, help="Run identifier used in output paths and titles.")
    parser.add_argument(
        "--model-name",
        required=True,
        choices=list(DEFAULT_BASE_MODELS),
        help="Base model to evaluate.",
    )
    parser.add_argument("--condition", required=True, help="Condition label such as baseline, full-ft, or lora.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token injected at runtime if required.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Torch device such as auto, cuda:0, mps, or cpu.")
    parser.add_argument(
        "--device-map",
        default=None,
        help="Deprecated alias for --device, retained for command compatibility.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
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


def main() -> None:
    args = parse_args()
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
