#!/usr/bin/env python3
"""Inspect PubMedQA prompts and parser behavior before model dry runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pubmedqa.answer_parser import parse_pubmedqa_answer
from pubmedqa.prompt_builder import build_messages, build_plain_prompt, example_from_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/processed/pqa_labeled/test.jsonl"),
        help="Generated PubMedQA JSONL split file.",
    )
    parser.add_argument("--index", type=int, default=0, help="Zero-based example index to inspect.")
    parser.add_argument(
        "--format",
        choices=("chat", "plain"),
        default="chat",
        help="Show chat messages or the plain fallback prompt.",
    )
    parser.add_argument(
        "--parse",
        dest="parse_text",
        default=None,
        help="Optional model output text to parse as a PubMedQA prediction.",
    )
    return parser.parse_args()


def load_record(path: Path, index: int) -> dict:
    with path.open() as f:
        for line_no, line in enumerate(f):
            if line_no == index:
                return json.loads(line)
    raise IndexError(f"{path} has fewer than {index + 1} records")


def main() -> None:
    args = parse_args()
    record = load_record(args.input_file, args.index)
    example = example_from_record(record)

    if args.format == "chat":
        print(json.dumps(build_messages(example), indent=2, ensure_ascii=False))
    else:
        print(build_plain_prompt(example))

    if example.final_decision is not None:
        print(f"\nGold: {example.final_decision}")

    if args.parse_text is not None:
        result = parse_pubmedqa_answer(args.parse_text)
        print("\nParse result:")
        print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
