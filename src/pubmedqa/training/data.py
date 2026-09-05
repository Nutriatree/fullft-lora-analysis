"""Supervised training examples, datasets, and collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from pubmedqa.domain.prompts import PubMedQAExample, build_tokenizer_prompt


def longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True)
class SupervisedExample:
    pubid: str
    prompt_text: str
    full_text: str
    label: str


class PubMedQASupervisedDataset(Dataset[SupervisedExample]):
    def __init__(self, examples: Sequence[PubMedQAExample], tokenizer: Any) -> None:
        self._examples = [
            SupervisedExample(
                pubid=example.pubid,
                prompt_text=build_tokenizer_prompt(tokenizer, example, include_answer=False),
                full_text=build_tokenizer_prompt(tokenizer, example, include_answer=True),
                label=example.final_decision or "",
            )
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> SupervisedExample:
        return self._examples[index]


class SupervisedDataCollator:
    """Tokenize examples and mask every non-assistant target token."""

    def __init__(self, tokenizer: Any, max_length: int | None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[SupervisedExample]) -> dict[str, torch.Tensor]:
        tokenize_kwargs: dict[str, Any] = {
            "padding": True,
            "truncation": self.max_length is not None,
            "return_tensors": "pt",
        }
        if self.max_length is not None:
            tokenize_kwargs["max_length"] = self.max_length

        prompt_batch = self.tokenizer(
            [item.prompt_text for item in batch],
            **tokenize_kwargs,
        )
        full_batch = self.tokenizer(
            [item.full_text for item in batch],
            **tokenize_kwargs,
        )

        input_ids = full_batch["input_ids"]
        attention_mask = full_batch["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        for row_index in range(input_ids.shape[0]):
            prompt_length = int(prompt_batch["attention_mask"][row_index].sum().item())
            full_length = int(attention_mask[row_index].sum().item())
            prompt_ids = prompt_batch["input_ids"][row_index][:prompt_length].tolist()
            full_ids = input_ids[row_index][:full_length].tolist()
            prefix_length = longest_common_prefix_length(prompt_ids, full_ids)
            labels[row_index, :prefix_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
