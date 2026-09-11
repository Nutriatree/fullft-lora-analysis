"""Supervised training examples, datasets, and collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from pubmedqa.data.prompts import build_tokenizer_prompt
from pubmedqa.data.records import PubMedQAExample


def shifted_target_counts(labels: torch.Tensor) -> torch.Tensor:
    """Causal LM predicts labels[:, 1:]; a label at index zero is not a target."""
    if labels.ndim != 2:
        raise ValueError("labels must have [batch, sequence] shape")
    return (labels[:, 1:] != -100).sum(dim=1)


def validate_targets(labels: torch.Tensor, pubids=None) -> int:
    counts = shifted_target_counts(labels)
    for row in (counts == 0).nonzero().flatten().tolist():
        identity = f"pubid={pubids[row]}" if pubids is not None else f"row={row}"
        # Never include question/context text in errors or distributed payloads.
        raise ValueError(
            f"No shifted target tokens: {identity}, length={labels.shape[1]}"
        )
    if counts.numel() == 0:
        raise ValueError("Empty supervised batch")
    return int(counts.sum().item())


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
        if not examples:
            raise ValueError("Empty supervised split")
        self._examples = [
            SupervisedExample(
                pubid=example.pubid,
                prompt_text=build_tokenizer_prompt(
                    tokenizer, example, include_answer=False
                ),
                full_text=build_tokenizer_prompt(
                    tokenizer, example, include_answer=True
                ),
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
        if not batch:
            raise ValueError("Empty supervised batch")
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
            prompt_mask = prompt_batch["attention_mask"][row_index].bool()
            full_mask = attention_mask[row_index].bool()
            prompt_ids = prompt_batch["input_ids"][row_index][prompt_mask].tolist()
            full_ids = input_ids[row_index][full_mask].tolist()
            # Token boundaries may change when the answer is appended. Mask the
            # shared prefix rather than assuming prompt character/token lengths
            # are additive; -100 keeps both prompt and padding out of the loss.
            prefix_length = longest_common_prefix_length(prompt_ids, full_ids)
            positions = full_mask.nonzero().flatten()
            labels[row_index, positions[:prefix_length]] = -100

        validate_targets(labels, [item.pubid for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def validate_supervised_dataset(dataset, collator, batch_size):
    """Preflight every row before any optimizer update; do not silently drop rows."""
    if len(dataset) == 0:
        raise ValueError("Empty supervised split")
    for start in range(0, len(dataset), batch_size):
        collator(
            [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        )


def build_train_dataloader(
    dataset,
    tokenizer,
    *,
    batch_size,
    seed,
    num_workers,
    max_input_tokens,
    world_size=1,
    rank=0,
):
    """Return loader and sampler; the epoch loop explicitly owns sampler epochs."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=SupervisedDataCollator(tokenizer, max_input_tokens),
        generator=generator,
    )
    return loader, sampler


def build_eval_dataloader(
    dataset, tokenizer, *, batch_size, num_workers, max_input_tokens
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SupervisedDataCollator(tokenizer, max_input_tokens),
    )
