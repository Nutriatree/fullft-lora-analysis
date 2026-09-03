from __future__ import annotations

import unittest

import torch

from pubmedqa.full_finetune import (
    SupervisedDataCollator,
    SupervisedExample,
    _longest_common_prefix_length,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, *, padding, truncation, return_tensors, max_length=None):
        encoded = []
        for text in texts:
            token_ids = [index + 1 for index, _ in enumerate(text.split())]
            if max_length is not None:
                token_ids = token_ids[:max_length]
            encoded.append(token_ids)

        max_width = max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        for row in encoded:
            pad_width = max_width - len(row)
            input_ids.append(row + [self.pad_token_id] * pad_width)
            attention_mask.append([1] * len(row) + [0] * pad_width)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class PubMedQAFullFineTuneTest(unittest.TestCase):
    def test_longest_common_prefix_length(self) -> None:
        self.assertEqual(3, _longest_common_prefix_length([1, 2, 3], [1, 2, 3, 4]))
        self.assertEqual(1, _longest_common_prefix_length([1, 9], [1, 2, 3]))
        self.assertEqual(0, _longest_common_prefix_length([5], [1, 2, 3]))

    def test_collator_masks_prompt_tokens_only(self) -> None:
        collator = SupervisedDataCollator(FakeTokenizer(), max_length=None)
        batch = [
            SupervisedExample(
                pubid="1",
                prompt_text="system user assistant",
                full_text="system user assistant yes",
                label="yes",
            )
        ]

        output = collator(batch)
        labels = output["labels"][0].tolist()

        self.assertEqual([-100, -100, -100, 4], labels)


if __name__ == "__main__":
    unittest.main()
