from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from pubmedqa.training.data import (
    SupervisedDataCollator,
    SupervisedExample,
    longest_common_prefix_length,
)
from pubmedqa.training.schedules import build_checkpoint_schedule
from pubmedqa.training.strategies.full import PubMedQAFullFineTuner


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
        self.assertEqual(3, longest_common_prefix_length([1, 2, 3], [1, 2, 3, 4]))
        self.assertEqual(1, longest_common_prefix_length([1, 9], [1, 2, 3]))
        self.assertEqual(0, longest_common_prefix_length([5], [1, 2, 3]))

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

    def test_checkpoint_schedule_rounds_up_to_optimizer_steps(self) -> None:
        self.assertEqual(
            [(25, 313), (50, 625), (75, 938), (100, 1250)],
            build_checkpoint_schedule(1250, (25, 50, 75, 100)),
        )

    def test_fsdp_evaluation_does_not_run_forward_inside_summon_full_params(self) -> None:
        source = inspect.getsource(PubMedQAFullFineTuner.evaluate_split)
        self.assertNotIn("FSDP.summon_full_params", source)

    def test_main_process_operation_broadcasts_to_distributed_ranks(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.config = SimpleNamespace(distributed_mode="fsdp")
        trainer.rank = 0

        with patch("pubmedqa.training.engine.dist.broadcast_object_list") as broadcast:
            with patch("pubmedqa.training.engine.dist.is_initialized", return_value=True):
                result = trainer._run_on_main_process(
                    lambda: "completed",
                    operation_name="validation",
                )

        self.assertEqual("completed", result)
        broadcast.assert_called_once_with(
            [{"ok": True, "result": "completed"}],
            src=0,
        )

    def test_non_main_process_does_not_execute_main_process_operation(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.config = SimpleNamespace(distributed_mode="fsdp")
        trainer.rank = 1
        operation = Mock(side_effect=AssertionError("rank 1 must not execute the operation"))

        def receive_result(payload, *, src):
            payload[0] = {"ok": True, "result": "rank-zero-result"}

        with (
            patch("pubmedqa.training.engine.dist.is_initialized", return_value=True),
            patch(
                "pubmedqa.training.engine.dist.broadcast_object_list",
                side_effect=receive_result,
            ),
        ):
            result = trainer._run_on_main_process(
                operation,
                operation_name="validation",
            )

        self.assertEqual("rank-zero-result", result)
        operation.assert_not_called()

    def test_main_process_operation_propagates_python_errors_to_all_ranks(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.config = SimpleNamespace(distributed_mode="fsdp")
        trainer.rank = 0

        with patch("pubmedqa.training.engine.dist.is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "validation failed on rank 0: ValueError: broken"):
                trainer._run_on_main_process(
                    lambda: (_ for _ in ()).throw(ValueError("broken")),
                    operation_name="validation",
                )

    def test_fsdp_wrap_preserves_full_shard_and_original_parameters(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.config = SimpleNamespace(
            distributed_mode="fsdp",
            dtype=torch.bfloat16,
            fsdp_cpu_offload=False,
        )
        trainer.device = torch.device("cpu")
        trainer.local_rank = 0
        model = torch.nn.Linear(2, 2)

        with patch("pubmedqa.training.engine.FullyShardedDataParallel") as fsdp_constructor:
            trainer.wrap_model_for_training(model)

        self.assertTrue(fsdp_constructor.call_args.kwargs["use_orig_params"])

    def test_training_model_reference_is_deleted_before_best_checkpoint_load(self) -> None:
        source = inspect.getsource(PubMedQAFullFineTuner.run)
        self.assertIn("del model", source)
        self.assertLess(source.index("del model"), source.index("def evaluate_best_checkpoint"))

    def test_close_destroys_owned_default_process_group(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer._owns_process_group = True

        with (
            patch("pubmedqa.training.engine.dist.is_initialized", return_value=True),
            patch("pubmedqa.training.engine.dist.destroy_process_group") as destroy,
        ):
            trainer.close()

        destroy.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
