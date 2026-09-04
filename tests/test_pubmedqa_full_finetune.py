from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from pubmedqa.full_finetune import (
    PubMedQAFullFineTuner,
    SupervisedDataCollator,
    SupervisedExample,
    _discover_transformer_layer_classes,
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


class FakeDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Linear(2, 2)
        self.mlp = torch.nn.Linear(2, 2)


class FakeDecoderModel(torch.nn.Module):
    _no_split_modules = ["FakeDecoderLayer"]

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeDecoderLayer(), FakeDecoderLayer()])


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

    def test_fsdp_evaluation_does_not_run_forward_inside_summon_full_params(self) -> None:
        source = inspect.getsource(PubMedQAFullFineTuner.evaluate_split)
        self.assertNotIn("FSDP.summon_full_params", source)

    def test_main_process_operation_broadcasts_over_control_group(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.config = SimpleNamespace(distributed_mode="fsdp")
        trainer.rank = 0
        trainer._control_group = object()

        with patch("pubmedqa.full_finetune.dist.broadcast_object_list") as broadcast:
            result = trainer._run_on_main_process(lambda: "completed", operation_name="validation")

        self.assertEqual("completed", result)
        broadcast.assert_called_once()
        self.assertIs(trainer._control_group, broadcast.call_args.kwargs["group"])

    def test_non_main_process_does_not_execute_main_process_operation(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.config = SimpleNamespace(distributed_mode="fsdp")
        trainer.rank = 1
        trainer._control_group = object()
        operation = Mock(side_effect=AssertionError("rank 1 must not execute the operation"))

        def receive_result(payload, *, src, group):
            payload[0] = {"ok": True, "result": "rank-zero-result"}

        with patch(
            "pubmedqa.full_finetune.dist.broadcast_object_list",
            side_effect=receive_result,
        ):
            result = trainer._run_on_main_process(operation, operation_name="validation")

        self.assertEqual("rank-zero-result", result)
        operation.assert_not_called()

    def test_main_process_operation_propagates_python_errors_to_all_ranks(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.config = SimpleNamespace(distributed_mode="fsdp")
        trainer.rank = 0
        trainer._control_group = object()

        with patch("pubmedqa.full_finetune.dist.broadcast_object_list"):
            with self.assertRaisesRegex(RuntimeError, "validation failed on rank 0: ValueError: broken"):
                trainer._run_on_main_process(
                    lambda: (_ for _ in ()).throw(ValueError("broken")),
                    operation_name="validation",
                )

    def test_discovers_transformer_decoder_layer_class_for_fsdp_auto_wrap(self) -> None:
        classes = _discover_transformer_layer_classes(FakeDecoderModel())
        self.assertEqual({FakeDecoderLayer}, classes)

    def test_transformer_layer_discovery_fails_instead_of_root_only_fallback(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "transformer decoder layer"):
            _discover_transformer_layer_classes(torch.nn.Sequential(torch.nn.Linear(2, 2)))

    def test_fsdp_wrap_uses_discovered_transformer_auto_wrap_policy(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.config = SimpleNamespace(
            distributed_mode="fsdp",
            dtype=torch.bfloat16,
            fsdp_cpu_offload=False,
        )
        trainer.device = torch.device("cpu")
        trainer._fsdp_layer_class_names = ()
        model = FakeDecoderModel()

        with patch("pubmedqa.full_finetune.FSDP") as fsdp_constructor:
            trainer.wrap_model_for_training(model)

        auto_wrap_policy = fsdp_constructor.call_args.kwargs["auto_wrap_policy"]
        self.assertEqual(
            {FakeDecoderLayer},
            auto_wrap_policy.keywords["transformer_layer_cls"],
        )
        self.assertEqual(("FakeDecoderLayer",), trainer._fsdp_layer_class_names)

    def test_training_model_reference_is_deleted_before_best_checkpoint_load(self) -> None:
        source = inspect.getsource(PubMedQAFullFineTuner.run)
        self.assertIn("del model", source)
        self.assertLess(source.index("del model"), source.index("def evaluate_best_checkpoint"))

    def test_close_destroys_control_and_owned_default_process_groups(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        control_group = object()
        trainer._control_group = control_group
        trainer._distributed_initialized_here = True

        with (
            patch("pubmedqa.full_finetune.dist.is_available", return_value=True),
            patch("pubmedqa.full_finetune.dist.is_initialized", return_value=True),
            patch("pubmedqa.full_finetune.dist.destroy_process_group") as destroy,
        ):
            trainer.close()

        self.assertEqual([((control_group,), {}), ((), {})], destroy.call_args_list)
        self.assertIsNone(trainer._control_group)
        self.assertFalse(trainer._distributed_initialized_here)


if __name__ == "__main__":
    unittest.main()
