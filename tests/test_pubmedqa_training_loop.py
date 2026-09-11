from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


class ScalarLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(loss=(self.weight * input_ids.float()).square().mean())


class TrainingLoopTest(unittest.TestCase):
    def test_accumulation_and_partial_batch_match_reference_optimizer(self):
        from pubmedqa.train.loop import train_epoch

        for accumulation in (1, 2):
            with self.subTest(accumulation=accumulation):
                model, reference = ScalarLoss(), ScalarLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                batches = [
                    {
                        "input_ids": torch.tensor([[i, i]]),
                        "attention_mask": torch.tensor([[0, 1]]),
                        "labels": torch.tensor([[-100, i]]),
                    }
                    for i in (1, 2, 3)
                ]
                for index, batch in enumerate(batches, 1):
                    (reference(**batch).loss / accumulation).backward()
                    if index % accumulation == 0 or index == len(batches):
                        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
                        expected_optimizer.step()
                        expected_optimizer.zero_grad(set_to_none=True)
                steps = list(
                    train_epoch(
                        model=model,
                        loader=batches,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=1,
                        global_step=0,
                        accumulation_steps=accumulation,
                        max_grad_norm=1.0,
                        device=torch.device("cpu"),
                        autocast_context=nullcontext,
                        ddp_enabled=False,
                    )
                )
                torch.testing.assert_close(model.weight, reference.weight)
                self.assertEqual(3 if accumulation == 1 else 2, len(steps))
                self.assertEqual(3, sum(step.micro_batches for step in steps))
                self.assertEqual(3, sum(step.log.input_tokens for step in steps))
                self.assertEqual(3, steps[-1].log.step_in_epoch)

    def test_ddp_no_sync_only_for_nonfinal_microbatch(self):
        from pubmedqa.train.loop import train_epoch

        model = ScalarLoss()
        model.no_sync = Mock(side_effect=nullcontext)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        batch = {
            "input_ids": torch.ones(1, 2),
            "attention_mask": torch.ones(1, 2),
            "labels": torch.tensor([[-100, 1]]),
        }
        steps = list(
            train_epoch(
                model=model,
                loader=[batch] * 3,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=2,
                global_step=4,
                accumulation_steps=2,
                max_grad_norm=1.0,
                device=torch.device("cpu"),
                autocast_context=nullcontext,
                ddp_enabled=True,
            )
        )
        model.no_sync.assert_called_once()
        self.assertEqual([5, 6], [step.log.global_step for step in steps])
