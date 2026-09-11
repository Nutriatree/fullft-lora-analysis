from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch

from helpers.offline import offline_cpu
from helpers.tiny_training import make_config, environment
from pubmedqa.full_finetune import PubMedQAFullFineTuner
from pubmedqa.lora_finetune import LoRAFineTuneConfig, PubMedQALoRAFineTuner


class TrainingDryRunTest(unittest.TestCase):
    def test_lora_merge_for_evaluation_preserves_logits(self):
        from dataclasses import replace

        with tempfile.TemporaryDirectory() as directory:
            trainer, summary = self.run_fixture(Path(directory), "lora")
            _, adapter = trainer.load_model_and_tokenizer(summary.best_checkpoint_dir)
            trainer.lora_config = replace(trainer.lora_config, merge_for_eval=True)
            _, merged = trainer.load_model_and_tokenizer(summary.best_checkpoint_dir)
            tokens = torch.tensor([[3, 4, 5]])
            adapter.eval()
            merged.eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    adapter(tokens).logits, merged(tokens).logits, atol=1e-6, rtol=1e-5
                )
            self.assertFalse(
                any("lora_" in name for name, _ in merged.named_parameters())
            )

    def run_fixture(self, root, method, **changes):
        config = make_config(root, **changes)
        if method != "full":
            values = asdict(config)
            values.update(
                lora_rank=2,
                lora_alpha=4.0,
                lora_dropout=0.0,
                target_modules=("q_proj", "v_proj"),
                target_layers=(0,) if method == "selective" else (),
                layer_scope="selected" if method == "selective" else "all",
            )
            config = LoRAFineTuneConfig(
                **values,
                lora_bias="none",
                lora_task_type="CAUSAL_LM",
                modules_to_save=(),
                merge_for_eval=False,
            )
            trainer = PubMedQALoRAFineTuner(config, environment())
        else:
            trainer = PubMedQAFullFineTuner(config, environment())
        # Compare a fresh loader's base weights to the persisted result. Adapter
        # A is randomly initialized, so only B (zero at initialization) is used
        # to prove a real update; every frozen base tensor must stay identical.
        _, initial = trainer.load_model_and_tokenizer(config.model_name)
        before = {
            name: value.detach().clone() for name, value in initial.named_parameters()
        }
        del initial
        try:
            with offline_cpu():
                summary = trainer.run()
        finally:
            trainer.close()
        _, restored = trainer.load_model_and_tokenizer(summary.best_checkpoint_dir)
        after = dict(restored.named_parameters())
        if method == "full":
            self.assertTrue(
                any(
                    not torch.equal(value, after[name])
                    for name, value in before.items()
                )
            )
        else:
            for name, value in before.items():
                if "lora_" not in name:
                    self.assertTrue(torch.equal(value, after[name]), name)
            self.assertTrue(
                any(
                    torch.count_nonzero(value).item() > 0
                    for name, value in after.items()
                    if "lora_B" in name
                )
            )
        del restored
        return trainer, summary

    def test_epoch_snapshot_and_scheduled_checkpoint_with_accumulation(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, summary = self.run_fixture(
                Path(directory),
                "full",
                checkpoint_percents=(100,),
                save_every_epoch=False,
                gradient_accumulation_steps=2,
                train_batch_size=1,
                num_epochs=2,
            )
            self.assertEqual(4, summary.optimizer_steps)
            self.assertEqual("scheduled", summary.best_checkpoint_kind)
            self.assertEqual(
                [], list((trainer.output_root / ".evaluation_snapshots").glob("*"))
            )

    def test_snapshot_is_removed_when_checkpoint_evaluation_fails(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            config = make_config(
                Path(directory),
                checkpoint_percents=(100,),
                save_every_epoch=False,
                num_epochs=2,
            )
            trainer = PubMedQAFullFineTuner(config, environment())
            try:
                with patch(
                    "pubmedqa.train.pipeline.evaluate_checkpoint_on_main",
                    side_effect=RuntimeError("evaluation failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                        trainer.run()
                self.assertEqual(
                    [], list((trainer.output_root / ".evaluation_snapshots").glob("*"))
                )
            finally:
                trainer.close()

    def test_real_cpu_full_lora_and_selective_pipeline(self):
        for method in ("full", "lora", "selective"):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                trainer, summary = self.run_fixture(Path(directory), method)
                self.assertEqual(2, summary.optimizer_steps)
                self.assertTrue(torch.isfinite(torch.tensor(summary.final_train_loss)))
                checkpoint = Path(summary.best_checkpoint_dir)
                self.assertTrue((checkpoint / "optimizer.pt").is_file())
                self.assertTrue((checkpoint / "training_state.json").is_file())
                self.assertTrue(
                    (trainer.evaluations_dir / "test_summary.json").is_file()
                )
                logs = [
                    json.loads(line)
                    for line in (trainer.logs_dir / "train_steps.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual([1, 2], [row["global_step"] for row in logs])
                self.assertEqual([2, 1], [row["batch_size"] for row in logs])
                self.assertGreater(summary.final_train_loss, 0)
                # Fixed-seed pre-refactor CPU reference; tolerate FP rounding,
                # but detect optimizer order, masking, or LR changes.
                expected = {
                    "full": (2.1923067569732666, 2.1954565048217773, 1464),
                    "lora": (2.192288041114807, 2.1958494186401367, 128),
                    "selective": (2.1922885179519653, 2.1958532333374023, 64),
                }[method]
                self.assertAlmostEqual(
                    expected[0], summary.final_train_loss, delta=2e-6
                )
                self.assertAlmostEqual(
                    expected[1], summary.final_validation_loss, delta=2e-6
                )
                self.assertEqual(expected[2], summary.trainable_params)
                if method != "full":
                    self.assertLess(summary.trainable_ratio, 1)
                    self.assertTrue((checkpoint / "adapter_config.json").is_file())
