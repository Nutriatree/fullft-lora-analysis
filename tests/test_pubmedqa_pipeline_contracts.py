"""Public, on-disk contracts to preserve when replacing the Trainer hierarchy.

These tests deliberately do not inspect inheritance, private helper names, or
the number of modules. The tiny model is real; external I/O and GPUs are not.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from helpers.offline import offline_cpu
from helpers.tiny_training import environment, make_config, with_lora

from pubmedqa.train.pipeline import run_training

ROOT = Path(__file__).resolve().parents[1]


class PipelineContractsTest(unittest.TestCase):
    def run_fixture(self, root, method, **changes):
        """Exercise the actual pipeline; preserve every on-disk assertion."""
        with offline_cpu():
            config = make_config(root, **changes)
            if method != "full":
                config = with_lora(
                    config,
                    target_layers=(0,) if method == "selective" else (),
                    layer_scope="selected" if method == "selective" else "all",
                )
            summary = run_training(config, environment())
        # The returned checkpoint, not an internal trainer field, locates output.
        output_root = Path(summary.best_checkpoint_dir).parent.parent
        return config, summary, output_root

    @staticmethod
    def read_documents(output_root, summary):
        return {
            "summary": json.loads((output_root / "summary.json").read_text()),
            "metadata": json.loads((output_root / "run_metadata.json").read_text()),
            "config": json.loads((output_root / "config.json").read_text()),
            "artifacts": json.loads((output_root / "artifacts.json").read_text()),
            "state": json.loads(
                (Path(summary.best_checkpoint_dir) / "training_state.json").read_text()
            ),
        }

    def assert_document_contract(self, documents, config, summary, method):
        persisted = documents["summary"]
        # JSON changes tuples into lists; additive LoRA fields remain permitted.
        returned = json.loads(json.dumps(asdict(summary)))
        self.assertEqual(returned, {key: persisted[key] for key in returned})
        self.assertEqual(2, persisted["optimizer_steps"])
        self.assertEqual(3, persisted["train_examples"])
        self.assertEqual(3, persisted["validation_examples"])
        expected_method = "full-ft" if method == "full" else "lora"
        self.assertEqual(expected_method, persisted["method_name"])

        metadata = documents["metadata"]
        self.assertEqual(expected_method, metadata["method_name"])
        self.assertEqual(2, metadata["analysis_schema_version"])
        self.assertEqual("token_mean", metadata["validation_loss_reduction"])
        self.assertEqual("microbatch_mean", metadata["training_loss_reduction"])
        self.assertEqual(config.run_id, metadata["run_id"])
        self.assertEqual(list(config.target_layers), metadata["target_layers"])
        self.assertEqual(list(config.target_modules), metadata["target_modules"])
        self.assertEqual(config.layer_scope, metadata["layer_scope"])

        persisted_config = documents["config"]
        self.assertEqual("float32", persisted_config["dtype"])
        self.assertEqual({"hf_token_set": False}, persisted_config["environment"])
        self.assertEqual(config.seed, persisted_config["seed"])
        self.assertEqual(
            config.gradient_accumulation_steps,
            persisted_config["gradient_accumulation_steps"],
        )

        artifacts = documents["artifacts"]
        records = artifacts["checkpoints"]
        self.assertEqual(
            ["reference", "scheduled"], [r["checkpoint_kind"] for r in records]
        )
        self.assertEqual([0, 2], [r["global_step"] for r in records])
        self.assertEqual([0.0, 100.0], [r["checkpoint_percent"] for r in records])
        self.assertEqual(records[-1], artifacts["best_checkpoint"])
        self.assertEqual(summary.best_checkpoint_dir, records[-1]["checkpoint_dir"])

        state = documents["state"]
        self.assertEqual("scheduled", state["checkpoint_kind"])
        self.assertEqual(
            (1, 2, 2), (state["epoch"], state["step_in_epoch"], state["global_step"])
        )
        self.assertEqual(config.run_id, state["run_id"])
        self.assertEqual(summary.title, state["title"])
        self.assertEqual(
            summary.best_validation_loss, state["validation_metrics"]["loss"]
        )
        self.assertEqual("token_mean", state["validation_metrics"]["loss_reduction"])
        for document in (state, metadata, persisted_config):
            runtime = document["distributed"]
            self.assertEqual("single", runtime["mode"])
            self.assertEqual(
                (0, 0, 1),
                (runtime["rank"], runtime["local_rank"], runtime["world_size"]),
            )
            self.assertEqual("cpu", runtime["evaluation_device"])
            self.assertEqual("torch.float32", runtime["evaluation_dtype"])
            self.assertIsNone(runtime["control_backend"])

        if method == "full":
            self.assertNotIn("lora", state)
            self.assertNotIn("adapter_params", persisted)
        else:
            self.assertEqual(summary.trainable_params, persisted["adapter_params"])
            self.assertGreater(persisted["adapter_checkpoint_size_bytes"], 0)
            self.assertIsNone(persisted["merged_checkpoint_size_bytes"])
            self.assertEqual("none", persisted["lora_bias"])
            self.assertEqual("CAUSAL_LM", persisted["lora_task_type"])
            self.assertEqual([], persisted["modules_to_save"])
            self.assertFalse(persisted["merge_for_eval"])
            self.assertEqual(
                Path(summary.best_checkpoint_dir).name, state["checkpoint_id"]
            )
            self.assertEqual(
                {
                    "target_modules": ["q_proj", "v_proj"],
                    "target_layers": [0] if method == "selective" else [],
                    "layer_scope": "selected" if method == "selective" else "all",
                    "rank": 2,
                    "alpha": 4.0,
                    "dropout": 0.0,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "modules_to_save": [],
                    "merge_for_eval": False,
                },
                state["lora"],
            )

    def assert_persisted_state(self, config, summary, output_root, method):
        documents = self.read_documents(output_root, summary)
        self.assert_document_contract(documents, config, summary, method)
        checkpoint = Path(summary.best_checkpoint_dir)
        with offline_cpu():
            # Validate saved state contents without implying exact FSDP resume.
            optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
            scheduler = torch.load(checkpoint / "scheduler.pt", weights_only=True)
        self.assertEqual({"state", "param_groups"}, set(optimizer))
        self.assertTrue(optimizer["state"])
        self.assertTrue(
            all(int(state["step"]) == 2 for state in optimizer["state"].values())
        )
        self.assertEqual(2, scheduler["last_epoch"])
        self.assertEqual([0.0], scheduler["_last_lr"])
        self.assertEqual(0.0, optimizer["param_groups"][0]["lr"])
        weights = (
            "model.safetensors" if method == "full" else "adapter_model.safetensors"
        )
        self.assertTrue((checkpoint / weights).is_file())
        self.assertTrue((checkpoint / "tokenizer.json").is_file())

        reference = Path(documents["artifacts"]["checkpoints"][0]["checkpoint_dir"])
        self.assertTrue((reference / "training_state.json").is_file())
        for filename in (weights, "optimizer.pt", "scheduler.pt"):
            self.assertFalse((reference / filename).exists())
        predictions = [
            json.loads(line)
            for line in (output_root / "evaluations/test_predictions.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(["0", "1", "2"], [row["pubid"] for row in predictions])
        self.assertEqual(
            ["yes", "no", "maybe"], [row["gold_label"] for row in predictions]
        )

    def test_public_entries_import_without_loading_models_or_starting_distributed(self):
        # A fresh interpreter matters: importing already-cached modules would
        # miss accidental top-level work introduced during CLI rewiring.
        script = """
from contextlib import ExitStack
import importlib
from unittest.mock import patch
from helpers.offline import offline_cpu

with offline_cpu(), ExitStack() as stack:
    for target in (
        'transformers.AutoTokenizer.from_pretrained',
        'transformers.AutoModelForCausalLM.from_pretrained',
    ):
        stack.enter_context(patch(target, side_effect=AssertionError('import loaded a model')))
    for name in (
        'pubmedqa',
        'pubmedqa.train.pipeline',
        'pubmedqa.model.lora',
        'scripts.run_pubmedqa_full_finetune',
        'scripts.run_pubmedqa_lora_finetune',
        'scripts.run_pubmedqa_experiments',
        'scripts.run_pubmedqa_eval',
        'scripts.pubmedqa_prompt',
        'scripts.select_pubmedqa_lora_layers',
        'scripts.validate_pubmedqa_runs',
    ):
        importlib.import_module(name)
"""
        with tempfile.TemporaryDirectory() as directory:
            env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    (str(ROOT / "src"), str(ROOT / "tests"), str(ROOT))
                ),
                "HF_HOME": directory,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_full_training_public_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assert_persisted_state(
                *self.run_fixture(Path(directory), "full"), "full"
            )

    def test_lora_and_selective_public_artifact_contract(self):
        for method in ("lora", "selective"):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.assert_persisted_state(
                    *self.run_fixture(Path(directory), method), method
                )

    def test_optional_optimizer_and_test_outputs_remain_absent_when_disabled(self):
        for method in ("full", "lora"):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                _, summary, output_root = self.run_fixture(
                    Path(directory), method, save_optimizer_state=False, test_path=None
                )
                self.assertEqual(0, summary.test_examples)
                self.assertIsNone(summary.test_loss)
                self.assertIsNone(summary.test_accuracy)
                self.assertIsNone(summary.test_macro_f1)
                self.assertFalse(
                    (output_root / "evaluations/test_summary.json").exists()
                )
                self.assertFalse(
                    (output_root / "evaluations/test_predictions.jsonl").exists()
                )
                for path in (output_root / "checkpoints").iterdir():
                    self.assertFalse((path / "optimizer.pt").exists())
                    self.assertFalse((path / "scheduler.pt").exists())

    def test_schedule_collisions_keep_first_percentage_for_each_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            _, summary, output_root = self.run_fixture(
                Path(directory), "full", checkpoint_percents=(25, 50, 100)
            )
            records = self.read_documents(output_root, summary)["artifacts"][
                "checkpoints"
            ]
            self.assertEqual([0, 1, 2], [row["global_step"] for row in records])
            self.assertEqual(
                [0.0, 25.0, 100.0], [row["checkpoint_percent"] for row in records]
            )
            self.assertEqual(
                {
                    "reference_pct_000_epoch_000_step_000000",
                    "scheduled_pct_025_epoch_001_step_000001",
                    "scheduled_pct_100_epoch_001_step_000002",
                },
                {path.name for path in (output_root / "checkpoints").iterdir()},
            )

    def test_contract_checks_reject_semantic_artifact_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            config, summary, output_root = self.run_fixture(
                Path(directory), "selective"
            )
            documents = self.read_documents(output_root, summary)
            self.assert_document_contract(documents, config, summary, "selective")
            # Mutate only in-memory copies of synthetic outputs, never production
            # code or prior experiment artifacts. A passing guard must reject each.
            mutations = (
                ("summary", "optimizer_steps", 99),
                ("metadata", "validation_loss_reduction", "batch_mean"),
                ("metadata", "target_layers", [1]),
                ("state", "global_step", 99),
                ("state", "lora", {}),
            )
            for document, key, value in mutations:
                with self.subTest(document=document, key=key):
                    corrupted = deepcopy(documents)
                    corrupted[document][key] = value
                    with self.assertRaises(AssertionError):
                        self.assert_document_contract(
                            corrupted, config, summary, "selective"
                        )
