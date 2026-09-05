from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pubmedqa.config import EVAL_CONFIG, TRAIN_FULL_FINE_TUNE_CONFIG
from pubmedqa.domain.labels import VALID_LABELS, normalize_label
from pubmedqa.domain.metrics import accuracy, classwise_f1, confusion_matrix, macro_f1
from pubmedqa.domain.prompts import build_messages, example_from_record
from pubmedqa.evaluation import EvalSummary as LegacyEvalSummary
from pubmedqa.full_finetune import TrainingSummary as LegacyTrainingSummary
from pubmedqa.inference import EvalSummary
from pubmedqa.labels import normalize_label as legacy_normalize_label
from pubmedqa.prompt_builder import PubMedQAExample as LegacyPubMedQAExample
from pubmedqa.runtime.io import safe_name, write_json, write_jsonl
from pubmedqa.runtime_settings import EVAL_CONFIG as LEGACY_EVAL_CONFIG
from pubmedqa.training import TrainingSummary
from pubmedqa.runtime.distributed import destroy_process_groups, run_on_rank_zero
from pubmedqa.runtime.torch_runtime import count_parameters, memory_snapshot, resolve_dtype


class PubMedQABoundaryTest(unittest.TestCase):
    def test_domain_package_is_independent_from_ml_frameworks(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import pubmedqa.domain; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'transformers' not in sys.modules; "
                    "assert 'peft' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        self.assertEqual(0, process.returncode, process.stderr)

    def test_domain_prompt_contract_matches_existing_format(self) -> None:
        example = example_from_record(
            {
                "pubid": "1",
                "question": "Is treatment effective?",
                "context": {"contexts": ["The treatment improved the outcome."]},
                "final_decision": "YES",
            }
        )

        messages = build_messages(example, include_answer=True)

        self.assertEqual(["system", "user", "assistant"], [item["role"] for item in messages])
        self.assertEqual("yes", messages[-1]["content"])

    def test_domain_metrics_cover_binary_and_three_class_gold_sets(self) -> None:
        binary_gold = ["yes", "yes", "no"]
        binary_predicted = ["yes", "no", "no"]
        self.assertAlmostEqual(2 / 3, accuracy(binary_gold, binary_predicted))
        self.assertEqual({"yes", "no"}, set(classwise_f1(binary_gold, binary_predicted)))
        self.assertGreater(macro_f1(binary_gold, binary_predicted), 0.0)

        matrix = confusion_matrix(
            ["yes", "maybe", "no"],
            ["yes", None, "maybe"],
        )
        self.assertEqual(1, matrix["maybe"]["invalid"])
        self.assertEqual(1, matrix["no"]["maybe"])

    def test_config_package_owns_existing_settings_objects(self) -> None:
        self.assertEqual("Qwen/Qwen3-1.7B", EVAL_CONFIG.default_model_name)
        self.assertEqual(1, TRAIN_FULL_FINE_TUNE_CONFIG.default_num_epochs)
        self.assertEqual(
            "PUBMEDQA_DISTRIBUTED_MODE",
            TRAIN_FULL_FINE_TUNE_CONFIG.env_distributed_mode,
        )
        self.assertEqual(("yes", "no", "maybe"), VALID_LABELS)
        self.assertEqual("maybe", normalize_label(" MAYBE "))

    def test_legacy_modules_reexport_new_public_contracts(self) -> None:
        self.assertIs(normalize_label, legacy_normalize_label)
        self.assertIs(EVAL_CONFIG, LEGACY_EVAL_CONFIG)
        self.assertIs(EvalSummary, LegacyEvalSummary)
        self.assertIs(TrainingSummary, LegacyTrainingSummary)
        self.assertEqual("PubMedQAExample", LegacyPubMedQAExample.__name__)

    def test_runtime_io_preserves_json_and_jsonl_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "nested" / "summary.json"
            jsonl_path = root / "nested" / "rows.jsonl"

            write_json(json_path, {"path": Path("a/b"), "value": 1})
            write_jsonl(jsonl_path, ({"index": index} for index in range(2)))

            self.assertEqual(
                {"path": "a/b", "value": 1},
                json.loads(json_path.read_text(encoding="utf-8")),
            )
            self.assertEqual(
                [{"index": 0}, {"index": 1}],
                [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()],
            )
            self.assertEqual("Qwen_Qwen3-1.7B", safe_name("Qwen/Qwen3-1.7B"))

    def test_torch_runtime_reports_parameters_and_cpu_memory(self) -> None:
        model = torch.nn.Linear(3, 2)
        model.bias.requires_grad = False

        total, trainable, ratio = count_parameters(model)

        self.assertEqual(8, total)
        self.assertEqual(6, trainable)
        self.assertEqual(0.75, ratio)
        self.assertIs(torch.bfloat16, resolve_dtype("bf16"))
        self.assertEqual(
            {
                "allocated_gb": None,
                "reserved_gb": None,
                "max_allocated_gb": None,
                "max_reserved_gb": None,
            },
            memory_snapshot(torch.device("cpu")),
        )

    def test_distributed_runtime_broadcasts_rank_zero_result(self) -> None:
        control_group = object()
        with patch("pubmedqa.runtime.distributed.dist.broadcast_object_list") as broadcast:
            result = run_on_rank_zero(
                lambda: "complete",
                is_main_process=True,
                control_group=control_group,
                operation_name="test operation",
            )

        self.assertEqual("complete", result)
        broadcast.assert_called_once_with(
            [{"ok": True, "result": "complete"}],
            src=0,
            group=control_group,
        )

    def test_distributed_runtime_destroys_control_then_default_group(self) -> None:
        control_group = object()
        with (
            patch("pubmedqa.runtime.distributed.dist.is_initialized", return_value=True),
            patch("pubmedqa.runtime.distributed.dist.destroy_process_group") as destroy,
        ):
            destroy_process_groups(control_group)

        self.assertEqual([((control_group,), {}), ((), {})], destroy.call_args_list)


if __name__ == "__main__":
    unittest.main()
