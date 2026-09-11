from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pubmedqa.config.eval import EVAL_CONFIG
from pubmedqa.config.train import TRAIN_CONFIG
from pubmedqa.data.prompts import build_messages, example_from_record
from pubmedqa.data.records import (
    VALID_LABELS,
    normalize_label,
    safe_name,
    write_json,
    write_jsonl,
)
from pubmedqa.eval.metrics import accuracy, classwise_f1, confusion_matrix, macro_f1
from pubmedqa.model.device import count_parameters, memory_snapshot, resolve_dtype
from pubmedqa.train.distributed import destroy_process_groups, run_on_rank_zero


class PubMedQABoundaryTest(unittest.TestCase):
    def test_eager_module_imports_have_no_cycles(self) -> None:
        graph = {}
        for path in Path("src/pubmedqa").rglob("*.py"):
            name = ".".join(path.relative_to("src").with_suffix("").parts)
            name = name.removesuffix(".__init__")
            dependencies = set()
            # Function-local imports do not participate in startup dependency cycles.
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    dependencies.add(node.module)
                elif isinstance(node, ast.Import):
                    dependencies.update(alias.name for alias in node.names)
            graph[name] = dependencies
        visited = set()

        def visit(name, chain):
            self.assertNotIn(name, chain, " -> ".join([*chain, name]))
            if name in visited or name not in graph:
                return
            for dependency in graph[name]:
                visit(dependency, [*chain, name])
            visited.add(name)

        for name in graph:
            visit(name, [])

    def test_optimizer_update_has_one_owner(self) -> None:
        owners = []
        for path in Path("src/pubmedqa/train").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (
                        node.func.attr == "step"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "optimizer"
                    ):
                        owners.append(path.name)
        self.assertEqual(["loop.py"], owners)

    def test_prompt_data_is_independent_from_ml_frameworks(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import pubmedqa.data.prompts; "
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

    def test_prompt_contract_matches_training_format(self) -> None:
        example = example_from_record(
            {
                "pubid": "1",
                "question": "Is treatment effective?",
                "context": {"contexts": ["The treatment improved the outcome."]},
                "final_decision": "YES",
            }
        )

        messages = build_messages(example, include_answer=True)

        self.assertEqual(
            ["system", "user", "assistant"], [item["role"] for item in messages]
        )
        self.assertEqual("yes", messages[-1]["content"])

    def test_metrics_cover_binary_and_three_class_gold_sets(self) -> None:
        binary_gold = ["yes", "yes", "no"]
        binary_predicted = ["yes", "no", "no"]
        self.assertAlmostEqual(2 / 3, accuracy(binary_gold, binary_predicted))
        self.assertEqual(
            {"yes", "no"}, set(classwise_f1(binary_gold, binary_predicted))
        )
        self.assertGreater(macro_f1(binary_gold, binary_predicted), 0.0)

        matrix = confusion_matrix(
            ["yes", "maybe", "no"],
            ["yes", None, "maybe"],
        )
        self.assertEqual(1, matrix["maybe"]["invalid"])
        self.assertEqual(1, matrix["no"]["maybe"])

    def test_config_package_owns_existing_settings_objects(self) -> None:
        self.assertEqual("Qwen/Qwen3-1.7B", EVAL_CONFIG.default_model_name)
        self.assertEqual(1, TRAIN_CONFIG.default_num_epochs)
        self.assertEqual(
            "PUBMEDQA_DISTRIBUTED_MODE",
            TRAIN_CONFIG.env_distributed_mode,
        )
        self.assertEqual(("yes", "no", "maybe"), VALID_LABELS)
        self.assertEqual("maybe", normalize_label(" MAYBE "))

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
                [
                    json.loads(line)
                    for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                ],
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
        with patch(
            "pubmedqa.train.distributed.dist.broadcast_object_list"
        ) as broadcast:
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
            patch("pubmedqa.train.distributed.dist.is_initialized", return_value=True),
            patch("pubmedqa.train.distributed.dist.destroy_process_group") as destroy,
        ):
            destroy_process_groups(control_group)

        self.assertEqual([((control_group,), {}), ((), {})], destroy.call_args_list)


if __name__ == "__main__":
    unittest.main()
