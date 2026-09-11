"""Physical ownership checks, in addition to the numerical dry-run contracts."""

import ast
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1] / "src" / "pubmedqa"


class TrainingDirectoryLayoutTest(unittest.TestCase):
    def test_retained_legacy_exports_point_to_the_same_objects(self):
        import pubmedqa
        from pubmedqa import full_finetune, lora_finetune
        from pubmedqa.config import full_ft, lora

        for module, config, trainer in (
            (full_finetune, full_ft.FullFineTuneConfig, "PubMedQAFullFineTuner"),
            (lora_finetune, lora.LoRAFineTuneConfig, "PubMedQALoRAFineTuner"),
        ):
            self.assertIs(getattr(pubmedqa, trainer), getattr(module, trainer))
            self.assertIs(config, getattr(module, config.__name__))

    def test_legacy_classes_live_outside_the_training_core(self):
        from pubmedqa import PubMedQAFullFineTuner, PubMedQALoRAFineTuner

        for cls, owner in (
            (PubMedQAFullFineTuner, "pubmedqa.full_finetune"),
            (PubMedQALoRAFineTuner, "pubmedqa.lora_finetune"),
        ):
            self.assertEqual(owner, cls.__module__)
        for path in (ROOT / "train").glob("*.py"):
            tree = ast.parse(path.read_text())
            self.assertFalse(
                any(
                    isinstance(n, ast.ClassDef) and "FineTuner" in n.name
                    for n in ast.walk(tree)
                )
            )

    def test_core_never_imports_compatibility_paths(self):
        prohibited = (
            "pubmedqa.compat",
            "pubmedqa.training",
            "pubmedqa.runtime",
            "pubmedqa.domain",
            "pubmedqa.models",
            "pubmedqa.inference",
            "pubmedqa.reporting",
            "pubmedqa.experiments",
            "pubmedqa.full_finetune",
            "pubmedqa.lora_finetune",
            "pubmedqa.runtime_settings",
            "pubmedqa.experiment_runs",
            "pubmedqa.evaluation",
            "pubmedqa.prompt_builder",
            "pubmedqa.answer_parser",
            "pubmedqa.labels",
        )
        for directory in ("data", "model", "train", "eval", "config"):
            paths = list((ROOT / directory).glob("*.py"))
            self.assertTrue(paths, directory)
            for path in paths:
                for node in ast.walk(ast.parse(path.read_text())):
                    dependencies = []
                    if isinstance(node, ast.ImportFrom):
                        dependencies.append(node.module or "")
                    elif isinstance(node, ast.Import):
                        dependencies.extend(a.name for a in node.names)
                    for dependency in dependencies:
                        self.assertFalse(dependency.startswith(prohibited), str(path))

    def test_training_resources_are_owned_next_to_the_pipeline(self):
        for module, name in (
            ("pubmedqa.train.distributed", "TrainingSession"),
            ("pubmedqa.train.distributed", "RuntimeSettings"),
            ("pubmedqa.train.loop", "TrainingMemory"),
        ):
            with self.subTest(module=module, name=name):
                owner = importlib.import_module(module)
                self.assertEqual(owner.__name__, getattr(owner, name).__module__)
        self.assertTrue((ROOT / "train/distributed.py").is_file())

    def test_training_core_uses_local_resources_and_keeps_shared_runtime(self):
        import pubmedqa.train.pipeline as pipeline
        from pubmedqa.train.distributed import TrainingSession
        from pubmedqa.train.loop import TrainingMemory

        self.assertIs(TrainingSession, pipeline.TrainingSession)
        self.assertIs(TrainingMemory, pipeline.TrainingMemory)
        for relative in ("train/distributed.py", "model/device.py", "data/records.py"):
            self.assertTrue((ROOT / relative).is_file())

    def test_settings_live_with_their_consumer_and_preserve_all_defaults(self):
        from pubmedqa import runtime_settings
        from pubmedqa.config import eval as inference_settings
        from pubmedqa.config import full_ft as training_settings
        from pubmedqa.config import lora as lora_settings

        expected = json.loads(
            (Path(__file__).parent / "fixtures/settings_defaults.json").read_text()
        )
        actual = {}
        for name in expected:
            owner = {
                "EVAL_CONFIG": inference_settings,
                "TRAIN_LORA_CONFIG": lora_settings,
            }.get(name, training_settings)
            value = getattr(owner, name)
            self.assertEqual(owner.__name__, type(value).__module__)
            self.assertIs(value, getattr(runtime_settings, name))
            actual[name] = asdict(value)
            if "default_cpu_threads" in actual[name]:
                self.assertEqual(
                    max(1, (os.cpu_count() or 1) - 2),
                    actual[name].pop("default_cpu_threads"),
                )
        # Snapshot precedes the move; compare every environment key and default.
        self.assertEqual(expected, json.loads(json.dumps(actual, default=str)))

    def test_settings_and_registry_import_without_ml_frameworks(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import pubmedqa.config.full_ft; "
                    "import pubmedqa.config.eval; import pubmedqa.config.experiments; "
                    "assert not {'torch', 'transformers', 'peft'} & set(sys.modules)"
                ),
            ],
            env={**os.environ, "PYTHONPATH": str(ROOT.parent)},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_core_imports_settings_from_the_concept_owner(self):
        import pubmedqa.config as environment

        self.assertFalse(hasattr(environment, "TRAIN_FULL_FINE_TUNE_CONFIG"))
        self.assertFalse(hasattr(environment, "TRAIN_LORA_CONFIG"))
        for package in ("data", "model", "train", "eval", "config"):
            for path in (ROOT / package).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotEqual("pubmedqa.runtime_settings", node.module)
