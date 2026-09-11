"""Guard five flat pipeline packages and method-specific ownership."""

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1] / "src/pubmedqa"


class PurposeLayoutTest(unittest.TestCase):
    def test_only_purpose_packages_remain(self):
        packages = {
            p.name for p in ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"
        }
        self.assertEqual({"data", "model", "train", "eval", "config"}, packages)
        for package in packages:
            nested = [
                p
                for p in (ROOT / package).iterdir()
                if p.is_dir() and p.name != "__pycache__"
            ]
            self.assertEqual([], nested, package)

    def test_study_configuration_and_execution_share_one_owner(self):
        from pubmedqa.train import study
        from pubmedqa.config import experiments
        from pubmedqa import experiment_runs

        for name in ("build_training_config", "execute_study", "ExperimentStudy"):
            self.assertEqual(study.__name__, getattr(study, name).__module__)
        self.assertEqual(experiments.__name__, experiment_runs.RunSpec.__module__)
        self.assertIs(study.build_full_ft_config, experiment_runs.build_full_ft_config)

    def test_method_configuration_remains_lightweight(self):
        from pubmedqa.config import full_ft, lora

        self.assertEqual(full_ft.__name__, full_ft.FullFineTuneConfig.__module__)
        self.assertEqual(lora.__name__, lora.LoRAFineTuneConfig.__module__)
        self.assertEqual(lora.__name__, lora.TrainLoraSettings.__module__)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import pubmedqa.config.full_ft, pubmedqa.config.lora, pubmedqa.config.eval, pubmedqa.config.experiments; assert not {'torch', 'transformers', 'peft'} & set(sys.modules)",
            ],
            env={**os.environ, "PYTHONPATH": str(ROOT.parent)},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_execution_and_storage_have_concrete_owners(self):
        from pubmedqa.train.distributed import TrainingSession, run_rank_local
        from pubmedqa.data import records

        self.assertEqual("pubmedqa.train.distributed", TrainingSession.__module__)
        self.assertEqual(TrainingSession.__module__, run_rank_local.__module__)
        for name in ("DEFAULT_ARTIFACT_WRITER", "FileArtifactWriter", "ArtifactWriter"):
            self.assertFalse(hasattr(records, name))

    def test_input_rules_have_direct_concept_owners(self):
        from pubmedqa.data.records import PubMedQAExample
        from pubmedqa.data import prompts, records
        from pubmedqa import prompt_builder, labels

        self.assertEqual(records.__name__, PubMedQAExample.__module__)
        self.assertIs(prompts.build_messages, prompt_builder.build_messages)
        self.assertIs(records.normalize_label, labels.normalize_label)
        tree = ast.parse((ROOT / "data/prompts.py").read_text())
        assigned = {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign)}
        self.assertIn("SYSTEM_PROMPT_TEMPLATE", assigned)
        self.assertIn("USER_PROMPT_TEMPLATE", assigned)

    def test_full_and_lora_are_separate_with_one_base_loader(self):
        loading = importlib.import_module("pubmedqa.model.loading")
        for module_name, function_name in (
            ("full_ft", "load_full_model"),
            ("lora", "load_lora_model"),
        ):
            module = importlib.import_module(f"pubmedqa.model.{module_name}")
            self.assertEqual(module.__name__, getattr(module, function_name).__module__)
            self.assertIs(loading._load_pretrained, module._load_pretrained)
            tree = ast.parse((ROOT / f"model/{module_name}.py").read_text())
            function = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == function_name
            )
            calls = {
                n.func.id
                for n in ast.walk(function)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            self.assertIn("_load_pretrained", calls)
