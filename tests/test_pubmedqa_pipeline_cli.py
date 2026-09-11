import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from helpers.tiny_training import make_config

from pubmedqa.eval.reports import RunArtifacts
from scripts.run_pubmedqa_experiments import main


class PipelineCliTest(unittest.TestCase):
    def test_direct_config_builders_keep_variants_and_reject_invalid_selection(self):
        from dataclasses import replace

        from helpers.tiny_training import environment

        from pubmedqa.config.experiments import (
            DEFAULT_PATHS,
            DEFAULT_SHARED_DEFAULTS,
            build_training_config,
            resolve_run_spec,
        )
        from pubmedqa.config.train import TrainingConfig
        from scripts.run_pubmedqa_experiments import build_baseline_runner

        defaults = replace(DEFAULT_SHARED_DEFAULTS, device="cpu", dtype="float32")
        with tempfile.TemporaryDirectory() as directory:
            for tag, expected in (
                ("B0", "baseline"),
                ("F1", "full-ft"),
                ("L1", "lora"),
                ("LL1", "lora"),
            ):
                spec = resolve_run_spec(tag)
                self.assertEqual(expected, spec.method)
                if tag == "B0":
                    runner = build_baseline_runner(
                        run_id="direct",
                        spec=spec,
                        defaults=defaults,
                        output_dir=Path(directory),
                        environment=environment(),
                    )
                    self.assertEqual(defaults.device, runner.runtime.device)
                else:
                    config = build_training_config(
                        run_id="direct",
                        run_tag=tag,
                        paths=DEFAULT_PATHS,
                        output_dir=Path(directory),
                        defaults=defaults,
                        target_layer_overrides={"LL1": (0,)},
                    )
                    self.assertIsInstance(config, TrainingConfig)
                    self.assertEqual(tag != "F1", config.adapter is not None)
            common = dict(
                run_id="bad",
                paths=DEFAULT_PATHS,
                output_dir=Path(directory),
                defaults=defaults,
            )
            with self.assertRaisesRegex(ValueError, "Unsupported training method"):
                build_training_config(run_tag="B0", **common)
            with self.assertRaisesRegex(ValueError, "not a selective"):
                build_training_config(
                    run_tag="L1", target_layer_overrides={"L1": (0,)}, **common
                )

    def test_environment_configurations_feed_the_functional_program(self):
        from types import SimpleNamespace

        from helpers.tiny_training import environment, with_lora

        from pubmedqa.config.train import TrainingCliConfig
        from pubmedqa.train.pipeline import run_training

        for method in ("full-ft", "lora"):
            with tempfile.TemporaryDirectory() as directory:
                cfg = make_config(Path(directory))
                if method == "lora":
                    cfg = with_lora(cfg, target_modules=("q_proj",))
                cli = SimpleNamespace(config=cfg, environment=environment())
                with (
                    patch.object(TrainingCliConfig, "from_env", return_value=cli),
                    redirect_stdout(io.StringIO()),
                ):
                    parsed = TrainingCliConfig.from_env(method)
                    run_training(parsed.config, parsed.environment)
                self.assertTrue(list(Path(directory).rglob("summary.json")))

    def test_standalone_commands_run_the_training_pipeline(self):
        from helpers.offline import offline_cpu

        from scripts import run_pubmedqa_full_finetune, run_pubmedqa_lora_finetune

        for command, extra in (
            (run_pubmedqa_full_finetune, []),
            (
                run_pubmedqa_lora_finetune,
                [
                    "--lora-rank",
                    "2",
                    "--lora-alpha",
                    "4",
                    "--target-modules",
                    "q_proj,v_proj",
                ],
            ),
        ):
            with (
                self.subTest(command=command.__name__),
                tempfile.TemporaryDirectory() as directory,
                offline_cpu(),
            ):
                cfg = make_config(Path(directory))
                argv = [
                    "train",
                    "--run-id",
                    "standalone",
                    "--train-file",
                    str(cfg.train_path),
                    "--validation-file",
                    str(cfg.validation_path),
                    "--test-file",
                    str(cfg.test_path),
                    "--output-dir",
                    str(Path(directory) / "outputs"),
                    "--model-name",
                    cfg.model_name,
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--attn-implementation",
                    "eager",
                    "--num-epochs",
                    "1",
                    "--train-batch-size",
                    "2",
                    "--eval-batch-size",
                    "3",
                    "--gradient-accumulation-steps",
                    "1",
                    "--num-workers",
                    "0",
                    "--cpu-threads",
                    "1",
                    "--max-new-tokens",
                    "1",
                    "--checkpoint-percents",
                    "100",
                    *extra,
                ]
                with (
                    patch("sys.argv", argv),
                    redirect_stdout(io.StringIO()),
                ):
                    command.main()
                summaries = list((Path(directory) / "outputs").rglob("summary.json"))
                self.assertEqual(1, len(summaries))
                self.assertEqual(
                    2, json.loads(summaries[0].read_text())["optimizer_steps"]
                )

    def test_runner_errors_close_resources_and_obey_continue_flag(self):
        from dataclasses import replace

        from helpers.tiny_training import environment

        from pubmedqa.config.experiments import DEFAULT_PATHS, DEFAULT_SHARED_DEFAULTS
        from pubmedqa.train.distributed import TrainingSession
        from scripts.run_pubmedqa_experiments import ExperimentStudy, execute_study

        with tempfile.TemporaryDirectory() as directory:
            for keep_going in (False, True):
                study = ExperimentStudy(
                    "failed",
                    ("F1", "L1"),
                    DEFAULT_PATHS,
                    replace(DEFAULT_SHARED_DEFAULTS, device="cpu", dtype="float32"),
                    Path(directory),
                    Path(directory),
                    {},
                    environment(),
                    keep_going,
                )
                with (
                    patch(
                        "pubmedqa.train.pipeline.load_full_model",
                        side_effect=ValueError("training failed"),
                    ) as full,
                    patch(
                        "pubmedqa.train.pipeline.load_lora_model",
                        side_effect=ValueError("training failed"),
                    ) as lora,
                    patch.object(
                        TrainingSession,
                        "close",
                        autospec=True,
                        side_effect=TrainingSession.close,
                    ) as close,
                    redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaisesRegex(RuntimeError, "training failed"):
                        execute_study(study)
                self.assertEqual(
                    2 if keep_going else 1, full.call_count + lora.call_count
                )
                self.assertEqual(full.call_count + lora.call_count, close.call_count)

    def test_all_run_tags_manifest_only_does_not_load_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "run",
                "--run-id",
                "manifest",
                "--dry-run",
                "--runs",
                "B0,F1,L1,L2,L3,L4,LL1,LL2",
                "--target-layer-override",
                "LL1=0",
                "--target-layer-override",
                "LL2=1",
                "--train-output-dir",
                str(root / "train"),
                "--baseline-output-dir",
                str(root / "eval"),
            ]
            with (
                patch("sys.argv", argv),
                patch(
                    "scripts.run_pubmedqa_experiments.run_training",
                    side_effect=AssertionError("dry-run must not construct a model"),
                ),
                patch(
                    "scripts.run_pubmedqa_experiments.build_baseline_runner",
                    side_effect=AssertionError("manifest constructed baseline"),
                ),
                patch(
                    "scripts.run_pubmedqa_experiments.distributed_control_group",
                    side_effect=AssertionError("manifest initialized distributed"),
                ),
                redirect_stdout(io.StringIO()),
            ):
                main()
            manifest = json.loads(
                (root / "train/manifest/run_manifest.json").read_text()
            )
            self.assertEqual(8, len(manifest["requested_runs"]))
            self.assertEqual([0], manifest["target_layer_overrides"]["LL1"])

    def test_cli_runs_local_full_lora_selective_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            argv = [
                "run",
                "--run-id",
                "cpu-cli",
                "--runs",
                "F1,L1,LL1",
                "--target-layer-override",
                "LL1=0",
                "--model-name",
                config.model_name,
                "--train-file",
                str(config.train_path),
                "--validation-file",
                str(config.validation_path),
                "--test-file",
                str(config.test_path),
                "--train-output-dir",
                str(root / "train"),
                "--baseline-output-dir",
                str(root / "eval"),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--attn-implementation",
                "eager",
                "--num-epochs",
                "1",
                "--train-batch-size",
                "2",
                "--eval-batch-size",
                "3",
                "--gradient-accumulation-steps",
                "1",
                "--num-workers",
                "0",
                "--cpu-threads",
                "1",
                "--max-new-tokens",
                "1",
                "--checkpoint-percents",
                "100",
            ]
            with (
                patch("sys.argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                main()
            summaries = list((root / "train").rglob("summary.json"))
            self.assertEqual(3, len(summaries))
            for path in summaries:
                reader = RunArtifacts(path.parent)
                self.assertEqual(2, reader.summary()["optimizer_steps"])
                self.assertEqual(3, reader.evaluation_summary("test")["num_examples"])
                self.assertEqual(2, len(reader.optimization_timeline()))
