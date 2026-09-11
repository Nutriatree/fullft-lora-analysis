from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helpers.tiny_training import make_config
from pubmedqa.eval.reports import RunArtifacts
from scripts.run_pubmedqa_experiments import main


class PipelineCliTest(unittest.TestCase):
    def test_legacy_factory_keeps_variants_and_rejects_invalid_training_selection(self):
        from dataclasses import replace
        from helpers.tiny_training import environment
        from pubmedqa.experiment_runs import build_runner
        from pubmedqa.train.study import build_training_config
        from pubmedqa.experiment_runs import DEFAULT_PATHS, DEFAULT_SHARED_DEFAULTS
        from pubmedqa.config.full_ft import FullFineTuneConfig
        from pubmedqa.config.lora import LoRAFineTuneConfig

        defaults = replace(DEFAULT_SHARED_DEFAULTS, device="cpu", dtype="float32")
        with tempfile.TemporaryDirectory() as directory:
            for tag, expected in (
                ("B0", "baseline"),
                ("F1", "full-ft"),
                ("L1", "lora"),
                ("LL1", "lora"),
            ):
                method, adapter = build_runner(
                    run_id="legacy",
                    run_tag=tag,
                    defaults=defaults,
                    environment=environment(),
                    train_output_dir=Path(directory),
                    baseline_output_dir=Path(directory),
                    target_layer_overrides={"LL1": (0,)},
                )
                self.assertEqual(expected, method)
                if method != "baseline":
                    self.assertIsInstance(
                        adapter.config,
                        FullFineTuneConfig if tag == "F1" else LoRAFineTuneConfig,
                    )
                    adapter.close()
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

    def test_environment_entrypoints_use_the_functional_program(self):
        from dataclasses import asdict
        from types import SimpleNamespace
        from helpers.tiny_training import environment
        from pubmedqa import full_finetune as engine
        from pubmedqa import lora_finetune as lora
        from pubmedqa.config.lora import LoRAFineTuneConfig

        for entry, config_type in (
            (engine, "FullFineTuneCliConfig"),
            (lora, "LoRAFineTuneCliConfig"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                cfg = make_config(Path(directory))
                if entry is lora:
                    cfg = LoRAFineTuneConfig(
                        **{
                            **asdict(cfg),
                            "lora_rank": 2,
                            "lora_alpha": 4,
                            "lora_dropout": 0,
                            "target_modules": ("q_proj",),
                        },
                        lora_bias="none",
                        lora_task_type="CAUSAL_LM",
                        modules_to_save=(),
                        merge_for_eval=False,
                    )
                cli = SimpleNamespace(config=cfg, environment=environment())
                with (
                    patch.object(
                        getattr(entry, config_type), "from_env", return_value=cli
                    ),
                    patch(
                        "pubmedqa.full_finetune.PubMedQATrainingEngine.__init__",
                        side_effect=AssertionError(
                            "environment entry constructed legacy trainer"
                        ),
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    entry.main()
                self.assertTrue(list(Path(directory).rglob("summary.json")))

    def test_standalone_commands_use_functional_pipeline_without_legacy_trainers(self):
        from scripts import run_pubmedqa_full_finetune, run_pubmedqa_lora_finetune
        from helpers.offline import offline_cpu

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
                    patch(
                        "pubmedqa.full_finetune.PubMedQATrainingEngine.__init__",
                        side_effect=AssertionError("CLI constructed a legacy trainer"),
                    ),
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
        from pubmedqa.train.distributed import TrainingSession
        from pubmedqa.train.study import ExperimentStudy, execute_study
        from pubmedqa.experiment_runs import DEFAULT_PATHS, DEFAULT_SHARED_DEFAULTS
        from helpers.tiny_training import environment

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
                    "pubmedqa.train.study.run_training",
                    side_effect=AssertionError("dry-run must not construct a model"),
                ),
                patch(
                    "pubmedqa.train.study.build_baseline_runner",
                    side_effect=AssertionError("manifest constructed baseline"),
                ),
                patch(
                    "pubmedqa.train.study.distributed_control_group",
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
                patch(
                    "pubmedqa.full_finetune.PubMedQATrainingEngine.__init__",
                    side_effect=AssertionError("study constructed legacy trainer"),
                ),
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
