from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from pubmedqa.config.env import EnvironmentConfig
from pubmedqa.config.experiments import (
    DEFAULT_PATHS,
    DEFAULT_SHARED_DEFAULTS,
    build_full_ft_config,
    build_lora_config,
    resolve_data_fraction,
    resolve_run_spec,
)
from scripts.run_pubmedqa_experiments import (
    ExperimentStudy,
    save_manifest,
)
from scripts.run_pubmedqa_experiments import (
    _destroy_distributed_process_groups,
    _run_on_rank_zero,
)
from scripts.run_pubmedqa_experiments import (
    parse_args as parse_experiment_args,
)


class PubMedQAExperimentRunsTest(unittest.TestCase):
    def test_registered_training_configs_preserve_method_specific_values(self) -> None:
        expected = {
            "F1": ("full-ft", (), None, DEFAULT_SHARED_DEFAULTS.full_ft_learning_rate),
            "L1": (
                "lora",
                ("q_proj", "v_proj"),
                8,
                DEFAULT_SHARED_DEFAULTS.lora_learning_rate,
            ),
            "L2": (
                "lora",
                ("q_proj", "k_proj", "v_proj", "o_proj"),
                8,
                DEFAULT_SHARED_DEFAULTS.lora_learning_rate,
            ),
            "L3": (
                "lora",
                ("q_proj", "v_proj"),
                4,
                DEFAULT_SHARED_DEFAULTS.lora_learning_rate,
            ),
            "L4": (
                "lora",
                ("q_proj", "v_proj"),
                16,
                DEFAULT_SHARED_DEFAULTS.lora_learning_rate,
            ),
            "LL1": (
                "lora",
                ("q_proj", "v_proj"),
                8,
                DEFAULT_SHARED_DEFAULTS.lora_learning_rate,
            ),
            "LL2": (
                "lora",
                ("q_proj", "v_proj"),
                8,
                DEFAULT_SHARED_DEFAULTS.lora_learning_rate,
            ),
        }
        self.assertEqual("baseline", resolve_run_spec("B0").method)
        for run_tag, (method, modules, rank, learning_rate) in expected.items():
            spec = resolve_run_spec(run_tag)
            if method == "full-ft":
                config = build_full_ft_config(
                    run_id="snapshot",
                    spec=spec,
                    paths=DEFAULT_PATHS,
                    output_dir=Path("outputs/pubmedqa_train"),
                    defaults=DEFAULT_SHARED_DEFAULTS,
                )
            else:
                config = build_lora_config(
                    run_id="snapshot",
                    spec=spec,
                    paths=DEFAULT_PATHS,
                    output_dir=Path("outputs/pubmedqa_train"),
                    defaults=DEFAULT_SHARED_DEFAULTS,
                    target_layers_override=(0,) if spec.layer_scope != "all" else None,
                )
            self.assertEqual(method, config.method_name, run_tag)
            self.assertEqual(modules, config.target_modules, run_tag)
            self.assertEqual(rank, config.lora_rank, run_tag)
            self.assertEqual(learning_rate, config.learning_rate, run_tag)

    def test_manifest_schema_is_stable_across_distributed_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            keys_by_mode: dict[str, set[str]] = {}
            for mode in ("single", "ddp", "fsdp"):
                study = ExperimentStudy(
                    run_id=f"manifest-{mode}",
                    run_tags=("B0", "F1", "L1"),
                    paths=DEFAULT_PATHS,
                    defaults=replace(DEFAULT_SHARED_DEFAULTS, distributed_mode=mode),
                    baseline_output_dir=Path(tmp_dir) / "baseline",
                    train_output_dir=Path(tmp_dir) / "train",
                    target_layer_overrides={},
                    environment=EnvironmentConfig(),
                )
                payload = json.loads(save_manifest(study).read_text(encoding="utf-8"))
                keys_by_mode[mode] = set(payload)
                self.assertEqual(mode, payload["shared_defaults"]["distributed_mode"])
                self.assertEqual(["B0", "F1", "L1"], payload["requested_runs"])

            self.assertEqual(keys_by_mode["single"], keys_by_mode["ddp"])
            self.assertEqual(keys_by_mode["single"], keys_by_mode["fsdp"])

    def test_resolve_data_fraction_for_low_data_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train.jsonl"
            train_path.write_text("{}\n" * 100, encoding="utf-8")
            paths = DEFAULT_PATHS.__class__(
                train_path=train_path,
                validation_path=DEFAULT_PATHS.validation_path,
                test_path=DEFAULT_PATHS.test_path,
                baseline_eval_path=DEFAULT_PATHS.baseline_eval_path,
            )
            spec = resolve_run_spec("F2")
            fraction = resolve_data_fraction(spec, paths)
            self.assertGreater(fraction, 0.0)
            self.assertLessEqual(fraction, 1.0)

    def test_build_full_ft_config_uses_run_spec(self) -> None:
        config = build_full_ft_config(
            run_id="test_run",
            spec=resolve_run_spec("F1"),
            paths=DEFAULT_PATHS,
            output_dir=Path("outputs/pubmedqa_train"),
            defaults=DEFAULT_SHARED_DEFAULTS,
        )
        self.assertEqual("F1", config.run_tag)
        self.assertEqual("full-ft", config.method_name)
        self.assertEqual("full-ft", config.condition)

    def test_build_lora_config_uses_lora_fields(self) -> None:
        config = build_lora_config(
            run_id="test_run",
            spec=resolve_run_spec("L2"),
            paths=DEFAULT_PATHS,
            output_dir=Path("outputs/pubmedqa_train"),
            defaults=DEFAULT_SHARED_DEFAULTS,
        )
        self.assertEqual("L2", config.run_tag)
        self.assertEqual("lora", config.method_name)
        self.assertEqual(
            ("q_proj", "k_proj", "v_proj", "o_proj"), config.target_modules
        )
        self.assertEqual(8, config.lora_rank)

    def test_selective_lora_requires_explicit_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires explicit target layers"):
            build_lora_config(
                run_id="test_run",
                spec=resolve_run_spec("LL1"),
                paths=DEFAULT_PATHS,
                output_dir=Path("outputs/pubmedqa_train"),
                defaults=DEFAULT_SHARED_DEFAULTS,
            )

    def test_selective_lora_accepts_layer_override(self) -> None:
        config = build_lora_config(
            run_id="test_run",
            spec=resolve_run_spec("LL1"),
            paths=DEFAULT_PATHS,
            output_dir=Path("outputs/pubmedqa_train"),
            defaults=DEFAULT_SHARED_DEFAULTS,
            target_layers_override=(20, 21, 22),
        )
        self.assertEqual((20, 21, 22), config.target_layers)

    def test_rank_zero_operation_uses_long_running_control_group(self) -> None:
        control_group = object()
        with patch(
            "pubmedqa.train.distributed.dist.broadcast_object_list"
        ) as broadcast:
            result = _run_on_rank_zero(
                lambda: "baseline-complete",
                is_main_process=True,
                control_group=control_group,
                operation_name="baseline",
            )

        self.assertEqual("baseline-complete", result)
        self.assertIs(control_group, broadcast.call_args.kwargs["group"])

    def test_nonzero_rank_does_not_run_baseline_operation(self) -> None:
        operation = Mock(side_effect=AssertionError("nonzero rank must remain idle"))

        def receive_result(payload, *, src, group):
            payload[0] = {"ok": True, "result": "rank-zero-result"}

        with patch(
            "pubmedqa.train.distributed.dist.broadcast_object_list",
            side_effect=receive_result,
        ):
            result = _run_on_rank_zero(
                operation,
                is_main_process=False,
                control_group=object(),
                operation_name="baseline",
            )

        self.assertEqual("rank-zero-result", result)
        operation.assert_not_called()

    def test_destroy_distributed_process_groups_destroys_control_then_default(
        self,
    ) -> None:
        control_group = object()
        with (
            patch("pubmedqa.train.distributed.dist.is_initialized", return_value=True),
            patch("pubmedqa.train.distributed.dist.destroy_process_group") as destroy,
        ):
            _destroy_distributed_process_groups(control_group)

        self.assertEqual([((control_group,), {}), ((), {})], destroy.call_args_list)

    def test_experiment_cli_accepts_smoke_test_dataset_limits(self) -> None:
        argv = [
            "run_pubmedqa_experiments.py",
            "--max-train-examples",
            "2",
            "--max-validation-examples",
            "3",
            "--max-test-examples",
            "4",
        ]
        with patch("sys.argv", argv):
            args = parse_experiment_args()

        self.assertEqual(2, args.max_train_examples)
        self.assertEqual(3, args.max_validation_examples)
        self.assertEqual(4, args.max_test_examples)

    def test_dataset_limits_propagate_to_full_ft_and_lora_configs(self) -> None:
        defaults = replace(
            DEFAULT_SHARED_DEFAULTS,
            max_train_examples=2,
            max_validation_examples=3,
            max_test_examples=4,
        )
        full_config = build_full_ft_config(
            run_id="smoke",
            spec=resolve_run_spec("F1"),
            paths=DEFAULT_PATHS,
            output_dir=Path("outputs/pubmedqa_train"),
            defaults=defaults,
        )
        lora_config = build_lora_config(
            run_id="smoke",
            spec=resolve_run_spec("L1"),
            paths=DEFAULT_PATHS,
            output_dir=Path("outputs/pubmedqa_train"),
            defaults=defaults,
        )

        for config in (full_config, lora_config):
            self.assertEqual(2, config.max_train_examples)
            self.assertEqual(3, config.max_validation_examples)
            self.assertEqual(4, config.max_test_examples)


if __name__ == "__main__":
    unittest.main()
