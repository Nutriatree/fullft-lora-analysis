from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pubmedqa.experiment_runs import (
    DEFAULT_PATHS,
    DEFAULT_SHARED_DEFAULTS,
    build_full_ft_config,
    build_lora_config,
    resolve_data_fraction,
    resolve_run_spec,
)


class PubMedQAExperimentRunsTest(unittest.TestCase):
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
        self.assertEqual(("q_proj", "k_proj", "v_proj", "o_proj"), config.target_modules)
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


if __name__ == "__main__":
    unittest.main()
