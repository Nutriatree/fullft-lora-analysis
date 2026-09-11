import unittest
from unittest.mock import patch


class TrainingConfigTest(unittest.TestCase):
    def test_config_has_its_own_module(self):
        from pubmedqa.config.train import TrainingCliConfig, TrainingConfig

        self.assertEqual("pubmedqa.config.train", TrainingConfig.__module__)
        with patch.dict(
            "os.environ",
            {
                "PUBMEDQA_TRAIN_PATH": "a",
                "PUBMEDQA_VALIDATION_PATH": "b",
                "PUBMEDQA_LEARNING_RATE": "0.03",
                "PUBMEDQA_DTYPE": "float32",
            },
            clear=True,
        ):
            config = TrainingCliConfig.from_env("full-ft").config
        self.assertEqual(0.03, config.learning_rate)
        self.assertEqual("single", config.distributed_mode)

    def test_supervised_data_has_one_owner(self):
        from pubmedqa.data.supervised import SupervisedDataCollator

        self.assertEqual("pubmedqa.data.supervised", SupervisedDataCollator.__module__)

    def test_lora_specific_settings_keep_precedence(self):
        from pubmedqa.config.train import TrainingCliConfig

        with patch.dict(
            "os.environ",
            {
                "PUBMEDQA_TRAIN_PATH": "a",
                "PUBMEDQA_VALIDATION_PATH": "b",
                "PUBMEDQA_LEARNING_RATE": "0.03",
                "PUBMEDQA_LORA_LEARNING_RATE": "0.01",
            },
            clear=True,
        ):
            config = TrainingCliConfig.from_env("lora").config
        self.assertEqual(0.01, config.learning_rate)
        self.assertIsNotNone(config.adapter)
