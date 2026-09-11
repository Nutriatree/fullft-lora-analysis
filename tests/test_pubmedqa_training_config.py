import unittest
from unittest.mock import patch


class TrainingConfigTest(unittest.TestCase):
    def test_config_has_its_own_module_and_legacy_alias(self):
        from pubmedqa.config.full_ft import FullFineTuneCliConfig, FullFineTuneConfig
        from pubmedqa.full_finetune import FullFineTuneConfig as Legacy

        self.assertIs(FullFineTuneConfig, Legacy)
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
            config = FullFineTuneCliConfig.from_env().config
        self.assertEqual(0.03, config.learning_rate)
        self.assertEqual("single", config.distributed_mode)

    def test_supervised_data_has_one_owner(self):
        from pubmedqa.data.supervised import SupervisedDataCollator
        from pubmedqa.full_finetune import SupervisedDataCollator as Legacy

        self.assertIs(SupervisedDataCollator, Legacy)

    def test_lora_specific_settings_keep_precedence(self):
        from pubmedqa.config.lora import LoRAFineTuneCliConfig

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
            config = LoRAFineTuneCliConfig.from_env().config
        self.assertEqual(0.01, config.learning_rate)
