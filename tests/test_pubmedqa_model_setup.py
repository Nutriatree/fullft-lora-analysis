from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helpers.tiny_training import make_config, environment
from pubmedqa.full_finetune import PubMedQAFullFineTuner


class ModelSetupTest(unittest.TestCase):
    def test_shared_loader_keeps_full_only_multimodal_fallback(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import torch
        from pubmedqa.model.loading import ModelLoadOptions
        from pubmedqa.model.full_ft import load_full_model
        from pubmedqa.model.lora import AdapterOptions, load_lora_model

        options = ModelLoadOptions(device=torch.device("cpu"), dtype=torch.float32)
        adapter = AdapterOptions(2, 4.0, 0.0, ("q_proj",))
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
        model = Mock()
        # No remote weights or GPU: exercise method dispatch around a failing
        # causal loader. Consolidation must not broaden LoRA's fallback policy.
        with (
            patch(
                "pubmedqa.model.loading.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            patch(
                "pubmedqa.model.loading.AutoModelForCausalLM.from_pretrained",
                side_effect=ValueError("unsupported architecture"),
            ),
            patch(
                "transformers.AutoModelForImageTextToText.from_pretrained",
                return_value=model,
            ) as fallback,
        ):
            self.assertIs(model, load_full_model("local", options=options)[1])
            fallback.assert_called_once()
            model.to.assert_called_once_with(torch.device("cpu"))
            fallback.reset_mock()
            with self.assertRaisesRegex(ValueError, "unsupported architecture"):
                load_lora_model("local", options=options, adapter=adapter)
            fallback.assert_not_called()

    def test_loading_lives_in_models_and_keeps_padding(self):
        from pubmedqa.model.loading import ModelLoadOptions
        from pubmedqa.model.full_ft import load_full_model
        import torch

        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            options = ModelLoadOptions(device=torch.device("cpu"), dtype=config.dtype)
            tokenizer, model = load_full_model(config.model_name, options=options)
            self.assertEqual("right", tokenizer.padding_side)
            self.assertEqual("cpu", next(model.parameters()).device.type)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_missing_padding_and_eos_reports_actionable_error(self):
        from types import SimpleNamespace
        from pubmedqa.model.loading import ModelLoadOptions
        from pubmedqa.model.full_ft import load_full_model
        import torch

        options = ModelLoadOptions(device=torch.device("cpu"), dtype=torch.float32)
        with patch(
            "pubmedqa.model.loading.AutoTokenizer.from_pretrained",
            return_value=SimpleNamespace(pad_token_id=None, eos_token_id=None),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "neither pad_token nor eos_token"
            ):
                load_full_model("local", options=options)
