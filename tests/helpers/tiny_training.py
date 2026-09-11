"""Synthetic, deterministic inputs for the actual training and saving pipeline."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pubmedqa.config import EnvironmentConfig
from pubmedqa.config.full_ft import FullFineTuneCliConfig


def make_local_model(root: Path) -> Path:
    """Save a two-layer Llama and tokenizer without contacting a model registry."""
    path = root / "tiny-model"
    vocabulary = {
        word: index
        for index, word in enumerate(
            ["[PAD]", "[UNK]", "[EOS]", "yes", "no", "maybe", "Assistant", ":", "test"]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        model_input_names=["input_ids", "attention_mask"],
    )
    torch.manual_seed(123)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            pad_token_id=0,
            eos_token_id=2,
            attention_dropout=0.0,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def make_config(root: Path, **changes):
    model_path = make_local_model(root)
    data_path = root / "synthetic.jsonl"
    records = [
        {
            "pubid": str(i),
            "question": "test",
            "context": {"contexts": ["test"]},
            "final_decision": label,
        }
        for i, label in enumerate(("yes", "no", "maybe"))
    ]
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    with patch.dict(
        "os.environ",
        {
            "PUBMEDQA_TRAIN_PATH": str(data_path),
            "PUBMEDQA_VALIDATION_PATH": str(data_path),
        },
        clear=True,
    ):
        config = FullFineTuneCliConfig.from_env().config
    return replace(
        config,
        **{
            "run_id": "dry-run",
            "model_name": str(model_path),
            "output_dir": root / "runs",
            "test_path": data_path,
            "num_epochs": 1,
            "train_batch_size": 2,
            "eval_batch_size": 3,
            "gradient_accumulation_steps": 1,
            "device": "cpu",
            "dtype": torch.float32,
            "attn_implementation": "eager",
            "cpu_threads": 1,
            "num_workers": 0,
            "max_new_tokens": 1,
            "gradient_checkpointing": False,
            "checkpoint_percents": (100,),
            "track_layerwise_updates": True,
            "log_every_steps": 1,
            "save_optimizer_state": True,
            **changes,
        },
    )


def environment():
    return EnvironmentConfig(hf_token=None)
