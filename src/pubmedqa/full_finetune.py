"""Public notebook adapter for Full FT; execution lives in train.pipeline.

These forwarding methods preserve the established API, not extension hooks for
new training code. Model/optimizer state remains local to run_training.
"""

from __future__ import annotations

import pubmedqa.eval.validation as validation_ops
import pubmedqa.model.full_ft as model_loading
import pubmedqa.train.checkpoints as checkpoint_ops
import pubmedqa.train.distributed as training_runtime
import pubmedqa.train.full_ft as analysis_ops
import pubmedqa.train.lora as lora_analysis_ops
from pubmedqa.config.full_ft import (
    TRAIN_FULL_FINE_TUNE_CONFIG,
    TRAIN_LAYER_CONFIG,
    FullFineTuneCliConfig,
    FullFineTuneConfig,
)
from pubmedqa.config.lora import LoRAFineTuneConfig
from pubmedqa.data.records import load_local_jsonl
from pubmedqa.data.supervised import (
    PubMedQASupervisedDataset,
    SupervisedDataCollator,
    SupervisedExample,
    build_eval_dataloader,
    build_train_dataloader,
)
from pubmedqa.data.supervised import (
    longest_common_prefix_length as _longest_common_prefix_length,
)
from pubmedqa.eval.validation import EvalMetrics, EvalPrediction, EvalResult
from pubmedqa.model.lora import AdapterOptions, load_lora_model
from pubmedqa.train.artifacts import RunFiles, TrainingSummary
from pubmedqa.train.checkpoints import CheckpointRecord
from pubmedqa.train.checkpoints import (
    build_checkpoint_schedule as _build_checkpoint_schedule,
)
from pubmedqa.train.checkpoints import (
    normalize_checkpoint_percents as _normalize_checkpoint_percents,
)
from pubmedqa.train.full_ft import (
    LayerwiseReference,
    LayerwiseUpdateRecord,
)
from pubmedqa.train.full_ft import (
    cosine_similarity as _cosine_similarity,
)
from pubmedqa.train.full_ft import (
    match_tracked_module as _match_tracked_module,
)
from pubmedqa.train.full_ft import (
    parse_layer_index as _parse_layer_index,
)
from pubmedqa.train.loop import TrainStepLog
from pubmedqa.train.pipeline import run_training

# Preserve only constants exported by this established public entry point.
DEFAULT_ATTN_IMPLEMENTATION = TRAIN_FULL_FINE_TUNE_CONFIG.default_attn_implementation
DEFAULT_CHECKPOINT_PERCENTS = TRAIN_LAYER_CONFIG.default_checkpoint_percents
DEFAULT_CONDITION = TRAIN_FULL_FINE_TUNE_CONFIG.default_condition
DEFAULT_CPU_THREADS = TRAIN_FULL_FINE_TUNE_CONFIG.default_cpu_threads
DEFAULT_DATA_FRACTION = TRAIN_FULL_FINE_TUNE_CONFIG.default_data_fraction
DEFAULT_DATA_REGIME = TRAIN_FULL_FINE_TUNE_CONFIG.default_data_regime
DEFAULT_DISTRIBUTED_MODE = "single"
DEFAULT_DEVICE = TRAIN_FULL_FINE_TUNE_CONFIG.default_device
DEFAULT_DTYPE = TRAIN_FULL_FINE_TUNE_CONFIG.default_dtype
DEFAULT_EVAL_BATCH_SIZE = TRAIN_FULL_FINE_TUNE_CONFIG.default_eval_batch_size
DEFAULT_FSDP_CPU_OFFLOAD = False
DEFAULT_GRAD_ACCUM_STEPS = TRAIN_FULL_FINE_TUNE_CONFIG.default_grad_accum_steps
DEFAULT_LAYER_SCOPE = TRAIN_LAYER_CONFIG.default_layer_scope
DEFAULT_LEARNING_RATE = TRAIN_FULL_FINE_TUNE_CONFIG.default_learning_rate
DEFAULT_LOG_EVERY_STEPS = TRAIN_FULL_FINE_TUNE_CONFIG.default_log_every_steps
DEFAULT_MAX_GRAD_NORM = TRAIN_FULL_FINE_TUNE_CONFIG.default_max_grad_norm
DEFAULT_MAX_NEW_TOKENS = TRAIN_FULL_FINE_TUNE_CONFIG.default_max_new_tokens
DEFAULT_METHOD_NAME = TRAIN_FULL_FINE_TUNE_CONFIG.default_method_name
DEFAULT_MODEL_NAME = TRAIN_FULL_FINE_TUNE_CONFIG.default_model_name
DEFAULT_NOTES = TRAIN_LAYER_CONFIG.default_notes
DEFAULT_NUM_EPOCHS = TRAIN_FULL_FINE_TUNE_CONFIG.default_num_epochs
DEFAULT_NUM_WORKERS = TRAIN_FULL_FINE_TUNE_CONFIG.default_num_workers
DEFAULT_OUTPUT_DIR = TRAIN_FULL_FINE_TUNE_CONFIG.default_output_dir
DEFAULT_RUN_TAG = TRAIN_FULL_FINE_TUNE_CONFIG.default_run_tag
DEFAULT_SAVE_OPTIMIZER_STATE = TRAIN_FULL_FINE_TUNE_CONFIG.default_save_optimizer_state
DEFAULT_SEED = TRAIN_FULL_FINE_TUNE_CONFIG.default_seed
DEFAULT_TRAIN_BATCH_SIZE = TRAIN_FULL_FINE_TUNE_CONFIG.default_train_batch_size
DEFAULT_WARMUP_RATIO = TRAIN_FULL_FINE_TUNE_CONFIG.default_warmup_ratio
DEFAULT_WEIGHT_DECAY = TRAIN_FULL_FINE_TUNE_CONFIG.default_weight_decay


class PubMedQATrainingEngine:
    """Compatibility adapter for notebooks; the training program owns no Trainer.

    New callers use run_training directly. These forwarding methods preserve
    historical imports without making them extension hooks for the core loop.
    """

    def __init__(self, config, environment):
        self.config = config
        self.environment = environment
        self.session = training_runtime.TrainingSession.from_config(config)
        self.files = RunFiles.from_config(config)
        self.analysis_history = lora_analysis_ops.AdapterHistory()

    def run(self):
        return run_training(self.config, self.environment, session=self.session)

    def close(self):
        self.session.close()

    def load_model_and_tokenizer(self, source):
        config = getattr(self, "lora_config", self.config)
        options = model_loading.ModelLoadOptions(
            device=self.session.device,
            dtype=config.dtype,
            distributed_mode=config.distributed_mode,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
            hf_token=self.environment.hf_token,
        )
        if isinstance(config, LoRAFineTuneConfig):
            return load_lora_model(
                source, options=options, adapter=AdapterOptions.from_config(config)
            )
        return model_loading.load_full_model(source, options=options)

    def build_eval_dataloader(self, dataset, tokenizer):
        return build_eval_dataloader(
            dataset,
            tokenizer,
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers,
            max_input_tokens=self.config.max_input_tokens,
        )

    def build_train_dataloader(self, dataset, tokenizer):
        loader, self.train_sampler = build_train_dataloader(
            dataset,
            tokenizer,
            batch_size=self.config.train_batch_size,
            seed=self.config.seed,
            num_workers=self.config.num_workers,
            max_input_tokens=self.config.max_input_tokens,
            world_size=self.session.world_size,
            rank=self.session.rank,
        )
        return loader

    def evaluate_split(self, **inputs):
        return validation_ops.evaluate_split(
            settings=validation_ops.EvaluationSettings.from_config(
                self.config, self.session.device
            ),
            evaluations_dir=self.files.evaluations_dir,
            **inputs,
        )

    def _evaluate_checkpoint_on_main(self, **inputs):
        return validation_ops.evaluate_checkpoint_on_main(
            session=self.session,
            settings=validation_ops.EvaluationSettings.from_config(
                self.config, self.session.device
            ),
            evaluations_dir=self.files.evaluations_dir,
            load_model=self.load_model_and_tokenizer,
            **inputs,
        )

    def _save_model_files_to_directory(self, **inputs):
        return checkpoint_ops.save_model_files_to_directory(
            session=self.session, **inputs
        )

    def _create_evaluation_snapshot(self, **inputs):
        return checkpoint_ops.create_evaluation_snapshot(
            session=self.session, files=self.files, **inputs
        )

    def _remove_evaluation_snapshot(self, path):
        return checkpoint_ops.remove_evaluation_snapshot(path, session=self.session)

    def capture_layerwise_references(self, model):
        analysis = (
            lora_analysis_ops
            if isinstance(self.config, LoRAFineTuneConfig)
            else analysis_ops
        )
        return analysis.capture_layerwise_references(
            model,
            session=self.session,
            files=self.files,
            enabled=self.config.track_layerwise_updates,
        )

    def write_layerwise_update_artifacts(self, **inputs):
        if isinstance(self.config, LoRAFineTuneConfig):
            return lora_analysis_ops.write_layerwise_update_artifacts(
                session=self.session,
                files=self.files,
                history=self.analysis_history,
                enabled=self.config.track_layerwise_updates,
                **inputs,
            )
        return analysis_ops.write_layerwise_update_artifacts(
            session=self.session,
            files=self.files,
            enabled=self.config.track_layerwise_updates,
            **inputs,
        )

    def wrap_model_for_training(self, model):
        return self.session.wrap_model(model)

    def _distributed_metadata(self):
        return self.session.metadata()

    def _run_on_main_process(self, operation, *, operation_name):
        return self.session.run_on_main_process(
            operation, operation_name=operation_name
        )

    def _load_examples(self):
        return tuple(
            load_local_jsonl(path)[:limit] if path else []
            for path, limit in (
                (self.config.train_path, self.config.max_train_examples),
                (self.config.validation_path, self.config.max_validation_examples),
                (self.config.test_path, self.config.max_test_examples),
            )
        )

    @property
    def device(self):
        return self.session.device

    @device.setter
    def device(self, value):
        self.session.device = value

    @property
    def rank(self):
        return self.session.rank

    @rank.setter
    def rank(self, value):
        self.session.rank = value

    @property
    def local_rank(self):
        return self.session.local_rank

    @local_rank.setter
    def local_rank(self, value):
        self.session.local_rank = value

    @property
    def world_size(self):
        return self.session.world_size

    @world_size.setter
    def world_size(self, value):
        self.session.world_size = value

    @property
    def control_group(self):
        return self.session.control_group

    @control_group.setter
    def control_group(self, value):
        self.session.control_group = value

    @property
    def _fsdp_layer_classes(self):
        return self.session._fsdp_layer_classes

    @property
    def output_root(self):
        return self.files.output_root

    @property
    def checkpoints_dir(self):
        return self.files.checkpoints_dir

    @property
    def layerwise_dir(self):
        return self.files.layerwise_dir

    @property
    def evaluations_dir(self):
        return self.files.evaluations_dir

    @property
    def logs_dir(self):
        return self.files.logs_dir

    @property
    def transitions_dir(self):
        return self.files.transitions_dir

    @property
    def title(self):
        return self.files.title


class PubMedQAFullFineTuner(PubMedQATrainingEngine):
    """Historical Full FT entry point, delegating to the functional pipeline."""


def main() -> None:
    cli_config = FullFineTuneCliConfig.from_env()
    summary = run_training(cli_config.config, cli_config.environment)
    print(summary.title)
    print(f"Best checkpoint: {summary.best_checkpoint_dir}")
    print(f"Best validation ACC: {summary.best_validation_accuracy:.4f}")
    print(f"Best validation Macro F1: {summary.best_validation_macro_f1:.4f}")
    if summary.test_accuracy is not None:
        print(f"Test ACC: {summary.test_accuracy:.4f}")
        print(f"Test Macro F1: {summary.test_macro_f1:.4f}")


if __name__ == "__main__":
    main()


__all__ = [
    "FullFineTuneCliConfig",
    "FullFineTuneConfig",
    "PubMedQAFullFineTuner",
    "PubMedQASupervisedDataset",
    "SupervisedDataCollator",
    "SupervisedExample",
    "TrainingSummary",
    "main",
]
