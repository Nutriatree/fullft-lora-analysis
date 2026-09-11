"""Full fine-tuning model preparation."""

from __future__ import annotations

from pubmedqa.model.loading import ModelLoadOptions, _load_pretrained


def load_full_model(model_name_or_path: str, *, options: ModelLoadOptions):
    return _load_pretrained(
        model_name_or_path, options=options, allow_multimodal_fallback=True
    )
