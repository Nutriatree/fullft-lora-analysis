"""Credentials and process-environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    hf_token: str | None = None

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        return cls(hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
