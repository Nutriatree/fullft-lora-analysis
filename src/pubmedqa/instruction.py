"""Compatibility facade for :mod:`pubmedqa.domain.instructions`."""

from pubmedqa.domain.instructions import (
    DIRECT_ASSISTANT_ANSWER_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)

__all__ = [
    "DIRECT_ASSISTANT_ANSWER_TEMPLATE",
    "SYSTEM_PROMPT_TEMPLATE",
    "USER_PROMPT_TEMPLATE",
]
