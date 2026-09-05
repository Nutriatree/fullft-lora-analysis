"""Compatibility facade for :mod:`pubmedqa.domain.prompts`."""

from pubmedqa.domain.prompts import (
    ChatMessage,
    PubMedQAExample,
    build_assistant_answer,
    build_messages,
    build_plain_prompt,
    build_system_prompt,
    build_tokenizer_prompt,
    build_user_prompt,
    example_from_record,
    format_context,
)

__all__ = [
    "ChatMessage",
    "PubMedQAExample",
    "build_assistant_answer",
    "build_messages",
    "build_plain_prompt",
    "build_system_prompt",
    "build_tokenizer_prompt",
    "build_user_prompt",
    "example_from_record",
    "format_context",
]
