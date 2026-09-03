"""Prompt builders for PubMedQA yes/no/maybe QA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pubmedqa.instruction import (
    DIRECT_ASSISTANT_ANSWER_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
)
from pubmedqa.labels import normalize_label, require_label


ChatMessage = dict[Literal["role", "content"], str]


@dataclass(frozen=True)
class PubMedQAExample:
    """Normalized example consumed by prompt builders."""

    pubid: str
    question: str
    contexts: tuple[str, ...]
    labels: tuple[str, ...] = ()
    meshes: tuple[str, ...] = ()
    final_decision: str | None = None
    long_answer: str | None = None
    dataset_id: str | None = None
    split: str | None = None


def example_from_record(record: dict[str, Any]) -> PubMedQAExample:
    """Build a prompt example from a generated PubMedQA JSON/JSONL row."""

    context = record.get("context") or {}
    contexts = context.get("contexts")
    if contexts is None:
        contexts = record.get("contexts", [])
    labels = context.get("labels") or record.get("labels") or ()
    meshes = context.get("meshes") or record.get("meshes") or ()

    return PubMedQAExample(
        pubid=str(record.get("pubid", "")),
        question=str(record["question"]),
        contexts=tuple(str(item) for item in contexts),
        labels=tuple(str(item) for item in labels),
        meshes=tuple(str(item) for item in meshes),
        final_decision=normalize_label(record.get("final_decision")),
        long_answer=record.get("long_answer"),
        dataset_id=record.get("dataset_id"),
        split=record.get("split"),
    )


def format_context(example: PubMedQAExample) -> str:
    """Format PubMedQA abstract snippets into a stable prompt context block."""

    blocks: list[str] = []
    for index, text in enumerate(example.contexts, start=1):
        blocks.append(f"{index}. {text}".strip())
    return "\n".join(blocks)


def build_system_prompt() -> str:
    """System instruction used for chat-template capable causal LMs."""

    return SYSTEM_PROMPT_TEMPLATE


def build_user_prompt(example: PubMedQAExample) -> str:
    """Build the user message for one PubMedQA example."""

    return USER_PROMPT_TEMPLATE.format(
        question=example.question,
        context=format_context(example),
    )


def build_assistant_answer(label: str) -> str:
    """Build the supervised target text used for full fine-tune and LoRA."""

    return DIRECT_ASSISTANT_ANSWER_TEMPLATE.format(answer=require_label(label))


def build_messages(
    example: PubMedQAExample,
    *,
    include_answer: bool = False,
) -> list[ChatMessage]:
    """Build chat messages for baseline inference or supervised fine-tuning.

    For baseline inference, use ``include_answer=False`` and pass the returned
    messages to ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``.
    For SFT/full fine-tune/LoRA, use ``include_answer=True`` and train on the
    assistant answer portion.
    """

    messages: list[ChatMessage] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(example)},
    ]
    if include_answer:
        if example.final_decision is None:
            raise ValueError("include_answer=True requires final_decision")
        messages.append({"role": "assistant", "content": build_assistant_answer(example.final_decision)})
    return messages


def build_plain_prompt(
    example: PubMedQAExample,
    *,
    include_answer: bool = False,
) -> str:
    """Build a non-chat prompt fallback for tokenizers without chat templates."""

    prompt = (
        f"System: {build_system_prompt()}\n\n"
        f"User: {build_user_prompt(example)}\n\n"
        "Assistant:"
    )
    if include_answer:
        if example.final_decision is None:
            raise ValueError("include_answer=True requires final_decision")
        prompt += f" {build_assistant_answer(example.final_decision)}"
    return prompt


def build_tokenizer_prompt(
    tokenizer: Any,
    example: PubMedQAExample,
    *,
    include_answer: bool = False,
    add_generation_prompt: bool | None = None,
) -> str:
    """Render messages with a tokenizer chat template when available.

    This keeps Qwen, Llama Instruct, Gemma Instruct, full fine-tune, and LoRA
    on the same semantic prompt while still respecting model-specific templates.
    """

    messages = build_messages(example, include_answer=include_answer)
    if add_generation_prompt is None:
        add_generation_prompt = not include_answer

    if getattr(tokenizer, "chat_template", None):
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).lower()
        if "qwen3" in tokenizer_name:
            chat_template_kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        except TypeError:
            chat_template_kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    return build_plain_prompt(example, include_answer=include_answer)
