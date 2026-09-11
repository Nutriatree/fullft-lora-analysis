"""Prompt construction shared by training and inference."""

from __future__ import annotations

from typing import Any, Literal

from pubmedqa.data.records import PubMedQAExample, example_from_record, require_label

SYSTEM_PROMPT_TEMPLATE = (
    "You are a biomedical question-answering assistant. "
    "Answer the question based only on the provided abstract context."
)

USER_PROMPT_TEMPLATE = """Choose exactly one answer: yes, no, or maybe.

Question:
{question}

Context:
{context}

Respond with only one word: yes, no, or maybe."""

DIRECT_ASSISTANT_ANSWER_TEMPLATE = "{answer}"


ChatMessage = dict[Literal["role", "content"], str]


def format_context(example: PubMedQAExample) -> str:
    return "\n".join(
        f"{index}. {text}".strip()
        for index, text in enumerate(example.contexts, start=1)
    )


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE


def build_user_prompt(example: PubMedQAExample) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=example.question,
        context=format_context(example),
    )


def build_assistant_answer(label: str) -> str:
    return DIRECT_ASSISTANT_ANSWER_TEMPLATE.format(answer=require_label(label))


def build_messages(
    example: PubMedQAExample,
    *,
    include_answer: bool = False,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(example)},
    ]
    if include_answer:
        if example.final_decision is None:
            raise ValueError("include_answer=True requires final_decision")
        messages.append(
            {
                "role": "assistant",
                "content": build_assistant_answer(example.final_decision),
            }
        )
    return messages


def build_plain_prompt(
    example: PubMedQAExample,
    *,
    include_answer: bool = False,
) -> str:
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
