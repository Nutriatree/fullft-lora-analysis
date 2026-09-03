"""Instruction text templates for PubMedQA prompting."""

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
