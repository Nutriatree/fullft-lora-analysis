from __future__ import annotations

import unittest

from pubmedqa.answer_parser import parse_pubmedqa_answer, require_pubmedqa_label
from pubmedqa.prompt_builder import (
    build_assistant_answer,
    build_messages,
    build_plain_prompt,
    build_tokenizer_prompt,
    example_from_record,
)


class PubMedQAPromptParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "pubid": "123",
            "dataset_id": "pqa_labeled",
            "split": "test",
            "question": "Does the intervention improve survival?",
            "context": {
                "contexts": ["The trial reported improved survival.", "No major harms were observed."],
                "labels": ["BACKGROUND", "RESULTS"],
                "meshes": ["Survival", "Clinical Trial"],
            },
            "long_answer": "The intervention improved survival in the trial.",
            "final_decision": "yes",
        }

    def test_build_messages_for_inference(self) -> None:
        example = example_from_record(self.record)
        messages = build_messages(example)

        self.assertEqual(["system", "user"], [message["role"] for message in messages])
        self.assertIn("Answer the question based only on the provided abstract context.", messages[0]["content"])
        self.assertNotIn("Examples:", messages[1]["content"])
        self.assertIn("Choose exactly one answer: yes, no, or maybe.", messages[1]["content"])
        self.assertIn("Does the intervention improve survival?", messages[1]["content"])
        self.assertIn("1. The trial reported improved survival.", messages[1]["content"])
        self.assertNotIn("MeSH terms", messages[1]["content"])
        self.assertNotIn("[BACKGROUND]", messages[1]["content"])
        self.assertNotIn("Reasoning:", messages[1]["content"])

    def test_build_messages_for_supervised_target(self) -> None:
        example = example_from_record(self.record)
        messages = build_messages(example, include_answer=True)

        self.assertEqual("assistant", messages[-1]["role"])
        self.assertEqual("yes", messages[-1]["content"])
        self.assertEqual("yes", build_assistant_answer("YES"))

    def test_plain_prompt(self) -> None:
        example = example_from_record(self.record)
        prompt = build_plain_prompt(example)

        self.assertIn("System:", prompt)
        self.assertIn("Assistant:", prompt)
        self.assertIn("Answer the question based only on the provided abstract context.", prompt)
        self.assertNotIn("Reasoning:", prompt)
        self.assertNotIn(" yes", prompt.split("Assistant:")[-1])

    def test_qwen3_chat_template_disables_thinking(self) -> None:
        example = example_from_record(self.record)

        class FakeTokenizer:
            chat_template = "stub"
            name_or_path = "Qwen/Qwen3-0.6B"

            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append({"messages": messages, **kwargs})
                return "rendered"

        tokenizer = FakeTokenizer()
        rendered = build_tokenizer_prompt(tokenizer, example)

        self.assertEqual("rendered", rendered)
        self.assertEqual(False, tokenizer.calls[0]["enable_thinking"])
        self.assertEqual(True, tokenizer.calls[0]["add_generation_prompt"])

    def test_parser_handles_answer_line_variant(self) -> None:
        result = parse_pubmedqa_answer("Answer: maybe")

        self.assertTrue(result.ok)
        self.assertEqual("maybe", result.label)
        self.assertEqual("answer_line", result.method)

    def test_parser_prefers_exact_label(self) -> None:
        result = parse_pubmedqa_answer("yes")

        self.assertTrue(result.ok)
        self.assertEqual("yes", result.label)
        self.assertEqual("exact_label", result.method)

    def test_parser_handles_json(self) -> None:
        result = parse_pubmedqa_answer('{"answer": "no"}')

        self.assertTrue(result.ok)
        self.assertEqual("no", result.label)
        self.assertEqual("json", result.method)

    def test_parser_strict_fallback(self) -> None:
        self.assertEqual("yes", require_pubmedqa_label("yes, based on the abstract.", strict=True))
        self.assertFalse(parse_pubmedqa_answer("The answer is yes.", strict=True).ok)


if __name__ == "__main__":
    unittest.main()
