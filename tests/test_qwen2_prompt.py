"""Offline tests for Qwen's upstream-compatible TestGenEval chat adapter."""

import os
from pathlib import Path
import unittest

from datasets import Dataset, DatasetDict

from inference.configs.qwen2_prompt import Qwen2Prompt


MODEL = Path(
    "/home/oliverbooklinux/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-Coder-7B-Instruct-AWQ/snapshots/"
    "8e8ed243bbe6f9a5aff549a0924562fc719b2b8a"
)


def dataset(code=None):
    return DatasetDict({"test": Dataset.from_list([{
        "id": "example-1", "code_file": "example.py",
        "code_src": code or "def example(value):\n    return value + 1\n",
        "local_imports": ["from example import example"],
        "preds_context": {
            "preamble": "from example import example\n",
            "last_minus_one": "def test_existing():\n    assert example(1) == 2\n",
            "last": "def test_existing():\n    assert example(1) == 2\n",
        },
    }])})


class CharacterTokenizer:
    """A deliberately non-ChatML template checks that boundaries are derived."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if tokenize or not add_generation_prompt:
            raise AssertionError("Incorrect chat-template options")
        return "".join(
            f"[ROLE={m['role']}]\n{m['content']}[END]\n" for m in messages
        ) + "[ROLE=assistant]\n"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(token) for token in tokens)


class Qwen2PromptTests(unittest.TestCase):
    def setUp(self):
        self.adapter = Qwen2Prompt()
        self.tokenizer = CharacterTokenizer()

    def prompts(self, code=None, no_import=False):
        return self.adapter.add_prompts_to_dataset(
            dataset(code), tokenizer=self.tokenizer, no_import=no_import
        )["test"][0]["preds_prompts"]

    def test_all_four_settings_get_their_system_role_and_generation_prompt(self):
        prompts = self.prompts()
        self.assertEqual(set(prompts), {"full", "first", "last", "extra"})
        for setting, prompt in prompts.items():
            system = (self.adapter.SYSTEM_MESSAGE_FULL if setting == "full"
                      else self.adapter.SYSTEM_MESSAGE)
            self.assertTrue(prompt.startswith(f"[ROLE=system]\n{system}[END]\n"))
            self.assertIn("[ROLE=user]\nBelow is a code file:", prompt)
            self.assertTrue(prompt.endswith("[ROLE=assistant]\n"))

    def test_missing_tokenizer_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            self.adapter.add_prompts_to_dataset(dataset())

    def test_upstream_strict_output_protocol_is_preserved(self):
        plain = "def test_one():\n    assert True\n"
        self.assertEqual(self.adapter.postprocess_output(plain, True), "compilation error")
        fenced = "```python\n" + plain + "def test_two():\n    assert False\n```"
        self.assertIn("test_two", self.adapter.postprocess_output(fenced, True))
        completion = self.adapter.postprocess_output(fenced, False)
        self.assertIn("test_one", completion)
        self.assertNotIn("test_two", completion)

    def test_fitting_prompt_is_unchanged(self):
        for setting, prompt in self.prompts().items():
            self.assertEqual(prompt, self.adapter.truncate_prompt(
                prompt, self.tokenizer, len(prompt), setting))

    def test_truncation_preserves_template_and_complete_instruction_tail(self):
        prompts = self.prompts("obsolete_context\n" * 700 + "recent_context\n")
        for setting, prompt in prompts.items():
            marked = self.adapter._render(self.tokenizer, self.adapter._USER_MARKER, setting)
            prefix, suffix = marked.split(self.adapter._USER_MARKER)
            body = prompt[len(prefix):-len(suffix)]
            tail = self.adapter._instruction_tail(body, setting)
            budget = len(prefix + tail + suffix) + 150
            shortened = self.adapter.truncate_prompt(prompt, self.tokenizer, budget, setting)
            self.assertLessEqual(len(self.tokenizer.encode(shortened)), budget)
            self.assertTrue(shortened.startswith(prefix))
            self.assertTrue(shortened.endswith(tail + suffix))
            self.assertLess(shortened.count("obsolete_context"),
                            prompt.count("obsolete_context"))

    def test_no_import_full_prompt_keeps_its_instructions_when_truncated(self):
        prompt = self.prompts("obsolete_context\n" * 700, no_import=True)["full"]
        shortened = self.adapter.truncate_prompt(prompt, self.tokenizer, 1000, "full")
        self.assertIn("Your job is to output a corresponding unit test file", shortened)
        self.assertIn("Only output the unit test Python file", shortened)
        self.assertNotIn("Here are some examples of how to import", shortened)

    def test_tiny_budget_cannot_silently_remove_task_instructions(self):
        prompt = self.prompts("old_context\n" * 700)["full"]
        with self.assertRaisesRegex(ValueError, "cannot preserve"):
            self.adapter.truncate_prompt(prompt, self.tokenizer, 20, "full")

    def test_malformed_template_and_wrong_setting_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.adapter.truncate_prompt("a bare text prompt", self.tokenizer, 1000, "full")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.adapter.truncate_prompt(self.prompts()["full"], self.tokenizer, 1000, "unknown")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.adapter.truncate_prompt(self.prompts()["full"], self.tokenizer, 0, "full")


class Qwen2LocalTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get("TESTGENEVAL_QWEN_TOKENIZER", str(MODEL)))
        if not path.is_dir():
            raise unittest.SkipTest("Set TESTGENEVAL_QWEN_TOKENIZER to a local Qwen tokenizer")
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)

    def test_real_chat_roles_survive_the_actual_22k_prompt_budgets(self):
        adapter = Qwen2Prompt()
        source = "def example(value):\n    return value + 1\n" * 4000
        prompts = adapter.add_prompts_to_dataset(
            dataset(source), tokenizer=self.tokenizer
        )["test"][0]["preds_prompts"]
        for setting, prompt in prompts.items():
            with self.subTest(setting=setting):
                budget = 22_528 - 8192 if setting == "full" else 22_528 - 1000
                self.assertGreater(len(self.tokenizer.encode(prompt)), budget)
                shortened = adapter.truncate_prompt(prompt, self.tokenizer, budget, setting)
                self.assertLessEqual(len(self.tokenizer.encode(shortened)), budget)
                self.assertTrue(shortened.startswith("<|im_start|>system\n"))
                self.assertIn("<|im_start|>user\n", shortened)
                self.assertTrue(shortened.endswith("<|im_start|>assistant\n"))
                self.assertIn("Only output", shortened)
                expected_system = (adapter.SYSTEM_MESSAGE_FULL if setting == "full"
                                   else adapter.SYSTEM_MESSAGE)
                self.assertIn(expected_system, shortened)


if __name__ == "__main__":
    unittest.main()
