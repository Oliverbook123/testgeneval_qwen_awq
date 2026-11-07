"""Exercise the real runner's token-budget path with a local Qwen tokenizer."""

from pathlib import Path
import unittest

from datasets import Dataset
from transformers import AutoTokenizer

from inference.configs.qwen2_prompt import Qwen2Prompt
from inference.huggingface.huggingface_utils import TokenizedDataset
from inference.huggingface.run_huggingface import truncate_prompts


MODEL = Path(
    '/home/oliverbooklinux/.cache/huggingface/hub/'
    'models--Qwen--Qwen2.5-Coder-7B-Instruct-AWQ/snapshots/'
    '8e8ed243bbe6f9a5aff549a0924562fc719b2b8a'
)


class QwenRunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL.is_dir():
            raise unittest.SkipTest('Requires the existing local Qwen tokenizer')
        cls.tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)

    def test_runner_truncation_and_tokenized_dataset_preserve_chat_structure(self):
        adapter = Qwen2Prompt()
        context = 'def example(value):\n    return value + 1\n' * 4000
        user = adapter.PROMPT_FULL.format(
            code_src=context, code_filename='example.py', imports='from example import example')
        prompt = adapter._render(self.tokenizer, user, 'full')
        budget = 22_528 - 8192
        ds = Dataset.from_list([{
            'id': 'budget-case', 'instance_id': 'example-1',
            'setting': 'full', 'prompt': prompt,
        }])
        ds = truncate_prompts(ds, self.tokenizer, budget, prompt_info=adapter)
        record = next(iter(TokenizedDataset(ds, self.tokenizer, n_tasks=1)))
        tokens = record['ids'][:int(record['input_len'])].tolist()
        self.assertLessEqual(len(tokens) + 8192, 22_528)
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        self.assertTrue(text.startswith('<|im_start|>system\n'))
        self.assertIn(adapter.SYSTEM_MESSAGE_FULL, text)
        self.assertIn('<|im_start|>user\n', text)
        self.assertIn('Only output the unit test Python file', text)
        self.assertTrue(text.endswith('<|im_start|>assistant\n'))

    def test_legacy_truncation_without_chat_adapter_still_keeps_tail(self):
        original = 'old source\n' * 3000 + 'most recent task context'
        ds = Dataset.from_list([{'prompt': original, 'setting': 'full'}])
        result = truncate_prompts(ds, self.tokenizer, 128)
        expected = self.tokenizer.decode(self.tokenizer.encode(original)[-128:])
        self.assertEqual(result[0]['prompt'], expected)


if __name__ == '__main__':
    unittest.main()
