"""Offline checks that original generations survive strict postprocessing."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from inference.huggingface.huggingface_utils import complete_code


def completion(text, token_ids=None, finish_reason="stop", stop_reason=None):
    return SimpleNamespace(
        text=text,
        token_ids=token_ids,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


class FakeVLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def generate(self, *, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [SimpleNamespace(outputs=values) for values in self.outputs]


class GenerationRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="testgeneval-generations-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "model__dataset__raw.jsonl"
        self.prompts = self.directory / "model__dataset__prompt.jsonl"
        self.generations = self.directory / "model__dataset__generations.jsonl"

    @staticmethod
    def batch(sample_ids=("sample-a",), settings=None):
        count = len(sample_ids)
        settings = settings or ["first"] * count
        return {
            "ids": torch.tensor([[11, 12, 13, 0], [21, 0, 0, 0]][:count]),
            "input_len": torch.tensor([3, 1][:count]),
            "instance_id": [f"repo-{index}" for index in range(count)],
            "id": list(sample_ids),
            "setting": list(settings),
            "prompt": [f"prompt for {sample_id}" for sample_id in sample_ids],
        }

    @staticmethod
    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    def generate(self, model, batch, postprocess, num_samples=2, is_full=False):
        complete_code(
            model=model,
            tokenizer=None,
            sampling_params=SimpleNamespace(n=num_samples),
            dataloader=[batch],
            num_samples=num_samples,
            n_tasks=len(batch["id"]),
            postprocess_fn=postprocess,
            is_full=is_full,
            output_file=self.output,
            prompt_file=self.prompts,
            use_huggingface=False,
        )

    def test_two_candidates_per_prompt_keep_trace_metadata_and_prompt_pairing(self):
        texts = [
            "```python\ndef test_a0():\n    assert True\n```",
            "```python\ndef test_a1():\n    assert False\n```",
            "```python\ndef test_b0():\n    assert '中文'\n```",
            "```python\ndef test_b1():\n    assert 1\n```",
        ]
        model = FakeVLLM([
            [completion(texts[0], [100, 101], "stop", 151645),
             completion(texts[1], [102, 103, 104], "length")],
            [completion(texts[2], [200], "stop", "END"),
             completion(texts[3], [], "stop")],
        ])

        def postprocess(text, is_full):
            self.assertFalse(is_full)
            # Reading during the callback proves that the original generation
            # has been flushed before a potentially destructive postprocessor.
            records = self.read_jsonl(self.generations)
            self.assertIn(text, [record["raw_text"] for record in records])
            return "processed:" + text

        self.generate(model, self.batch(("sample-a", "sample-b"), ["first", "last"]),
                      postprocess)
        traces = self.read_jsonl(self.generations)
        predictions = self.read_jsonl(self.output)
        prompts = self.read_jsonl(self.prompts)

        self.assertEqual(model.calls[0][0], [
            {"prompt_token_ids": [11, 12, 13]}, {"prompt_token_ids": [21]},
        ])
        self.assertEqual([row["raw_text"] for row in traces], texts)
        self.assertEqual([row["candidate_index"] for row in traces], [0, 1, 0, 1])
        self.assertEqual([row["generated_tokens"] for row in traces], [2, 3, 1, 0])
        self.assertEqual([row["prompt_tokens"] for row in traces], [3, 3, 1, 1])
        self.assertEqual([row["finish_reason"] for row in traces],
                         ["stop", "length", "stop", "stop"])
        self.assertEqual([row["stop_reason"] for row in traces],
                         [151645, None, "END", None])
        expected = [
            ("sample-a", "first", 0), ("sample-a", "first", 1),
            ("sample-b", "last", 0), ("sample-b", "last", 1),
        ]
        self.assertEqual([(row["id"], row["setting"], row["candidate_index"])
                          for row in traces], expected)
        self.assertEqual(len(prompts), 4)
        self.assertEqual(len(predictions), 4)
        for trace, pred, prompt in zip(traces, predictions, prompts):
            for key in ("instance_id", "id", "setting", "candidate_index"):
                self.assertEqual(trace[key], pred[key])
            for key in ("id", "setting"):
                self.assertEqual(trace[key], prompt[key])
            self.assertEqual(prompt["prompt"], f"prompt for {trace['id']}")
            self.assertEqual(pred["pred"], "processed:" + trace["raw_text"])
            self.assertIsNone(pred["postprocessing_error"])

    def test_failure_sentinels_keep_original_text_and_explicit_reason(self):
        model = FakeVLLM([[
            completion("A real model answer without a code fence.", [1, 2]),
            completion("```python\n# A comment, but no function.\n```", [3, 4]),
        ]])

        def strict_postprocess(text, is_full):
            return "compilation error" if "```" not in text else "this does not compile"

        self.generate(model, self.batch(), strict_postprocess)
        traces = self.read_jsonl(self.generations)
        predictions = self.read_jsonl(self.output)
        self.assertEqual([row["raw_text"] for row in traces], [
            "A real model answer without a code fence.",
            "```python\n# A comment, but no function.\n```",
        ])
        self.assertEqual([row["pred"] for row in predictions],
                         ["compilation error", "this does not compile"])
        self.assertEqual([row["postprocessing_error"] for row in predictions],
                         ["missing_code_fence", "no_function_extracted"])

    def test_empty_postprocessed_output_has_empty_output_reason(self):
        model = FakeVLLM([[completion("```python\n\n```", [1])]])
        self.generate(model, self.batch(), lambda text, is_full: "", num_samples=1,
                      is_full=True)
        prediction = self.read_jsonl(self.output)[0]
        self.assertEqual(prediction["pred"], "")
        self.assertEqual(prediction["postprocessing_error"], "empty_output")
        self.assertEqual(self.read_jsonl(self.generations)[0]["raw_text"],
                         "```python\n\n```")

    def test_postprocessing_exception_cannot_erase_flushed_original_generation(self):
        original = "```python\ndef test_saved():\n    assert True\n```"
        second_original = "```python\ndef test_also_saved():\n    assert 2 + 2 == 4\n```"
        model = FakeVLLM([[
            completion(original, [5, 6, 7]),
            completion(second_original, [8, 9], "length"),
        ]])

        def fail_postprocessing(text, is_full):
            self.assertEqual(text, original)
            self.assertEqual(
                [record["raw_text"] for record in self.read_jsonl(self.generations)],
                [original, second_original],
            )
            raise ValueError("Strict extraction failed")

        with self.assertRaisesRegex(ValueError, "Strict extraction failed"):
            self.generate(model, self.batch(), fail_postprocessing, num_samples=2)
        traces = self.read_jsonl(self.generations)
        self.assertEqual(len(traces), 2)
        self.assertEqual([row["raw_text"] for row in traces],
                         [original, second_original])
        self.assertEqual([row["candidate_index"] for row in traces], [0, 1])
        self.assertEqual([row["generated_tokens"] for row in traces], [3, 2])

    def test_missing_vllm_token_metadata_is_unknown(self):
        output = SimpleNamespace(text="```python\ndef test_ok(): pass\n```", index=0)
        model = FakeVLLM([[output]])
        self.generate(model, self.batch(), lambda text, is_full: text, num_samples=1)
        trace = self.read_jsonl(self.generations)[0]
        self.assertIsNone(trace["generated_tokens"])
        self.assertIsNone(trace["finish_reason"])
        self.assertIsNone(trace["stop_reason"])
        self.assertEqual(trace["prompt_tokens"], 3)

    def test_repeated_calls_append_all_files_without_overwriting(self):
        for sample_id in ("sample-a", "sample-b"):
            model = FakeVLLM([[
                completion(f"{sample_id} answer 0", [1]),
                completion(f"{sample_id} answer 1", [2]),
            ]])
            self.generate(model, self.batch((sample_id,)),
                          lambda text, is_full: "processed:" + text)

        traces = self.read_jsonl(self.generations)
        predictions = self.read_jsonl(self.output)
        prompts = self.read_jsonl(self.prompts)
        self.assertEqual([row["id"] for row in traces],
                         ["sample-a", "sample-a", "sample-b", "sample-b"])
        self.assertEqual(len(predictions), 4)
        self.assertEqual(len(prompts), 4)
        self.assertEqual([row["raw_text"] for row in traces], [
            "sample-a answer 0", "sample-a answer 1",
            "sample-b answer 0", "sample-b answer 1",
        ])

if __name__ == "__main__":
    unittest.main()
