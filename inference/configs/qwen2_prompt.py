# Copyright (c) Meta Platforms, Inc. and affiliates.

from datasets import Dataset, DatasetDict

from inference.configs.instruct_prompt import InstructPrompt


class Qwen2Prompt(InstructPrompt):
    """Use Qwen's chat template with the upstream TestGenEval instructions.

    Output postprocessing deliberately remains InstructPrompt's strict fenced
    Python protocol, so adding a model adapter does not change scoring rules.
    """

    SETTINGS = {"full", "first", "last", "extra"}
    _USER_MARKER = "__TESTGENEVAL_QWEN_USER_CONTENT_8473b8351d__"

    def _messages(self, user_content, setting):
        if setting not in self.SETTINGS:
            raise ValueError(f"Unknown TestGenEval setting: {setting}")
        system = self.SYSTEM_MESSAGE_FULL if setting == "full" else self.SYSTEM_MESSAGE
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _render(self, tokenizer, user_content, setting):
        if tokenizer is None:
            raise ValueError("Qwen2Prompt requires the model's tokenizer")
        return tokenizer.apply_chat_template(
            self._messages(user_content, setting),
            tokenize=False,
            add_generation_prompt=True,
        )

    def add_prompts_to_dataset(self, dataset, no_import=False, tokenizer=None):
        if tokenizer is None:
            raise ValueError("Qwen2Prompt requires the model's tokenizer")
        upstream_dataset = super().add_prompts_to_dataset(
            dataset, no_import=no_import, tokenizer=tokenizer
        )
        rows = []
        for row in upstream_dataset["test"]:
            row["preds_prompts"] = {
                setting: self._render(tokenizer, user_content, setting)
                for setting, user_content in row["preds_prompts"].items()
            }
            rows.append(row)
        return DatasetDict({"test": Dataset.from_list(rows)})

    def _instruction_tail(self, user_content, setting):
        """Keep the upstream task instructions intact when removing context."""
        if setting == "full":
            tails = (
                self.PROMPT_FULL.rsplit("{imports}", 1)[-1],
                self.PROMPT_FULL_NO_IMPORT.rsplit("{code_filename}", 1)[-1],
            )
        else:
            tails = (self.PROMPT_COMPLETION.rsplit("{test_src}", 1)[-1],)
        for tail in tails:
            if tail and user_content.endswith(tail):
                return tail
        raise ValueError("Qwen prompt is missing the TestGenEval instruction suffix")

    def truncate_prompt(self, prompt, tokenizer, max_tokens, setting):
        """Truncate only user context, preserving chat roles and task suffix.

        The wrappers come from the tokenizer template rather than hardcoded
        ChatML strings. The final rendered text is encoded again because BPE
        token counts at concatenation boundaries are not necessarily additive.
        """
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("The prompt token budget must be a positive integer")
        marked = self._render(tokenizer, self._USER_MARKER, setting)
        if marked.count(self._USER_MARKER) != 1:
            raise ValueError("The tokenizer template must preserve user message content")
        prefix, suffix = marked.split(self._USER_MARKER, 1)
        if not prefix or not suffix:
            raise ValueError("The tokenizer template must include chat role boundaries")
        if not prompt.startswith(prefix) or not prompt.endswith(suffix):
            raise ValueError("The prompt does not match this setting's Qwen chat template")

        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        if len(encode(prompt)) <= max_tokens:
            return prompt

        user_content = prompt[len(prefix):-len(suffix)]
        task_tail = self._instruction_tail(user_content, setting)
        minimal_prompt = prefix + task_tail + suffix
        minimal_tokens = len(encode(minimal_prompt))
        if minimal_tokens > max_tokens:
            raise ValueError(
                f"Prompt budget {max_tokens} cannot preserve chat roles and task "
                f"instructions ({minimal_tokens} tokens required)"
            )

        context_tokens = encode(user_content[:-len(task_tail)])
        retained = min(len(context_tokens), max_tokens - minimal_tokens)
        while True:
            context = tokenizer.decode(
                context_tokens[-retained:] if retained else [],
                skip_special_tokens=False,
            )
            candidate = prefix + context + task_tail + suffix
            overflow = len(encode(candidate)) - max_tokens
            if overflow <= 0:
                return candidate
            # Every iteration drops at least one context token. Empty context
            # is known to fit from the minimal-prompt check above.
            retained = max(0, retained - max(1, overflow))
