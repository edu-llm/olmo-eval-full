"""Checkpoint load + batched log-likelihood MCQ scoring.

This mirrors the checkpoint-loading patterns in ``tests/OnNode/checkpoint_infer.py``:
the HuggingFace path is a real implementation and the raw ``olmo_core`` path is a
marked integration point that raises ``NotImplementedError`` with guidance.

Scoring follows the log-likelihood MCQ convention used across ``olmo_eval``
(``RequestType.LOGLIKELIHOOD`` in ``src/olmo_eval/evals/tasks``): for each item,
score the continuation for every choice given the question prompt and pick the
highest-likelihood choice. Heavy dependencies (``torch``, ``transformers``) are
imported lazily so this module imports without a GPU stack installed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import BenchmarkItem, ItemResponse, ScoringModel

log = logging.getLogger("mcq_cat.inference")


@dataclass
class InferenceConfig:
    """Configuration for checkpoint loading and MCQ scoring."""

    checkpoint_kind: str = "hf"  # "hf" or "olmo_core"
    batch_size: int = 16
    max_length: int | None = None
    seed: int = 1234
    device_map: str = "auto"
    prompt_template: str = "{question}\nAnswer:"
    choice_prefix: str = " "


def format_mcq_prompt(item: BenchmarkItem, config: InferenceConfig) -> tuple[str, tuple[str, ...]]:
    """Return ``(prompt, continuations)`` for an item using the log-likelihood convention."""
    prompt = config.prompt_template.format(question=item.question)
    continuations = tuple(f"{config.choice_prefix}{choice}" for choice in item.choices)
    return prompt, continuations


def load_scoring_model(checkpoint_dir: Path, config: InferenceConfig) -> ScoringModel:
    """Load an MCQ log-likelihood scoring model for the configured checkpoint kind."""
    if config.checkpoint_kind == "hf":
        return _HFScoringModel(checkpoint_dir, config)
    if config.checkpoint_kind == "olmo_core":
        return _load_olmo_core(checkpoint_dir, config)
    raise ValueError(
        f"Unknown checkpoint_kind: {config.checkpoint_kind!r} (expected 'hf' or 'olmo_core')"
    )


class _HFScoringModel:
    """HuggingFace-backed batched log-likelihood MCQ scorer."""

    def __init__(self, checkpoint_dir: Path, config: InferenceConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self._torch = torch
        self.config = config
        set_seed(config.seed)

        self.tokenizer: Any = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype="auto",
            device_map=config.device_map,
        )
        self.model.eval()

    def _continuation_logprob(self, prompt: str, continuation: str) -> float:
        """Sum the log-probabilities of ``continuation`` tokens given ``prompt``."""
        torch = self._torch
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        full_ids = self.tokenizer(prompt + continuation, return_tensors="pt")["input_ids"]

        if self.config.max_length is not None and full_ids.shape[1] > self.config.max_length:
            full_ids = full_ids[:, -self.config.max_length :]

        full_ids = full_ids.to(self.model.device)
        cont_len = full_ids.shape[1] - prompt_ids.shape[1]
        if cont_len <= 0:
            return 0.0

        with torch.no_grad():
            logits = self.model(full_ids).logits

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = full_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        continuation_log_probs = token_log_probs[:, -cont_len:]
        return float(continuation_log_probs.sum().item())

    def score_items(self, items: Sequence[BenchmarkItem]) -> list[ItemResponse]:
        """Grade each item by scoring every choice's continuation log-likelihood."""
        responses: list[ItemResponse] = []
        for item in items:
            prompt, continuations = format_mcq_prompt(item, self.config)
            choice_logprobs = tuple(
                self._continuation_logprob(prompt, continuation) for continuation in continuations
            )
            chosen_index = max(range(len(choice_logprobs)), key=lambda i: choice_logprobs[i])
            responses.append(
                ItemResponse(
                    item_id=item.item_id,
                    chosen_index=chosen_index,
                    correct=chosen_index == item.gold_index,
                    choice_logprobs=choice_logprobs,
                )
            )
        return responses


def _load_olmo_core(checkpoint_dir: Path, config: InferenceConfig) -> ScoringModel:
    """Load a raw OLMo-core checkpoint for scoring (integration point).

    Mirrors ``_load_olmo_core`` in ``tests/OnNode/checkpoint_infer.py``: reconstruct
    the model and tokenizer from the run config and load the checkpoint weights.
    This depends on the run's config layout and is left as an integration point.
    """
    raise NotImplementedError(
        "olmo_core scoring is a training-env integration point. Provide the run config "
        "and checkpoint layout, then implement _load_olmo_core (mirror the HF scorer's "
        "score_items). See tests/OnNode/checkpoint_infer.py and the plan's open questions."
    )
