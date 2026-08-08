"""Shared fixtures: tiny toy banks on disk and simulated scorers.

The toy banks have the same on-disk shape as a vendored one, so they exercise the real
loaders rather than bypassing them. There is one per modality: an MCQ bank scored by
:class:`SimScorer` and a generative bank answered by :class:`SimCompleter`. Both draw
their outcomes from the bank's own 3PL at a known true theta, which is what makes the
CAT's job well posed enough for a test to assert convergence.

:class:`SimGenerativeTaker` and :class:`SimMcqTaker` are the same idea pointed at a
*committed* bank rather than a toy one. A five-item toy bank cannot pin an ability and a
synthetic 60-item ladder cannot say anything about the parameters actually shipped, so
the recovery tests for the real banks run over ``calibrated_datasets/`` itself: real
difficulties, real discriminations, real prompts, and only the tokens simulated.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ....base import BenchmarkItem, ItemResponse
from ..convention import CONVENTION_KEY, load_config, manifest_block
from ..datasets import SUPPORTED, DatasetSpec
from ..irt import prob

#: Where the committed banks live. Mirrors ``resolve.CALIBRATED_DATASETS``, spelled out
#: here so a test can check for a bank without importing the resolver.
CALIBRATED_DATASETS = Path(__file__).resolve().parents[5] / "calibrated_datasets"

#: A synthetic dataset name the generative tests register on the allowlist. The gsm8k
#: bank is vendored separately, so nothing here may depend on it existing yet.
GENERATIVE_DATASET = "toy_generative"

#: A 5-item bank spanning easy to hard, with one 2PL-shaped item (guessing 0).
TOY_PARAMS = [
    {"item_id": "toy_0", "difficulty": -1.5, "discrimination": 1.2, "guessing": 0.20},
    {"item_id": "toy_1", "difficulty": -0.5, "discrimination": 1.8, "guessing": 0.25},
    {"item_id": "toy_2", "difficulty": 0.0, "discrimination": 2.0, "guessing": 0.00},
    {"item_id": "toy_3", "difficulty": 0.7, "discrimination": 1.5, "guessing": 0.20},
    {"item_id": "toy_4", "difficulty": 1.6, "discrimination": 1.1, "guessing": 0.25},
]

TOY_ITEMS = [
    {
        "id": f"toy_{i}",
        "question": f"Toy question {i}?",
        "choices": ["alpha", "beta", "gamma", "delta"],
        "gold_index": i % 4,
    }
    for i in range(5)
]


#: The same five items with no choice set, in the shape a vendored generative bank
#: uses: the expected answer lives in metadata, and gold_index is -1 because there is
#: no choice to point at. Gold answers are distinct so a completer cannot pass by luck.
TOY_GENERATIVE_ITEMS = [
    {
        "id": f"toy_{i}",
        "question": f"Toy word problem {i}?",
        "choices": [],
        "gold_index": -1,
        "metadata": {
            "gold_answer": str(40 + i),
            "answer_type": "numeric",
            "modality": "generative",
        },
    }
    for i in range(5)
]


def write_bank(
    root: Path,
    *,
    dataset: str = "toy",
    fit_family: str = "3pl",
    modality: str = "mcq",
) -> Path:
    """Write a vendored-bank directory and return it.

    ``modality`` selects which item shape is written and is recorded in the manifest,
    exactly as vendoring does, so the modality guard sees a realistic bank.

    The scoring convention is recorded too, and built by the same code vendoring uses
    against the shipped ``config.yaml``, so a toy bank passes the startup check for the
    reason a real one does rather than because the check was stubbed. The spec it is
    built from is synthesized from the requested modality rather than looked up, so a
    bank written under a name the allowlist happens to know keeps the shape the caller
    asked for.
    """
    items = TOY_ITEMS if modality == "mcq" else TOY_GENERATIVE_ITEMS
    bank_dir = root / dataset
    bank_dir.mkdir(parents=True, exist_ok=True)
    (bank_dir / "params.json").write_text(json.dumps(TOY_PARAMS, indent=2), encoding="utf-8")
    (bank_dir / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )
    (bank_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "task": dataset,
                "route": "C",
                "fit_family": fit_family,
                "modality": modality,
                "source_ref": "test",
                "source_commit": "0" * 40,
                "bank_dir": "test",
                "upstream_bank_rows": len(TOY_PARAMS),
                "bridge_rows": len(TOY_PARAMS),
                "items": len(TOY_PARAMS),
                "dropped": {},
                "positional_ids": False,
                CONVENTION_KEY: manifest_block(
                    make_spec(dataset, modality=modality),
                    load_config(),
                    recorded_by="test",
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bank_dir


def make_spec(name: str, *, modality: str = "mcq") -> DatasetSpec:
    """Build a :class:`DatasetSpec` directly, bypassing the shipped allowlist.

    Tests register their own spec rather than reaching for a real dataset entry, so
    they neither depend on which datasets happen to be vendored nor break when the
    allowlist changes.
    """
    return DatasetSpec(
        name=name,
        task=name,
        route="C",
        bank_dir="test",
        bridge_path="test",
        bridge_kind="atlas",
        fit_family="3pl",
        expected_bank_rows=len(TOY_PARAMS),
        positional_ids=False,
        notes="Synthetic bank used by the uni_mcq tests.",
        modality=modality,
    )


def unblock(monkeypatch: pytest.MonkeyPatch, dataset: str) -> DatasetSpec:
    """Clear ``dataset``'s blocker for the duration of one test, if it has one.

    A no-op for a dataset that is not blocked, which several callers pass; winogrande
    and gsm8k are the two that are, withheld over how far their rebuilt bridge can be
    checked rather than over anything known to be wrong with it.

    What its callers are pinning is why a blocked bank is tested rather than skipped.
    They assert which prompt format a dataset is scored behind, which grader its
    modality selects, that its committed artifacts resolve and record where their
    ordering came from, that a committed bank still passes the startup convention
    check -- claims about the spec, the config and the artifacts that stay true and
    stay worth testing while a bank is out of service over its *join*. Skipping them
    instead would mean that the next bridge found wanting takes that coverage down with
    it, and that an unblocking ships a bank nobody has checked a second thing about.
    """
    spec = replace(SUPPORTED[dataset], blocked=None)
    monkeypatch.setitem(SUPPORTED, dataset, spec)
    return spec


@pytest.fixture
def toy_bank_dir(tmp_path: Path) -> Path:
    """A directory containing one vendored toy bank named ``toy``."""
    write_bank(tmp_path)
    return tmp_path


def stage_hf_checkpoint(root: Path, name: str = "ckpt") -> Path:
    """Create the thinnest directory that reads as an HF checkpoint, and return it.

    For tests that stub ``s3_io.resolve_checkpoint`` and then let the runner continue.
    The step after the fetch is ``convert.prepare_checkpoint``, which asks what layout
    arrived and refuses one it cannot name -- so a stub returning a bare path now stops
    the run before the scorer, which is the correct behaviour and the wrong fixture.

    Nothing reads the contents. These tests inject the scorer, so the files only have to
    satisfy the detection in :mod:`diagnostics.mcq_cat.common.convert`. Staging them here
    rather than stubbing preparation away keeps the runner's real ordering under test,
    which is what several of these tests exist to pin.
    """
    checkpoint = root / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["OlmoForCausalLM"]}), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_bytes(b"\x00")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    return checkpoint


class SimScorer:
    """Answers each item by drawing from the bank's own 3PL at a known true theta.

    This is the offline stand-in for a checkpoint: it makes the CAT's job well posed,
    so a test can assert that the estimate converges toward a value we chose.
    """

    def __init__(
        self,
        true_theta: float,
        params: dict[str, tuple[float, float, float]],
        *,
        seed: int = 20260806,
    ) -> None:
        self.true_theta = true_theta
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.calls: list[str] = []

    def score_items(self, items) -> list[ItemResponse]:
        responses = []
        for item in items:
            self.calls.append(item.item_id)
            a, b, c = self.params[item.item_id]
            p = float(prob(self.true_theta, np.array([a]), np.array([b]), np.array([c]))[0])
            correct = bool(self.rng.random() < p)
            chosen = item.gold_index if correct else (item.gold_index + 1) % len(item.choices)
            responses.append(
                ItemResponse(item_id=item.item_id, chosen_index=chosen, correct=correct)
            )
        return responses


class SimCompleter:
    """Writes a GSM8K-shaped completion whose final number is right or wrong per the 3PL.

    The generative counterpart of :class:`SimScorer`, and deliberately a *completer*
    rather than a scorer: it hands back raw text, so a test using it runs through the
    real prompt format, stop-sequence truncation and answer extraction instead of
    stubbing them out. The completions include comma-grouped intermediate numbers and
    a run-on into a hallucinated next question, both of which the extractor has to
    survive.
    """

    def __init__(
        self,
        true_theta: float,
        params: dict[str, tuple[float, float, float]],
        items: list[dict],
        *,
        seed: int = 20260806,
    ) -> None:
        self.true_theta = true_theta
        self.params = params
        self.by_question = {
            item["question"]: (item["id"], item["metadata"]["gold_answer"]) for item in items
        }
        self.rng = np.random.default_rng(seed)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        question = prompt.rsplit("Question: ", 1)[-1].split("\nAnswer:")[0]
        item_id, gold = self.by_question[question]
        self.calls.append(item_id)

        a, b, c = self.params[item_id]
        p = float(prob(self.true_theta, np.array([a]), np.array([b]), np.array([c]))[0])
        correct = bool(self.rng.random() < p)
        answer = gold if correct else str(int(gold) + 1000)
        return (
            f" Start from 1,200 and take away 1,100 to get 100. "
            f"So the answer is {answer}.\n\n"
            f"Question: a hallucinated follow-up ending in 999\nAnswer: 999"
        )


def vendored_params(dataset: str) -> dict[str, tuple[float, float, float]]:
    """Return ``item_id -> (a, b, c)`` from a committed bank, or skip if it is absent.

    The simulated taker needs the same parameters the CAT is estimating against, so it
    reads them from the artifact rather than being handed a copy: a test that made up
    its own would still converge, and would keep converging after a re-vendor changed
    the bank underneath it.
    """
    root = CALIBRATED_DATASETS / dataset
    if not (root / "params.json").is_file():
        pytest.skip(f"{dataset} has not been vendored")
    params = json.loads((root / "params.json").read_text(encoding="utf-8"))
    return {
        row["item_id"]: (row["discrimination"], row["difficulty"], row["guessing"])
        for row in params
    }


class SimGenerativeTaker:
    """Answers a committed generative bank's prompts from its own 3PL at a known theta.

    A *completer*, not a scorer, for the reason :class:`SimCompleter` is one: it hands
    back raw text, so everything between the prompt and the binary -- the real prompt
    template, stop-sequence truncation, answer extraction or constraint verification --
    is exercised rather than stubbed. Only the tokens are simulated, and only in the
    two places a benchmark differs: how a question is recovered from its prompt, and
    what a right or wrong answer to it looks like.

    Args:
        true_theta: The ability the responses are drawn at, and what recovery is
            measured against.
        params: ``item_id -> (a, b, c)``, from :func:`vendored_params`.
        items: The bank's items, used to map a rendered prompt back to its item.
        recover_question: Pulls the live question back out of a rendered prompt. The
            few-shot block and the framing around it are the template's, so this is
            written per prompt style.
        answer: Writes the completion for an item, right or wrong. Right has to be
            right by the real grader's standard, which is the point of the test.
    """

    def __init__(
        self,
        true_theta: float,
        params: dict[str, tuple[float, float, float]],
        items: Sequence[BenchmarkItem],
        *,
        recover_question: Callable[[str], str],
        answer: Callable[[BenchmarkItem, bool], str],
        seed: int = 20260807,
    ) -> None:
        self.true_theta = true_theta
        self.params = params
        self.by_question = {item.question: item for item in items}
        self.recover_question = recover_question
        self.answer = answer
        self.rng = np.random.default_rng(seed)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        item = self.by_question[self.recover_question(prompt)]
        self.calls.append(item.item_id)
        a, b, c = self.params[item.item_id]
        p = float(prob(self.true_theta, np.array([a]), np.array([b]), np.array([c]))[0])
        return self.answer(item, bool(self.rng.random() < p))


class SimMcqTaker:
    """A stand-in forward pass whose per-character scores come from the bank's own 3PL.

    The MCQ counterpart of :class:`SimGenerativeTaker`, and a *forward pass* rather than
    a scorer for the same reason: everything above it -- the prompt style, the
    normalization, the argmax, the CAT -- is the shipped code running on the committed
    bank, and only the log-probabilities are invented.

    They are invented per character, which is what makes the intended choice win under
    ``acc_norm`` regardless of how long the choices happen to be. That is deliberate: a
    taker built to be decisive under an unnormalized sum would make a test agree with
    itself whichever normalization was configured.

    Shared by the two banks scored under ``acc_norm``. MuSR needs the per-character
    construction because its choices are whole clauses of very unequal length; BBH needs
    it because a handful of its subtasks rank "0" against "18" while the rest rank option
    letters of equal length, so a taker that only worked on equal-length choices would
    pass most of that bank while telling nothing about the part where the rule matters.
    """

    #: Per character of continuation, for the choice this taker means to pick and for
    #: the rest. Both negative, as a log-probability is, and separated widely enough
    #: that the ordering cannot turn on a rounding difference.
    TARGET = -0.5
    OTHER = -1.0

    def __init__(
        self,
        true_theta: float,
        params: dict[str, tuple[float, float, float]],
        items: list[BenchmarkItem],
        config: Any,
        *,
        seed: int = 20260807,
    ) -> None:
        from ....common import inference

        self.true_theta = true_theta
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.by_prompt = {
            inference.scored_choices(item, config)[0].prompt: (
                item,
                [choice.continuation for choice in inference.scored_choices(item, config)],
            )
            for item in items
        }
        self.intended: dict[str, int] = {}
        self.calls: list[str] = []

    def _intended(self, item: BenchmarkItem) -> int:
        """Draw once per item whether this taker answers it, and which choice it names."""
        if item.item_id not in self.intended:
            a, b, c = self.params[item.item_id]
            p = float(prob(self.true_theta, np.array([a]), np.array([b]), np.array([c]))[0])
            correct = bool(self.rng.random() < p)
            self.intended[item.item_id] = (
                item.gold_index if correct else (item.gold_index + 1) % len(item.choices)
            )
            self.calls.append(item.item_id)
        return self.intended[item.item_id]

    def __call__(self, prompt: str, continuation: str) -> float:
        item, continuations = self.by_prompt[prompt]
        per_char = (
            self.TARGET if continuations.index(continuation) == self._intended(item) else self.OTHER
        )
        return per_char * len(continuation)


@pytest.fixture
def toy_params() -> dict[str, tuple[float, float, float]]:
    """``item_id -> (a, b, c)`` for the toy bank."""
    return {
        row["item_id"]: (row["discrimination"], row["difficulty"], row["guessing"])
        for row in TOY_PARAMS
    }


@pytest.fixture
def toy_benchmark_items() -> list[BenchmarkItem]:
    """The toy items as in-memory :class:`BenchmarkItem` objects."""
    return [
        BenchmarkItem(
            item_id=row["id"],
            question=row["question"],
            choices=tuple(row["choices"]),
            gold_index=row["gold_index"],
        )
        for row in TOY_ITEMS
    ]


@pytest.fixture
def toy_generative_items() -> list[BenchmarkItem]:
    """The generative toy items as in-memory :class:`BenchmarkItem` objects."""
    return [
        BenchmarkItem(
            item_id=row["id"],
            question=row["question"],
            choices=(),
            gold_index=row["gold_index"],
            metadata=dict(row["metadata"]),
        )
        for row in TOY_GENERATIVE_ITEMS
    ]
