"""Per-dataset MCQ prompt formats, pinned against the olmo-eval tasks themselves.

The three MCQ benchmarks here each define their own ``format_request`` and each bank was
calibrated behind the one its task builds, so "which prompt" is not cosmetic: EAP treats
an item's difficulty as fixed, and a prompt the difficulty was not estimated under moves
theta by the whole difference while the standard error stays healthy. There is no way to
see that in a report, which is why the check is here.

The parity tests construct an ``olmo_eval`` ``Instance`` from a committed item and
compare our ``(prompt, continuation)`` pairs against the task's own ``LMRequest``,
rather than against a string typed out here. A hardcoded expectation would keep passing
after the task changed, which is the exact failure being guarded against. They skip
where ``olmo_eval`` is not importable.

:class:`TestWinograndeLengthBias` is the one test that is not about parity. It pins the
second thing partial evaluation buys, beyond measuring coreference at all: with the
options as continuations they differ in length, and an unnormalized sum then hands the
item to whichever option is shorter, whatever the model thinks. Under partial evaluation
the continuations are byte-identical, so the same sum is length-invariant by
construction. Both halves run through the shipped scorer with only the forward pass
replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ....base import BenchmarkItem
from ....common import grading, inference
from .. import datasets, resolve
from ..convention import CONFIG_PATH
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS, unblock, write_bank

#: The MCQ datasets whose prompt format this file pins, and the style each must use.
EXPECTED_STYLES = {
    "arc_challenge": "question_answer",
    "hellaswag": "bare_context",
    "winogrande": "blank_substitution",
}


def load_config() -> dict[str, Any]:
    """The committed ``config.yaml``, read as the style reads it."""
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def mcq_datasets_block() -> dict[str, Any]:
    """The per-dataset half of the committed MCQ settings block."""
    return dict((load_config().get("mcq") or {}).get(grading.PER_DATASET_KEY) or {})


def vendored_items(dataset: str) -> list[BenchmarkItem]:
    """The committed items for ``dataset``, loaded through the real loader."""
    path = CALIBRATED_DATASETS / dataset / "items.jsonl"
    if not path.is_file():
        pytest.skip(f"{dataset} has not been vendored")
    from ....common.benchmark_download import load_items_from_jsonl

    return list(load_items_from_jsonl(path, name=dataset).items)


def as_instance(item: BenchmarkItem) -> Any:
    """Rebuild the ``olmo_eval`` instance a committed item was vendored from.

    Only the fields ``format_request`` reads are restored, which is all of them for
    these three tasks: the stem, the choice tuple and the gold index.
    """
    types = pytest.importorskip("olmo_eval.common.types")
    return types.Instance(
        question=item.question,
        choices=item.choices,
        gold_answer=str(item.gold_index),
        metadata={"gold_idx": item.gold_index},
    )


def task_pairs(task_name: str, item: BenchmarkItem) -> tuple[tuple[str, str], ...]:
    """Return the task's own ``(prompt, continuation)`` pairs for ``item``.

    ``LMRequest`` spells the two prompt shapes in one structure: a shared ``prompt``
    with a continuation per choice, or a ``continuation_prompts`` tuple that varies the
    prompt instead. Flattening both to pairs here is what lets one comparison cover
    both, and it is the same reduction :class:`inference.ScoredChoice` performs.
    """
    registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
    request = registry.get_task(task_name).format_request(as_instance(item))
    prompts = request.continuation_prompts or ((request.prompt,) * len(request.continuations))
    return tuple(zip(prompts, request.continuations, strict=True))


def our_choices(item: BenchmarkItem, prompt_style: str) -> tuple[inference.ScoredChoice, ...]:
    """Return this harness's :class:`ScoredChoice` objects for ``item``."""
    return inference.scored_choices(item, inference.InferenceConfig(prompt_style=prompt_style))


def our_pairs(item: BenchmarkItem, prompt_style: str) -> tuple[tuple[str, str], ...]:
    """Return this harness's ``(prompt, continuation)`` pairs for ``item``."""
    return tuple((c.prompt, c.continuation) for c in our_choices(item, prompt_style))


class TestParityWithTheTask:
    """Our pairs must be the task's pairs, item for item."""

    @pytest.mark.parametrize("dataset", sorted(EXPECTED_STYLES))
    def test_the_first_committed_item_matches(self, dataset: str) -> None:
        item = vendored_items(dataset)[0]
        assert our_pairs(item, EXPECTED_STYLES[dataset]) == task_pairs(dataset, item)

    @pytest.mark.parametrize("dataset", sorted(EXPECTED_STYLES))
    def test_a_sample_of_the_bank_matches(self, dataset: str) -> None:
        """Every fiftieth item, so a format that only breaks on some stems is caught."""
        items = vendored_items(dataset)
        for item in items[::50]:
            assert our_pairs(item, EXPECTED_STYLES[dataset]) == task_pairs(dataset, item), (
                item.item_id
            )

    def test_arc_carries_the_question_prefix_the_shared_helper_writes(self) -> None:
        """``format_helpers.format_rc``, which arc.py's non-MC branch calls directly."""
        item = vendored_items("arc_challenge")[0]
        prompt, continuation = our_pairs(item, "question_answer")[0]
        assert prompt == f"Question: {item.question}\nAnswer:"
        assert continuation == f" {item.choices[0]}"

    def test_hellaswag_has_no_answer_cue_at_all(self) -> None:
        """``hellaswag._format_rc`` returns the bare query; its contexts end mid-clause."""
        item = vendored_items("hellaswag")[0]
        prompt, _ = our_pairs(item, "bare_context")[0]
        assert prompt == item.question
        assert "Answer:" not in prompt

    def test_winogrande_varies_the_prompt_and_shares_the_continuation(self) -> None:
        """Trinh & Le partial evaluation, which inverts what the scorer used to assume."""
        item = vendored_items("winogrande")[0]
        pairs = our_pairs(item, "blank_substitution")

        assert len({continuation for _, continuation in pairs}) == 1
        assert len({prompt for prompt, _ in pairs}) == len(item.choices)
        head, _, tail = item.question.partition("_")
        for (prompt, continuation), choice in zip(pairs, item.choices, strict=True):
            assert prompt == head + choice
            assert continuation == " " + tail.strip()


class TestWinograndeLengthBias:
    """A short wrong option beats a long right one under the old scheme, and not the new.

    The simulated checkpoint below is the smallest thing that can show it: every
    character of continuation costs the same, and text that reads as the correct
    completion of the sentence earns a fixed bonus. That is the shape of a real
    log-probability -- longer continuations score lower, coherent ones score higher --
    with the two effects separated so the test can say which one decided the item.
    """

    #: Per character of continuation. Sums, like the metric being modelled.
    COST_PER_CHAR = 0.5
    #: Awarded to text that reads as the sentence with the correct name substituted in.
    COHERENCE_BONUS = 1.0

    @pytest.fixture
    def item(self) -> BenchmarkItem:
        """A committed item whose correct option is at least three characters longer.

        Chosen from the bank rather than written here, so the case stays real, and by
        rule rather than by id, so a re-vendor cannot leave the test pinning an item
        that no longer exists. Three characters is where the length term outweighs the
        coherence bonus at the constants above; 114 of the 865 items qualify.
        """
        for candidate in vendored_items("winogrande"):
            gold = candidate.choices[candidate.gold_index]
            other = candidate.choices[1 - candidate.gold_index]
            if len(gold) - len(other) >= 3:
                return candidate
        pytest.skip("no winogrande item has a correct option three characters longer")

    def scorer(self, item: BenchmarkItem) -> Any:
        """The shipped scorer with only the forward pass replaced."""
        gold_reading = item.question.replace("_", item.choices[item.gold_index], 1)

        def logprob(prompt: str, continuation: str) -> float:
            score = -self.COST_PER_CHAR * len(continuation)
            text = prompt + continuation
            if gold_reading in text or continuation.strip() == item.choices[item.gold_index]:
                score += self.COHERENCE_BONUS
            return score

        model = inference._HFScoringModel.__new__(inference._HFScoringModel)
        model._continuation_logprob = logprob  # type: ignore[method-assign]
        return model

    def chosen(self, item: BenchmarkItem, prompt_style: str) -> int:
        model = self.scorer(item)
        model.config = inference.InferenceConfig(prompt_style=prompt_style)
        return model.score_items([item])[0].chosen_index

    def test_option_as_continuation_picks_the_shorter_option(self, item: BenchmarkItem) -> None:
        """Coherence is on the correct option and it loses anyway, on token count."""
        assert self.chosen(item, "question_answer") == 1 - item.gold_index

    def test_partial_evaluation_picks_the_correct_option(self, item: BenchmarkItem) -> None:
        assert self.chosen(item, "blank_substitution") == item.gold_index

    def test_the_length_term_cannot_decide_under_partial_evaluation(
        self, item: BenchmarkItem
    ) -> None:
        """The structural reason, not just the outcome: the continuations are identical."""
        continuations = {c.continuation for c in our_choices(item, "blank_substitution")}
        assert len(continuations) == 1

    def test_the_correct_option_really_is_the_longer_one(self, item: BenchmarkItem) -> None:
        """Otherwise the two tests above would agree for an uninteresting reason."""
        gold = item.choices[item.gold_index]
        other = item.choices[1 - item.gold_index]
        assert len(gold) > len(other)


class TestTheStyleTable:
    def test_every_style_is_named_after_itself(self) -> None:
        for key, style in inference.MCQ_PROMPT_STYLES.items():
            assert style.name == key

    def test_an_unknown_style_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="Unknown prompt_style 'minerva'"):
            inference.get_mcq_prompt_style("minerva")

    def test_the_default_is_the_shared_question_answer_helper(self) -> None:
        """olmo-eval's own default for a task with no bespoke framing."""
        assert inference.DEFAULT_PROMPT_STYLE == "question_answer"
        assert isinstance(
            inference.get_mcq_prompt_style(inference.DEFAULT_PROMPT_STYLE),
            inference.SharedPromptStyle,
        )

    def test_substitution_refuses_a_stem_with_no_blank(self) -> None:
        """The bank would have to have been vendored from a task that dropped it."""
        item = BenchmarkItem(
            item_id="w0", question="No blank here.", choices=("a", "b"), gold_index=0
        )
        with pytest.raises(ValueError, match="standalone"):
            our_choices(item, "blank_substitution")


class TestEveryMcqDatasetDeclaresItsFormat:
    """No supported MCQ dataset may reach the default by omission.

    The default is defensible for a dataset nobody has looked at and wrong for one whose
    bank is committed here, and the difference between those two cases is not visible at
    run time. Pinning it as a test rather than as a runtime check keeps the check total:
    it covers every allowlisted dataset at once, including the ones a given run does not
    touch.
    """

    def test_each_supported_mcq_dataset_has_an_entry(self) -> None:
        listed = mcq_datasets_block()
        expected = sorted(
            name for name, spec in datasets.SUPPORTED.items() if spec.modality == "mcq"
        )
        assert sorted(listed) == expected

    def test_each_entry_names_a_known_style(self) -> None:
        for dataset, entry in mcq_datasets_block().items():
            assert entry["prompt_style"] in inference.MCQ_PROMPT_STYLES, dataset

    def test_the_entries_are_the_formats_their_tasks_use(self) -> None:
        listed = mcq_datasets_block()
        for dataset, prompt_style in EXPECTED_STYLES.items():
            assert listed[dataset]["prompt_style"] == prompt_style


class TestTheClozeGuard:
    """The fallback failure that would otherwise be invisible."""

    def cloze_items(self, n: int) -> list[BenchmarkItem]:
        return [
            BenchmarkItem(
                item_id=f"c{i}",
                question=f"The dog chased the cat because _ was faster {i}.",
                choices=("dog", "cat"),
                gold_index=0,
            )
            for i in range(n)
        ]

    def plain_items(self, n: int) -> list[BenchmarkItem]:
        return [
            BenchmarkItem(
                item_id=f"p{i}", question=f"What is {i}?", choices=("a", "b"), gold_index=0
            )
            for i in range(n)
        ]

    def test_a_cloze_bank_under_a_shared_prompt_style_is_refused(self) -> None:
        with pytest.raises(ValueError, match="standalone blank"):
            inference.check_prompt_style_fits("question_answer", self.cloze_items(10))

    def test_the_same_bank_under_a_substitution_style_passes(self) -> None:
        inference.check_prompt_style_fits("blank_substitution", self.cloze_items(10))

    def test_one_stray_underscore_does_not_fail_a_bank(self) -> None:
        """A lone underscore turns up in one HellaSwag stem out of 4,840."""
        inference.check_prompt_style_fits(
            "bare_context", self.plain_items(99) + self.cloze_items(1)
        )

    def test_an_identifier_underscore_is_not_a_blank(self) -> None:
        items = [
            BenchmarkItem(
                item_id=f"i{i}",
                question="Which call initialises the object, __init__ or new?",
                choices=("a", "b"),
                gold_index=0,
            )
            for i in range(10)
        ]
        inference.check_prompt_style_fits("question_answer", items)

    def test_an_empty_bank_is_not_a_cloze_bank(self) -> None:
        inference.check_prompt_style_fits("question_answer", [])

    def test_the_committed_winogrande_bank_would_be_refused(self) -> None:
        """The live case: this is what the previous format did to every one of them."""
        with pytest.raises(ValueError, match="standalone blank"):
            inference.check_prompt_style_fits("question_answer", vendored_items("winogrande"))


class TestTheStyleWiresItThrough:
    """From ``--benchmark NAME`` to the constructed scorer's config."""

    @pytest.fixture
    def style(self, monkeypatch, tmp_path: Path):
        def build(dataset: str) -> UniMcqStyle:
            unblock(monkeypatch, dataset)
            write_bank(tmp_path, dataset=dataset)
            monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
            built = UniMcqStyle()
            built.download_benchmark(dataset)
            return built

        return build

    @pytest.mark.parametrize(("dataset", "prompt_style"), sorted(EXPECTED_STYLES.items()))
    def test_the_config_entry_reaches_the_inference_config(
        self, monkeypatch, style, tmp_path: Path, dataset: str, prompt_style: str
    ) -> None:
        seen: list[inference.InferenceConfig] = []
        monkeypatch.setattr(
            inference,
            "load_scoring_model",
            lambda _dir, config: seen.append(config) or object(),
        )
        request = style(dataset).grading_request()
        grading.load_grader(request, tmp_path / "ckpt", grading.GradingSettings())

        assert request.modality == "mcq"
        assert seen[0].prompt_style == prompt_style

    def test_a_generative_bank_carries_no_mcq_settings(self, monkeypatch, tmp_path: Path) -> None:
        from .conftest import GENERATIVE_DATASET, make_spec

        monkeypatch.setitem(
            datasets.SUPPORTED,
            GENERATIVE_DATASET,
            make_spec(GENERATIVE_DATASET, modality="generative"),
        )
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        write_bank(tmp_path, dataset=GENERATIVE_DATASET, modality="generative")

        built = UniMcqStyle()
        built.download_benchmark(GENERATIVE_DATASET)
        assert built.grading_request().mcq == {}

    def test_a_misspelled_mcq_setting_raises_rather_than_being_ignored(
        self, tmp_path: Path
    ) -> None:
        request = grading.GradingRequest(
            dataset="arc_challenge", modality="mcq", mcq={"prompt_stile": "question_answer"}
        )
        with pytest.raises(ValueError, match="Unknown mcq settings"):
            grading.load_grader(request, tmp_path, grading.GradingSettings())

    def test_a_non_mapping_per_dataset_block_raises(self, tmp_path: Path) -> None:
        request = grading.GradingRequest(
            dataset="arc_challenge",
            modality="mcq",
            mcq={grading.PER_DATASET_KEY: ["arc_challenge"]},
        )
        with pytest.raises(ValueError, match="must be a mapping"):
            grading.load_grader(request, tmp_path, grading.GradingSettings())

    def test_the_report_records_which_format_produced_the_theta(self, style) -> None:
        built = style("hellaswag")
        assert built.mcq_prompt_style() == "bare_context"

    def test_a_dataset_with_no_entry_falls_back_to_the_default(self, style) -> None:
        """Nothing shipped takes this path; the guard above is what keeps it that way."""
        built = style("gsm8k")
        assert built.mcq_prompt_style() == inference.DEFAULT_PROMPT_STYLE


class TestARunOverTheCommittedWinograndeBank:
    """The whole path on the real bank: the substitution format's only live consumer.

    A smoke rather than a recovery test: the stubbed forward pass below is a fixed
    arithmetic rule, so which items come back correct means nothing. What it exercises
    is everything around that -- resolution, the cloze guard against 865 real stems, the
    substitution format on each administered item, and the report -- with only the
    tokens replaced.

    The bank is blocked, so the run has to lift the blocker to reach the path at all.
    That is the right shape here: the stems and choices are not what the blocker is
    about, and the substitution format has no other live consumer, so retiring this
    would leave that format exercised only against items written in a test file.
    """

    def test_it_resolves_scores_and_reports_the_format_it_used(self, monkeypatch) -> None:
        if not (CALIBRATED_DATASETS / "winogrande" / "params.json").is_file():
            pytest.skip("winogrande has not been vendored")
        from ....common import cat_loop

        unblock(monkeypatch, "winogrande")

        style = UniMcqStyle()
        bank = style.download_benchmark("winogrande")
        irt = style.load_irt_params("winogrande")

        request = style.grading_request()
        assert request.modality == "mcq"
        grading.check_bank_modality(request, bank.items)

        def logprob(prompt: str, continuation: str) -> float:
            return -0.01 * len(prompt) - len(continuation)

        model = inference._HFScoringModel.__new__(inference._HFScoringModel)
        model.config = inference.InferenceConfig(prompt_style="blank_substitution")
        model._continuation_logprob = logprob

        report = cat_loop.run_cat(
            style, bank=bank, irt_bank=irt, model=model, se_threshold=0.3, max_items=12
        )

        assert report.metadata["prompt_style"] == "blank_substitution"
        assert report.metadata["stop_reason"] in {"precision_reached", "max_items_reached"}
        assert style.min_items <= report.num_items_administered <= 12
        for response in report.responses:
            assert len(response.choice_logprobs) == len(bank.get(response.item_id).choices)


class TestTheVendoredWinograndeBank:
    """The format needs the blank, so the bank has to have kept it."""

    def test_every_stem_still_carries_the_placeholder(self) -> None:
        items = vendored_items("winogrande")
        assert items
        assert all("_" in item.question for item in items)

    def test_the_manifest_still_names_the_winogrande_task(self) -> None:
        path = CALIBRATED_DATASETS / "winogrande" / "manifest.json"
        if not path.is_file():
            pytest.skip("winogrande has not been vendored")
        assert json.loads(path.read_text(encoding="utf-8"))["task"] == "winogrande"
