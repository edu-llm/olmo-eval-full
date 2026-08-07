"""The BBH bank: its frozen 3-shot stems, its 24-subtask join, and its honesty labelling.

Three things about BBH are unlike every other bank here.

Its prompt is not built at run time. leaderboard_bbh is 3-shot behind a one-line
description that differs per *subtask*, and this style's ``config.yaml`` is per dataset,
so nothing here could express it; vendoring calls each task's own ``format_request`` and
keeps the whole rendered prompt as the item's stem. Everything below that concerns the
prompt is really one question -- is the stem still what the task renders -- because a
frozen prompt is out of reach of every guard that would otherwise notice it drifting.

Its ``min_items`` is 24 where every other bank's is 8. That is the one harness setting
this bank moves, and the tests hold both halves: that BBH gets 24, and that nothing else
moved with it.

And it is the one bank that is runnable and not trustworthy in the same breath. Its
predicted accuracy is usable and its theta is not, no report field distinguishes those,
and the tests below pin the ``bank_caveat`` that does -- in the report itself, because a
reader of ``cat_report.json`` should not have to open ``datasets.py`` to learn it.

The prompt-parity tests reconstruct nothing. They ask each task what it renders in front
of a probe question and compare that against the prefix the vendored stems actually
carry, which is the only comparison that can fail if the two ever disagree; an expected
prompt typed out here would keep passing after the task changed. They skip where
``olmo_eval`` is not importable, and they never enumerate, so nothing touches the network.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from ....base import BenchmarkItem
from ....common import cat_loop, grading, inference
from ..datasets import SUPPORTED
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS, SimMcqTaker, vendored_params

DATASET = "bbh"

#: A stem the dataset cannot contain, so a rendered prompt can be split at it without
#: the split being ambiguous.
PROBE = "<<probe question>>"

#: Every subtask's bridge rows and its vendored item count, which for this bank are also
#: its enumeration span and its per-subtask overlap: 24 of 24 subtasks join at 100%.
#:
#: The three short ones are the point of pinning all 24 rather than a sample.
#: ``causal_judgement`` at 187, ``snarks`` at 178 and ``penguins_in_a_table`` at 146 are
#: those subtasks' full upstream size and not a truncation, so a future run that "fixed"
#: them up to 250 would be enumerating something else.
SUBTASK_ROWS: dict[str, int] = {
    "boolean_expressions": 250,
    "causal_judgement": 187,
    "date_understanding": 250,
    "disambiguation_qa": 250,
    "formal_fallacies": 250,
    "geometric_shapes": 250,
    "hyperbaton": 250,
    "logical_deduction_five_objects": 250,
    "logical_deduction_seven_objects": 250,
    "logical_deduction_three_objects": 250,
    "movie_recommendation": 250,
    "navigate": 250,
    "object_counting": 250,
    "penguins_in_a_table": 146,
    "reasoning_about_colored_objects": 250,
    "ruin_names": 250,
    "salient_translation_error_detection": 250,
    "snarks": 178,
    "sports_understanding": 250,
    "temporal_sequences": 250,
    "tracking_shuffled_objects_five_objects": 250,
    "tracking_shuffled_objects_seven_objects": 250,
    "tracking_shuffled_objects_three_objects": 250,
    "web_of_lies": 250,
}

SUBTASK_ITEMS: dict[str, int] = {
    "boolean_expressions": 135,
    "causal_judgement": 97,
    "date_understanding": 197,
    "disambiguation_qa": 191,
    "formal_fallacies": 133,
    "geometric_shapes": 180,
    "hyperbaton": 130,
    "logical_deduction_five_objects": 187,
    "logical_deduction_seven_objects": 202,
    "logical_deduction_three_objects": 193,
    "movie_recommendation": 232,
    "navigate": 145,
    "object_counting": 244,
    "penguins_in_a_table": 111,
    "reasoning_about_colored_objects": 222,
    "ruin_names": 226,
    "salient_translation_error_detection": 194,
    "snarks": 87,
    "sports_understanding": 113,
    "temporal_sequences": 225,
    "tracking_shuffled_objects_five_objects": 120,
    "tracking_shuffled_objects_seven_objects": 156,
    "tracking_shuffled_objects_three_objects": 117,
    "web_of_lies": 128,
}

#: BIG-Bench Hard's generative subtasks, which have no closed answer set to rank.
GENERATIVE_SUBTASKS = ("dyck_languages", "multistep_arithmetic_two", "word_sorting")


def vendored_items() -> list[BenchmarkItem]:
    """The committed BBH items, loaded through the real loader."""
    from ....common.benchmark_download import load_items_from_jsonl

    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    if not path.is_file():
        pytest.skip("bbh has not been vendored")
    return list(load_items_from_jsonl(path, name=DATASET).items)


def by_subtask() -> dict[str, list[BenchmarkItem]]:
    """The committed items grouped by the subtask half of their composite id."""
    grouped: dict[str, list[BenchmarkItem]] = {}
    for item in vendored_items():
        grouped.setdefault(item.item_id.split("|", 1)[0], []).append(item)
    return grouped


def manifest() -> dict[str, Any]:
    """The committed BBH manifest, or skip."""
    path = CALIBRATED_DATASETS / DATASET / "manifest.json"
    if not path.is_file():
        pytest.skip("bbh has not been vendored")
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_mcq_config() -> inference.InferenceConfig:
    """The MCQ settings the committed ``config.yaml`` resolves for BBH."""
    settings = grading._apply_mcq_overrides(
        grading.GradingSettings(), UniMcqStyle().mcq_settings, dataset=DATASET
    )
    return settings.mcq


def rendered_prefix(subtask: str, choices: tuple[str, ...]) -> str:
    """Return what ``bbh_<subtask>``'s own ``format_request`` puts in front of an item.

    Asked of the task rather than assembled from the constants, and asked without
    enumerating anything: ``process_doc`` and ``format_request`` need only a question and
    a target, so a probe question is enough to render the description, the three
    exemplars and the answer cue and then cut the probe back out. That makes this a
    comparison between the vendored stems and the code that produced them, which is the
    only comparison that fails when the two disagree.
    """
    registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
    task = registry.get_task(f"bbh_{subtask}")
    instance = task.process_doc({"input": PROBE, "target": choices[0]}, index=0)
    prompt = task.format_request(instance).prompt
    tail = f"Q: {PROBE}\nA:"
    assert prompt.endswith(tail), subtask
    return prompt[: -len(tail)]


def our_pairs(item: BenchmarkItem) -> tuple[tuple[str, str], ...]:
    """Return this harness's ``(prompt, continuation)`` pairs under the resolved settings."""
    choices = inference.scored_choices(item, resolved_mcq_config())
    return tuple((choice.prompt, choice.continuation) for choice in choices)


class TestTheFrozenPrompt:
    """The stem is the task's rendered prompt, and run time adds only the choice."""

    def test_every_stem_carries_its_own_subtask_prefix(self) -> None:
        """All 24, because the builder is bound to a task and there are 24 of them.

        A builder bound to the wrong subtask renders well-formed prompts, joins totally
        and clears every floor, having attached one subtask's instructions and worked
        examples to another's questions.
        """
        for subtask, items in by_subtask().items():
            prefix = rendered_prefix(subtask, items[0].choices)
            for item in items[:3]:
                assert item.question.startswith(prefix), item.item_id

    def test_the_prefixes_repeat_only_where_lm_eval_repeats_them(self) -> None:
        """Twenty distinct prefixes over 24 subtasks, and the two coincidences are real.

        Without this the test above would pass on a bank rendered entirely behind one
        task, so the count is the guard. It is 20 rather than 24 because
        ``leaderboard_bbh`` gives the five- and seven-object variants of both sized
        families the *three*-object exemplars, description and all -- checked against
        lm-evaluation-harness's own YAML rather than assumed -- so the prefix is shared
        by construction upstream and reproducing it is the whole point of freezing.
        """
        grouped = by_subtask()
        shared: dict[str, set[str]] = {}
        for subtask, items in grouped.items():
            shared.setdefault(rendered_prefix(subtask, items[0].choices), set()).add(subtask)

        assert len(shared) == 20
        assert sorted(g for g in shared.values() if len(g) > 1) == sorted(
            [
                {f"logical_deduction_{n}_objects" for n in ("three", "five", "seven")},
                {f"tracking_shuffled_objects_{n}_objects" for n in ("three", "five", "seven")},
            ],
            key=sorted,
        )

    def test_a_sized_variant_keeps_the_shorter_exemplars_upstream_shows_it(self) -> None:
        """The sharp end of that: an item offering (A) to (E) behind examples ending at (C).

        It reads like something to fix and is not. Every difficulty in the four subtasks
        shown the wrong-size exemplars was estimated behind exactly this prompt, so
        completing them to match the item's option count would change 665 vendored items'
        presentation while leaving their parameters describing the old one.
        """
        items = by_subtask()["logical_deduction_five_objects"]
        prefix = rendered_prefix("logical_deduction_five_objects", items[0].choices)

        assert items[0].choices == ("(A)", "(B)", "(C)", "(D)", "(E)")
        assert "(C) Eve finished last" in prefix
        assert "(D)" not in prefix

    def test_every_stem_ends_at_the_answer_cue(self) -> None:
        """The scored span begins right after it, so a stem that stopped short of the
        cue would have the model rank choices against a bare question."""
        assert all(item.question.endswith("\nA:") for item in vendored_items())

    def test_the_description_runs_into_the_first_exemplar(self) -> None:
        """lm-eval inserts no separator, and this is the bank's own copy of that artifact.

        The task-level regression test guards the renderer; this guards the 3,965 strings
        it produced, which are what the difficulties are attached to and what a
        re-vendoring after a tidy-up would silently replace.
        """
        stem = by_subtask()["boolean_expressions"][0].question
        assert stem.startswith("Evaluate the result of a random Boolean expression.Q: ")

    def test_the_three_exemplars_are_in_the_stem(self) -> None:
        """Three answered questions ahead of the unanswered one, so four cues in all."""
        for items in by_subtask().values():
            assert items[0].question.count("\nA:") == 4

    def test_run_time_appends_the_choice_and_nothing_else(self) -> None:
        """What ``prompt_style: bbh`` has to mean if the freeze is to be worth anything."""
        item = vendored_items()[0]
        pairs = our_pairs(item)

        assert {prompt for prompt, _ in pairs} == {item.question}
        assert [continuation for _, continuation in pairs] == [f" {c}" for c in item.choices]

    def test_the_stems_are_whole_prompts_rather_than_questions(self) -> None:
        """A bare-question bank would be 0-shot against 3-shot difficulties and would
        pass every other test here, so the length is worth asserting outright."""
        assert min(len(item.question) for item in vendored_items()) > 200

    def test_the_spec_and_the_manifest_both_record_the_freeze(self) -> None:
        assert SUPPORTED[DATASET].frozen_prompt is True
        assert manifest()["frozen_prompt"] is True


class FakeRequest:
    """The three fields :func:`~..scripts.vendor_bank.frozen_stem` reads off an LMRequest."""

    def __init__(
        self,
        prompt: str,
        continuations: tuple[str, ...],
        continuation_prompts: tuple[str, ...] | None = None,
    ) -> None:
        self.prompt = prompt
        self.continuations = continuations
        self.continuation_prompts = continuation_prompts


class FakeTask:
    """A task that renders one fixed request, so a guard can be aimed at one failure."""

    def __init__(self, request: FakeRequest) -> None:
        self.request = request

    def format_request(self, instance: Any) -> FakeRequest:
        return self.request


class FakeInstance:
    def __init__(self, question: str, choices: tuple[str, ...]) -> None:
        self.question = question
        self.choices = choices
        self.metadata = {"gold_idx": 0}
        self.gold_answer = choices[0]


class TestTheVendoringGuardsBehindTheFreeze:
    """Freezing a prompt puts it out of reach of every later guard, so these are it.

    Each covers a way the frozen stem and the run-time half could come to describe
    different requests while the bank still vendored, joined and scored cleanly. The
    happy path is checked against the committed artifacts above; these are aimed at the
    task changing underneath a re-vendoring, which is when they would fire.
    """

    QUESTION = "What is the answer?"
    CHOICES = ("(A)", "(B)")

    def frozen(self, request: FakeRequest) -> str:
        from ..scripts.vendor_bank import frozen_stem

        return frozen_stem(
            SUPPORTED[DATASET], FakeTask(request), FakeInstance(self.QUESTION, self.CHOICES)
        )

    def test_a_well_formed_render_is_kept_verbatim(self) -> None:
        prompt = f"Description.Q: ex\nA: (A)\n\nQ: {self.QUESTION}\nA:"

        assert self.frozen(FakeRequest(prompt, (" (A)", " (B)"))) == prompt

    def test_a_scored_span_that_is_not_the_choice_aborts(self) -> None:
        """Only the prompt is frozen, so the halves have to agree about the other one."""
        prompt = f"Description.Q: ex\nA: (A)\n\nQ: {self.QUESTION}\nA:"

        with pytest.raises(SystemExit, match="scored span"):
            self.frozen(FakeRequest(prompt, ("A", "B")))

    def test_a_per_choice_prompt_aborts(self) -> None:
        """One stem cannot carry a substitution layout, and collapsing it would score a
        different question than the bank holds a difficulty for."""
        prompt = f"Description.Q: {self.QUESTION}\nA:"

        with pytest.raises(SystemExit, match="varies the prompt per choice"):
            self.frozen(FakeRequest(prompt, (" (A)", " (B)"), ("p1", "p2")))

    def test_a_render_that_lost_its_few_shot_block_aborts(self) -> None:
        """The quiet one: a bare stem here is a 0-shot bank scored against 3-shot
        difficulties, and every count, join and floor downstream would still pass."""
        with pytest.raises(SystemExit, match="0-shot bank"):
            self.frozen(FakeRequest(self.QUESTION, (" (A)", " (B)")))

    def test_a_render_that_dropped_the_question_aborts(self) -> None:
        with pytest.raises(SystemExit, match="not that question inside a framing"):
            self.frozen(FakeRequest("Description.Q: something else\nA:", (" (A)", " (B)")))

    def test_a_generative_spec_may_not_declare_the_freeze(self) -> None:
        """That path settles its own stem and never reads the flag, so a spec setting it
        would be vendored unfrozen with nothing to show for the declaration."""
        from dataclasses import replace

        with pytest.raises(ValueError, match="MCQ-only"):
            replace(SUPPORTED[DATASET], modality="generative", answer_type="numeric")


class TestTheShotCount:
    """3, not 0, everywhere the count is written down."""

    def test_the_style_declares_the_shots_its_stems_carry(self) -> None:
        assert inference.fewshot_count("bbh") == 3
        assert inference.MCQ_PROMPT_STYLES["bbh"].num_fewshot == 3

    def test_the_manifest_records_three(self) -> None:
        """0 here would describe a prompt no model is shown."""
        assert manifest()["scoring_convention"]["runtime"]["num_fewshot"] == 3

    def test_the_task_is_three_shot(self) -> None:
        """The config entry is only right if the task it freezes still says so."""
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        assert registry.get_task("bbh_boolean_expressions").config.num_fewshot == 3

    def test_every_other_mcq_style_still_reports_zero(self) -> None:
        """The four banks that predate this must not have their manifests invalidated."""
        for name, style in inference.MCQ_PROMPT_STYLES.items():
            if name != "bbh":
                assert style.num_fewshot == inference.NUM_FEWSHOT, name

    def test_a_stem_framing_style_would_be_refused_against_this_manifest(self) -> None:
        """The guard that keeps a config edit from quietly dropping the frozen prefix.

        Swapping bbh onto a style that frames the stem is the one edit that would leave
        the items intact, the join intact and the prompts 0-shot, so it is caught by the
        recorded count rather than by anything about the items.
        """
        from .. import convention

        recorded = manifest()["scoring_convention"]["runtime"]
        stem_framing = convention._mcq_convention(
            inference.InferenceConfig(
                prompt_style="bare_context",
                score_normalization=recorded["score_normalization"],
            )
        )
        assert convention._differences(recorded, stem_framing)


class TestTheConfiguredConvention:
    def test_it_is_scored_by_acc_norm_behind_its_own_style(self) -> None:
        config = resolved_mcq_config()
        assert config.prompt_style == "bbh"
        assert config.score_normalization == "continuation_logprob_per_character"

    def test_every_subtask_declares_the_metric_the_config_reproduces(self) -> None:
        metrics = pytest.importorskip("olmo_eval.common.metrics")
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        for subtask in SUBTASK_ITEMS:
            (metric,) = registry.get_task(f"bbh_{subtask}").config.metrics
            assert isinstance(metric, metrics.LogprobPerCharMCAccuracyMetric), subtask

    def test_the_calibration_metric_is_the_one_it_is_graded_under(self) -> None:
        assert SUPPORTED[DATASET].calibration.metric == "acc_norm"

    def test_the_calibration_shot_count_stays_unrecorded(self) -> None:
        """3 is lm-eval's leaderboard configuration, inferred from where the harvest came
        from, and the local calibration directory states nothing -- so the run-time half
        says 3 and the calibration half does not pretend to know."""
        from ..datasets import UNRECORDED

        assert SUPPORTED[DATASET].calibration.num_fewshot == UNRECORDED
        assert SUPPORTED[DATASET].calibration.prompt_style == UNRECORDED

    def test_the_report_note_says_three_shot_and_where_they_came_from(self) -> None:
        from ..style import _scoring_note

        note = _scoring_note(
            grading.MCQ,
            [],
            score_normalization="continuation_logprob_per_character",
            prompt_style="bbh",
        )
        assert "3-shot continuation log-likelihood" in note
        assert "frozen into each item's stem at vendoring" in note

    def test_the_note_for_a_stem_framing_bank_is_unchanged(self) -> None:
        """The four banks scored 0-shot keep the sentence they have always carried."""
        from ..style import _scoring_note

        note = _scoring_note(grading.MCQ, [], prompt_style="question_answer")
        assert note.startswith("Items are scored here by 0-shot continuation log-likelihood")
        assert "frozen" not in note


class TestTheVendoredBank:
    def test_the_counts_are_the_ones_upstream_reports(self) -> None:
        recorded = manifest()
        assert recorded["upstream_bank_rows"] == 5761
        assert recorded["bridge_rows"] == 5761
        assert recorded["dropped"]["non_positive_discrimination"] == 1796
        assert recorded["dropped"]["non_finite_discrimination"] == 0
        assert recorded["dropped"]["not_in_bridge"] == 0
        assert recorded["dropped"]["not_in_task"] == 0
        assert recorded["dropped"]["ambiguous_item_id"] == 0
        assert recorded["items"] == 3965

    def test_the_drop_accounts_for_every_upstream_row(self) -> None:
        recorded = manifest()
        assert recorded["items"] + sum(recorded["dropped"].values()) == 5761

    def test_every_subtask_joined_completely(self) -> None:
        counts = {subtask: len(items) for subtask, items in by_subtask().items()}
        assert counts == SUBTASK_ITEMS
        assert sum(counts.values()) == 3965

    def test_the_three_short_subtasks_are_full_sizes_and_not_truncations(self) -> None:
        """Their bridge spans stop where their upstream splits do, so the alignment guard
        sees a dense 0..n-1 run rather than a gap it would have to tolerate."""
        for subtask in ("causal_judgement", "snarks", "penguins_in_a_table"):
            positions = [int(i.item_id.split("|", 1)[1]) for i in by_subtask()[subtask]]
            assert max(positions) < SUBTASK_ROWS[subtask]

    def test_the_generative_subtasks_are_absent(self) -> None:
        assert set(by_subtask()).isdisjoint(GENERATIVE_SUBTASKS)
        assert all(sub not in manifest()["task"] for sub in GENERATIVE_SUBTASKS)

    def test_it_spans_the_twenty_four_registered_tasks(self) -> None:
        assert manifest()["task"] == ", ".join(f"bbh_{label}" for label in SUBTASK_ITEMS)

    def test_every_item_carries_a_choice_set_and_a_gold_index(self) -> None:
        for item in vendored_items():
            assert len(item.choices) >= 2
            assert 0 <= item.gold_index < len(item.choices)

    def test_the_modality_guard_admits_it(self) -> None:
        request = grading.GradingRequest(dataset=DATASET, modality=grading.MCQ)
        grading.check_bank_modality(request, vendored_items())

    def test_the_cloze_guard_admits_it(self) -> None:
        """It is scored by an appending style, so the guard applies; no BBH stem is cloze."""
        inference.check_prompt_style_fits("bbh", vendored_items())


class TestTheItemFloor:
    def test_this_bank_asks_for_one_item_per_subtask(self) -> None:
        style = UniMcqStyle()
        style.download_benchmark(DATASET)

        assert style.min_items == len(SUBTASK_ITEMS) == 24

    def test_both_entry_points_settle_the_same_floor(self) -> None:
        """``load_irt_params`` can be the first call to name a bank, and a session that
        reached it that way and kept the shared floor would stop early with nothing in
        its report to distinguish that from stopping on precision.

        It then fails on the empty item map, which is the ordinary consequence of
        skipping ``download_benchmark`` and not what is being tested: the floor is
        settled when the bank is resolved, which is before that point.
        """
        downloaded = UniMcqStyle()
        downloaded.download_benchmark(DATASET)

        named = UniMcqStyle()
        with pytest.raises(ValueError, match="No item has both a stem"):
            named.load_irt_params(DATASET)

        assert named.min_items == downloaded.min_items == 24

    def test_the_report_records_the_floor_the_session_actually_used(self) -> None:
        assert run_real_bank(0.5)["metadata"]["cat_settings"]["min_items"] == 24


def run_real_bank(true_theta: float, *, max_items: int = 40) -> dict:
    """A full CAT over the committed 3,965-item bank, with only the forward pass simulated."""
    style = UniMcqStyle()
    bank = style.download_benchmark(DATASET)
    irt = style.load_irt_params(DATASET)
    config = resolved_mcq_config()

    model = inference._HFScoringModel.__new__(inference._HFScoringModel)
    model.config = config
    model._continuation_logprob = SimMcqTaker(
        true_theta, vendored_params(DATASET), list(bank.items), config
    )

    report = cat_loop.run_cat(
        style, bank=bank, irt_bank=irt, model=model, se_threshold=0.3, max_items=max_items
    )
    return report.to_dict()


class TestAbilityRecoveryOnTheRealBank:
    """Theta recovery over the committed bank, through the real engine and scorer.

    Read what this establishes and not more, which on this bank is a narrower thing than
    on the others. The taker draws every response from the bank's own 3PL at one ability,
    so it satisfies the unidimensional assumption exactly, and what recovery proves is
    that the shipped parameters, the selection rule and EAP compose correctly over them.
    BBH's published theta_mae of 0.684 is what happens when the takers are real models
    whose ability differs across 24 unrelated subtasks, and no simulation drawn from a
    single theta can reproduce that -- which is why the caveat is asserted here beside
    the recovery rather than left to the notes.
    """

    run = staticmethod(run_real_bank)

    def test_a_session_converges_and_stops_on_precision(self) -> None:
        report = self.run(0.5)

        assert report["metadata"]["stop_reason"] == "precision_reached"
        assert report["metadata"]["standard_error"] <= 0.3
        assert report["metadata"]["bank_size"] == 3965
        assert np.isfinite(report["ability"]["theta"])

    def test_the_session_runs_to_the_floor_rather_than_stopping_at_eight(self) -> None:
        """The inflated discriminations, seen from the harness: precision is reached long
        before the floor is, so the floor is what sets the length."""
        assert self.run(0.5)["metadata"]["n_items_administered"] == 24

    @pytest.mark.parametrize("true_theta", [-1.0, 0.0, 1.0])
    def test_the_estimate_lands_near_the_truth(self, true_theta: float) -> None:
        assert self.run(true_theta)["metadata"]["theta"] == pytest.approx(true_theta, abs=0.6)

    def test_theta_recovers_monotonically(self) -> None:
        estimates = [self.run(theta)["metadata"]["theta"] for theta in (-1.5, -0.5, 0.5, 1.5)]

        assert all(b > a for a, b in pairwise(estimates)), estimates

    def test_the_floor_widens_coverage_without_completing_it(self) -> None:
        """Both halves of the claim the raised floor is allowed to make.

        Eight items cannot reach a third of the suite by arithmetic; 24 can and does not,
        because Fisher-information selection takes whatever is most informative at the
        current theta and that clusters. The assertion is deliberately weak on the upper
        side -- more subtasks than the floor of 8 could reach, and fewer than all 24 --
        because the honest claim is that raising the floor makes coverage possible rather
        than that it delivers it.
        """
        touched = {i.split("|", 1)[0] for i in self.run(0.0)["metadata"]["selected_item_ids"]}

        assert 8 < len(touched) < 24


class TestItSaysHowFarToTrustIt:
    """The labelling, in the report rather than only in the allowlist."""

    def test_the_report_carries_the_caveat(self) -> None:
        caveat = self.report_caveat()

        assert "predicted accuracy" in caveat
        assert "theta" in caveat

    def test_it_names_both_figures_and_which_one_holds(self) -> None:
        caveat = self.report_caveat()

        assert "0.86" in caveat
        assert "0.674" in caveat
        assert "0.684" in caveat

    def test_it_says_no_cross_validation_was_run(self) -> None:
        """The r beside it is from one 90/10 split, where its siblings have ten folds."""
        caveat = self.report_caveat()

        assert "cross-validation" in caveat
        assert "90/10" in caveat

    def test_the_manifest_records_it_too(self) -> None:
        assert manifest()["bank_caveat"] == SUPPORTED[DATASET].report_caveat

    def test_no_other_bank_carries_the_key(self) -> None:
        """A caveat on every report would be read as boilerplate and stop being read."""
        with_caveat = {name for name, spec in SUPPORTED.items() if spec.report_caveat}

        assert with_caveat == {DATASET}

    @staticmethod
    def report_caveat() -> str:
        return run_real_bank(0.5)["metadata"]["bank_caveat"]
