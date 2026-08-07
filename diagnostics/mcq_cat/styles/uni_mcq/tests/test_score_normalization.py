"""How an MCQ choice's log-probability becomes the number that is ranked, per dataset.

Normalization used to be a constant of the scoring scheme, and for the three ATLAS MCQ
banks it is the right one: their tasks declare ``LogprobMCAccuracyMetric``, the plain sum.
MuSR's declares ``LogprobPerCharMCAccuracyMetric`` and its bank was fit on the
leaderboard's ``acc_norm``, so the constant was wrong for exactly one dataset -- which is
the worst case, because the two rules produce the same shape of output and differ only in
which choice wins.

Two properties, and the second is why this file exists as much as the first.

The rule has to be selectable per dataset, checked against the manifest before a
checkpoint is fetched, and it has to actually change the ranking -- otherwise the setting
is decorative and MuSR is still scored under ARC's convention.

And the banks that were vendored before it was selectable have to be **provably**
unaffected: the same prompts to the byte, the same numbers out of the scorer, and the
same string in their manifests. A change that silently re-ranked those three would move
every theta they report with nothing in the output to show it, which is the same failure
the setting was added to prevent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ....base import BenchmarkItem
from ....common import grading, inference
from .. import datasets
from ..convention import CONFIG_PATH
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS

#: The MCQ banks whose tasks declare the unnormalized sum, and which therefore must be
#: scored exactly as they were before the rule became a setting.
UNNORMALIZED = ("arc_challenge", "hellaswag", "winogrande")

#: The prompt style each of them resolves, restated here so the proof below does not
#: depend on reading the config it is checking.
UNNORMALIZED_STYLES = {
    "arc_challenge": "question_answer",
    "hellaswag": "bare_context",
    "winogrande": "blank_substitution",
}


def vendored_items(dataset: str) -> list[BenchmarkItem]:
    """The committed items for ``dataset``, loaded through the real loader."""
    path = CALIBRATED_DATASETS / dataset / "items.jsonl"
    if not path.is_file():
        pytest.skip(f"{dataset} has not been vendored")
    from ....common.benchmark_download import load_items_from_jsonl

    return list(load_items_from_jsonl(path, name=dataset).items)


def resolved_mcq_config(dataset: str) -> inference.InferenceConfig:
    """The MCQ settings the committed ``config.yaml`` resolves for ``dataset``."""
    settings = grading._apply_mcq_overrides(
        grading.GradingSettings(), UniMcqStyle().mcq_settings, dataset=dataset
    )
    return settings.mcq


def mcq_datasets_block() -> dict[str, Any]:
    """The per-dataset half of the committed MCQ settings block."""
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return dict((config.get("mcq") or {}).get(grading.PER_DATASET_KEY) or {})


def scorer(config: inference.InferenceConfig, logprob: Any) -> Any:
    """The shipped scorer with only the forward pass replaced."""
    model = inference._HFScoringModel.__new__(inference._HFScoringModel)
    model.config = config
    model._continuation_logprob = logprob
    return model


class TestTheTable:
    def test_every_rule_is_named_after_itself(self) -> None:
        for key, rule in inference.MCQ_SCORE_NORMALIZATIONS.items():
            assert rule.name == key

    def test_an_unknown_rule_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="Unknown score_normalization 'acc_uncond'"):
            inference.get_mcq_score_normalization("acc_uncond")

    def test_the_default_is_the_unnormalized_sum(self) -> None:
        rule = inference.get_mcq_score_normalization(inference.DEFAULT_SCORE_NORMALIZATION)
        assert isinstance(rule, inference.UnnormalizedSum)

    def test_the_default_name_is_the_string_the_committed_manifests_carry(self) -> None:
        """Renaming it would fail every bank vendored before this at startup.

        Those manifests record the string, and the startup check compares strings, so a
        rename is a behaviour change to four banks whose scoring did not change.
        """
        assert inference.DEFAULT_SCORE_NORMALIZATION == "unnormalized_sum_of_continuation_logprobs"

    def test_every_rule_has_a_report_clause(self) -> None:
        """A theta means nothing without it, so no rule may reach a report unnamed."""
        for name in inference.MCQ_SCORE_NORMALIZATIONS:
            assert inference.normalization_note(name) != f"ranked by {name}"

    def test_the_field_defaults_to_it(self) -> None:
        assert (
            inference.InferenceConfig().score_normalization == inference.DEFAULT_SCORE_NORMALIZATION
        )


class TestTheRules:
    def test_the_sum_returns_the_total_unchanged(self) -> None:
        rule = inference.get_mcq_score_normalization("unnormalized_sum_of_continuation_logprobs")
        assert rule.score(-12.5, " a longer continuation") == -12.5

    def test_per_character_divides_by_the_scored_span(self) -> None:
        rule = inference.get_mcq_score_normalization("continuation_logprob_per_character")
        assert rule.score(-12.0, " abc") == pytest.approx(-3.0)

    def test_the_divisor_includes_the_leading_space(self) -> None:
        """The documented off-by-one against lm-eval, pinned so it stays deliberate.

        ``LogprobPerCharMCAccuracyMetric`` divides by ``len(output.text)`` and the output
        text is the scored span, which every olmo-eval MCQ task builds as ``f" {choice}"``.
        lm-eval divides by ``len(choice)``. Matching the repo keeps one definition of
        acc_norm in this checkout; inventing a second would be the larger divergence.
        """
        rule = inference.get_mcq_score_normalization("continuation_logprob_per_character")
        choice = "gluon"
        assert rule.score(-6.0, f" {choice}") == pytest.approx(-6.0 / (len(choice) + 1))

    def test_an_empty_continuation_does_not_divide_by_zero(self) -> None:
        """A degenerate choice must score finitely rather than take down a session."""
        rule = inference.get_mcq_score_normalization("continuation_logprob_per_character")
        assert rule.score(-4.0, "") == -4.0

    def test_it_matches_the_olmo_eval_metric_it_reproduces(self) -> None:
        """Parity against ``LogprobPerCharMCAccuracyMetric`` rather than against a formula.

        The metric is the definition of ``acc_norm`` here, so what is checked is that it
        and this rule pick the same choice on a case where the sum and the per-character
        mean disagree -- which is exactly the case the setting exists for.
        """
        metrics = pytest.importorskip("olmo_eval.common.metrics")
        types = pytest.importorskip("olmo_eval.common.types")

        # A short wrong option against a long right one: the sum prefers the short one.
        continuations = (" no", " yes, for the reason given above")
        totals = (-6.0, -12.0)
        outputs = [
            types.LMOutput(text=text, logprobs=[{"logprob": total}])
            for text, total in zip(continuations, totals, strict=True)
        ]
        response = types.Response(
            request=types.LMRequest(
                request_type=types.RequestType.LOGLIKELIHOOD,
                prompt="q\n\n1 - no\n2 - yes\n\nAnswer:",
                continuations=continuations,
            ),
            instance=types.Instance(question="q", choices=continuations, metadata={"gold_idx": 1}),
            outputs=outputs,
        )
        assert metrics.LogprobPerCharMCAccuracyMetric().compute_instance(response) == 1.0

        rule = inference.get_mcq_score_normalization("continuation_logprob_per_character")
        scores = [
            rule.score(total, text) for total, text in zip(totals, continuations, strict=True)
        ]
        assert scores.index(max(scores)) == 1
        assert totals.index(max(totals)) == 0


class TestItChangesTheRanking:
    """A setting that could not change an outcome would not need a guard."""

    @pytest.fixture
    def item(self) -> BenchmarkItem:
        return BenchmarkItem(
            item_id="m0",
            question="A long narrative, then a question.",
            choices=("no", "yes, for the reason given above"),
            gold_index=1,
        )

    def logprob(self, prompt: str, continuation: str) -> float:
        """A uniform cost per character, which is the shape a real log-probability has."""
        return -1.0 * len(continuation)

    def chosen(self, item: BenchmarkItem, score_normalization: str) -> int:
        config = inference.InferenceConfig(
            prompt_style="musr", score_normalization=score_normalization
        )
        return scorer(config, self.logprob).score_items([item])[0].chosen_index

    def test_the_sum_hands_the_item_to_the_shorter_choice(self, item: BenchmarkItem) -> None:
        assert self.chosen(item, "unnormalized_sum_of_continuation_logprobs") == 0

    def test_per_character_leaves_them_tied_on_length_alone(self, item: BenchmarkItem) -> None:
        """Under acc_norm a pure length cost cannot decide, so the first choice holds.

        The point is not which index wins a tie but that the length term has been
        removed: it is what decided the item above and it decides nothing here.
        """
        scores = self.scores(item, "continuation_logprob_per_character")
        assert len(set(scores)) == 1

    def scores(self, item: BenchmarkItem, score_normalization: str) -> tuple[float, ...]:
        config = inference.InferenceConfig(
            prompt_style="musr", score_normalization=score_normalization
        )
        return scorer(config, self.logprob).score_items([item])[0].choice_logprobs

    def test_the_response_carries_the_numbers_the_argmax_compared(
        self, item: BenchmarkItem
    ) -> None:
        """Otherwise a report cannot be re-checked against the rule that produced it."""
        summed = self.scores(item, "unnormalized_sum_of_continuation_logprobs")
        per_char = self.scores(item, "continuation_logprob_per_character")

        assert summed == (-len(" no"), -len(" yes, for the reason given above"))
        assert per_char == (-1.0, -1.0)


class TestTheAtlasBanksAreUnaffected:
    """The three MCQ banks vendored before the rule was a setting, held byte for byte."""

    @pytest.mark.parametrize("dataset", UNNORMALIZED)
    def test_the_resolved_rule_is_still_the_default(self, dataset: str) -> None:
        config = resolved_mcq_config(dataset)
        assert config.score_normalization == inference.DEFAULT_SCORE_NORMALIZATION
        assert config.prompt_style == UNNORMALIZED_STYLES[dataset]

    @pytest.mark.parametrize("dataset", UNNORMALIZED)
    def test_their_config_entries_name_no_normalization_at_all(self, dataset: str) -> None:
        """They inherit the default rather than restating it, so it cannot drift per bank."""
        assert "score_normalization" not in mcq_datasets_block()[dataset]

    @pytest.mark.parametrize("dataset", UNNORMALIZED)
    def test_their_manifests_still_record_the_unnormalized_sum(self, dataset: str) -> None:
        path = CALIBRATED_DATASETS / dataset / "manifest.json"
        if not path.is_file():
            pytest.skip(f"{dataset} has not been vendored")
        recorded = json.loads(path.read_text(encoding="utf-8"))["scoring_convention"]["runtime"]
        assert recorded["score_normalization"] == inference.DEFAULT_SCORE_NORMALIZATION

    @pytest.mark.parametrize("dataset", UNNORMALIZED)
    def test_their_prompts_are_byte_identical_under_the_resolved_settings(
        self, dataset: str
    ) -> None:
        """Normalization must not have reached the prompt, which is a separate axis."""
        config = resolved_mcq_config(dataset)
        bare = inference.InferenceConfig(prompt_style=config.prompt_style)
        for item in vendored_items(dataset)[::50]:
            assert inference.scored_choices(item, config) == inference.scored_choices(item, bare)

    @pytest.mark.parametrize("dataset", UNNORMALIZED)
    def test_the_scorer_returns_the_raw_sums_it_always_did(self, dataset: str) -> None:
        """The identity that makes "unaffected" a fact rather than an intention.

        A stand-in forward pass that varies with both halves of the pair, so a rule that
        divided by anything would show up immediately; the scores out of the shipped
        scorer must equal the numbers it was handed.
        """

        def logprob(prompt: str, continuation: str) -> float:
            return -0.013 * len(prompt) - 1.7 * len(continuation)

        config = resolved_mcq_config(dataset)
        model = scorer(config, logprob)
        for item in vendored_items(dataset)[::50]:
            expected = tuple(
                logprob(choice.prompt, choice.continuation)
                for choice in inference.scored_choices(item, config)
            )
            response = model.score_items([item])[0]
            assert response.choice_logprobs == expected
            assert response.chosen_index == expected.index(max(expected))

    def test_gsm8k_never_reads_the_mcq_block_at_all(self) -> None:
        """The fourth ATLAS dataset is generative, so the setting cannot reach it."""
        assert datasets.SUPPORTED["gsm8k"].modality == "generative"
        assert "gsm8k" not in mcq_datasets_block()


class TestTheStartupGuard:
    """A wrong rule has to stop the run before the checkpoint, as a wrong prompt does."""

    def test_a_disagreeing_rule_is_refused_against_the_manifest(self, monkeypatch) -> None:
        """The live path: edit config.yaml after vendoring and the bank stops resolving."""
        import copy

        from .. import convention, resolve

        if not (CALIBRATED_DATASETS / "musr" / "manifest.json").is_file():
            pytest.skip("musr has not been vendored")

        edited = copy.deepcopy(convention.load_config())
        edited["mcq"]["datasets"]["musr"]["score_normalization"] = (
            inference.DEFAULT_SCORE_NORMALIZATION
        )
        monkeypatch.setattr(convention, "load_config", lambda *a, **k: edited)

        style = UniMcqStyle()
        style.download_benchmark("musr")
        request = style.grading_request()
        with pytest.raises(resolve.DatasetNotAvailable, match="score_normalization"):
            style.check_scoring_convention(
                request, grading.resolve_settings(request, grading.GradingSettings())
            )

    def test_an_unknown_rule_is_refused_before_the_checkpoint(self, monkeypatch) -> None:
        """Left to ``score_items`` this would surface after the GPU had been booted."""
        import copy

        from .. import convention

        if not (CALIBRATED_DATASETS / "musr" / "manifest.json").is_file():
            pytest.skip("musr has not been vendored")

        edited = copy.deepcopy(convention.load_config())
        edited["mcq"]["datasets"]["musr"]["score_normalization"] = "acc_per_word"
        monkeypatch.setattr(convention, "load_config", lambda *a, **k: edited)

        style = UniMcqStyle()
        style.download_benchmark("musr")
        with pytest.raises(ValueError, match="Unknown score_normalization"):
            style.grading_request()

    def test_a_misspelled_key_is_still_refused(self) -> None:
        """The existing unknown-key guard covers the new field for free."""
        request = grading.GradingRequest(
            dataset="musr", modality="mcq", mcq={"score_normalisation": "x"}
        )
        with pytest.raises(ValueError, match="Unknown mcq settings"):
            grading.resolve_settings(request, grading.GradingSettings())


class TestTheReportSaysWhichRuleRanked:
    """A report that described the wrong rule would be the failure, one step later."""

    def test_the_note_changes_with_the_rule(self) -> None:
        summed = inference.normalization_note("unnormalized_sum_of_continuation_logprobs")
        per_char = inference.normalization_note("continuation_logprob_per_character")

        assert "not length-normalized" in summed
        assert "acc_norm" in per_char
        assert summed != per_char
