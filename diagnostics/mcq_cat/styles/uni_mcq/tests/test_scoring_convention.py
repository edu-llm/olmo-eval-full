"""The convention a bank was vendored under, as a recorded property and a startup check.

The failure being guarded against leaves no trace in a report. An item's difficulty was
estimated behind one presentation and grading rule; EAP treats it as fixed, so a run
using another puts the whole difference into theta while the standard error, the p-IRT
accuracy and every other number stay plausible. Nothing downstream can notice, which is
why these tests are about the manifest and the startup guard rather than about outputs.

Three properties, and all three have to hold together. Every committed manifest records
the convention the shipped ``config.yaml`` resolves for it, so drift in either file is
caught here rather than in a run. A run whose resolved settings disagree with that
record fails, and fails before the checkpoint is fetched. And the settings that only
decide how long a session runs go on differing freely, because turning every difference
into an error would make the guard something people route around.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from .... import runner
from ....common import grading, inference
from .. import convention, datasets, resolve
from ..datasets import UNRECORDED, CalibrationConvention
from ..scripts import migrate_manifests, vendor_bank
from ..style import UniMcqStyle
from .conftest import (
    CALIBRATED_DATASETS,
    SimScorer,
    make_spec,
    stage_hf_checkpoint,
    unblock,
    write_bank,
)

#: The banks committed to this checkout. Read off disk rather than off the allowlist so
#: a dataset that has not been vendored yet skips instead of failing.
VENDORED = sorted(
    path.parent.name
    for path in CALIBRATED_DATASETS.glob("*/manifest.json")
    if path.parent.name in datasets.SUPPORTED
)


def manifest_of(dataset: str) -> dict[str, Any]:
    """The committed manifest for ``dataset``, or skip if it has not been vendored."""
    path = CALIBRATED_DATASETS / dataset / "manifest.json"
    if not path.is_file():
        pytest.skip(f"{dataset} has not been vendored")
    return json.loads(path.read_text(encoding="utf-8"))


def recorded_runtime(dataset: str) -> dict[str, Any]:
    """The run-time half of ``dataset``'s recorded convention."""
    return manifest_of(dataset)[convention.CONVENTION_KEY][convention.RUNTIME_KEY]


def resolved_settings(style: Any) -> grading.GradingSettings:
    """The settings ``style``'s bank would actually be graded with, as the runner folds them."""
    return grading.resolve_settings(style.grading_request(), grading.GradingSettings())


@pytest.mark.skipif(not VENDORED, reason="no bank has been vendored")
class TestEveryCommittedBankRecordsItsConvention:
    """The record has to exist and has to be the shipped configuration, not a copy of it."""

    @pytest.mark.parametrize("dataset", VENDORED)
    def test_the_block_is_present_and_has_both_halves(self, dataset: str) -> None:
        block = manifest_of(dataset)[convention.CONVENTION_KEY]
        assert convention.RUNTIME_KEY in block
        assert convention.CALIBRATION_KEY in block
        assert block[convention.RECORDED_BY_KEY]

    @pytest.mark.parametrize("dataset", VENDORED)
    def test_it_agrees_with_the_shipped_config(self, dataset: str) -> None:
        """Drift in either the manifest or ``config.yaml`` fails here, not in a GPU run."""
        spec = datasets.get_spec(dataset)
        expected = convention.configured_convention(spec, convention.load_config())
        assert recorded_runtime(dataset) == expected

    @pytest.mark.parametrize("dataset", VENDORED)
    def test_the_recorded_modality_is_the_manifest_modality(self, dataset: str) -> None:
        """Two records of the same fact in one file may not disagree."""
        manifest = manifest_of(dataset)
        assert recorded_runtime(dataset)["modality"] == manifest.get("modality", grading.MCQ)

    @pytest.mark.parametrize("dataset", VENDORED)
    def test_the_startup_check_passes_on_it(self, monkeypatch, dataset: str) -> None:
        """The whole chain on the real artifacts, which is what a run does.

        The comparison above is against one builder; this is against the other end of
        it -- the style resolving a committed bank and the settings the scorer would be
        constructed from -- so a break anywhere between ``config.yaml`` and the check
        shows up as every shipped bank refusing to run.

        A blocked bank is checked too, with its blocker lifted for the call. The
        convention it records is independent of whether its bridge joins the right
        questions, and it has to still agree with ``config.yaml`` on the day the bridge
        is rebuilt -- an artifact allowed to drift while it sits out would come back
        failing this for a second, unrelated reason.
        """
        unblock(monkeypatch, dataset)
        style = UniMcqStyle()
        style.download_benchmark(dataset)
        request = style.grading_request()
        style.check_scoring_convention(
            request, grading.resolve_settings(request, grading.GradingSettings())
        )

    def test_the_atlas_mcq_banks_record_their_own_prompt_format(self) -> None:
        """The formats are per benchmark, and this is where that survives a re-read."""
        for dataset, prompt_style in (
            ("arc_challenge", "question_answer"),
            ("hellaswag", "bare_context"),
            ("winogrande", "blank_substitution"),
        ):
            runtime = recorded_runtime(dataset)
            assert runtime["prompt_style"] == prompt_style
            assert runtime["score_normalization"] == inference.DEFAULT_SCORE_NORMALIZATION
            assert runtime["num_fewshot"] == 0

    def test_musr_records_the_acc_norm_it_was_calibrated_under(self) -> None:
        """The one MCQ bank that departs from the unnormalized sum, recorded as such."""
        runtime = recorded_runtime("musr")
        assert runtime["prompt_style"] == "musr"
        assert runtime["score_normalization"] == "continuation_logprob_per_character"

    def test_each_generative_bank_records_the_grader_that_decides_it(self) -> None:
        for dataset, grader in (
            ("gsm8k", "last_number_exact_match"),
            ("leaderboard_math", "math_latex_equivalence"),
            ("ifeval", "ifeval_prompt_strict"),
        ):
            assert recorded_runtime(dataset)["grader"] == grader

    def test_ifevals_empty_stop_list_is_recorded_as_empty_not_absent(self) -> None:
        """Deliberately no stop sequences, which is not the same as not saying."""
        runtime = recorded_runtime("ifeval")
        assert runtime["stop_sequences"] == []
        assert runtime["num_fewshot"] == 0

    def test_no_generative_bank_is_scored_through_a_chat_template(self) -> None:
        """``chat_format`` decides which checkpoints a bank can be scored on at all.

        Both of the two banks that ever set it have since given it up, for different
        reasons, and the pair is worth pinning together. IFEval is the only bank whose
        calibration is known to contain *both* framings -- Open LLM Leaderboard v2
        templated its chat submissions and not its pretrained ones -- so no run-time
        value matches all of it and the completion half was chosen, being also lm-eval's
        ``leaderboard_ifeval`` unmodified.

        GPQA held out longer and then left the generative side altogether, which was the
        right answer rather than a second flip: it has no completion presentation of the
        chain of thought to adopt, but it does have a presentation, because lm-eval
        scores it as a log-likelihood ranking over four option letters with no system
        prompt for any submission. Removing the template would have invented a framing;
        changing the modality adopted the one the difficulties were fit behind.
        """
        assert recorded_runtime("ifeval")["chat_format"] is False
        assert recorded_runtime("ifeval")["system_prompt_source"] is None
        assert recorded_runtime("gpqa")["modality"] == "mcq"
        assert "chat_format" not in recorded_runtime("gpqa")
        assert not any(
            recorded_runtime(name).get("chat_format")
            for name, spec in datasets.SUPPORTED.items()
            if spec.modality == "generative"
        )

    def test_ifevals_mixed_calibration_framing_is_recorded_beside_it(self) -> None:
        """The scale shift has to be readable off the bank, not just off a commit.

        arc_challenge's 25-shot calibration against an 0-shot run is recorded this way
        and this is the same case: the run-time half says what this harness does, the
        calibration half says what the difficulties were fit behind, and a reader
        comparing a theta against a published number needs the difference in front of
        them. A note that only said "unrecorded" would let the completion-format theta
        read as though it were on the bank's own scale.
        """
        recorded = manifest_of("ifeval")[convention.CONVENTION_KEY][convention.CALIBRATION_KEY]
        assert recorded["prompt_style"] != UNRECORDED
        assert "chat" in recorded["note"]
        assert "completion" in recorded["note"]

    def test_a_zero_shot_bank_names_no_fewshot_block(self) -> None:
        """The inherited source is never read at 0 shots, so pinning it would be noise."""
        assert recorded_runtime("ifeval")["fewshot_source"] is None
        assert recorded_runtime("gsm8k")["fewshot_source"] == "gsm8k"


class TestTheCalibrationHalfIsSeparate:
    """What the bank was fit under is a different claim from what this harness does."""

    @pytest.mark.parametrize("name", sorted(datasets.SUPPORTED))
    def test_every_unknown_is_spelled_out(self, name: str) -> None:
        """``None`` and ``""`` read as not applicable; these facts are simply unknown."""
        recorded = datasets.SUPPORTED[name].calibration.as_dict()
        for field in ("prompt_style", "num_fewshot", "metric"):
            assert recorded[field] not in (None, ""), f"{name}.{field}"

    @pytest.mark.parametrize("name", sorted(datasets.SUPPORTED))
    def test_every_dataset_says_where_its_facts_come_from(self, name: str) -> None:
        assert datasets.SUPPORTED[name].calibration.note

    def test_two_banks_record_a_shot_count_and_they_know_it_differently(self) -> None:
        """One was stated upstream and one was read off the harvest, which is not the
        same standing.

        ARC's 25 comes from the ATLAS release describing its own calibration. GPQA's 0
        comes from having identified the harvest itself -- its response matrix is the
        Open LLM Leaderboard v2 ``acc_norm`` outcome of ``leaderboard_gpqa``, and that
        task is 0-shot -- so the fit records nothing and the number is nonetheless a
        fact about it rather than an assumption. Everything else is honestly unknown,
        and pinning the set is what stops a plausible guess being written into one.
        """
        with_counts = {
            name
            for name, spec in datasets.SUPPORTED.items()
            if spec.calibration.num_fewshot != UNRECORDED
        }
        assert with_counts == {"arc_challenge", "gpqa"}
        assert datasets.SUPPORTED["arc_challenge"].calibration.num_fewshot == 25
        assert datasets.SUPPORTED["gpqa"].calibration.num_fewshot == 0

    def test_arcs_prompt_and_metric_stay_unknown_beside_its_shot_count(self) -> None:
        """A recorded shot count is not licence to reconstruct the rest."""
        recorded = datasets.SUPPORTED["arc_challenge"].calibration
        assert recorded.prompt_style == UNRECORDED
        assert recorded.metric == UNRECORDED

    def test_math_records_the_metric_it_is_graded_against_differently(self) -> None:
        """The one live deviation: math_verify calibrated, Minerva graded."""
        recorded = datasets.SUPPORTED["leaderboard_math"].calibration
        assert "math_verify" in recorded.metric
        assert recorded.metric != recorded_runtime("leaderboard_math")["grader"]

    def test_ifeval_records_the_metric_its_grader_matches(self) -> None:
        assert datasets.SUPPORTED["ifeval"].calibration.metric == "prompt_level_strict_acc"

    def test_the_three_banks_with_nothing_upstream_claim_nothing(self) -> None:
        for name in ("hellaswag", "winogrande", "gsm8k"):
            recorded = datasets.SUPPORTED[name].calibration
            assert recorded.prompt_style == UNRECORDED, name
            assert recorded.num_fewshot == UNRECORDED, name
            assert recorded.metric == UNRECORDED, name


class TestUnrecordedSurvivesARoundTrip:
    """An honest unknown is only useful if it is still there when it is read back."""

    def test_json_neither_drops_it_nor_coerces_it(self) -> None:
        restored = json.loads(json.dumps(CalibrationConvention().as_dict()))
        assert restored == {
            "prompt_style": UNRECORDED,
            "num_fewshot": UNRECORDED,
            "metric": UNRECORDED,
            "note": "",
        }

    def test_a_recorded_int_and_an_unknown_coexist(self) -> None:
        """arc's shape: one field an integer, the two beside it strings."""
        recorded = CalibrationConvention(num_fewshot=25).as_dict()
        restored = json.loads(json.dumps(recorded))
        assert restored["num_fewshot"] == 25
        assert restored["prompt_style"] == UNRECORDED

    @pytest.mark.parametrize("dataset", VENDORED)
    def test_the_committed_manifests_hold_the_literal_string(self, dataset: str) -> None:
        recorded = manifest_of(dataset)[convention.CONVENTION_KEY][convention.CALIBRATION_KEY]
        expected = datasets.get_spec(dataset).calibration.as_dict()
        assert recorded == expected
        for field in ("prompt_style", "num_fewshot", "metric"):
            assert recorded[field] not in (None, "")


class TestAMissingBlockIsAnError:
    """The pre-existing state of every bank, and it must not read as agreement."""

    @pytest.fixture
    def stripped(self, monkeypatch, tmp_path: Path) -> resolve.ResolvedBank:
        bank = write_bank(tmp_path, dataset="arc_challenge")
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        del manifest[convention.CONVENTION_KEY]
        (bank / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        return resolve.resolve("arc_challenge")

    def test_it_raises_rather_than_passing_quietly(self, stripped) -> None:
        with pytest.raises(resolve.DatasetNotAvailable):
            convention.check_runtime_convention(
                stripped, grading.GradingSettings(), modality=grading.MCQ
            )

    def test_the_message_names_the_dataset_and_the_way_out(self, stripped) -> None:
        with pytest.raises(resolve.DatasetNotAvailable) as exc:
            convention.check_runtime_convention(
                stripped, grading.GradingSettings(), modality=grading.MCQ
            )
        message = exc.value.args[0]
        assert "arc_challenge" in message
        assert "migrate_manifests" in message

    def test_a_block_with_no_runtime_half_is_the_same_failure(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Half a record is not a record; the fields it does carry prove nothing."""
        bank = write_bank(tmp_path, dataset="arc_challenge")
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        del manifest[convention.CONVENTION_KEY][convention.RUNTIME_KEY]
        (bank / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        with pytest.raises(resolve.DatasetNotAvailable, match="records no"):
            convention.check_runtime_convention(
                resolve.resolve("arc_challenge"), grading.GradingSettings(), modality=grading.MCQ
            )


class TestAMismatchIsRefused:
    """A difference in the convention is an error, on ``check_fit_family``'s terms."""

    @pytest.fixture
    def resolved(self, monkeypatch, tmp_path: Path) -> resolve.ResolvedBank:
        write_bank(tmp_path, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        return resolve.resolve("arc_challenge")

    def settings(self, **overrides: Any) -> grading.GradingSettings:
        return grading.GradingSettings(mcq=inference.InferenceConfig(**overrides))

    def test_the_recorded_configuration_passes(self, resolved) -> None:
        convention.check_runtime_convention(
            resolved, self.settings(prompt_style="question_answer"), modality=grading.MCQ
        )

    def test_a_changed_prompt_style_raises(self, resolved) -> None:
        with pytest.raises(resolve.DatasetNotAvailable) as exc:
            convention.check_runtime_convention(
                resolved, self.settings(prompt_style="bare_context"), modality=grading.MCQ
            )
        message = exc.value.args[0]
        assert "arc_challenge" in message
        assert "prompt_style" in message
        assert "'question_answer'" in message
        assert "'bare_context'" in message

    def test_the_message_says_what_to_do_about_it(self, resolved) -> None:
        with pytest.raises(resolve.DatasetNotAvailable) as exc:
            convention.check_runtime_convention(
                resolved, self.settings(prompt_style="bare_context"), modality=grading.MCQ
            )
        message = exc.value.args[0]
        assert "config.yaml" in message
        assert "vendor_bank" in message

    def test_a_changed_shot_count_raises(self, monkeypatch, tmp_path: Path) -> None:
        """The generative side of the same guard, and the setting with the most reach."""
        from .conftest import GENERATIVE_DATASET

        monkeypatch.setitem(
            datasets.SUPPORTED,
            GENERATIVE_DATASET,
            make_spec(GENERATIVE_DATASET, modality="generative"),
        )
        write_bank(tmp_path, dataset=GENERATIVE_DATASET, modality="generative")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        resolved = resolve.resolve(GENERATIVE_DATASET)

        from ....common import generative

        with pytest.raises(resolve.DatasetNotAvailable, match="num_fewshot"):
            convention.check_runtime_convention(
                resolved,
                grading.GradingSettings(generation=generative.GenerationConfig(num_fewshot=5)),
                modality=grading.GENERATIVE,
            )

    def test_a_field_the_manifest_never_recorded_is_reported(self, resolved) -> None:
        """A manifest written against an older schema cannot be assumed to agree."""
        del resolved.manifest[convention.CONVENTION_KEY][convention.RUNTIME_KEY]["prompt_style"]
        with pytest.raises(resolve.DatasetNotAvailable, match="records nothing"):
            convention.check_runtime_convention(
                resolved, self.settings(prompt_style="question_answer"), modality=grading.MCQ
            )

    def test_a_field_the_harness_dropped_is_reported(self, resolved) -> None:
        resolved.manifest[convention.CONVENTION_KEY][convention.RUNTIME_KEY]["retired"] = "x"
        with pytest.raises(resolve.DatasetNotAvailable, match="no longer has a setting for"):
            convention.check_runtime_convention(
                resolved, self.settings(prompt_style="question_answer"), modality=grading.MCQ
            )


class TestSettingsThatMayDiffer:
    """Precision and plumbing are the caller's; the scale theta is measured on is not."""

    @pytest.fixture
    def resolved(self, monkeypatch, tmp_path: Path) -> resolve.ResolvedBank:
        write_bank(tmp_path, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        return resolve.resolve("arc_challenge")

    def test_batch_size_seed_and_checkpoint_kind_are_not_the_convention(self, resolved) -> None:
        """None of them can change whether an item is answered correctly."""
        convention.check_runtime_convention(
            resolved,
            grading.GradingSettings(
                mcq=inference.InferenceConfig(
                    prompt_style="question_answer",
                    batch_size=1,
                    seed=99,
                    device_map="cpu",
                    checkpoint_kind="olmo_core",
                )
            ),
            modality=grading.MCQ,
        )

    def test_a_prompt_length_cap_is_the_convention(self, resolved) -> None:
        """It reads as a memory knob and truncates the prompt, few-shot block included."""
        with pytest.raises(resolve.DatasetNotAvailable, match="max_length"):
            convention.check_runtime_convention(
                resolved,
                grading.GradingSettings(
                    mcq=inference.InferenceConfig(prompt_style="question_answer", max_length=128)
                ),
                modality=grading.MCQ,
            )

    def test_the_recorded_fields_are_only_those_that_move_a_score(self) -> None:
        """Pinned as a list, because widening it later is how a guard gets disabled."""
        assert set(recorded_runtime("arc_challenge")) == {
            "modality",
            "prompt_style",
            "num_fewshot",
            "score_normalization",
            "max_length",
        }
        assert set(recorded_runtime("gsm8k")) == {
            "modality",
            "prompt_style",
            "num_fewshot",
            "fewshot_source",
            "chat_format",
            "system_prompt_source",
            "stop_sequences",
            "max_new_tokens",
            "max_length",
            "grader",
        }


class TestThroughTheRunner:
    """Where the check has to fire: beside the modality guard, before the checkpoint."""

    @pytest.fixture
    def banks(self, monkeypatch, tmp_path: Path) -> Path:
        root = tmp_path / "banks"
        write_bank(root, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", root)
        return root

    def run(self, tmp_path: Path, *extra: str) -> int:
        return runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/step_1000",
                "--s3-out",
                str(tmp_path / "out"),
                "--benchmark",
                "arc_challenge",
                *extra,
            ]
        )

    def test_a_config_edited_after_vendoring_fails_the_run(
        self, monkeypatch, banks: Path, tmp_path: Path
    ) -> None:
        """The scenario in full: someone changes the prompt style and reruns the bank."""
        from ....common import s3_io

        edited = copy.deepcopy(convention.load_config())
        edited["mcq"]["datasets"]["arc_challenge"]["prompt_style"] = "bare_context"
        monkeypatch.setattr(convention, "load_config", lambda *a, **k: edited)

        def _boom(*args: Any, **kwargs: Any):
            raise AssertionError("the checkpoint must not be fetched under a wrong convention")

        monkeypatch.setattr(s3_io, "resolve_checkpoint", _boom)
        monkeypatch.setattr(inference, "load_scoring_model", _boom)

        assert self.run(tmp_path) == 1
        assert not (tmp_path / "out").exists()

    def test_a_runtime_only_difference_still_runs(
        self, monkeypatch, banks: Path, tmp_path: Path, toy_params
    ) -> None:
        """``max_items`` moves precision, not scale, and the report already flags it."""
        from ....common import s3_io

        staged = stage_hf_checkpoint(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)
        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params)
        )

        assert self.run(tmp_path, "--max-items", "3", "--batch-size", "2") == 0

        report = json.loads((tmp_path / "out" / "cat_report.json").read_text(encoding="utf-8"))
        settings = report["metadata"]["cat_settings"]
        assert settings["max_items"] == 3
        assert settings["max_items_is_pinned_value"] is False

    def test_the_convention_reaches_the_report(
        self, monkeypatch, banks: Path, tmp_path: Path, toy_params
    ) -> None:
        """A theta is not interpretable without it, so it travels with the number."""
        from ....common import s3_io

        staged = stage_hf_checkpoint(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)
        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params)
        )
        assert self.run(tmp_path) == 0

        report = json.loads((tmp_path / "out" / "cat_report.json").read_text(encoding="utf-8"))
        block = report["metadata"]["bank_provenance"][convention.CONVENTION_KEY]
        assert block[convention.RUNTIME_KEY]["prompt_style"] == "question_answer"
        assert block[convention.CALIBRATION_KEY]["prompt_style"] == UNRECORDED

    def test_a_style_with_no_hook_is_left_alone(self) -> None:
        """The hook is optional, as ``grading_request`` is; a style may keep no record."""
        grading.check_scoring_convention(
            object(), grading.GradingRequest(dataset="x"), grading.GradingSettings()
        )


class TestVendoringWritesIt:
    """The manifest has to be produced by vendoring, not maintained beside it."""

    def spec(self) -> datasets.DatasetSpec:
        return make_spec("arc_challenge")

    def result(self) -> vendor_bank.VendorResult:
        return vendor_bank.VendorResult(
            items=[{"id": "0", "question": "q", "choices": ["a", "b"], "gold_index": 0}],
            params=[{"item_id": "0", "difficulty": 0.0, "discrimination": 1.0, "guessing": 0.0}],
            drops=vendor_bank.DropCounts(),
            upstream_bank_rows=1,
            bridge_rows=1,
        )

    def written(self, tmp_path: Path) -> dict[str, Any]:
        vendor_bank.write_artifacts(
            self.spec(),
            self.result(),
            source_ref="test",
            source_commit="0" * 40,
            fit_family="3pl",
            out_dir=tmp_path,
        )
        return json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    def test_the_block_is_written(self, tmp_path: Path) -> None:
        block = self.written(tmp_path)[convention.CONVENTION_KEY]
        assert block[convention.RECORDED_BY_KEY] == vendor_bank.VENDOR_RECORDED_BY
        assert block[convention.RUNTIME_KEY]["prompt_style"] == "question_answer"

    def test_what_it_writes_is_what_a_run_would_resolve(self, tmp_path: Path) -> None:
        """The property the whole guard rests on: one builder, both sides."""
        recorded = self.written(tmp_path)[convention.CONVENTION_KEY][convention.RUNTIME_KEY]
        assert recorded == convention.configured_convention(self.spec(), convention.load_config())


@pytest.mark.skipif(not VENDORED, reason="no bank has been vendored")
class TestTheMigration:
    """It exists to add a record, so it must be able to prove it added nothing else."""

    @pytest.fixture
    def bank(self, tmp_path: Path) -> Path:
        """A byte-for-byte copy of a committed bank, so the real digests are in play."""
        import shutil

        root = tmp_path / "arc_challenge"
        shutil.copytree(CALIBRATED_DATASETS / "arc_challenge", root)
        return root

    def test_the_committed_manifests_are_already_migrated(self) -> None:
        assert migrate_manifests.main(["--dry-run"]) == 0, (
            "the dry run must complete for every committed bank"
        )
        for dataset in VENDORED:
            assert convention.CONVENTION_KEY in manifest_of(dataset)

    def test_rerunning_changes_nothing(self, bank: Path) -> None:
        before = (bank / "manifest.json").read_text(encoding="utf-8")
        migrate_manifests.migrate(
            "arc_challenge", convention.load_config(), root=bank, dry_run=False
        )
        assert (bank / "manifest.json").read_text(encoding="utf-8") == before

    def test_it_preserves_a_vendored_attribution(self, bank: Path) -> None:
        """Re-running must not relabel a block that vendoring actually wrote."""
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        manifest[convention.CONVENTION_KEY][convention.RECORDED_BY_KEY] = (
            vendor_bank.VENDOR_RECORDED_BY
        )
        (bank / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        migrate_manifests.migrate(
            "arc_challenge", convention.load_config(), root=bank, dry_run=False
        )

        rewritten = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        assert (
            rewritten[convention.CONVENTION_KEY][convention.RECORDED_BY_KEY]
            == vendor_bank.VENDOR_RECORDED_BY
        )

    def test_an_artifact_that_moved_aborts_instead_of_being_papered_over(self, bank: Path) -> None:
        """A bank that already misdescribes itself is not one to add a claim to."""
        (bank / "items.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="changed after it was vendored"):
            migrate_manifests.migrate(
                "arc_challenge", convention.load_config(), root=bank, dry_run=True
            )

    def test_a_count_that_no_longer_adds_up_aborts(self, bank: Path) -> None:
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        manifest["items"] = 1
        (bank / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(SystemExit, match="must agree"):
            migrate_manifests.migrate(
                "arc_challenge", convention.load_config(), root=bank, dry_run=True
            )

    def test_a_re_record_leaves_the_block_where_vendoring_puts_it(self, tmp_path: Path) -> None:
        """Re-recording a bank that carries a ``bank_caveat`` must not step over it.

        The script inserts the block before a named key, and it named only ``notes``
        while every bank it had ever migrated predated ``bank_caveat``. Re-recording one
        vendored with a caveat -- which is what changing a live setting now requires --
        put the block on the far side of it, leaving that one manifest shaped unlike its
        siblings. Nothing reads a manifest positionally, so this is only about a
        difference a reader would have to account for and could not.
        """
        import shutil

        root = tmp_path / "bbh"
        shutil.copytree(CALIBRATED_DATASETS / "bbh", root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["bank_caveat"], "bbh is the bank with a caveat; pick another if it moves"
        before = list(manifest)
        del manifest[convention.CONVENTION_KEY]
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        migrate_manifests.migrate("bbh", convention.load_config(), root=root, dry_run=False)

        assert list(json.loads((root / "manifest.json").read_text(encoding="utf-8"))) == before

    def test_a_dry_run_writes_nothing(self, bank: Path) -> None:
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        del manifest[convention.CONVENTION_KEY]
        payload = json.dumps(manifest, indent=2)
        (bank / "manifest.json").write_text(payload, encoding="utf-8")

        assert migrate_manifests.migrate(
            "arc_challenge", convention.load_config(), root=bank, dry_run=True
        )
        assert (bank / "manifest.json").read_text(encoding="utf-8") == payload
