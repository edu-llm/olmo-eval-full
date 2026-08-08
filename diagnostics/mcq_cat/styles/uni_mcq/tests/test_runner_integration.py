"""The style as reached through the frozen runner CLI.

These are the checks the branch plan lists before a PR: the style is discoverable by
name, ``--dry-run`` resolves without loading a model, and a full run writes a report.
They exercise ``runner.main`` rather than the style directly, so a mismatch between the
CLI's expectations and the style's signatures would surface here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .... import runner
from .. import resolve
from .conftest import write_bank


def test_list_styles_reports_uni_mcq(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main(["--list-styles"]) == 0
    assert "uni_mcq" in capsys.readouterr().out


def test_dry_run_resolves_without_loading_a_model(tmp_path: Path) -> None:
    """A dry run must not touch the checkpoint, so a bogus path is fine."""
    exit_code = runner.main(
        [
            "--cat-style",
            "uni_mcq",
            "--checkpoint",
            "s3://bucket/does-not-exist/step_1000",
            "--s3-out",
            str(tmp_path / "out"),
            "--benchmark",
            "arc_challenge",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not (tmp_path / "out").exists()


def test_missing_required_arguments_is_rejected() -> None:
    assert runner.main(["--cat-style", "uni_mcq"]) == 2


def test_unknown_style_fails_cleanly() -> None:
    assert (
        runner.main(
            [
                "--cat-style",
                "no_such_style",
                "--checkpoint",
                "x",
                "--s3-out",
                "y",
                "--dry-run",
            ]
        )
        == 1
    )


def test_full_run_writes_a_report_locally(monkeypatch, tmp_path: Path, toy_params) -> None:
    """End to end through the CLI with the model and checkpoint stubbed out."""
    banks = tmp_path / "banks"
    write_bank(banks, dataset="arc_challenge")
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", banks)

    from ....common import inference, s3_io
    from .conftest import SimScorer, stage_hf_checkpoint

    staged = stage_hf_checkpoint(tmp_path)
    monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)
    monkeypatch.setattr(inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params))

    out_dir = tmp_path / "out"
    exit_code = runner.main(
        [
            "--cat-style",
            "uni_mcq",
            "--checkpoint",
            "s3://bucket/run/step_1000",
            "--s3-out",
            str(out_dir),
            "--benchmark",
            "arc_challenge",
            "--se-threshold",
            "0.3",
            "--max-items",
            "40",
        ]
    )

    assert exit_code == 0
    report_path = out_dir / "cat_report.json"
    assert report_path.is_file()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["cat_style"] == "uni_mcq"
    assert payload["benchmark"] == "arc_challenge"
    assert payload["run"]["checkpoint"] == "s3://bucket/run/step_1000"

    metadata = payload["metadata"]
    assert metadata["bank_size"] == 5
    assert 0.0 <= metadata["pirt_accuracy"] <= 1.0
    assert isinstance(metadata["theta"], float)


class TestThePreparationSeamIsWiredIn:
    """The runner's two seams: what prepares the checkpoint, and what loads it.

    These pin the composition rather than either half. Preparation and the backend vary
    independently -- convert-then-``hf`` is one cell of that table, and the ones this
    branch does not use (native, served) have to stay one flag away, which is only true
    if the runner reads the flag and records what it did.
    """

    def _run(self, tmp_path: Path, out_dir: Path, *extra: str) -> int:
        return runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/step_1000",
                "--s3-out",
                str(out_dir),
                "--benchmark",
                "arc_challenge",
                *extra,
            ]
        )

    @pytest.fixture
    def wired(self, monkeypatch, tmp_path: Path, toy_params):
        """A run with the fetch and the scorer stubbed, and preparation spied on."""
        from ....common import convert, inference, s3_io
        from .conftest import SimScorer, stage_hf_checkpoint

        write_bank(tmp_path / "banks", dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")

        staged = stage_hf_checkpoint(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)

        seen: dict[str, object] = {}
        real = convert.prepare_checkpoint

        def _spy(local_dir, work_dir, *, policy=convert.PREP_AUTO, **kwargs):
            seen["policy"] = policy
            seen["local_dir"] = local_dir
            return real(local_dir, work_dir, policy=policy, **kwargs)

        monkeypatch.setattr(runner.convert, "prepare_checkpoint", _spy)
        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params)
        )
        return seen, staged

    def test_preparation_runs_between_the_fetch_and_the_load(self, wired, tmp_path: Path) -> None:
        """It has to see what the fetch produced, not the URI the user passed."""
        seen, staged = wired
        assert self._run(tmp_path, tmp_path / "out") == 0
        assert seen["local_dir"] == staged

    def test_the_policy_defaults_to_auto(self, wired, tmp_path: Path) -> None:
        seen, _ = wired
        assert self._run(tmp_path, tmp_path / "out") == 0
        assert seen["policy"] == "auto"

    def test_the_flag_reaches_the_seam(self, wired, tmp_path: Path) -> None:
        seen, _ = wired
        assert self._run(tmp_path, tmp_path / "out", "--checkpoint-prep", "none") == 0
        assert seen["policy"] == "none"

    def test_the_report_records_which_policy_ran(self, wired, tmp_path: Path) -> None:
        """A report that cannot say whether its checkpoint was converted is not
        comparable with one that was, and the two produce the same-shaped number."""
        out_dir = tmp_path / "out"
        assert self._run(tmp_path, out_dir, "--checkpoint-prep", "none") == 0
        payload = json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))
        assert payload["run"]["checkpoint_prep"] == "none"
        assert payload["run"]["checkpoint_kind"] == "hf"

    def test_an_unknown_policy_is_refused_by_the_parser(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            self._run(tmp_path, tmp_path / "out", "--checkpoint-prep", "always")


class TestTheTwoSeamsAreIndependent:
    """The four cells of ``{auto, none} x {hf, olmo_core}``, from the runner's side.

    ``test_olmo_core_scoring.py`` runs the native cell end to end against the real
    scorer; this runs the same flags with the backend replaced by a sentinel, so the
    claim that the two seams do not know about each other is checked in a checkout with
    neither ``olmo_eval`` nor a GPU stack -- which is where the design constraint is
    actually load-bearing, since that is where anyone would notice it being violated.

    What would break these is any coupling in either direction: preparation reading the
    backend name and skipping itself for a native one, or the loader inferring a kind
    from what preparation returned. Both are the sort of shortcut that looks like a
    simplification and quietly removes a configuration from the table.
    """

    def _native_dir(self, tmp_path: Path) -> Path:
        """A directory the layout detector reads as a raw OLMo-core checkpoint."""
        native = tmp_path / "native"
        native.mkdir()
        (native / "config.json").write_text(
            json.dumps({"model": {"d_model": 576}, "dataset": {}}), encoding="utf-8"
        )
        (native / "model_and_optim").mkdir()
        (native / "model_and_optim" / ".metadata").write_bytes(b"\x00")
        return native

    @pytest.fixture
    def native(self, monkeypatch, tmp_path: Path, toy_params):
        """A native checkpoint staged, and the MCQ registry watched rather than stubbed.

        The registry entry is replaced rather than ``load_scoring_model`` itself, so the
        lookup that chooses a backend is the shipped one and what is recorded is the
        directory the chosen backend was handed.
        """
        from ....common import convert, inference, s3_io
        from .conftest import SimScorer

        write_bank(tmp_path / "banks", dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")

        staged = self._native_dir(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)

        seen: dict[str, object] = {"staged": staged, "converted": []}

        def _backend(checkpoint_dir: Path, config: object):
            seen["loaded_from"] = checkpoint_dir
            seen["kind"] = config.checkpoint_kind
            return SimScorer(0.5, toy_params)

        monkeypatch.setitem(inference.MCQ_SCORING_BACKENDS, "olmo_core", _backend)
        monkeypatch.setitem(inference.MCQ_SCORING_BACKENDS, "hf", _backend)

        def _convert(local_dir: Path, out_dir: Path, **kwargs: object) -> Path:
            seen["converted"].append(local_dir)  # type: ignore[union-attr]
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

        monkeypatch.setattr(convert, "convert_olmo_core_to_hf", _convert)
        return seen

    def _run(self, tmp_path: Path, out_dir: Path, *extra: str) -> int:
        return runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/checkpoints/step305176",
                "--s3-out",
                str(out_dir),
                "--benchmark",
                "arc_challenge",
                *extra,
            ]
        )

    def test_prep_none_hands_the_native_backend_the_staged_directory(
        self, native, tmp_path: Path
    ) -> None:
        assert (
            self._run(
                tmp_path,
                tmp_path / "out",
                "--checkpoint-prep",
                "none",
                "--checkpoint-kind",
                "olmo_core",
            )
            == 0
        )
        assert native["converted"] == []
        assert native["loaded_from"] == native["staged"]
        assert native["kind"] == "olmo_core"

    def test_prep_auto_converts_first_whichever_backend_is_named(
        self, native, tmp_path: Path
    ) -> None:
        """The mis-composition, and it is the reason both flags have to be set.

        ``--checkpoint-kind olmo_core`` alone leaves preparation at its default, so the
        native reader is handed the *converted* directory rather than the shards. That is
        a real thing somebody will type, and what makes it recoverable is that the
        directory is visibly not the staged one rather than that anything guessed.
        """
        assert self._run(tmp_path, tmp_path / "out", "--checkpoint-kind", "olmo_core") == 0
        assert native["converted"] == [native["staged"]]
        assert native["loaded_from"] != native["staged"]
        assert native["kind"] == "olmo_core"

    def test_prep_none_with_the_hf_backend_is_still_reachable(self, native, tmp_path: Path) -> None:
        """The fourth cell: a checkpoint prepared out of band, or already HF."""
        assert self._run(tmp_path, tmp_path / "out", "--checkpoint-prep", "none") == 0
        assert native["converted"] == []
        assert native["loaded_from"] == native["staged"]
        assert native["kind"] == "hf"

    def test_the_report_names_the_cell(self, native, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        assert (
            self._run(
                tmp_path,
                out_dir,
                "--checkpoint-prep",
                "none",
                "--checkpoint-kind",
                "olmo_core",
            )
            == 0
        )
        run = json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))["run"]
        assert (run["checkpoint_prep"], run["checkpoint_kind"]) == ("none", "olmo_core")

    def test_the_dry_run_prints_both_flags(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A dry run whose output cannot distinguish two cells is not a plan."""
        with caplog.at_level("INFO"):
            assert (
                self._run(
                    tmp_path,
                    tmp_path / "out",
                    "--checkpoint-prep",
                    "none",
                    "--checkpoint-kind",
                    "olmo_core",
                    "--dry-run",
                )
                == 0
            )
        assert "kind=olmo_core" in caplog.text
        assert "prep=none" in caplog.text


class TestThePrecisionIsOnTheCommandLine:
    """``--dtype``, and the reason it is a flag rather than a default in code.

    The platform's ``bfloat16_not_in_the_hardware`` guard reads the *text* of ``command:``
    in ``.edullm/run.yaml``. It cannot see a precision the program picks for itself, so
    while conversion took ``prepare_checkpoint``'s bfloat16 default a Turing card priced,
    passed, was admitted, was given a machine, and only then refused the first kernel
    wanting the format -- which is the exact failure the guard exists to prevent.

    Naming the flag in the yaml is half of the fix and the cheap half. These pin the
    other half: that the flag is real, that it reaches the converter, and that a value
    the converter cannot honour is refused by the parser. A ``--dtype`` the guard can
    read and the runner cannot parse would trade a failure on one card for a failure on
    every card, which is why the threading is what is under test here and not the text.
    """

    def _run(self, out_dir: Path, *extra: str) -> int:
        return runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/step_1000",
                "--s3-out",
                str(out_dir),
                "--benchmark",
                "arc_challenge",
                *extra,
            ]
        )

    @pytest.fixture
    def seam(self, monkeypatch, tmp_path: Path, toy_params):
        """A run with the fetch and the scorer stubbed, spying on the preparation seam."""
        from ....common import convert, inference, s3_io
        from .conftest import SimScorer, stage_hf_checkpoint

        write_bank(tmp_path / "banks", dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")
        monkeypatch.setattr(
            s3_io, "resolve_checkpoint", lambda *a, **k: stage_hf_checkpoint(tmp_path)
        )
        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params)
        )

        seen: dict[str, object] = {}
        real = convert.prepare_checkpoint

        def _spy(local_dir, work_dir, **kwargs):
            seen["dtype"] = kwargs.get("dtype")
            return real(local_dir, work_dir, **kwargs)

        monkeypatch.setattr(runner.convert, "prepare_checkpoint", _spy)
        return seen

    @pytest.fixture
    def body(self, monkeypatch, tmp_path: Path, toy_params):
        """The same, but staged as OLMo-core and spying on the conversion body itself.

        The seam fixture above stops one call short. Between them sits
        ``ensure_hf_checkpoint``, which takes ``dtype`` positionally in one signature and
        by keyword in the other, so the hop is worth its own assertion.
        """
        import json

        from ....common import convert, inference, s3_io
        from .conftest import SimScorer

        write_bank(tmp_path / "banks", dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")

        native = tmp_path / "native"
        native.mkdir()
        (native / "config.json").write_text(
            json.dumps({"model": {"d_model": 576}, "dataset": {}}), encoding="utf-8"
        )
        (native / "model_and_optim").mkdir()
        (native / "model_and_optim" / ".metadata").write_bytes(b"\x00")
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: native)
        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params)
        )

        seen: list[str] = []

        def _convert(local_dir: Path, out_dir: Path, *, dtype: str = "bfloat16") -> Path:
            seen.append(dtype)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

        monkeypatch.setattr(convert, "convert_olmo_core_to_hf", _convert)
        return seen

    def test_the_default_is_bfloat16(self, seam, tmp_path: Path) -> None:
        """Today's behaviour exactly. Adding the flag must not change what a run does."""
        assert self._run(tmp_path / "out") == 0
        assert seam["dtype"] == "bfloat16"

    def test_an_explicit_precision_reaches_the_seam(self, seam, tmp_path: Path) -> None:
        assert self._run(tmp_path / "out", "--dtype", "float32") == 0
        assert seam["dtype"] == "float32"

    def test_it_reaches_the_conversion_body_and_not_only_the_seam(
        self, body, tmp_path: Path
    ) -> None:
        assert self._run(tmp_path / "out", "--dtype", "float16") == 0
        assert body == ["float16"]

    def test_the_body_takes_the_default_when_nothing_is_asked_for(
        self, body, tmp_path: Path
    ) -> None:
        assert self._run(tmp_path / "out") == 0
        assert body == ["bfloat16"]

    def test_a_precision_the_converter_cannot_honour_is_refused_by_the_parser(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Argparse, not torch, and ``float8_e4m3fn`` rather than a nonsense string.

        It is a real ``olmo_core.config.DType``, so anything that validated by asking
        torch would admit it and the run would get as far as writing weights nothing can
        load. The refusal has to come from the accepted set, at parse time, before a
        byte is fetched.
        """
        with pytest.raises(SystemExit) as excinfo:
            self._run(tmp_path / "out", "--dtype", "float8_e4m3fn")
        assert excinfo.value.code == 2
        stderr = capsys.readouterr().err
        assert "--dtype" in stderr
        assert "bfloat16" in stderr

    def test_the_report_records_the_precision_its_weights_were_written_at(
        self, seam, tmp_path: Path
    ) -> None:
        """Same reason ``checkpoint_prep`` is recorded. Two thetas from the same
        checkpoint at two precisions are two measurements, and a report that cannot name
        which one it is cannot be compared with the other."""
        out_dir = tmp_path / "out"
        assert self._run(out_dir, "--dtype", "float16") == 0
        payload = json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))
        assert payload["run"]["dtype"] == "float16"

    def test_the_report_records_the_default_too(self, seam, tmp_path: Path) -> None:
        """A field only written when the flag was passed would leave every default run
        looking like a run whose precision is unknown."""
        out_dir = tmp_path / "out"
        assert self._run(out_dir) == 0
        payload = json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))
        assert payload["run"]["dtype"] == "bfloat16"

    def test_the_dry_run_names_the_precision(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The dry run prints the plan, and the precision is part of the plan -- the same
        argument that put ``prep=`` there."""
        with caplog.at_level("INFO"):
            assert self._run(tmp_path / "out", "--dtype", "float16", "--dry-run") == 0
        assert "dtype=float16" in caplog.text


class TestTheRunnerClosesWhatItOpens:
    """Inert for both in-process backends, and the reason a served one needs no rewrite."""

    def _model(self, toy_params, closed: list[bool]):
        from .conftest import SimScorer

        scorer = SimScorer(0.5, toy_params)
        scorer.close = lambda: closed.append(True)  # type: ignore[attr-defined]
        return scorer

    @pytest.fixture
    def staged_run(self, monkeypatch, tmp_path: Path):
        from ....common import s3_io
        from .conftest import stage_hf_checkpoint

        write_bank(tmp_path / "banks", dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")
        staged = stage_hf_checkpoint(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)

        def _run(out_dir: Path) -> int:
            return runner.main(
                [
                    "--cat-style",
                    "uni_mcq",
                    "--checkpoint",
                    "s3://bucket/run/step_1000",
                    "--s3-out",
                    str(out_dir),
                    "--benchmark",
                    "arc_challenge",
                ]
            )

        return _run

    def test_a_backend_with_close_is_closed(
        self, monkeypatch, staged_run, tmp_path: Path, toy_params
    ) -> None:
        from ....common import inference

        closed: list[bool] = []
        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: self._model(toy_params, closed)
        )
        assert staged_run(tmp_path / "out") == 0
        assert closed == [True]

    def test_it_is_closed_even_when_the_session_raises(
        self, monkeypatch, staged_run, tmp_path: Path, toy_params
    ) -> None:
        """The case that matters. A served backend leaked on the failure path is a
        process nobody owns, and the failure path is where a real run ends up."""
        from ....common import cat_loop, inference

        closed: list[bool] = []
        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: self._model(toy_params, closed)
        )

        def _explode(*args: object, **kwargs: object):
            raise RuntimeError("the session died")

        monkeypatch.setattr(cat_loop, "run_cat", _explode)

        assert staged_run(tmp_path / "out") == 1
        assert closed == [True]

    def test_a_backend_without_close_is_fine(
        self, monkeypatch, staged_run, tmp_path: Path, toy_params
    ) -> None:
        """Neither in-process scorer defines one, so this is the live path today."""
        from ....common import inference
        from .conftest import SimScorer

        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params)
        )
        assert staged_run(tmp_path / "out") == 0


def test_unsupported_dataset_fails_before_touching_the_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    """The allowlist check must run before anything expensive happens."""
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")

    from ....common import inference, s3_io

    def _boom(*args, **kwargs):
        raise AssertionError("the checkpoint must not be fetched for a bad dataset")

    monkeypatch.setattr(s3_io, "resolve_checkpoint", _boom)
    monkeypatch.setattr(inference, "load_scoring_model", _boom)

    assert (
        runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/step_1000",
                "--s3-out",
                str(tmp_path / "out"),
                "--benchmark",
                "piqa",
            ]
        )
        == 1
    )
