"""The preparation seam: which layout a staged checkpoint is, and what happens to it.

Every case here is a directory a run can actually be handed. Two of them come from the
happy paths -- a training run that already writes HF, and one that writes raw OLMo-core --
and the rest are the ways a checkpoint arrives broken: an upload that stopped after the
config, a prefix that was empty, a path that was a typo. They are worth pinning because
the expensive failure in this pipeline is not a crash, it is a run that converts or loads
the wrong thing and reports a plausible theta for it.

Nothing here needs torch. The detection and the policy are pure filesystem questions, and
the conversion body is injected, which is the whole reason those two halves live in
separate functions.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ....common import convert

#: The conversion body's dependencies. Where they are present the two tests below cannot
#: reach the branch they pin, because the imports succeed.
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_no_torch = pytest.mark.skipif(
    HAS_TORCH, reason="torch is installed, so the missing-dependency branch is unreachable"
)


def write_hf(root: Path, name: str = "hf") -> Path:
    """A directory ``transformers`` would open: architecture, weights and a tokenizer."""
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps({"architectures": ["OlmoForCausalLM"], "hidden_size": 576}), encoding="utf-8"
    )
    (d / "model.safetensors").write_bytes(b"\x00")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    return d


def write_olmo_core(root: Path, name: str = "native", sharded: bool = True) -> Path:
    """A raw OLMo-core step directory: experiment config plus a sharded DCP checkpoint."""
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps(
            {
                "model": {"d_model": 576, "n_layers": 30},
                "dataset": {"tokenizer": {"identifier": "HuggingFaceTB/SmolLM2-135M"}},
            }
        ),
        encoding="utf-8",
    )
    if sharded:
        (d / "model_and_optim").mkdir()
        (d / "model_and_optim" / ".metadata").write_bytes(b"\x00")
        (d / "model_and_optim" / "__0_0.distcp").write_bytes(b"\x00")
    return d


class TestLayoutDetection:
    """Which of the two known formats a directory is, if either."""

    def test_an_hf_directory_is_hf_and_not_olmo_core(self, tmp_path: Path) -> None:
        d = write_hf(tmp_path)
        assert convert.is_hf_checkpoint(d)
        assert not convert.is_olmo_core_checkpoint(d)

    def test_an_olmo_core_directory_is_olmo_core_and_not_hf(self, tmp_path: Path) -> None:
        d = write_olmo_core(tmp_path)
        assert convert.is_olmo_core_checkpoint(d)
        assert not convert.is_hf_checkpoint(d)

    def test_olmo_core_is_recognized_from_its_config_without_the_shards(
        self, tmp_path: Path
    ) -> None:
        """``model_and_optim/`` is positive evidence; its absence is not the opposite."""
        assert convert.is_olmo_core_checkpoint(write_olmo_core(tmp_path, sharded=False))

    def test_weights_without_a_tokenizer_are_not_hf(self, tmp_path: Path) -> None:
        """The shape of an upload that stopped early, and it must not read as loadable."""
        d = tmp_path / "half"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"architectures": ["X"]}), encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"\x00")
        assert not convert.is_hf_checkpoint(d)

    def test_a_tokenizer_without_weights_is_not_hf(self, tmp_path: Path) -> None:
        d = tmp_path / "half"
        d.mkdir()
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
        assert not convert.is_hf_checkpoint(d)

    def test_an_hf_config_naming_an_architecture_is_never_olmo_core(self, tmp_path: Path) -> None:
        """The clause that keeps the ladder's branches disjoint.

        A converted directory has both a ``model``-ish config and HF weights. Without the
        ``architectures`` check it would answer ``True`` to both detectors, and which
        branch ran would depend on the order they happen to be tested in.
        """
        d = tmp_path / "converted"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"architectures": ["OlmoForCausalLM"], "model": {}, "dataset": {}}),
            encoding="utf-8",
        )
        assert not convert.is_olmo_core_checkpoint(d)

    def test_an_unparseable_config_is_neither(self, tmp_path: Path) -> None:
        """A truncated download, which is a broken checkpoint rather than a third format."""
        d = tmp_path / "truncated"
        d.mkdir()
        (d / "config.json").write_text('{"model": {"d_mod', encoding="utf-8")
        assert not convert.is_hf_checkpoint(d)
        assert not convert.is_olmo_core_checkpoint(d)


class TestTheLadderRoutesEachLayout:
    """:func:`ensure_hf_checkpoint` with the conversion body injected."""

    @pytest.fixture
    def spy(self, monkeypatch):
        calls: list[tuple[Path, Path, str]] = []

        def _convert(local_dir: Path, out_dir: Path, *, dtype: str = "bfloat16") -> Path:
            calls.append((local_dir, out_dir, dtype))
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

        monkeypatch.setattr(convert, "convert_olmo_core_to_hf", _convert)
        return calls

    def test_hf_passes_through_without_converting(self, tmp_path: Path, spy, caplog) -> None:
        """Returned as staged, not copied. The common case must not cost a rewrite.

        The log assertion is not decoration. The pass-through branch below returns
        ``local_dir`` too, so an HF directory that stopped being *recognized* as HF would
        still come back unchanged and every run would look identical -- except for a
        warning, on every healthy checkpoint, saying its layout was not understood.
        """
        d = write_hf(tmp_path)
        with caplog.at_level("INFO"):
            assert convert.ensure_hf_checkpoint(d, tmp_path / "out") == d
        assert spy == []
        assert not (tmp_path / "out").exists()
        assert "already HF format" in caplog.text
        assert "neither clearly HF nor OLMo-core" not in caplog.text

    def test_olmo_core_is_converted_into_the_output_directory(self, tmp_path: Path, spy) -> None:
        d = write_olmo_core(tmp_path)
        out = tmp_path / "out"
        assert convert.ensure_hf_checkpoint(d, out) == out
        assert spy == [(d, out, "bfloat16")]

    def test_the_dtype_reaches_the_converter(self, tmp_path: Path, spy) -> None:
        """Not hardcoded here. A Turing card cannot take the bfloat16 default."""
        d = write_olmo_core(tmp_path)
        convert.ensure_hf_checkpoint(d, tmp_path / "out", dtype="float16")
        assert spy[0][2] == "float16"

    def test_a_config_matching_neither_is_passed_through_with_a_warning(
        self, tmp_path: Path, spy, caplog
    ) -> None:
        """Let the loader report what it cannot read; it knows more than this function."""
        d = tmp_path / "odd"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"something": "else"}), encoding="utf-8")

        with caplog.at_level("WARNING"):
            assert convert.ensure_hf_checkpoint(d, tmp_path / "out") == d

        assert spy == []
        assert "neither clearly HF nor OLMo-core" in caplog.text

    def test_the_partially_synced_warning_names_what_was_found(
        self, tmp_path: Path, spy, caplog
    ) -> None:
        """The case a real S3 interruption produces: a config and nothing else.

        It reaches the pass-through branch, which is correct -- but a warning saying only
        that the layout was unrecognized would leave the reader unable to tell a bad
        prefix from a dead upload, so the contents have to be in the message.
        """
        d = tmp_path / "interrupted"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"architectures": ["X"]}), encoding="utf-8")

        with caplog.at_level("WARNING"):
            convert.ensure_hf_checkpoint(d, tmp_path / "out")

        assert "weights=none" in caplog.text
        assert "tokenizer=none" in caplog.text

    def test_an_unrecognizable_directory_raises_naming_the_path_and_the_layout(
        self, tmp_path: Path, spy
    ) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="Unrecognized checkpoint layout") as excinfo:
            convert.ensure_hf_checkpoint(d, tmp_path / "out")
        assert str(d) in str(excinfo.value)
        assert "empty directory" in str(excinfo.value)
        assert spy == []

    def test_a_missing_path_raises_rather_than_being_converted(self, tmp_path: Path, spy) -> None:
        with pytest.raises(ValueError, match="path does not exist"):
            convert.ensure_hf_checkpoint(tmp_path / "nope", tmp_path / "out")
        assert spy == []


class TestThePreparationPolicy:
    """The seam that makes a native backend a flag rather than a revert."""

    @pytest.fixture
    def spy(self, monkeypatch):
        calls: list[Path] = []

        def _convert(local_dir: Path, out_dir: Path, *, dtype: str = "bfloat16") -> Path:
            calls.append(local_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

        monkeypatch.setattr(convert, "convert_olmo_core_to_hf", _convert)
        return calls

    def test_auto_converts_a_native_checkpoint(self, tmp_path: Path, spy) -> None:
        d = write_olmo_core(tmp_path)
        out = tmp_path / "out"
        assert convert.prepare_checkpoint(d, out, policy="auto") == out
        assert spy == [d]

    def test_auto_is_the_default(self, tmp_path: Path, spy) -> None:
        d = write_olmo_core(tmp_path)
        assert convert.prepare_checkpoint(d, tmp_path / "out") == tmp_path / "out"
        assert spy == [d]

    def test_none_hands_a_native_checkpoint_over_untouched(self, tmp_path: Path, spy) -> None:
        """The assertion that fails if this policy is ever 'simplified' away.

        ``none`` is the entire mechanism by which a backend that reads OLMo-core natively
        is reachable -- a flag rather than reverting this module. It looks like dead code
        precisely because the backend that needs it is parked, so it is pinned here.
        """
        d = write_olmo_core(tmp_path)
        assert convert.prepare_checkpoint(d, tmp_path / "out", policy="none") == d
        assert spy == []
        assert not (tmp_path / "out").exists()

    def test_none_says_so(self, tmp_path: Path, spy, caplog) -> None:
        """Turned-off preparation and an unreadable format fail the same way downstream."""
        d = write_olmo_core(tmp_path)
        with caplog.at_level("INFO"):
            convert.prepare_checkpoint(d, tmp_path / "out", policy="none")
        assert "preparation is off" in caplog.text

    def test_none_does_not_inspect_the_directory_at_all(self, tmp_path: Path, spy) -> None:
        """Not even to validate it. A path this policy cannot read is the backend's to
        report, and a missing directory here would otherwise raise from the wrong place."""
        missing = tmp_path / "nope"
        assert convert.prepare_checkpoint(missing, tmp_path / "out", policy="none") == missing

    def test_an_unknown_policy_is_refused_by_name(self, tmp_path: Path, spy) -> None:
        with pytest.raises(ValueError, match="Unknown checkpoint preparation policy"):
            convert.prepare_checkpoint(write_hf(tmp_path), tmp_path / "out", policy="always")


class TestThePrecisionTheSeamWritesAt:
    """``dtype`` through the policy, and the two ways it can be got wrong.

    The precision matters twice over. It decides what the output weighs and what
    hardware can read it, and -- because the platform's hardware guard reads the text of
    a command rather than the behaviour of a program -- it is the difference between a
    card without bfloat16 being refused for free and being billed for a machine that
    dies on the first kernel.
    """

    @pytest.fixture
    def spy(self, monkeypatch):
        calls: list[str] = []

        def _convert(local_dir: Path, out_dir: Path, *, dtype: str = "bfloat16") -> Path:
            calls.append(dtype)
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

        monkeypatch.setattr(convert, "convert_olmo_core_to_hf", _convert)
        return calls

    def test_auto_defaults_to_bfloat16(self, tmp_path: Path, spy) -> None:
        convert.prepare_checkpoint(write_olmo_core(tmp_path), tmp_path / "out")
        assert spy == ["bfloat16"]

    def test_auto_carries_the_requested_precision_to_the_converter(
        self, tmp_path: Path, spy
    ) -> None:
        """The hop the runner actually takes; ``ensure_hf_checkpoint`` is only the middle."""
        convert.prepare_checkpoint(write_olmo_core(tmp_path), tmp_path / "out", dtype="float16")
        assert spy == ["float16"]

    def test_an_unknown_precision_is_refused_by_name(self, tmp_path: Path, spy) -> None:
        with pytest.raises(ValueError, match="Unknown conversion dtype"):
            convert.prepare_checkpoint(write_olmo_core(tmp_path), tmp_path / "out", dtype="bf16")
        assert spy == []

    def test_a_precision_torch_has_and_this_pipeline_cannot_load_back_is_refused(
        self, tmp_path: Path, spy
    ) -> None:
        """``float8_e4m3fn`` is a real ``DType``, which is the point of testing with it.

        A check that only asked whether the string names a torch dtype would pass it
        through, ``save_hf_model`` would happily cast to it, and the directory that came
        out would carry no quantization block for ``from_pretrained`` to read -- so the
        conversion would succeed and the load two lines later would not.
        """
        with pytest.raises(ValueError, match="float8_e4m3fn"):
            convert.prepare_checkpoint(
                write_olmo_core(tmp_path), tmp_path / "out", dtype="float8_e4m3fn"
            )
        assert spy == []

    def test_the_body_refuses_before_importing_or_touching_the_disk(self, tmp_path: Path) -> None:
        """Not spied: the real body, with the check ahead of everything it costs.

        Same ordering argument as the dependency check and the config patch. A precision
        validated at ``DType(dtype)`` would be validated as an argument to
        ``save_hf_model``, which is after the model has been rebuilt and 1.74 GB of
        shards read into it.
        """
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="Unknown conversion dtype"):
            convert.convert_olmo_core_to_hf(write_olmo_core(tmp_path), out, dtype="int8")
        assert not out.exists()

    def test_none_warns_when_a_precision_was_actually_asked_for(
        self, tmp_path: Path, spy, caplog
    ) -> None:
        """Silence here is the failure this flag exists to prevent, one policy over.

        Somebody sets ``--dtype float16`` because the card they were given has no
        bfloat16. Under ``none`` nothing is converted, so this function has honoured
        nothing -- and if they are on the HF backend the way they find out is a kernel
        refusing a format, which reads like the flag is broken rather than inapplicable.
        """
        with caplog.at_level("WARNING"):
            convert.prepare_checkpoint(
                write_olmo_core(tmp_path), tmp_path / "out", policy="none", dtype="float16"
            )
        assert "converts nothing under --checkpoint-prep none" in caplog.text
        assert spy == []

    def test_the_warning_does_not_claim_the_flag_is_dead(self, tmp_path: Path, spy, caplog) -> None:
        """The native backend honours the precision, so "no effect" is now false.

        ``_OlmoCoreScoringModel`` passes ``InferenceConfig.dtype`` to
        ``from_checkpoint``, and the runner fills that from this same flag. A warning
        asserting the opposite would be the doc contradicting the behaviour, which is
        the state this replaced -- the flag reached ``prepare_checkpoint`` and stopped,
        and a native run recorded bfloat16 while scoring in float32.
        """
        with caplog.at_level("WARNING"):
            convert.prepare_checkpoint(
                write_olmo_core(tmp_path), tmp_path / "out", policy="none", dtype="float16"
            )
        assert "no effect" not in caplog.text
        assert "olmo_core" in caplog.text, "the backend that does honour it must be named"

    def test_none_is_quiet_at_the_default(self, tmp_path: Path, spy, caplog) -> None:
        """The default is what a caller gets for not asking, so warning about it would
        fire on every ``none`` run and tell nobody anything."""
        with caplog.at_level("WARNING"):
            convert.prepare_checkpoint(write_olmo_core(tmp_path), tmp_path / "out", policy="none")
        assert "converts nothing under" not in caplog.text

    def test_the_default_is_named_once(self) -> None:
        """``DTYPE_DEFAULT`` is what the runner's default, the warning's threshold and
        the three signatures all read, so a change lands in one place or not at all."""
        assert convert.DTYPE_DEFAULT == "bfloat16"
        assert convert.CONVERSION_DTYPES[0] == convert.DTYPE_DEFAULT


class TestTheConversionBodyIsReachableWithoutTorch:
    """The body needs torch and ai2-olmo-core; nothing else in the module may.

    This split is the reason the ladder above is testable at all. If the heavy imports sat
    at module scope, every test in this file would need a CUDA-pinned dependency tree to
    ask a question about a directory listing.
    """

    def test_importing_the_module_pulls_in_nothing_heavy(self) -> None:
        importlib.reload(convert)
        if not HAS_TORCH:
            assert "torch" not in sys.modules
        assert "olmo_core" not in sys.modules

    @needs_no_torch
    def test_calling_it_without_the_extra_says_which_extra(self, tmp_path: Path) -> None:
        """A legible RuntimeError, not an ImportError from three frames down.

        The message has to name the way out, because there are three and they are not
        interchangeable: install the extra, pre-convert elsewhere, or turn preparation off
        and use a backend that reads the native format.
        """
        d = write_olmo_core(tmp_path)
        with pytest.raises(RuntimeError, match="ai2-olmo-core") as excinfo:
            convert.convert_olmo_core_to_hf(d, tmp_path / "out")
        message = str(excinfo.value)
        assert "--extra olmo_core" in message
        assert "--checkpoint-prep none" in message

    @needs_no_torch
    def test_the_dependency_check_precedes_any_filesystem_work(self, tmp_path: Path) -> None:
        """Nothing is created before the imports resolve.

        Ordering worth pinning: a run that cannot convert should leave no half-written
        output directory behind for the next attempt to trip over.
        """
        out = tmp_path / "out"
        with pytest.raises(RuntimeError):
            convert.convert_olmo_core_to_hf(write_olmo_core(tmp_path), out)
        assert not out.exists()
