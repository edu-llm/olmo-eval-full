"""The ``uni_frq`` CAT style: unidimensional (2PL) free-response adaptive testing.

Runs a Fisher-information CAT over TutorEval's calibrated unidimensional item bank
(1,186 criteria fitted with 2PL MML-EM) and reports a scalar ability ``theta`` with
its posterior standard error. FRQ grading is two-stage (the checkpoint under test
generates one response per scenario; a judge grades each criterion pass/fail). This style
supplies only the genuinely style-specific pieces: the IRT model (2PL EAP), the item
selector (max Fisher information), the stopping rule, and its graduated bank + judge
selection.

Two engines drive it. The shared ``common/cat_loop.py`` (via ``frq_cat.runner``) is the
frozen scaffolding; ``styles/uni_frq/session.py`` (via ``run_uni_frq``) is the hardened
path used on AWS, which skips unservable scenarios and treats an unscorable verdict as
missing data. The style behaves identically under both.

Numerics are pure-Python over a small fixed ``theta`` grid, so the style imports
without heavy dependencies and its unit tests run without a GPU or extra packages.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from ...base import (
    AbilityEstimate,
    CATReport,
    CATState,
    CriterionResponse,
    Discrimination,
    FrqBank,
    FrqCatStyle,
    IRTBank,
    IRTItemParams,
)
from ...common import bank_loader, irt_params, s3_io
from ...common.judge import PROMPT_VERSION as _RUNTIME_PROMPT_VERSION

#: The prompt the bank was actually fitted under, per FLOW_PACKAGE.md. If the shared
#: client's prompt differs, theta is not on the calibrated scale and the report must say
#: so rather than let the judge YAML's provenance imply otherwise.
_CALIBRATION_PROMPT_VERSION = "judge-validation-v3"

log = logging.getLogger("uni_frq.style")

_HERE = Path(__file__).resolve().parent
_BANK_DIR = _HERE / "bank"
_SCENARIOS = _BANK_DIR / "scenarios.jsonl"
_PARAMS = _BANK_DIR / "params.jsonl"
_CONFIG = _HERE / "config.yaml"

#: EAP / Fisher grid over the latent ability axis. The recovery study in ``evidence/``
#: used 61 nodes; this grid is wider and finer (0.1 spacing) because a [-4, 4] grid
#: censors extreme checkpoints: a 40-item all-pass run reported theta=3.996 with
#: SE=0.024 (pinned at the edge) where the true posterior is theta=5.13, SE=0.32.
#: ``estimate_ability`` flags a result whose posterior still leans on the boundary.
_GRID_MIN, _GRID_MAX, _GRID_N = -8.0, 8.0, 161

#: Posterior mass beyond this |theta| means the estimate is boundary-limited.
_BOUNDARY_AT = 0.9 * _GRID_MAX
_BOUNDARY_MASS_LIMIT = 0.01


def _disc_scalar(discrimination: Discrimination) -> float:
    """Return the scalar discrimination for a unidimensional item.

    The frozen loader represents skill-keyed discrimination (``{"ability": a}``) as a
    length-1 tuple; accept either that or a bare scalar.
    """
    if isinstance(discrimination, tuple):
        return float(discrimination[0]) if discrimination else 1.0
    return float(discrimination)


def _prob_correct(theta: float, params: IRTItemParams) -> float:
    """2PL/3PL success probability P(pass | theta) for one criterion."""
    a = _disc_scalar(params.discrimination)
    b = params.difficulty
    c = params.guessing
    z = a * (theta - b)
    if z >= 0.0:  # numerically stable logistic
        p = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        p = ez / (1.0 + ez)
    return c + (1.0 - c) * p


def _fisher_information(theta: float, params: IRTItemParams) -> float:
    """Item information at ``theta``.

    Reduces to ``a^2 p(1-p)`` for a 2PL item, which is every item in the shipped bank.
    The general form matters only if a 3PL bank is ever loaded, where ``a^2 p(1-p)``
    overstates information for easy items by a large factor.
    """
    a = _disc_scalar(params.discrimination)
    p = _prob_correct(theta, params)
    if p <= 0.0 or p >= 1.0:
        return 0.0  # a saturated item carries no information (and guards the division)
    c = params.guessing
    if c <= 0.0:
        return a * a * p * (1.0 - p)
    ratio = (p - c) / (1.0 - c)
    return a * a * ratio * ratio * (1.0 - p) / p


def _softplus(x: float) -> float:
    """``log(1 + exp(x))`` without overflow."""
    return x + math.log1p(math.exp(-x)) if x > 0.0 else math.log1p(math.exp(x))


def _log_prob_pair(theta: float, params: IRTItemParams) -> tuple[float, float]:
    """Return ``(log P(pass), log P(fail))``, computed without cancellation.

    Taking ``log(1 - p)`` from a probability loses everything once ``p`` rounds to 1.0,
    which happens for real items in this bank: ``te_0173_c01`` (a=4.67, b=-11.98)
    saturates across the whole grid, so a single FAIL on it produced a likelihood of
    exactly zero and the estimate silently collapsed back to the prior. Deriving both
    tails straight from ``z`` keeps the failure branch finite.
    """
    a = _disc_scalar(params.discrimination)
    z = a * (theta - params.difficulty)
    c = params.guessing
    log_sig = -_softplus(-z)  # log P(pass) for a 2PL item
    log_one_minus_sig = -_softplus(z)  # log P(fail) for a 2PL item
    if c <= 0.0:
        return log_sig, log_one_minus_sig
    # 3PL: P = c + (1-c)sigmoid(z), Q = (1-c)(1 - sigmoid(z))
    return math.log(c + (1.0 - c) * math.exp(log_sig)), math.log1p(-c) + log_one_minus_sig


class UniFrqStyle(FrqCatStyle):
    """Unidimensional 2PL FRQ CAT style (registered as ``uni_frq``)."""

    name: ClassVar[str] = "uni_frq"

    def __init__(self) -> None:
        self._config = self._load_config()
        self._min_items = int(self._config.get("min_items", 8))
        self._se_threshold_default = float(self._config.get("se_threshold", 0.3))
        step = (_GRID_MAX - _GRID_MIN) / (_GRID_N - 1)
        self._grid: tuple[float, ...] = tuple(_GRID_MIN + step * i for i in range(_GRID_N))
        # Unnormalized log standard-normal prior; the constant cancels on normalization
        # and log space is what stops a long test from underflowing the likelihood.
        self._log_prior: tuple[float, ...] = tuple(-0.5 * t * t for t in self._grid)
        self._provenance: dict[str, Any] | None = None

    @property
    def config(self) -> dict[str, Any]:
        """The vendored operating point (fit family, stop policy, judge selection)."""
        return dict(self._config)

    def bank_provenance(self) -> dict[str, Any]:
        """Identify the bank this style scores against, so a report stands alone.

        Hashes are what make a theta re-checkable later: without them a report names a
        benchmark but not the item parameters it was actually computed from.
        """
        if self._provenance is None:
            with _PARAMS.open(encoding="utf-8") as handle:
                fitted = json.loads(handle.readline()).get("irt_params", {})
            self._provenance = {
                "benchmark": str(self._config.get("benchmark", "tutoreval")),
                "criteria": sum(1 for _ in _PARAMS.open(encoding="utf-8")),
                "scenarios": sum(1 for _ in _SCENARIOS.open(encoding="utf-8")),
                "params_sha256": bank_loader.bank_sha256(_PARAMS),
                "scenarios_sha256": bank_loader.bank_sha256(_SCENARIOS),
                "fit_method": fitted.get("method"),
                "n_persons": fitted.get("n_persons"),
                "flags": list(fitted.get("flags", [])),
            }
        return dict(self._provenance)

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load the vendored ``config.yaml`` (yaml imported lazily)."""
        import yaml

        return yaml.safe_load(_CONFIG.read_text(encoding="utf-8")) or {}

    def download_bank(self, benchmark: str, *, dest: Path | None = None) -> FrqBank:
        """Load the graduated scenarios + criteria into an :class:`FrqBank`.

        The bank ships inside the style package, so ``dest`` (a scratch dir the runner
        offers) is unused; the criteria are the IRT items.
        """
        name = benchmark or str(self._config.get("benchmark", "tutoreval"))
        return bank_loader.load_bank(name, _SCENARIOS, _PARAMS)

    def load_irt_params(self, source: str | Path) -> IRTBank:
        """Load 2PL parameters from an explicit source, else the bundled bank.

        A caller-supplied path that does not resolve raises instead of silently falling
        back: a typo in ``--irt-params`` must not score a checkpoint against a different
        bank than the operator asked for. The runner passes the benchmark *name* when no
        source is given, so a bare name (no separator, no suffix) still selects the
        bundled bank.
        """
        src = str(source).strip() if source else ""
        if not src:
            return irt_params.load_irt_params(_PARAMS)
        if s3_io.is_s3_uri(src):
            return irt_params.load_irt_params(src)
        candidate = Path(src)
        if candidate.exists():
            return irt_params.load_irt_params(candidate)
        looks_like_path = candidate.is_absolute() or "/" in src or src.endswith((".json", ".jsonl"))
        if looks_like_path:
            raise FileNotFoundError(f"IRT parameter source not found: {src}")
        return irt_params.load_irt_params(_PARAMS)

    def estimate_ability(
        self,
        bank: IRTBank,
        responses: Sequence[CriterionResponse],
        *,
        previous: AbilityEstimate | None = None,
    ) -> AbilityEstimate:
        """Unidimensional 2PL EAP: posterior mean of ``theta`` and its SD (the SE).

        Accumulated in log space and normalized with log-sum-exp, so a long test or a
        saturated item cannot underflow the likelihood to zero. A missing fitted item is
        an error rather than a silent skip: scoring against a bank that does not contain
        an administered criterion would quietly understate the evidence.
        """
        log_post = list(self._log_prior)
        for response in responses:
            params = bank.get(response.criterion_id)  # KeyError = bank/response mismatch
            for i, theta in enumerate(self._grid):
                log_p, log_q = _log_prob_pair(theta, params)
                log_post[i] += log_p if response.correct else log_q

        peak = max(log_post)
        weights = [math.exp(lp - peak) for lp in log_post]
        total = math.fsum(weights)
        mean = math.fsum(t * w for t, w in zip(self._grid, weights, strict=True)) / total
        variance = (
            math.fsum(w * (t - mean) ** 2 for t, w in zip(self._grid, weights, strict=True)) / total
        )
        se = math.sqrt(max(variance, 0.0))

        boundary_mass = (
            math.fsum(w for t, w in zip(self._grid, weights, strict=True) if abs(t) >= _BOUNDARY_AT)
            / total
        )
        metadata: dict[str, Any] = {
            "estimator": "2pl-eap-log",
            "grid_nodes": _GRID_N,
            "grid_range": [_GRID_MIN, _GRID_MAX],
            "n_responses": len(responses),
            "boundary_mass": round(boundary_mass, 6),
        }
        if boundary_mass > _BOUNDARY_MASS_LIMIT:
            # theta is at the edge of the scale: the point estimate and SE are censored.
            metadata["boundary_limited"] = True
            log.warning(
                "ability is boundary-limited (%.1f%% of posterior mass beyond |theta|>=%.1f); "
                "theta=%.3f se=%.3f understates the true uncertainty",
                100.0 * boundary_mass,
                _BOUNDARY_AT,
                mean,
                se,
            )
        return AbilityEstimate(theta=mean, standard_error=se, metadata=metadata)

    def select_next_item(self, bank: IRTBank, state: CATState) -> str | None:
        """Pick the un-administered criterion with max Fisher information at current theta."""
        theta = 0.0
        if state.ability is not None and isinstance(state.ability.theta, int | float):
            theta = float(state.ability.theta)

        administered = state.administered_ids
        best_id: str | None = None
        best_info = -1.0
        for item_id, params in bank.params.items():
            if item_id in administered:
                continue
            info = _fisher_information(theta, params)
            if info > best_info:
                best_info = info
                best_id = item_id
        return best_id

    def stopping_rule(self, state: CATState) -> bool:
        """Stop when SE(theta) < threshold, after a minimum-item floor."""
        if state.step < self._min_items:
            return False
        threshold = (
            state.se_threshold if state.se_threshold is not None else self._se_threshold_default
        )
        if threshold is None or state.ability is None:
            return False
        se = state.ability.standard_error
        return isinstance(se, int | float) and se < threshold

    def report(self, state: CATState) -> CATReport:
        """Summarize the session into a serializable :class:`CATReport`."""
        ability = state.ability or AbilityEstimate(
            theta=0.0,
            standard_error=float("inf"),
            metadata={"estimator": "2pl-eap", "note": "no items administered"},
        )
        threshold = (
            state.se_threshold if state.se_threshold is not None else self._se_threshold_default
        )
        provenance = self.bank_provenance()
        pinned_max = self._config.get("max_items")
        # Field names mirror the MCQ report (modality / stop_reason / ungradable /
        # selected_item_ids / cat_settings / bank_provenance) so one reader can consume
        # either flow. MCQ-only keys (prompt_style, score_normalization, pirt_accuracy)
        # are deliberately absent: they describe choice scoring, which FRQ does not do.
        metadata: dict[str, Any] = {
            "modality": "frq",
            "fit_family": str(self._config.get("fit_family", "2pl")),
            "dimensions": 1,
            "stop_reason": str(state.metadata.get("stop_reason", "unknown")),
            "selected_item_ids": [response.criterion_id for response in state.administered],
            "cat_settings": {
                "se_threshold": threshold,
                "min_items": self._min_items,
                "max_items": state.max_items,
                "max_items_pinned_by_style": pinned_max,
                "max_items_is_pinned_value": state.max_items == pinned_max,
                "grid": {"min": _GRID_MIN, "max": _GRID_MAX, "n": _GRID_N},
            },
            "bank_provenance": provenance,
            "judge_prompt": {
                "runtime": _RUNTIME_PROMPT_VERSION,
                "calibration": _CALIBRATION_PROMPT_VERSION,
                "matches_calibration": _RUNTIME_PROMPT_VERSION == _CALIBRATION_PROMPT_VERSION,
            },
            "scoring_note": (
                "Criteria are graded pass/fail by an LLM judge, so theta is comparable only "
                "across runs sharing the same judge and prompt."
                if _RUNTIME_PROMPT_VERSION == _CALIBRATION_PROMPT_VERSION
                else (
                    f"UNCALIBRATED: the bank was fitted under prompt "
                    f"{_CALIBRATION_PROMPT_VERSION} but this run graded with "
                    f"{_RUNTIME_PROMPT_VERSION}, which has no evidence gate. Theta is an "
                    f"ordinal score comparable only to other runs using this same prompt; "
                    f"it is not on the TutorEval calibrated scale."
                )
            ),
        }
        if "bank_size" in state.metadata:
            metadata["bank_size"] = state.metadata["bank_size"]
        if "ungradable" in state.metadata:
            # How much of the attempted test could not be scored at all. Without this a
            # theta from 8 scored criteria reads the same as one from 8 out of 40.
            metadata["ungradable"] = state.metadata["ungradable"]
        if "low_n" in provenance.get("flags", []):
            metadata["bank_caveat"] = (
                f"calibrated on n_persons={provenance.get('n_persons')}, below the ~150 "
                f"identifiability floor; treat abilities as coarse"
            )
        return CATReport(
            cat_style=self.name,
            benchmark=state.benchmark,
            ability=ability,
            num_items_administered=state.step,
            responses=tuple(state.administered),
            metadata=metadata,
        )
