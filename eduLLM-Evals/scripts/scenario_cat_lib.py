"""Benchmark-agnostic offline CAT replay and ability estimators.

This module supplies the post-calibration primitives that are intentionally absent from
the benchmark-pruned branches.  It runs the real :mod:`tutor_cat.engine` over an already
graded model-by-criterion response matrix; it never calls a tutor model or an LLM judge.

The public pieces are:

``load_fitted_bank``
    Load a fitted-only rubric bank with any number of named latent dimensions.
``build_quadrature`` / ``batch_eap``
    Gauss-Hermite batch EAP using the fitted latent correlation matrix.
``mwle``
    Multidimensional Warm/Firth weighted-likelihood estimation.
``engine_bank_for_row`` / ``run_recorded_model``
    Build a per-model scenario bank without converting missing judgments to failures and
    replay CAT or the seeded-random baseline through the production engine.

The normal ``tutor_cat.dataio.load_bank`` path still uses the package's historical
three-skill schema.  This module deliberately constructs ``Rubric`` arrays directly so a
five-dimensional InFoBench bank can use the same engine without repurposing skill names.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor_cat.dataio import ItemBank  # noqa: E402
from tutor_cat.engine import RunConfig, run_evaluation  # noqa: E402
from tutor_cat.schemas import JudgeVerdict, Rubric, Scenario  # noqa: E402


class OfflineStudyError(RuntimeError):
    """Raised when a fitted bank or recorded response study is not internally valid."""


@dataclass
class FittedBank:
    """Numeric, fitted-only bank in one explicit latent-dimension order."""

    records: list[dict[str, Any]]
    dims: tuple[str, ...]
    criterion_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    Q: np.ndarray
    A: np.ndarray
    b: np.ndarray
    latent_correlation: np.ndarray
    source_path: str = ""
    dropped_negative_items: tuple[str, ...] = ()
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {cid: i for i, cid in enumerate(self.criterion_ids)}

    @property
    def n_dims(self) -> int:
        return len(self.dims)

    @property
    def n_items(self) -> int:
        return len(self.criterion_ids)

    @property
    def index(self) -> Mapping[str, int]:
        return self._index

    def scenario_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for j, sid in enumerate(self.scenario_ids):
            groups.setdefault(sid, []).append(j)
        return groups


@dataclass(frozen=True)
class Quadrature:
    """Tensor-product Gauss-Hermite grid and normalized correlated-prior weights."""

    grid: np.ndarray
    log_prior: np.ndarray
    correlation: np.ndarray
    nodes_per_dim: int


@dataclass
class AbilityEstimate:
    theta: np.ndarray
    se: np.ndarray
    covariance: np.ndarray
    method: str
    n_items: int
    converged: bool
    message: str = ""
    log_marginal: float | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise OfflineStudyError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _validate_correlation(value: np.ndarray, n_dims: int) -> np.ndarray:
    corr = np.asarray(value, dtype=float)
    if corr.shape != (n_dims, n_dims):
        raise OfflineStudyError(
            f"latent correlation shape {corr.shape} != ({n_dims}, {n_dims})"
        )
    if not np.all(np.isfinite(corr)) or not np.allclose(corr, corr.T, atol=1e-9):
        raise OfflineStudyError("latent correlation must be finite and symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-6):
        raise OfflineStudyError("latent correlation must have a unit diagonal")
    if float(np.linalg.eigvalsh(corr).min()) <= 1e-10:
        raise OfflineStudyError("latent correlation must be positive definite")
    return corr


def _inferred_correlation(records: Sequence[dict[str, Any]], n_dims: int) -> np.ndarray:
    found: list[np.ndarray] = []
    for rec in records:
        irt = rec.get("irt_params") or {}
        candidate = irt.get("latent_correlation")
        if candidate is None:
            candidate = (irt.get("provenance") or {}).get("latent_correlation")
        if candidate is not None:
            found.append(_validate_correlation(np.asarray(candidate, dtype=float), n_dims))
    if not found:
        return np.eye(n_dims)
    first = found[0]
    if any(not np.allclose(first, other, atol=1e-8) for other in found[1:]):
        raise OfflineStudyError("fitted-bank records disagree on latent_correlation")
    return first


def _looks_calibrated(rec: Mapping[str, Any]) -> bool:
    irt = rec.get("irt_params") or {}
    source = str(irt.get("source") or "").lower()
    return bool(irt.get("calibrated")) or source.startswith("calibrated")


def load_fitted_bank(
    path: str | Path,
    *,
    skills: Sequence[str] | None = None,
    negative_policy: str = "error",
    latent_correlation: np.ndarray | None = None,
    require_calibrated: bool = True,
) -> FittedBank:
    """Load and strictly validate a fitted rubric bank.

    ``negative_policy`` is deliberately explicit: ``error`` (default), ``drop`` the
    entire affected criterion, or ``keep`` it.  There is no silent clamping because that
    changes the fitted model.  ``q_modeled`` is preferred; ``q_mapping`` is accepted when
    its keys already match the modeled dimensions.
    """

    if negative_policy not in {"error", "drop", "keep"}:
        raise ValueError("negative_policy must be 'error', 'drop', or 'keep'")
    bank_path = Path(path)
    records = _read_jsonl(bank_path)
    if not records:
        raise OfflineStudyError(f"no records in fitted bank: {bank_path}")

    if skills is not None:
        dims = tuple(str(s).strip() for s in skills if str(s).strip())
    else:
        irt_skills = (records[0].get("irt_params") or {}).get("skills_order")
        dims = tuple(irt_skills or (records[0].get("discrimination") or {}).keys())
    if not dims or len(set(dims)) != len(dims):
        raise OfflineStudyError(f"invalid modeled skill order: {dims}")

    kept: list[dict[str, Any]] = []
    negative_ids: list[str] = []
    seen: set[str] = set()
    for rec in records:
        cid = str(rec.get("criterion_id") or "")
        if not cid or cid in seen:
            raise OfflineStudyError(f"blank or duplicate criterion_id: {cid!r}")
        seen.add(cid)
        if not rec.get("scenario_id"):
            raise OfflineStudyError(f"{cid}: missing scenario_id")
        if require_calibrated and not _looks_calibrated(rec):
            raise OfflineStudyError(
                f"{cid}: bank is not marked calibrated; refusing to mix synthetic or "
                "unverified parameters"
            )
        disc = rec.get("discrimination") or {}
        qmap = rec.get("q_modeled") or rec.get("q_mapping") or {}
        missing_a = [d for d in dims if d not in disc]
        missing_q = [d for d in dims if d not in qmap]
        if missing_a or missing_q:
            raise OfflineStudyError(
                f"{cid}: missing modeled keys; discrimination={missing_a}, q={missing_q}"
            )
        try:
            avec = np.asarray([float(disc[d]) for d in dims], dtype=float)
            qvec = np.asarray([int(qmap[d]) for d in dims], dtype=int)
            difficulty = float(rec["difficulty"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OfflineStudyError(f"{cid}: invalid fitted parameters: {exc}") from exc
        if not np.all(np.isfinite(avec)) or not np.isfinite(difficulty):
            raise OfflineStudyError(f"{cid}: non-finite fitted parameters")
        if not set(qvec.tolist()) <= {0, 1} or int(qvec.sum()) == 0:
            raise OfflineStudyError(f"{cid}: q row must be binary and map to >=1 skill")
        if np.any(avec[qvec == 1] <= 0):
            negative_ids.append(cid)
            if negative_policy == "drop":
                continue
        kept.append(rec)

    if negative_ids and negative_policy == "error":
        raise OfflineStudyError(
            f"{len(negative_ids)} fitted item(s) have nonpositive modeled "
            f"discrimination; first: {negative_ids[:10]}. Choose an explicit policy."
        )
    if not kept:
        raise OfflineStudyError("no fitted criteria remain after applying bank policy")

    ids = tuple(str(rec["criterion_id"]) for rec in kept)
    sids = tuple(str(rec["scenario_id"]) for rec in kept)
    Q = np.asarray(
        [[int((rec.get("q_modeled") or rec["q_mapping"])[d]) for d in dims] for rec in kept],
        dtype=int,
    )
    raw_A = np.asarray(
        [[float(rec["discrimination"][d]) for d in dims] for rec in kept], dtype=float
    )
    # Store the effective loading used by both EAP/MWLE and the engine.
    A = Q * raw_A
    b = np.asarray([float(rec["difficulty"]) for rec in kept], dtype=float)
    corr = (
        _validate_correlation(np.asarray(latent_correlation, dtype=float), len(dims))
        if latent_correlation is not None
        else _inferred_correlation(kept, len(dims))
    )
    return FittedBank(
        records=kept,
        dims=dims,
        criterion_ids=ids,
        scenario_ids=sids,
        Q=Q,
        A=A,
        b=b,
        latent_correlation=corr,
        source_path=str(bank_path),
        dropped_negative_items=tuple(negative_ids if negative_policy == "drop" else ()),
    )


def load_response_matrix(path: str | Path) -> pd.DataFrame:
    """Load a model-by-criterion matrix and reject anything except 0/1/missing."""

    matrix = pd.read_csv(path, index_col=0)
    if matrix.empty or matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise OfflineStudyError("response matrix is empty or has duplicate rows/columns")
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    # ``to_numeric`` must not turn a non-empty bad token into a missing value.
    raw = matrix.astype(str)
    bad_coercions = numeric.isna() & matrix.notna() & ~raw.apply(
        lambda col: col.str.strip().isin({"", "nan", "NaN", "NA", "N/A", "null", "None"})
    )
    if bool(bad_coercions.to_numpy().any()):
        r, c = np.argwhere(bad_coercions.to_numpy())[0]
        raise OfflineStudyError(
            f"non-binary response at model={matrix.index[r]!r}, "
            f"criterion={matrix.columns[c]!r}: {matrix.iat[r, c]!r}"
        )
    finite = numeric.to_numpy(dtype=float)
    observed = finite[np.isfinite(finite)]
    if not set(np.unique(observed).tolist()) <= {0.0, 1.0}:
        raise OfflineStudyError("response matrix contains observed values outside {0,1}")
    numeric.index = matrix.index.astype(str)
    numeric.index.name = "model"
    return numeric


def responses_for_bank(row: pd.Series, bank: FittedBank) -> np.ndarray:
    """Align one matrix row to the fitted bank; absent columns remain NaN."""

    return pd.to_numeric(row.reindex(bank.criterion_ids), errors="coerce").to_numpy(float)


def build_quadrature(
    n_dims: int,
    nodes_per_dim: int,
    correlation: np.ndarray | None = None,
    *,
    max_nodes: int = 50_000,
) -> Quadrature:
    """Build a correlated-normal Gauss-Hermite quadrature rule.

    The tensor grid grows as ``nodes_per_dim ** n_dims``.  The explicit guard prevents
    accidentally requesting historical two-dimensional defaults such as ``61**5``.
    """

    if n_dims < 1 or nodes_per_dim < 2:
        raise ValueError("n_dims must be >=1 and nodes_per_dim must be >=2")
    total = nodes_per_dim**n_dims
    if total > max_nodes:
        raise OfflineStudyError(
            f"quadrature request {nodes_per_dim}^{n_dims}={total:,} exceeds "
            f"max_nodes={max_nodes:,}"
        )
    corr = _validate_correlation(
        np.eye(n_dims) if correlation is None else np.asarray(correlation, dtype=float),
        n_dims,
    )
    nodes, weights = np.polynomial.hermite.hermgauss(nodes_per_dim)
    axis = math.sqrt(2.0) * nodes
    one_logw = np.log(weights) - 0.5 * math.log(math.pi)
    meshes = np.meshgrid(*([axis] * n_dims), indexing="ij")
    grid = np.stack([m.reshape(-1) for m in meshes], axis=1)
    wmeshes = np.meshgrid(*([one_logw] * n_dims), indexing="ij")
    logw = np.sum(np.stack([m.reshape(-1) for m in wmeshes], axis=1), axis=1)

    if not np.allclose(corr, np.eye(n_dims)):
        inv = np.linalg.inv(corr)
        sign, logdet = np.linalg.slogdet(corr)
        if sign <= 0:
            raise OfflineStudyError("latent correlation determinant is not positive")
        ratio_quad = np.einsum("gi,ij,gj->g", grid, inv - np.eye(n_dims), grid)
        # Importance reweighting: correlated MVN density / independent MVN density.
        logw = logw - 0.5 * ratio_quad - 0.5 * logdet
    logw = logw - logsumexp(logw)
    return Quadrature(grid=grid, log_prior=logw, correlation=corr, nodes_per_dim=nodes_per_dim)


def _observed_subset(
    responses: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    item_indices: Sequence[int] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(responses, dtype=float)
    if values.ndim != 1 or values.shape[0] != A.shape[0]:
        raise ValueError("responses must be a vector aligned to all rows of A")
    idx = (
        np.arange(A.shape[0], dtype=int)
        if item_indices is None
        else np.asarray(item_indices, dtype=int)
    )
    if idx.ndim != 1 or (idx.size and (idx.min() < 0 or idx.max() >= A.shape[0])):
        raise ValueError("item_indices are outside the fitted bank")
    keep = np.isfinite(values[idx])
    idx = idx[keep]
    y = values[idx]
    if not set(np.unique(y).tolist()) <= {0.0, 1.0}:
        raise ValueError("observed responses must be binary")
    return idx, y, A[idx], b[idx]


def batch_eap(
    responses: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    quadrature: Quadrature,
    item_indices: Sequence[int] | np.ndarray | None = None,
    *,
    node_chunk: int = 4096,
) -> AbilityEstimate:
    """Order-invariant posterior mean/covariance over observed administered items."""

    idx, y, Ai, bi = _observed_subset(responses, A, b, item_indices)
    n_nodes = quadrature.grid.shape[0]
    ll = np.zeros(n_nodes, dtype=float)
    for start in range(0, n_nodes, node_chunk):
        stop = min(start + node_chunk, n_nodes)
        eta = Ai @ quadrature.grid[start:stop].T - bi[:, None]
        ll[start:stop] = y @ log_expit(eta) + (1.0 - y) @ log_expit(-eta)
    joint = ll + quadrature.log_prior
    log_marginal = float(logsumexp(joint))
    post = np.exp(joint - log_marginal)
    theta = post @ quadrature.grid
    centered = quadrature.grid - theta
    covariance = (centered * post[:, None]).T @ centered
    covariance = (covariance + covariance.T) / 2.0
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return AbilityEstimate(
        theta=theta,
        se=se,
        covariance=covariance,
        method="batch_eap",
        n_items=int(idx.size),
        converged=True,
        log_marginal=log_marginal,
    )


def _mwle_neg_obj_grad(
    theta: np.ndarray, A: np.ndarray, b: np.ndarray, y: np.ndarray, ridge: float
) -> tuple[float, np.ndarray]:
    """Negative Warm/Firth objective and analytic gradient."""

    n_dims = theta.shape[0]
    eta = A @ theta - b
    p = expit(eta)
    w = p * (1.0 - p)
    info = (A * w[:, None]).T @ A + ridge * np.eye(n_dims)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e100, np.zeros(n_dims)
    loglik = float(y @ log_expit(eta) + (1.0 - y) @ log_expit(-eta))
    objective = loglik + 0.5 * logdet
    inv_info = np.linalg.inv(info)
    c = w * (1.0 - 2.0 * p)
    leverage = np.einsum("ij,jk,ik->i", A, inv_info, A)
    gradient = A.T @ (y - p) + 0.5 * (A * (c * leverage)[:, None]).sum(axis=0)
    return -objective, -gradient


def mwle(
    responses: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    item_indices: Sequence[int] | np.ndarray | None = None,
    *,
    theta0: np.ndarray | None = None,
    ridge: float = 1e-6,
    bound: float = 12.0,
) -> AbilityEstimate:
    """Estimate multidimensional ability with a Warm/Firth likelihood adjustment.

    A rank-deficient administered set returns ``converged=False`` and an explicit
    message.  Callers may display the supplied start value, but must not mistake it for a
    successful MWLE estimate.
    """

    idx, y, Ai, bi = _observed_subset(responses, A, b, item_indices)
    n_dims = A.shape[1]
    start = np.zeros(n_dims) if theta0 is None else np.asarray(theta0, dtype=float)
    if start.shape != (n_dims,):
        raise ValueError(f"theta0 shape {start.shape} != ({n_dims},)")
    rank = int(np.linalg.matrix_rank(Ai)) if Ai.size else 0
    if idx.size < n_dims or rank < n_dims:
        return AbilityEstimate(
            theta=start.copy(),
            se=np.full(n_dims, np.nan),
            covariance=np.full((n_dims, n_dims), np.nan),
            method="mwle",
            n_items=int(idx.size),
            converged=False,
            message=f"administered loading matrix is rank {rank}/{n_dims}",
        )
    try:
        result = minimize(
            _mwle_neg_obj_grad,
            start,
            args=(Ai, bi, y, ridge),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-bound, bound)] * n_dims,
        )
    except Exception as exc:  # pragma: no cover - scipy failures are environment-specific
        return AbilityEstimate(
            theta=start.copy(),
            se=np.full(n_dims, np.nan),
            covariance=np.full((n_dims, n_dims), np.nan),
            method="mwle",
            n_items=int(idx.size),
            converged=False,
            message=f"optimizer raised {type(exc).__name__}: {exc}",
        )
    theta = np.asarray(result.x, dtype=float)
    p = expit(Ai @ theta - bi)
    info = (Ai * (p * (1.0 - p))[:, None]).T @ Ai + ridge * np.eye(n_dims)
    covariance = np.linalg.pinv(info)
    covariance = (covariance + covariance.T) / 2.0
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    ok = bool(result.success and np.all(np.isfinite(theta)))
    return AbilityEstimate(
        theta=theta if np.all(np.isfinite(theta)) else start.copy(),
        se=se,
        covariance=covariance,
        method="mwle",
        n_items=int(idx.size),
        converged=ok,
        message="" if ok else str(result.message),
    )


def pass_rate_check(
    responses: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    theta: np.ndarray,
    item_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compare observed and p-IRT-predicted mean pass rate on fitted items.

    Only observed binary cells contribute.  The predicted rate is
    ``mean(sigmoid(A_j @ theta - b_j))`` over that exact same item set, so missing
    judgments can never become implicit failures or alter the denominator.
    """

    idx, y, Ai, bi = _observed_subset(responses, A, b, item_indices)
    ability = np.asarray(theta, dtype=float)
    if ability.shape != (A.shape[1],):
        raise ValueError(f"theta shape {ability.shape} != ({A.shape[1]},)")
    if idx.size == 0:
        return {
            "n_items": 0,
            "observed_pass_rate": float("nan"),
            "predicted_pass_rate": float("nan"),
            "error": float("nan"),
        }
    observed = float(y.mean())
    predicted = (
        float(expit(Ai @ ability - bi).mean())
        if np.all(np.isfinite(ability))
        else float("nan")
    )
    return {
        "n_items": int(idx.size),
        "observed_pass_rate": observed,
        "predicted_pass_rate": predicted,
        "error": predicted - observed,
    }


def load_scenario_records(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(Path(path))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("scenario_id") or "")
        if not sid or sid in out:
            raise OfflineStudyError(f"blank or duplicate scenario_id: {sid!r}")
        out[sid] = row
    return out


def engine_bank_for_row(
    fitted: FittedBank,
    scenario_records: Mapping[str, dict[str, Any]],
    row: pd.Series,
) -> ItemBank:
    """Build the exact per-model bank, dropping missing cells instead of imputing them."""

    responses = responses_for_bank(row, fitted)
    present = np.isfinite(responses)
    rubrics: dict[str, Rubric] = {}
    scenario_to_ids: dict[str, list[str]] = {}
    for j, rec in enumerate(fitted.records):
        if not present[j]:
            continue
        cid = fitted.criterion_ids[j]
        sid = fitted.scenario_ids[j]
        rubrics[cid] = Rubric(
            criterion_id=cid,
            scenario_id=sid,
            criterion=str(rec.get("criterion") or cid),
            q=fitted.Q[j].copy(),
            a=fitted.A[j].copy(),
            b=float(fitted.b[j]),
            primary_skill=str(rec.get("primary_skill") or ""),
            scoring_type="binary",
            criticality=str(rec.get("criticality") or "standard"),
            calibration_version=str((rec.get("irt_params") or {}).get("source") or ""),
            status=str(rec.get("status") or "approved"),
        )
        scenario_to_ids.setdefault(sid, []).append(cid)

    scenarios: dict[str, Scenario] = {}
    for sid, cids in scenario_to_ids.items():
        if sid not in scenario_records:
            raise OfflineStudyError(f"fitted bank references missing scenario {sid}")
        scenario = Scenario.from_json(dict(scenario_records[sid]))
        scenario.criterion_ids = sorted(cids)
        scenarios[sid] = scenario
    if not scenarios:
        raise OfflineStudyError(f"model {row.name!r} has no observed fitted criteria")
    return ItemBank(
        scenarios=scenarios,
        rubrics=rubrics,
        skills=fitted.dims,
    )


class RecordedTutor:
    def __init__(self, model: str):
        self.model = model
        self.name = model.replace("/", "_")

    def respond(self, scenario: Scenario) -> str:
        return f"[offline recorded response: {self.model} / {scenario.scenario_id}]"


class RecordedJudge:
    """Matrix-backed judge that raises if a missing cell reaches the engine."""

    def __init__(self, row: pd.Series, model: str):
        self.row = row
        self.model = model
        self.name = "recorded-binary-matrix"
        self.prompt_version = "recorded"
        self.seed = 0
        self.n_lookups = 0

    def evaluate(self, scenario: Scenario, rubric: Rubric, response: str) -> JudgeVerdict:
        self.n_lookups += 1
        value = self.row.get(rubric.criterion_id, np.nan)
        if pd.isna(value):
            raise OfflineStudyError(
                f"missing matrix cell reached engine: {self.model}/{rubric.criterion_id}"
            )
        if float(value) not in {0.0, 1.0}:
            raise OfflineStudyError(
                f"non-binary matrix cell reached engine: {self.model}/{rubric.criterion_id}"
            )
        return JudgeVerdict(
            verdict="pass" if int(value) == 1 else "fail",
            evidence="recorded criterion verdict",
            rationale="offline response-matrix replay",
        )


@dataclass(frozen=True)
class RunSpec:
    seed: int = 42
    top_n: int = 5
    max_se: float | Mapping[str, float] = 0.30
    min_evals_per_skill: int = 15
    min_scenarios: int = 0
    max_scenarios: int = 50
    selection: str = "trace"
    mode: str = "cat"

    def to_config(self, dims: Sequence[str]) -> RunConfig:
        if self.selection not in {"trace", "dopt"}:
            raise ValueError("selection must be 'trace' or 'dopt'")
        if self.mode not in {"cat", "baseline"}:
            raise ValueError("mode must be 'cat' or 'baseline'")
        max_se = (
            {d: float(self.max_se) for d in dims}
            if isinstance(self.max_se, (int, float))
            else {d: float(self.max_se[d]) for d in dims}
        )
        return RunConfig(
            seed=self.seed,
            top_n=self.top_n,
            max_se=max_se,
            min_evals_per_skill=self.min_evals_per_skill,
            min_scenarios=self.min_scenarios,
            max_scenarios=self.max_scenarios,
            skills=tuple(dims),
            selection=self.selection,
            write_logs=False,
            unmapped_criteria="skip",
        )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def run_recorded_model(
    model: str,
    row: pd.Series,
    fitted: FittedBank,
    scenario_records: Mapping[str, dict[str, Any]],
    quadrature: Quadrature,
    spec: RunSpec,
    *,
    mwle_ridge: float = 1e-6,
) -> dict[str, Any]:
    """Replay one recorded model and compute online, batch-EAP, MWLE, and reference EAP."""

    engine_bank = engine_bank_for_row(fitted, scenario_records, row)
    tutor = RecordedTutor(model)
    judge = RecordedJudge(row, model)
    final = run_evaluation(
        engine_bank,
        tutor,
        judge,
        spec.to_config(fitted.dims),
        mode=spec.mode,
        run_id=f"offline_{tutor.name}_{spec.mode}_{spec.selection}_s{spec.seed}",
    )
    administered_ids = list(final["administered_criteria"])
    administered_idx = np.asarray([fitted.index[cid] for cid in administered_ids], dtype=int)
    responses = responses_for_bank(row, fitted)
    eap = batch_eap(responses, fitted.A, fitted.b, quadrature, administered_idx)
    full_eap = batch_eap(responses, fitted.A, fitted.b, quadrature)
    weighted = mwle(
        responses,
        fitted.A,
        fitted.b,
        administered_idx,
        theta0=eap.theta,
        ridge=mwle_ridge,
    )
    online_theta = np.asarray([final["theta"][d] for d in fitted.dims], dtype=float)
    rate_checks = {
        "online": pass_rate_check(responses, fitted.A, fitted.b, online_theta),
        "eap": pass_rate_check(responses, fitted.A, fitted.b, eap.theta),
        "mwle": pass_rate_check(responses, fitted.A, fitted.b, weighted.theta),
        "full_eap": pass_rate_check(responses, fitted.A, fitted.b, full_eap.theta),
    }
    if not weighted.converged:
        # The numeric MWLE field is its documented start/fallback value. It must not
        # enter an observable-calibration aggregate as if optimization succeeded.
        rate_checks["mwle"]["predicted_pass_rate"] = float("nan")
        rate_checks["mwle"]["error"] = float("nan")
    scenario_order = _ordered_unique(fitted.scenario_ids[fitted.index[c]] for c in administered_ids)
    return {
        "model": model,
        "mode": spec.mode,
        "selection": spec.selection,
        "seed": spec.seed,
        "stop_reason": final["stop_reason"],
        "precision_reached": bool(final["precision_reached"]),
        "scenarios_administered": int(final["scenarios_administered"]),
        "criteria_administered": len(administered_ids),
        "scenario_order": scenario_order,
        "criterion_order": administered_ids,
        "judge_lookups": judge.n_lookups,
        "theta_online": {d: float(final["theta"][d]) for d in fitted.dims},
        "se_online": {d: float(final["se"][d]) for d in fitted.dims},
        "theta_eap": {d: float(eap.theta[k]) for k, d in enumerate(fitted.dims)},
        "se_eap": {d: float(eap.se[k]) for k, d in enumerate(fitted.dims)},
        "theta_mwle": {d: float(weighted.theta[k]) for k, d in enumerate(fitted.dims)},
        "se_mwle": {d: float(weighted.se[k]) for k, d in enumerate(fitted.dims)},
        "mwle_converged": weighted.converged,
        "mwle_message": weighted.message,
        "theta_full_eap": {d: float(full_eap.theta[k]) for k, d in enumerate(fitted.dims)},
        "se_full_eap": {d: float(full_eap.se[k]) for k, d in enumerate(fitted.dims)},
        "n_observed_fitted_items": int(rate_checks["full_eap"]["n_items"]),
        "observed_fitted_pass_rate": float(rate_checks["full_eap"]["observed_pass_rate"]),
        "pass_rate_checks": rate_checks,
    }


def flatten_result(result: Mapping[str, Any], dims: Sequence[str]) -> dict[str, Any]:
    """Flatten a replay result to one CSV-friendly row."""

    flat = {
        key: result[key]
        for key in (
            "model",
            "mode",
            "selection",
            "seed",
            "stop_reason",
            "precision_reached",
            "scenarios_administered",
            "criteria_administered",
            "judge_lookups",
            "mwle_converged",
            "mwle_message",
        )
    }
    flat["scenario_order"] = json.dumps(result["scenario_order"], ensure_ascii=False)
    flat["criterion_order"] = json.dumps(result["criterion_order"], ensure_ascii=False)
    flat["n_observed_fitted_items"] = result["n_observed_fitted_items"]
    flat["observed_fitted_pass_rate"] = result["observed_fitted_pass_rate"]
    for estimator in ("online", "eap", "mwle", "full_eap"):
        check = result["pass_rate_checks"][estimator]
        flat[f"pirt_predicted_pass_rate_{estimator}"] = check["predicted_pass_rate"]
        flat[f"pirt_pass_rate_error_{estimator}"] = check["error"]
    for prefix in (
        "theta_online",
        "se_online",
        "theta_eap",
        "se_eap",
        "theta_mwle",
        "se_mwle",
        "theta_full_eap",
        "se_full_eap",
    ):
        for dim in dims:
            flat[f"{prefix}_{dim}"] = result[prefix][dim]
    return flat
