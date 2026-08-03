"""Shared library for SCENARIO-LEVEL CAT analyses on a saved response matrix.

Every analysis here drives the REAL production engine (``tutor_cat.engine.run_evaluation``
+ ``tutor_cat.selector.select_next``), which selects whole SCENARIOS and grades all their
criteria — exactly what an online CAT does. This replaces the earlier item-level harnesses
that picked individual criteria (a granularity the online system can never use).

What this module provides
-------------------------
* Fitted-bank IO (modeled-skill keys + ``q_modeled``), negative-loading policy, and
  per-model ``ItemBank`` construction — lifted from ``offline_engine_driver`` so all
  scenario scripts share one implementation.
* The reference/post-hoc estimator math (dense-grid EAP, batch EAP over an administered
  set, multidimensional MWLE) — copied from ``regen_cat_figures`` so scenario scripts do
  not depend on that (now-superseded, item-level) file.
* ``run_models`` — runs the engine for many models, optionally in parallel across models
  with ``ProcessPoolExecutor``. Each model is independent; the worker builds the shared
  read-only bank once per process via an initializer.

Nothing here reimplements CAT selection or the M2PL update — those stay in ``tutor_cat``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor_cat import SKILLS  # noqa: E402
from tutor_cat.dataio import ItemBank  # noqa: E402
from tutor_cat.engine import RunConfig, run_evaluation  # noqa: E402
from tutor_cat.schemas import JudgeVerdict, Rubric, Scenario  # noqa: E402


# ---------------------------------------------------------------------------
# estimator math (dense-grid EAP + MWLE) — self-contained, no regen dependency
# ---------------------------------------------------------------------------


def build_grid(n_dims: int, nodes_per_dim: int, half_range: float = 6.0):
    """Uniform product grid on [-half_range, half_range]^n_dims with a standard-normal
    (log) prior, normalised. Returns (grid (n_nodes, n_dims), log_prior (n_nodes,))."""
    axis = np.linspace(-half_range, half_range, nodes_per_dim)
    logphi = -0.5 * axis**2 - 0.5 * np.log(2 * np.pi)
    mesh = np.meshgrid(*([axis] * n_dims), indexing="ij")
    grid = np.stack([m.reshape(-1) for m in mesh], axis=1)
    logmesh = np.meshgrid(*([logphi] * n_dims), indexing="ij")
    log_prior = np.sum(np.stack([m.reshape(-1) for m in logmesh], axis=1), axis=1)
    return grid, log_prior - logsumexp(log_prior)


def eap_all_models(Y, mask, A, b, grid, log_prior, chunk: int = 4096) -> np.ndarray:
    """Full-bank EAP ability for every model (the selection-independent reference)."""
    ym = np.where(mask, Y, 0.0)
    nm = np.where(mask, 1.0 - Y, 0.0)
    n_models, n_nodes = Y.shape[0], grid.shape[0]
    ll = np.empty((n_models, n_nodes), dtype=float)
    for start in range(0, n_nodes, chunk):
        stop = min(start + chunk, n_nodes)
        eta = A @ grid[start:stop].T - b[:, None]
        ll[:, start:stop] = ym @ log_expit(eta) + nm @ log_expit(-eta)
    joint = ll + log_prior[None, :]
    post = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    return post @ grid


def eap_subset(y, idx, A, b, grid, log_prior):
    """Batch EAP posterior mean over ONLY the administered items (order-invariant)."""
    Ai, bi, yi = A[idx], b[idx], y[idx]
    eta = Ai @ grid.T - bi[:, None]
    ll = yi @ log_expit(eta) + (1.0 - yi) @ log_expit(-eta)
    joint = ll + log_prior
    post = np.exp(joint - logsumexp(joint))
    return post @ grid


def eap_subset_mean_var(y, idx, A, b, grid, log_prior):
    """Batch EAP posterior mean AND variance over the administered items."""
    Ai, bi, yi = A[idx], b[idx], y[idx]
    eta = Ai @ grid.T - bi[:, None]
    ll = yi @ log_expit(eta) + (1.0 - yi) @ log_expit(-eta)
    post = np.exp(ll + log_prior - logsumexp(ll + log_prior))
    mean = post @ grid
    var = post @ (grid**2) - mean**2
    return mean, np.clip(var, 0.0, None)


def _mwle_neg_obj_grad(theta, A, b, y, ridge):
    """Negative multidimensional-WLE objective + gradient (Warm/Firth penalty 1/2 ln det I)."""
    n_dims = theta.shape[0]
    eta = A @ theta - b
    p = expit(eta)
    w = p * (1.0 - p)
    info = (A * w[:, None]).T @ A + ridge * np.eye(n_dims)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e10, np.zeros(n_dims)
    ll = float(y @ log_expit(eta) + (1.0 - y) @ log_expit(-eta))
    obj = ll + 0.5 * logdet
    iinv = np.linalg.inv(info)
    c = w * (1.0 - 2.0 * p)
    quad = np.einsum("ij,jk,ik->i", A, iinv, A)
    grad = A.T @ (y - p) + 0.5 * (A * (c * quad)[:, None]).sum(axis=0)
    return -obj, -grad


def mwle_subset(y, idx, A, b, theta0, ridge=1e-6, bound=12.0):
    """Multidimensional WLE over the administered items, started from ``theta0``."""
    from scipy.optimize import minimize

    Ai, bi, yi = A[idx], b[idx], y[idx].astype(float)
    if Ai.shape[0] < Ai.shape[1]:
        return np.asarray(theta0, dtype=float), False
    try:
        res = minimize(_mwle_neg_obj_grad, np.asarray(theta0, dtype=float),
                       args=(Ai, bi, yi, ridge), jac=True, method="L-BFGS-B",
                       bounds=[(-bound, bound)] * Ai.shape[1])
    except Exception:
        return np.asarray(theta0, dtype=float), False
    if not np.all(np.isfinite(res.x)):
        return np.asarray(theta0, dtype=float), False
    return res.x, bool(res.success)


# ---------------------------------------------------------------------------
# fitted-bank IO (modeled-skill keys) + negative-loading policy
# ---------------------------------------------------------------------------


def load_fitted_bank(path: Path, negative_policy: str = "clamp"):
    """Parse a ``*_fitted`` rubric bank. Returns (records, dims, stats)."""
    records = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    if not records:
        raise SystemExit(f"no records in {path}")
    dims = list(records[0]["discrimination"].keys())
    for r in records:
        if list(r["discrimination"].keys()) != dims:
            raise SystemExit(f"inconsistent discrimination keys at {r['criterion_id']}")
    n_neg = sum(1 for r in records for d in dims
                if float(r["discrimination"][d] or 0.0) < 0)
    n_neg_items = sum(1 for r in records
                      if any(float(r["discrimination"][d] or 0.0) < 0 for d in dims))
    if negative_policy == "clamp":
        for r in records:
            for d in dims:
                if float(r["discrimination"][d] or 0.0) < 0:
                    r["discrimination"][d] = 0.0
    elif negative_policy == "drop":
        records = [r for r in records
                   if not any(float(r["discrimination"][d] or 0.0) < 0 for d in dims)]
    stats = {"n_negative_cells": n_neg, "n_negative_items": n_neg_items,
             "negative_policy": negative_policy, "n_records_after": len(records)}
    return records, dims, stats


def assemble_arrays(records: list[dict], dims: list[str]):
    """(ids, A (n_items, n_dims), b (n_items,)) in modeled-skill order."""
    ids = [r["criterion_id"] for r in records]
    A = np.array([[float(r["discrimination"][d] or 0.0) for d in dims] for r in records])
    b = np.array([float(r["difficulty"]) for r in records])
    return ids, A, b


def build_rubrics(records: list[dict], dims: list[str]) -> dict[str, Rubric]:
    """Rubric objects with q/a as len(dims) vectors in MODELED-skill order."""
    out: dict[str, Rubric] = {}
    for r in records:
        out[r["criterion_id"]] = Rubric(
            criterion_id=r["criterion_id"], scenario_id=r["scenario_id"],
            criterion=r.get("criterion", ""),
            q=np.array([int(r["q_modeled"][d]) for d in dims], dtype=int),
            a=np.array([float(r["discrimination"][d] or 0.0) for d in dims], dtype=float),
            b=float(r["difficulty"]), primary_skill=r.get("primary_skill", ""),
            scoring_type=r.get("scoring_type", "binary"),
            criticality=r.get("criticality", "standard"),
            calibration_version=str((r.get("irt_params") or {}).get("source", "")),
            status=r.get("status", "approved"),
        )
    return out


def load_scenarios(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[obj["scenario_id"]] = obj
    return out


def bank_for_model(rubrics: dict[str, Rubric], scen_raw: dict[str, dict],
                   row: pd.Series) -> ItemBank:
    """Bank restricted to criteria this model actually has a recorded verdict for."""
    graded = {cid for cid in rubrics if cid in row.index and not pd.isna(row[cid])}
    scenarios: dict[str, Scenario] = {}
    for sid, obj in scen_raw.items():
        cids = [c for c in obj["criterion_ids"] if c in graded]
        if not cids:
            continue
        s = Scenario.from_json(obj)
        s.criterion_ids = sorted(cids)
        scenarios[sid] = s
    kept = {cid: rubrics[cid] for s in scenarios.values() for cid in s.criterion_ids}
    return ItemBank(scenarios, kept)


class MatrixTutor:
    def __init__(self, model: str):
        self.model = model
        self.name = model.replace("/", "_")

    def respond(self, scenario: Scenario) -> str:
        return f"[offline] {self.model} on {scenario.scenario_id}"


class MatrixJudge:
    def __init__(self, row: pd.Series, model: str):
        self._row = row
        self.name = "offline-matrix"
        self.prompt_version = "recorded"
        self.seed = 0
        self.model = model
        self.n_lookups = 0
        self.n_missing = 0

    def evaluate(self, scenario: Scenario, rubric: Rubric, response: str) -> JudgeVerdict:
        self.n_lookups += 1
        val = self._row.get(rubric.criterion_id, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            self.n_missing += 1
            return JudgeVerdict(verdict="fail", unscorable_reason="not_in_matrix")
        return JudgeVerdict(verdict="pass" if int(val) == 1 else "fail",
                            evidence="recorded", rationale="offline matrix lookup")


def administered_from_log(run_dir: Path) -> list[str]:
    """Criterion ids the engine actually administered, in update order."""
    out: list[str] = []
    p = Path(run_dir) / "criterion_updates.jsonl"
    if not p.is_file():
        return out
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line)["criterion_id"])
    return out


# ---------------------------------------------------------------------------
# run spec + single-model engine run
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    seed: int = 42
    top_n: int = 5
    max_se: float = 0.30
    min_evals_per_skill: int = 15
    min_scenarios: int = 0
    max_scenarios: int = 50
    unmapped_criteria: str = "judge"
    selection: str = "trace"
    mode: str = "cat"  # "cat" (adaptive) or "baseline" (seeded-random scenario order)
    runs_dir: str = str(ROOT / "staging" / "engine_runs")
    write_logs: bool = False  # offline replays skip the per-line flushed JSONL logs

    def to_config(self, dims: list[str]) -> RunConfig:
        return RunConfig(
            seed=self.seed, top_n=self.top_n,
            max_se={s: self.max_se for s in dims},
            min_evals_per_skill=self.min_evals_per_skill,
            min_scenarios=self.min_scenarios,
            max_scenarios=self.max_scenarios,
            output_dir=self.runs_dir,
            unmapped_criteria=self.unmapped_criteria,
            selection=self.selection,
            skills=tuple(dims),
            write_logs=self.write_logs,
        )


def run_one_model(model: str, rubrics: dict, scen_raw: dict, row: pd.Series,
                  dims: list[str], spec: RunSpec, cleanup: bool = True) -> dict:
    """Run the real scenario-level engine for one model; return a compact record."""
    import shutil

    cfg = spec.to_config(dims)
    bank = bank_for_model(rubrics, scen_raw, row)
    tutor = MatrixTutor(model)
    judge = MatrixJudge(row, model)
    run_id = f"offline_{tutor.name}_{spec.selection}_{spec.mode}_s{spec.seed}"
    final = run_evaluation(bank, tutor, judge, cfg, mode=spec.mode, run_id=run_id)
    # Prefer the order returned in the result (write_logs=False); fall back to the log file.
    order = final.get("administered_criteria")
    if order is None:
        order = administered_from_log(Path(spec.runs_dir) / run_id)
    rec = {
        "model": model,
        "order": order,
        "scenarios_administered": final["scenarios_administered"],
        "criteria_administered": len(order),
        "stop_reason": final["stop_reason"],
        "precision_reached": bool(final["precision_reached"]),
        "theta_online": [float(final["theta"][d]) for d in dims],
        "se_online": [float(final["se"][d]) for d in dims],
        "scorable_evals": [int(final["scorable_evaluations"][d]) for d in dims],
        "judge_lookups": judge.n_lookups,
        "judge_missing": judge.n_missing,
    }
    if cleanup and spec.write_logs:
        shutil.rmtree(Path(spec.runs_dir) / run_id, ignore_errors=True)
    return rec


# ---- parallel driver over models ------------------------------------------

_W: dict = {}


def _limit_blas_threads():
    """Pin BLAS to a single thread per worker. With many worker processes the default
    (one thread pool per core, per process) oversubscribes and OpenBLAS fails to allocate.
    Each engine run is tiny linear algebra, so 1 thread/worker is both safe and faster."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _init_worker(bank_path, matrix_path, scenarios_path, negative_policy, dims, spec_dict):
    _limit_blas_threads()
    records, _dims, _ = load_fitted_bank(Path(bank_path), negative_policy)
    _W["rubrics"] = build_rubrics(records, dims)
    _W["scen_raw"] = load_scenarios(Path(scenarios_path))
    _W["matrix"] = pd.read_csv(matrix_path, index_col=0)
    _W["dims"] = dims
    _W["spec"] = RunSpec(**spec_dict)


def _worker(model: str) -> dict:
    return run_one_model(model, _W["rubrics"], _W["scen_raw"],
                         _W["matrix"].loc[model], _W["dims"], _W["spec"])


def run_models(models: list[str], bank_path: Path, matrix_path: Path,
               scenarios_path: Path, negative_policy: str, dims: list[str],
               spec: RunSpec, workers: int = 1) -> list[dict]:
    """Run the engine for ``models``. ``workers`` > 1 parallelises across models."""
    if workers and workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(str(bank_path), str(matrix_path), str(scenarios_path),
                      negative_policy, dims, asdict(spec)),
        ) as ex:
            return list(ex.map(_worker, models))
    # serial
    records, _, _ = load_fitted_bank(bank_path, negative_policy)
    rubrics = build_rubrics(records, dims)
    scen_raw = load_scenarios(scenarios_path)
    matrix = pd.read_csv(matrix_path, index_col=0)
    return [run_one_model(m, rubrics, scen_raw, matrix.loc[m], dims, spec) for m in models]


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------------------
# post-hoc estimators from an administered order (parent side, fast)
# ---------------------------------------------------------------------------


def posthoc_estimators(rec: dict, col: dict, Yrow: np.ndarray, A: np.ndarray,
                       b: np.ndarray, grid: np.ndarray, log_prior: np.ndarray,
                       mwle_ridge: float = 1e-6):
    """(theta_online, theta_batch, theta_mwle, mwle_ok) for one model's administered set."""
    th_on = np.asarray(rec["theta_online"], float)
    idx = np.array([col[c] for c in rec["order"] if c in col], dtype=int)
    if idx.size == 0:
        return th_on, th_on.copy(), th_on.copy(), True
    th_ba = eap_subset(Yrow, idx, A, b, grid, log_prior)
    th_mw, ok = mwle_subset(Yrow, idx, A, b, th_ba, ridge=mwle_ridge)
    return th_on, th_ba, th_mw, ok


def verify_provenance(bank_path: Path, matrix_path: Path) -> dict:
    """Cross-check the matrix sha256 against the bank's recorded provenance."""
    import hashlib

    recs = [json.loads(l) for l in Path(bank_path).open(encoding="utf-8") if l.strip()]
    prov = (recs[0].get("irt_params") or {}).get("provenance") or {}
    actual = hashlib.sha256(Path(matrix_path).read_bytes()).hexdigest()
    return {"expected_matrix": prov.get("matrix_csv"),
            "expected_sha256": prov.get("matrix_sha256"),
            "actual_sha256": actual,
            "aligned": prov.get("matrix_sha256") == actual}
