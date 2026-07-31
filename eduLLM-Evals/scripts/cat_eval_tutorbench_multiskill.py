"""Skills-parameterized TutorBench CAT evaluation harness with an OPERATIONAL CAT policy.

A single entrypoint over BOTH TutorBench CAT instruments:

* ``--skills 2`` -> the 2-skill [correctness, scaffolding] M2PL CAT eval.
* ``--skills 3`` -> the Run 6 3-skill [correctness, scaffolding, presentation] M2PL CAT eval.

Unlike the earlier thin dispatcher, this harness re-implements the (N-dimensional) CAT
administration so that a proper, configurable CAT policy can be enforced. The M2PL math
itself is still reused verbatim (``tutor_cat.mirt.update`` / ``standard_errors`` and the
Gauss-Hermite quadrature helpers in ``scripts/calibrate_mirt.py``); only *item selection*
and *stopping* gain policy.

Why (the deceptive-convergence problem)
---------------------------------------
The naive harness registers a skill as "converged" the first step its posterior SE drops
below the target -- regardless of how many administered items actually LOAD that skill.
Scaffolding items are scarce but highly discriminating, so 3-4 of them crush the
scaffolding SE below 0.3, and scaffolding "converges" on almost no scaffolding evidence.
Its recovery then looks deceptively good. A real CAT would keep administering
scaffolding-loading items until a realistic floor is met. This harness enforces that.

Policy components (all configurable; each defaults ON, ``--legacy`` turns them all OFF)
-------------------------------------------------------------------------------------
1. Point-biserial item filter (``--min-point-biserial``, default 0.05) -- ports
   ``tutor_cat.mcq_irt.matrix.filter_items``: drop all-pass / all-fail items and items
   with a low/negative item-rest point-biserial from the CAT bank (NaN-aware for the
   partial matrix). Non-discriminating / mis-keyed items are not administrable.
2. Deterministic tie-break in selection -- ports the ``np.lexsort`` idea from
   ``tutor_cat.mcq_irt.cat.run_cat``: equal-information ties break by ascending bank
   index, i.e. by ``criterion_id`` (the bank CSV is criterion_id-ordered). Always on so
   runs are reproducible; ``--legacy`` still uses it (it only affects exact ties).
3. Per-skill minimum-item floor (``--min-items-per-skill``, default 15) -- ports
   ``tutor_cat.engine.RunConfig.min_evals_per_skill``: a skill may not register as
   converged (nor satisfy the overall stop) until at least this many ADMINISTERED items
   actually load it (``a_skill > 0``).
4. Content balancing (``--content-balance`` / ``--no-content-balance``, default ON) --
   mirrors ``tutor_cat.engine`` targeting + ``tutor_cat.selector.select_next``: each step
   targets the least-measured skill (max SE among skills still below their floor, else max
   SE above the SE target) and restricts the candidate pool to items loading that skill,
   so selection isn't dominated by whichever skill Fisher info happens to favour.
5. Exposure control (``--exposure-top-n``, default 5) -- ports
   ``tutor_cat.selector.select_next``'s ``top_n``: instead of the single argmax, choose
   uniformly (seeded, per-model) among the top-N most-informative candidates, capping the
   over-exposure of the single highest-info item across the fleet.

Everything else (full-bank EAP recovery reference, OOS k-fold honesty check, the summary
table, and all figures) mirrors ``scripts/cat_eval_tutorbench.py`` /
``scripts/cat_eval_tutorbench_3skill.py`` so results stay directly comparable.

READ-ONLY: nothing is written back to the bank, the rubric bank, or the fitters.

Usage
-----
    python scripts/cat_eval_tutorbench_multiskill.py --skills 2
    python scripts/cat_eval_tutorbench_multiskill.py --skills 3
    # before/after (legacy vs policy) in one shot, the deceptive-convergence report:
    python scripts/cat_eval_tutorbench_multiskill.py --skills 3 --compare
    # reproduce the OLD (deceptive) behaviour exactly:
    python scripts/cat_eval_tutorbench_multiskill.py --skills 2 --legacy
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


cm = _load_module("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cp = cm.cp
from tutor_cat import mirt as tc_mirt  # noqa: E402

# ---------------------------------------------------------------------------
# per-skill-set configuration (defaults reproduce the two original scripts)
# ---------------------------------------------------------------------------

DIM_COLORS = {
    "correctness": "#4c72b0",
    "scaffolding": "#dd8452",
    "presentation": "#55a868",
}

# The curated bank carries an ``exclude_from_fit`` flag on the 60 reverse/degenerate
# items; those must not be eligible for CAT administration either.
DEFAULT_CURATED = ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl"


def skillset_config(skills: int) -> dict:
    if skills == 2:
        return {
            "dims": ("correctness", "scaffolding"),
            "bank": ROOT / "staging" / "calibration_mirt_full2skill.csv",
            "a_cols": ["a_correctness", "a_scaffolding"],
            "matrix": ROOT / "staging" / "response_matrix_full_nonopt.csv",
            "kfold_dir": ROOT / "staging" / "kfold_full",
            "fold_a_cols": ["a_correctness", "a_scaffolding"],
            "items_kept": 3497,
            "items_total": 6180,
            "bank_health": {
                "correctness": "healthy (cross-fold a r=0.80)",
                "scaffolding": "needs refit (cross-fold a r=0.48)",
            },
        }
    if skills == 3:
        return {
            "dims": ("correctness", "scaffolding", "presentation"),
            # Run 6 slot-repurposing: a_content->correctness, a_diagnosis->scaffolding,
            # a_scaffolding->presentation.
            "bank": ROOT / "staging" / "run6_presentation" / "calibration_mirt.csv",
            "a_cols": ["a_content", "a_diagnosis", "a_scaffolding"],
            "matrix": ROOT / "staging" / "response_matrix_full.csv",
            "kfold_dir": ROOT / "staging" / "kfold_full_3skill",
            "fold_a_cols": ["a_correctness", "a_scaffolding", "a_presentation"],
            "items_kept": 4156,
            "items_total": 6845,
            "bank_health": {
                "correctness": "healthy (cross-fold a r=0.77)",
                "scaffolding": "needs refit (cross-fold a r=0.49)",
                "presentation": "stable (cross-fold a r=0.71)",
            },
        }
    raise ValueError("skills must be 2 or 3")


# ---------------------------------------------------------------------------
# CAT policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """Operational CAT policy knobs. ``Policy.legacy()`` reproduces the old behaviour."""

    pbis_filter: bool = True
    min_point_biserial: float = 0.05
    min_items_per_skill: int = 15
    content_balance: bool = True
    exposure_top_n: int = 5
    # deterministic tie-break is always on (only affects exact-tie selection); kept as a
    # field for provenance in the metrics JSON.
    deterministic_tiebreak: bool = True

    @classmethod
    def legacy(cls) -> "Policy":
        return cls(
            pbis_filter=False,
            min_point_biserial=0.05,
            min_items_per_skill=0,
            content_balance=False,
            exposure_top_n=1,
            deterministic_tiebreak=True,
        )

    def to_dict(self) -> dict:
        return {
            "pbis_filter": self.pbis_filter,
            "min_point_biserial": self.min_point_biserial,
            "min_items_per_skill": self.min_items_per_skill,
            "content_balance": self.content_balance,
            "exposure_top_n": self.exposure_top_n,
            "deterministic_tiebreak": self.deterministic_tiebreak,
        }


# ---------------------------------------------------------------------------
# bank / exclusion / point-biserial filter
# ---------------------------------------------------------------------------


def load_excluded_criteria(path: Path) -> set[str]:
    """criterion_ids flagged ``exclude_from_fit: true`` in the curated bank (the 60
    reverse/degenerate items). These must be skipped when building the CAT item bank."""
    excluded: set[str] = set()
    if not path.is_file():
        return excluded
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("exclude_from_fit") is True:
                excluded.add(str(rec["criterion_id"]))
    return excluded


def load_bank(path: Path, a_cols: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    items = df["criterion_id"].astype(str).tolist()
    A = df[a_cols].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    return items, A, b


def point_biserial_partial(Yraw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-item (p_value, item-rest point-biserial) for a partial matrix.

    Mirrors ``tutor_cat.mcq_irt.matrix._point_biserial`` (rest score = person's total
    minus the item) but computed over only the observed cells of each item, so we do not
    have to drop the (mostly-incomplete) model rows the dense version would discard.
    """
    obs = ~np.isnan(Yraw)
    Yz = np.nan_to_num(Yraw, nan=0.0)
    person_sum = Yz.sum(axis=1)
    person_cnt = obs.sum(axis=1)
    n_items = Yraw.shape[1]
    p_val = np.full(n_items, np.nan)
    pbis = np.full(n_items, np.nan)
    for j in range(n_items):
        rows = obs[:, j]
        if rows.sum() < 3:
            continue
        xj = Yz[rows, j]
        rest_cnt = person_cnt[rows] - 1
        rest_mean = np.where(rest_cnt > 0, (person_sum[rows] - xj) / np.maximum(rest_cnt, 1), np.nan)
        ok = np.isfinite(rest_mean)
        xj, rest_mean = xj[ok], rest_mean[ok]
        p_val[j] = float(xj.mean()) if xj.size else np.nan
        if xj.size >= 3 and xj.std() > 0 and rest_mean.std() > 0:
            pbis[j] = float(np.corrcoef(xj, rest_mean)[0, 1])
    return p_val, pbis


def apply_pbis_filter(
    Yraw: np.ndarray, items: list[str], min_pbis: float
) -> tuple[np.ndarray, dict]:
    """Return a boolean keep-mask over bank items + a small report dict."""
    p_val, pbis = point_biserial_partial(Yraw)
    all_pass = p_val >= 1.0
    all_fail = p_val <= 0.0
    low_pbis = ~(all_pass | all_fail) & (~np.isfinite(pbis) | (pbis < min_pbis))
    keep = ~(all_pass | all_fail | low_pbis)
    report = {
        "min_point_biserial": min_pbis,
        "n_all_pass": int(np.nansum(all_pass)),
        "n_all_fail": int(np.nansum(all_fail)),
        "n_low_point_biserial": int(low_pbis.sum()),
        "n_kept": int(keep.sum()),
        "n_bank_in": int(len(items)),
    }
    return keep, report


# ---------------------------------------------------------------------------
# ability estimation (reused math)
# ---------------------------------------------------------------------------


def eap_full(
    y_obs: np.ndarray,
    mask: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
) -> np.ndarray:
    """Full-response EAP ability over ALL observed items (the recovery reference)."""
    ym = np.where(mask, y_obs, 0.0)
    nm = np.where(mask, 1.0 - y_obs, 0.0)
    eta = A @ grid.T - b[:, None]
    ll = ym @ log_expit(eta) + nm @ log_expit(-eta)
    joint = ll + log_prior
    post = np.exp(joint - logsumexp(joint))
    return post @ grid


def run_cat_person(
    y: np.ndarray,
    mask: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    n_dims: int,
    pol: Policy,
    min_items: int,
    max_items: int,
    se_target: float,
    rng: np.random.Generator,
) -> dict:
    """Policy-driven adaptive administration for one person over their observed items.

    Selection: (optionally) target the least-measured skill and restrict the candidate
    pool to items loading it (content balancing), score candidates by multidimensional
    Fisher information ``p(1-p) * (m . m)``, break ties deterministically by ascending
    bank index (== criterion_id), and pick uniformly among the top-N (exposure control).
    A skill only registers as converged once its SE < target AND at least
    ``min_items_per_skill`` administered items load it (the per-skill floor). Stops when
    ALL skills are converged (after ``min_items``) or at ``max_items``.
    """
    obs_idx = np.where(mask)[0]
    remaining = sorted(int(i) for i in obs_idx)  # deterministic order
    theta = np.zeros(n_dims)
    U = np.eye(n_dims)
    ones_q = np.ones(n_dims)
    floor = pol.min_items_per_skill

    se_trace: list[list[float]] = []
    theta_trace: list[list[float]] = []
    n_cross: list[int | None] = [None] * n_dims
    exposure = np.zeros(n_dims, dtype=int)
    pattern_counts: dict[str, int] = {}
    administered: list[int] = []

    n_admin = 0
    while remaining and n_admin < max_items:
        se = tc_mirt.standard_errors(U)

        # --- content balancing: choose the target skill + candidate pool ---
        if pol.content_balance:
            needy = [d for d in range(n_dims) if exposure[d] < floor]
            if needy:
                target = max(needy, key=lambda d: (se[d], -d))
            else:
                above = [d for d in range(n_dims) if se[d] >= se_target]
                target = (
                    max(above, key=lambda d: (se[d], -d))
                    if above
                    else int(np.argmax(se))
                )
            cand = [i for i in remaining if A[i, target] > 0]
            pool = cand if cand else remaining
        else:
            pool = remaining

        rem = np.asarray(pool, dtype=int)
        m = A[rem]
        p = expit(m @ theta - b[rem])
        info = p * (1.0 - p) * np.sum(m * m, axis=1)
        # deterministic: information descending, ascending bank index (criterion_id) on ties
        order = np.lexsort((rem, -info))
        top = order[: max(1, pol.exposure_top_n)]
        chosen = int(top[int(rng.integers(len(top)))]) if pol.exposure_top_n > 1 else int(order[0])
        pick = int(rem[chosen])

        loads = A[pick] > 0
        exposure += loads.astype(int)
        key = "".join("1" if loads[d] else "0" for d in range(n_dims))
        pattern_counts[key] = pattern_counts.get(key, 0) + 1
        theta, U, _ = tc_mirt.update(theta, U, A[pick], ones_q, float(b[pick]), int(y[pick]))
        remaining.remove(pick)
        administered.append(pick)
        n_admin += 1

        se = tc_mirt.standard_errors(U)
        se_trace.append([float(se[d]) for d in range(n_dims)])
        theta_trace.append([float(theta[d]) for d in range(n_dims)])
        for d in range(n_dims):
            if n_cross[d] is None and float(se[d]) < se_target and exposure[d] >= floor:
                n_cross[d] = n_admin
        converged_all = all(
            float(se[d]) < se_target and exposure[d] >= floor for d in range(n_dims)
        )
        if n_admin >= min_items and converged_all:
            break

    se_final = tc_mirt.standard_errors(U)
    all_crossed = all(c is not None for c in n_cross)
    n_overall = max(c for c in n_cross) if all_crossed else None
    return {
        "theta_cat": theta,
        "n_items": n_admin,
        "se_final": [float(se_final[d]) for d in range(n_dims)],
        "n_cross": n_cross,
        "n_cross_overall": n_overall,
        "converged_overall": all_crossed,
        "exposure": exposure,
        "pattern_counts": pattern_counts,
        "administered": administered,
        "se_trace": se_trace,
        "theta_trace": theta_trace,
    }


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------


def pearson(x, y) -> float | None:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 2 or xa.std() == 0 or ya.std() == 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _clean(vals) -> list[float]:
    out = []
    for v in vals:
        if v is None:
            continue
        fv = float(v)
        if np.isnan(fv):
            continue
        out.append(fv)
    return out


def _median(vals) -> float | None:
    vals = _clean(vals)
    return float(np.median(vals)) if vals else None


def _mean(vals) -> float | None:
    vals = _clean(vals)
    return float(np.mean(vals)) if vals else None


def _min(vals) -> float | None:
    vals = _clean(vals)
    return float(np.min(vals)) if vals else None


# ---------------------------------------------------------------------------
# per-model run + aggregation
# ---------------------------------------------------------------------------


def _seed_for(base_seed: int, model: str) -> int:
    import zlib

    return (base_seed * 1_000_003 + zlib.crc32(model.encode())) % (2**32)


def run_cat_over_models(
    models, Y, M, A, b, dims, grid, log_prior, pol, min_items, max_items, se_target, base_seed
):
    n_dims = len(dims)
    rows = []
    for r, model in enumerate(models):
        mask = M[r]
        rng = np.random.default_rng(_seed_for(base_seed, model))
        theta_full = eap_full(Y[r], mask, A, b, grid, log_prior)
        res = run_cat_person(
            Y[r], mask, A, b, n_dims, pol, min_items, max_items, se_target, rng
        )
        theta_cat = res["theta_cat"]
        obs = np.where(mask)[0]
        obs_acc = float(Y[r][obs].mean())
        pred_acc_cat = float(expit(A[obs] @ theta_cat - b[obs]).mean())
        pred_acc_full = float(expit(A[obs] @ theta_full - b[obs]).mean())
        row = {
            "model": model,
            "n_obs_items": int(mask.sum()),
            "cat_n_items": res["n_items"],
            "n_cross_overall": res["n_cross_overall"],
            "converged_overall": res["converged_overall"],
            "obs_acc": obs_acc,
            "pred_acc_cat": pred_acc_cat,
            "pred_acc_full": pred_acc_full,
            "se_trace": res["se_trace"],
            "theta_trace": res["theta_trace"],
            "administered": res["administered"],
        }
        for i, dim in enumerate(dims):
            row[f"theta_full_{dim}"] = float(theta_full[i])
            row[f"theta_cat_{dim}"] = float(theta_cat[i])
            row[f"n_cross_{dim}"] = res["n_cross"][i]
            row[f"final_se_{dim}"] = res["se_final"][i]
            row[f"exposure_{dim}"] = int(res["exposure"][i])
        rows.append(row)
    return rows


def aggregate(df: pd.DataFrame, dims) -> dict:
    out = {}
    out["recovery_r"] = {d: pearson(df[f"theta_full_{d}"], df[f"theta_cat_{d}"]) for d in dims}
    full_all = np.concatenate([df[f"theta_full_{d}"].to_numpy() for d in dims])
    cat_all = np.concatenate([df[f"theta_cat_{d}"].to_numpy() for d in dims])
    out["recovery_r"]["overall_stacked"] = pearson(full_all, cat_all)

    out["cat_items"] = {"n_models": int(len(df))}
    for d in dims:
        out["cat_items"][d] = {
            "mean": _mean(df[f"n_cross_{d}"]),
            "median": _median(df[f"n_cross_{d}"]),
            "converged": int(df[f"n_cross_{d}"].notna().sum()),
        }
    out["cat_items"]["overall"] = {
        "mean": _mean(df["n_cross_overall"]),
        "median": _median(df["n_cross_overall"]),
        "converged": int(df["n_cross_overall"].notna().sum()),
    }
    # per-skill items-administered (items LOADING each skill that were administered)
    out["items_administered_by_skill"] = {
        d: {
            "mean": _mean(df[f"exposure_{d}"]),
            "median": _median(df[f"exposure_{d}"]),
            "min": _min(df[f"exposure_{d}"]),
        }
        for d in dims
    }
    d_cat = (df["pred_acc_cat"] - df["obs_acc"]).abs()
    d_full = (df["pred_acc_full"] - df["obs_acc"]).abs()
    out["pirt_mae_cat"] = float(d_cat.mean())
    out["pirt_mae_full"] = float(d_full.mean())
    out["pirt_corr_cat"] = pearson(df["pred_acc_cat"], df["obs_acc"])
    return out


def a_medians(A: np.ndarray, dims) -> dict:
    out = {}
    for i, dim in enumerate(dims):
        col = A[:, i]
        nz = col[col > 0]
        out[f"a_{dim}_median_nonzero"] = float(np.median(nz)) if nz.size else None
        out[f"a_{dim}_n_loading"] = int((col > 0).sum())
    return out


def exposure_summary(df: pd.DataFrame, A: np.ndarray, dims) -> dict:
    total_admin = int(df["cat_n_items"].sum())
    exp_counts = {d: int(df[f"exposure_{d}"].sum()) for d in dims}
    exp_share = {d: (exp_counts[d] / total_admin if total_admin else None) for d in dims}
    n_items = A.shape[0]
    bank_share = {d: float((A[:, i] > 0).sum()) / n_items for i, d in enumerate(dims)}
    return {
        "total_items_administered": total_admin,
        "exposure_count_by_skill": exp_counts,
        "exposure_share_by_skill": exp_share,
        "bank_loading_share_by_skill": bank_share,
    }


def item_exposure_population(rows: list[dict], items: list[str], top_k: int = 15) -> dict:
    """Population-level item over-exposure (exposure control target): how often each bank
    item was administered across all models, and the most-exposed items."""
    counts = np.zeros(len(items), dtype=int)
    n_models = len(rows)
    for r in rows:
        for idx in r["administered"]:
            counts[idx] += 1
    order = np.argsort(-counts)
    top = [
        {"criterion_id": items[i], "n_administered": int(counts[i]),
         "exposure_rate": float(counts[i] / n_models) if n_models else 0.0}
        for i in order[:top_k]
        if counts[i] > 0
    ]
    used = counts > 0
    return {
        "n_models": n_models,
        "n_items_ever_administered": int(used.sum()),
        "max_exposure_count": int(counts.max()) if counts.size else 0,
        "max_exposure_rate": float(counts.max() / n_models) if (counts.size and n_models) else 0.0,
        "top_exposed_items": top,
    }


# ---------------------------------------------------------------------------
# OOS honesty check
# ---------------------------------------------------------------------------


def run_oos(
    mat, oos_models, model_to_fold, kfold_dir, fold_a_cols, df_full, dims,
    excluded, keep_ids, pol, min_items, max_items, se_target, base_seed,
):
    n_dims = len(dims)
    full_theta = {r["model"]: {d: r[f"theta_full_{d}"] for d in dims} for _, r in df_full.iterrows()}
    rows = []
    for f in sorted(set(model_to_fold[m] for m in oos_models)):
        fp = pd.read_csv(kfold_dir / f"fold_{f}_item_params.csv")
        fp["criterion_id"] = fp["criterion_id"].astype(str)
        # apply the same bank policy (exclusion + pbis-derived keep set) to the fold bank
        fp = fp[~fp["criterion_id"].isin(excluded)]
        if keep_ids is not None:
            fp = fp[fp["criterion_id"].isin(keep_ids)]
        items = fp["criterion_id"].tolist()
        fA = fp[fold_a_cols].to_numpy(dtype=float)
        fb = fp["b"].to_numpy(dtype=float)
        test_models = [m for m in oos_models if model_to_fold[m] == f]
        sub = mat.loc[test_models].reindex(columns=items)
        Yraw = sub.to_numpy(dtype=float)
        Mobs = ~np.isnan(Yraw)
        Yv = np.nan_to_num(Yraw, nan=0.0)
        for ridx, model in enumerate(test_models):
            rng = np.random.default_rng(_seed_for(base_seed, "oos:" + model))
            res = run_cat_person(
                Yv[ridx], Mobs[ridx], fA, fb, n_dims, pol, min_items, max_items, se_target, rng
            )
            tc = res["theta_cat"]
            row = {"model": model, "fold": int(f), "cat_n_items": res["n_items"],
                   "n_cross_overall": res["n_cross_overall"]}
            for i, dim in enumerate(dims):
                row[f"theta_full_{dim}"] = full_theta[model][dim]
                row[f"theta_cat_{dim}"] = float(tc[i])
                row[f"n_cross_{dim}"] = res["n_cross"][i]
                row[f"final_se_{dim}"] = res["se_final"][i]
                row[f"exposure_{dim}"] = int(res["exposure"][i])
            rows.append(row)
    oos_df = pd.DataFrame(rows)
    oos = {
        "recovery_r": {d: pearson(oos_df[f"theta_full_{d}"], oos_df[f"theta_cat_{d}"]) for d in dims},
        "cat_items": {
            d: {"mean": _mean(oos_df[f"n_cross_{d}"]), "median": _median(oos_df[f"n_cross_{d}"]),
                "converged": int(oos_df[f"n_cross_{d}"].notna().sum())}
            for d in dims
        },
        "items_administered_by_skill": {
            d: {"mean": _mean(oos_df[f"exposure_{d}"]), "median": _median(oos_df[f"exposure_{d}"]),
                "min": _min(oos_df[f"exposure_{d}"])}
            for d in dims
        },
        "n_models": int(len(oos_df)),
    }
    return oos_df, oos


# ---------------------------------------------------------------------------
# one full evaluation pass (bank policy already applied to A/b/items)
# ---------------------------------------------------------------------------


def run_pass(
    label, items, A, b, dims, mat, models, Y, M, grid, log_prior,
    cfg, pol, args, excluded, keep_ids,
):
    print(f"\n[{label}] policy = {pol.to_dict()}")
    rows = run_cat_over_models(
        models, Y, M, A, b, dims, grid, log_prior, pol,
        args.min_items, args.max_items, args.se_target, args.seed,
    )
    df = pd.DataFrame(rows)
    agg = {"n_models": int(len(df)), "in_sample": aggregate(df, dims)}
    agg["in_sample"]["item_exposure"] = exposure_summary(df, A, dims)
    agg["in_sample"]["item_exposure_population"] = item_exposure_population(rows, items)

    oos_df = None
    oos = None
    if not args.no_oos:
        with (cfg["kfold_dir"] / "fold_assignments.json").open(encoding="utf-8") as fh:
            fa = json.load(fh)
        model_to_fold = fa["model_to_fold"]
        oos_models = [m for m in models if m in model_to_fold]
        oos_df, oos = run_oos(
            mat, oos_models, model_to_fold, cfg["kfold_dir"], cfg["fold_a_cols"],
            df, dims, excluded, keep_ids, pol, args.min_items, args.max_items,
            args.se_target, args.seed,
        )
        agg["out_of_sample"] = oos
    return df, agg, oos_df, oos


# ---------------------------------------------------------------------------
# figures (mirror the 2-/3-skill scripts, N-dim)
# ---------------------------------------------------------------------------


def make_figures(df, agg, A, dims, fig_dir, se_target, max_items):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def _save(fig, name):
        p = fig_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths[name] = str(p.resolve())

    ins = agg["in_sample"]

    for i, dim in enumerate(dims):
        full = df[f"theta_full_{dim}"].to_numpy()
        cat = df[f"theta_cat_{dim}"].to_numpy()
        r = ins["recovery_r"][dim]
        rstr = f"{r:.3f}" if r is not None else "n/a"
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.scatter(full, cat, s=28, alpha=0.75, edgecolor="k", linewidth=0.3,
                   color=DIM_COLORS.get(dim, "#333333"))
        lo = min(full.min(), cat.min()) - 0.3
        hi = max(full.max(), cat.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ability ({dim})")
        ax.set_ylabel(f"CAT ability ({dim})")
        ax.set_title(f"TutorBench {len(dims)}-skill CAT recovery: {dim}\n"
                     f"Pearson r = {rstr} (n = {len(df)} models)")
        ax.legend(loc="upper left", fontsize=9)
        _save(fig, f"recovery_scatter_{dim}.png")

    lens = df["n_cross_correctness"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if lens.size:
        ax.hist(lens, bins=range(0, int(lens.max()) + 2), color="#4c72b0", edgecolor="white")
        mean_v, med_v = float(lens.mean()), float(np.median(lens))
        ax.axvline(mean_v, color="crimson", ls="--", lw=1.5, label=f"mean = {mean_v:.1f}")
        ax.axvline(med_v, color="green", ls=":", lw=1.5, label=f"median = {med_v:.0f}")
        ax.legend()
    ax.set_xlabel(f"# items to reach correctness SE < {se_target}")
    ax.set_ylabel("# models")
    conv = int(df["n_cross_correctness"].notna().sum())
    ax.set_title(f"TutorBench {len(dims)}-skill CAT length (correctness axis)\n"
                 f"{conv}/{len(df)} models converged; cap = {max_items} items")
    _save(fig, "cat_length_hist.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    max_len = int(df["cat_n_items"].max())
    for idx, dim in enumerate(dims):
        means, xs = [], []
        for step in range(max_len):
            vals = [tr[step][idx] for tr in df["se_trace"] if len(tr) > step]
            if len(vals) >= 5:
                means.append(float(np.mean(vals)))
                xs.append(step + 1)
        ax.plot(xs, means, label=f"{dim} (mean SE)", color=DIM_COLORS.get(dim, "#333333"), lw=2)
    ax.axhline(se_target, color="gray", ls="--", lw=1, label=f"SE target = {se_target}")
    ax.set_xlabel("# items administered")
    ax.set_ylabel("mean posterior SE")
    ax.set_title(f"TutorBench {len(dims)}-skill CAT: mean SE reduction by dimension")
    ax.legend()
    _save(fig, "se_reduction_curve.png")

    pred = df["pred_acc_cat"].to_numpy()
    act = df["obs_acc"].to_numpy()
    mae = ins["pirt_mae_cat"]
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(act, pred, s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
    lo = min(pred.min(), act.min()) - 0.02
    hi = max(pred.max(), act.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("actual observed pass-rate")
    ax.set_ylabel("predicted accuracy (from CAT ability)")
    ax.set_title(f"TutorBench {len(dims)}-skill pIRT calibration\nMAE = {mae:.4f} (n = {len(df)} models)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "pirt_calibration.png")

    a_by_dim = {dim: A[:, i][A[:, i] > 0] for i, dim in enumerate(dims)}
    amax = max((v.max() for v in a_by_dim.values() if v.size), default=1.0)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bins = np.linspace(0, amax + 0.1, 40)
    for dim in dims:
        v = a_by_dim[dim]
        if not v.size:
            continue
        ax.hist(v, bins=bins, alpha=0.55,
                label=f"a_{dim} (n={v.size}, med={np.median(v):.2f})",
                color=DIM_COLORS.get(dim, "#333333"))
    ax.set_xlabel("discrimination loading a (non-zero loadings only)")
    ax.set_ylabel("# items")
    ax.set_title(f"TutorBench {len(dims)}-skill item discrimination by skill axis")
    ax.legend()
    _save(fig, "item_info_by_skill.png")

    exp = ins["item_exposure"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xlabs = list(dims)
    cat_share = [exp["exposure_share_by_skill"][d] for d in dims]
    bank_share = [exp["bank_loading_share_by_skill"][d] for d in dims]
    x = np.arange(len(xlabs))
    w = 0.38
    ax.bar(x - w / 2, bank_share, width=w, label="bank availability (share of items)",
           color="#bbbbbb", edgecolor="k", linewidth=0.3)
    ax.bar(x + w / 2, cat_share, width=w, label="CAT selections (share of administered)",
           color=[DIM_COLORS.get(d, "#333333") for d in dims], edgecolor="k", linewidth=0.3)
    for xi, cs in zip(x, cat_share, strict=False):
        ax.text(xi + w / 2, cs + 0.01, f"{cs * 100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(xlabs)
    ax.set_ylabel("share of items loading skill")
    ax.set_title(f"TutorBench {len(dims)}-skill CAT: item-exposure by skill")
    ax.legend(fontsize=8)
    _save(fig, "item_exposure_by_skill.png")

    # per-skill items-administered (floor visibility)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    data = [df[f"exposure_{d}"].to_numpy() for d in dims]
    ax.boxplot(data, tick_labels=list(dims), showmeans=True)
    ax.set_ylabel("# administered items loading skill")
    ax.set_title(f"TutorBench {len(dims)}-skill CAT: items administered per skill (floor visibility)")
    _save(fig, "items_administered_by_skill.png")

    return paths


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def table_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def build_summary_table(agg, am, dims, cfg) -> pd.DataFrame:
    ins = agg["in_sample"]
    n_models = ins["cat_items"]["n_models"]

    def fmt_items(dim_key):
        c = ins["cat_items"][dim_key]
        if c["mean"] is None:
            return f"n/a (0/{n_models} conv.)"
        return f"{c['mean']:.1f} (median {c['median']:.0f}; {c['converged']}/{n_models} conv.)"

    rows = [{
        "Benchmark": f"TutorBench ({len(dims)}-skill, overall)",
        "Models": agg["n_models"],
        "a median": " / ".join(f"{d[:5]} {am[f'a_{d}_median_nonzero']:.2f}" for d in dims),
        "Recovery r": " / ".join(
            f"{d[:5]} " + (f"{ins['recovery_r'][d]:.3f}" if ins["recovery_r"][d] is not None else "n/a")
            for d in dims),
        "CAT items": fmt_items("overall"),
        "pIRT MAE": f"{ins['pirt_mae_cat']:.4f}",
    }]
    for dim in dims:
        rv = ins["recovery_r"][dim]
        ia = ins["items_administered_by_skill"][dim]
        rows.append({
            "Benchmark": f"  - {dim}",
            "Models": agg["n_models"],
            "a median": f"{am[f'a_{dim}_median_nonzero']:.3f} ({am[f'a_{dim}_n_loading']} loading)",
            "Recovery r": f"{rv:.3f}" if rv is not None else "n/a",
            "CAT items": fmt_items(dim),
            "pIRT MAE": (f"items-admin mean {ia['mean']:.1f} / min {ia['min']:.0f}"
                         if ia["mean"] is not None else "-"),
        })
    return pd.DataFrame(rows)


def build_before_after(before, after, before_oos, after_oos, dims) -> pd.DataFrame:
    def g(agg, oos, dim, key):
        if key == "rec_in":
            v = agg["in_sample"]["recovery_r"][dim]
        elif key == "rec_oos":
            v = oos["recovery_r"][dim] if oos else None
        elif key == "ia_mean":
            v = agg["in_sample"]["items_administered_by_skill"][dim]["mean"]
        elif key == "ia_min":
            v = agg["in_sample"]["items_administered_by_skill"][dim]["min"]
        elif key == "conv":
            c = agg["in_sample"]["cat_items"][dim]
            return f"{c['converged']}/{agg['n_models']}"
        return v

    rows = []
    for dim in dims:
        rows.append({
            "Skill": dim,
            "items-admin mean (before)": _fmt(g(before, before_oos, dim, "ia_mean")),
            "items-admin mean (after)": _fmt(g(after, after_oos, dim, "ia_mean")),
            "items-admin min (before)": _fmt(g(before, before_oos, dim, "ia_min"), 0),
            "items-admin min (after)": _fmt(g(after, after_oos, dim, "ia_min"), 0),
            "OOS recovery r (before)": _fmt(g(before, before_oos, dim, "rec_oos")),
            "OOS recovery r (after)": _fmt(g(after, after_oos, dim, "rec_oos")),
            "in-sample recovery r (before)": _fmt(g(before, before_oos, dim, "rec_in")),
            "in-sample recovery r (after)": _fmt(g(after, after_oos, dim, "rec_in")),
            "converged (before)": g(before, before_oos, dim, "conv"),
            "converged (after)": g(after, after_oos, dim, "conv"),
        })
    return pd.DataFrame(rows)


def _fmt(v, ndigits: int = 3) -> str:
    if v is None:
        return "n/a"
    if ndigits == 0:
        return f"{v:.0f}"
    return f"{v:.{ndigits}f}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--skills", "--skill-set", dest="skills", type=int, choices=(2, 3),
                   required=True, help="Which TutorBench CAT instrument to run.")
    p.add_argument("--bank", type=Path, default=None)
    p.add_argument("--matrix", type=Path, default=None)
    p.add_argument("--kfold-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--curated", type=Path, default=DEFAULT_CURATED,
                   help="curated bank carrying the exclude_from_fit flag (60 items).")
    p.add_argument("--grid", type=int, default=7,
                   help="Gauss-Hermite nodes per dim (default 7).")
    p.add_argument("--se-target", type=float, default=0.3,
                   help="stop when ALL per-dim SEs < this (after --min-items and floors).")
    p.add_argument("--min-items", type=int, default=1, help="global minimum items before any stop.")
    p.add_argument("--max-items", type=int, default=100, help="hard cap on adaptive items.")
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--no-oos", action="store_true", help="skip the OOS k-fold check.")
    # --- policy knobs ---
    p.add_argument("--legacy", action="store_true",
                   help="reproduce the OLD (deceptive) behaviour: all policy knobs off.")
    p.add_argument("--no-pbis-filter", dest="pbis_filter", action="store_false")
    p.add_argument("--min-point-biserial", type=float, default=0.05)
    p.add_argument("--min-items-per-skill", type=int, default=15,
                   help="per-skill administered-item floor before a skill may converge.")
    p.add_argument("--no-content-balance", dest="content_balance", action="store_false")
    p.add_argument("--exposure-top-n", type=int, default=5,
                   help="pick uniformly among the top-N most-informative candidates.")
    p.add_argument("--compare", action="store_true",
                   help="run BOTH legacy (before) and policy (after) and emit a before/after report.")
    p.set_defaults(pbis_filter=True, content_balance=True)
    args = p.parse_args()

    cfg = skillset_config(args.skills)
    dims = cfg["dims"]
    n_dims = len(dims)
    bank_path = args.bank or cfg["bank"]
    matrix_path = args.matrix or cfg["matrix"]
    cfg["kfold_dir"] = args.kfold_dir or cfg["kfold_dir"]
    out_dir = args.out_dir or (ROOT / "reports" / "cat_eval_policy" / f"{args.skills}skill")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    policy = Policy(
        pbis_filter=args.pbis_filter,
        min_point_biserial=args.min_point_biserial,
        min_items_per_skill=args.min_items_per_skill,
        content_balance=args.content_balance,
        exposure_top_n=args.exposure_top_n,
    )
    if args.legacy:
        policy = Policy.legacy()

    print("=" * 78)
    print(f"CAT policy evaluation - TutorBench {n_dims}-skill {list(dims)}")
    print("=" * 78)

    # --- bank + exclusion ---
    items, A, b = load_bank(bank_path, cfg["a_cols"])
    n_bank_raw = len(items)
    excluded = load_excluded_criteria(args.curated)
    excl_present = [it for it in items if it in excluded]
    keep_mask = np.array([it not in excluded for it in items])
    items = [it for it, k in zip(items, keep_mask, strict=False) if k]
    A = A[keep_mask]
    b = b[keep_mask]
    print(f"bank: {n_bank_raw} fitted items; excluded_from_fit flagged={len(excluded)}, "
          f"present-in-bank & removed={len(excl_present)} -> {len(items)} eligible")

    # --- matrix ---
    mat = cp.load_matrix(matrix_path)
    models = list(mat.index)
    sub = mat.reindex(columns=items)
    Yraw = sub.to_numpy(dtype=float)

    # --- point-biserial filter (policy) ---
    keep_ids = None
    pbis_report = None
    if policy.pbis_filter:
        keep, pbis_report = apply_pbis_filter(Yraw, items, policy.min_point_biserial)
        keep_ids = {it for it, k in zip(items, keep, strict=False) if k}
        items = [it for it, k in zip(items, keep, strict=False) if k]
        A = A[keep]
        b = b[keep]
        Yraw = Yraw[:, keep]
        print(f"point-biserial filter (>= {policy.min_point_biserial}): "
              f"dropped all_pass={pbis_report['n_all_pass']}, all_fail={pbis_report['n_all_fail']}, "
              f"low_pbis={pbis_report['n_low_point_biserial']} -> {len(items)} eligible")

    M = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    print(f"matrix: {len(models)} models x {len(items)} bank items "
          f"({int(M.sum())} observed cells, {M.mean() * 100:.1f}% filled)")
    print(f"loadings: " + ", ".join(f"{d}={(A[:, i] > 0).sum()}" for i, d in enumerate(dims)))

    grid = cm.build_grid(n_dims, args.grid)
    base_logw = cm.base_log_weights(n_dims, args.grid)
    log_prior = cm.prior_log_weights(grid, base_logw, np.eye(n_dims))

    am = a_medians(A, dims)
    bank_info = {
        "n_bank_raw": n_bank_raw,
        "n_excluded_flagged": len(excluded),
        "n_excluded_removed_from_bank": len(excl_present),
        "excluded_present_ids_sample": sorted(excl_present)[:10],
        "n_eligible": len(items),
        "pbis_filter": pbis_report,
        "items_kept_ref": cfg["items_kept"],
        "items_total_ref": cfg["items_total"],
        "bank_health": cfg["bank_health"],
        **am,
    }

    # --- passes ---
    if args.compare:
        legacy_pol = Policy.legacy()
        # legacy must NOT pbis-filter; re-load the (unfiltered-but-exclusion-applied) bank
        items_l, A_l, b_l = load_bank(bank_path, cfg["a_cols"])
        km = np.array([it not in excluded for it in items_l])
        items_l = [it for it, k in zip(items_l, km, strict=False) if k]
        A_l, b_l = A_l[km], b_l[km]
        sub_l = mat.reindex(columns=items_l)
        Yraw_l = sub_l.to_numpy(dtype=float)
        M_l = ~np.isnan(Yraw_l)
        Y_l = np.nan_to_num(Yraw_l, nan=0.0)
        df_b, agg_b, oosdf_b, oos_b = run_pass(
            "BEFORE/legacy", items_l, A_l, b_l, dims, mat, models, Y_l, M_l,
            grid, log_prior, cfg, legacy_pol, args, excluded, None)
        df_a, agg_a, oosdf_a, oos_a = run_pass(
            "AFTER/policy", items, A, b, dims, mat, models, Y, M,
            grid, log_prior, cfg, policy, args, excluded, keep_ids)
        ba = build_before_after(agg_b, agg_a, oos_b, oos_a, dims)
        ba.to_csv(out_dir / "before_after_by_skill.csv", index=False)
        with (out_dir / "before_after_by_skill.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# TutorBench {n_dims}-skill CAT: before (legacy) vs after (policy)\n\n")
            fh.write(table_to_markdown(ba) + "\n")
        print("\nBEFORE/AFTER by skill:\n")
        print(table_to_markdown(ba))
        # headline outputs are the AFTER (policy) run
        df, agg, oos_df, oos = df_a, agg_a, oosdf_a, oos_a
        agg["before_legacy"] = {"in_sample": agg_b["in_sample"],
                                "out_of_sample": agg_b.get("out_of_sample")}
    else:
        df, agg, oos_df, oos = run_pass(
            "run", items, A, b, dims, mat, models, Y, M,
            grid, log_prior, cfg, policy, args, excluded, keep_ids)

    # --- write per-model CSVs ---
    csv_cols = [c for c in df.columns if c not in ("se_trace", "theta_trace", "administered")]
    df[csv_cols].to_csv(out_dir / "cat_per_model.csv", index=False)
    if oos_df is not None:
        oos_df.to_csv(out_dir / "cat_per_model_oos.csv", index=False)

    # --- figures ---
    print("\ngenerating figures ...")
    fig_paths = make_figures(df, agg, A, dims, out_dir / "figures", args.se_target, args.max_items)

    # --- metrics JSON ---
    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "skills": args.skills,
        "dims": list(dims),
        "config": {
            "bank": str(bank_path), "matrix": str(matrix_path),
            "kfold_dir": str(cfg["kfold_dir"]), "curated": str(args.curated),
            "grid": args.grid, "se_target": args.se_target, "min_items": args.min_items,
            "max_items": args.max_items, "seed": args.seed, "compare": args.compare,
        },
        "policy": policy.to_dict(),
        "bank": bank_info,
        "n_models": agg["n_models"],
        **agg,
        "figures": fig_paths,
    }
    with (out_dir / "cat_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # --- summary table ---
    table_df = build_summary_table(agg, am, dims, cfg)
    table_df.to_csv(out_dir / "cat_summary_table.csv", index=False)
    md = table_to_markdown(table_df)
    with (out_dir / "cat_summary_table.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# TutorBench {n_dims}-skill CAT summary table (policy ON)\n\n")
        fh.write(md + "\n")
        if oos is not None:
            fh.write("\n## Out-of-sample (k-fold fold-trained params) recovery\n\n")
            for d in dims:
                rv = oos["recovery_r"][d]
                fh.write(f"- Recovery r ({d}): " + (f"{rv:.3f}\n" if rv is not None else "n/a\n"))

    # --- console summary ---
    ins = agg["in_sample"]
    print("\n" + "=" * 78)
    print("RESULTS (in-sample headline; policy ON)" if not args.legacy else "RESULTS (legacy)")
    print("=" * 78)
    print("Recovery r  : " + "  ".join(
        f"{d}=" + (f"{ins['recovery_r'][d]:.3f}" if ins["recovery_r"][d] is not None else "n/a")
        for d in dims))
    for d in dims:
        c = ins["cat_items"][d]
        ia = ins["items_administered_by_skill"][d]
        mv = f"{c['mean']:.1f}" if c["mean"] is not None else "n/a"
        print(f"  {d:12s}: converge#={mv} (median {c['median']}, {c['converged']}/{agg['n_models']} conv.)"
              f"  items-admin mean/min = {ia['mean']:.1f}/{ia['min']:.0f}")
    print(f"pIRT MAE    : CAT={ins['pirt_mae_cat']:.4f}  full-bank={ins['pirt_mae_full']:.4f}")
    pop = ins["item_exposure_population"]
    print(f"exposure    : items-ever-used={pop['n_items_ever_administered']}, "
          f"max exposure rate={pop['max_exposure_rate']:.2f}")
    if oos is not None:
        print("OOS recovery: " + "  ".join(
            f"{d}=" + (f"{oos['recovery_r'][d]:.3f}" if oos["recovery_r"][d] is not None else "n/a")
            for d in dims))

    print(f"\nwrote outputs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
