"""LOCAL STUDY export: materialise the of-record InfoBench unidimensional 2PL
fitted-parameter bank as a persisted per-criterion JSONL.

InfoBench ships no persisted per-item fitted-param bank: the of-record OOS grid
(``run_oos_grid.py``) refits the log-shrinkage-2PL bank in-process from the frozen
response matrix and never writes the item parameters out. This script performs the
SAME fit ONCE on the FULL data (all 52 models, no fold split), reusing the exact
``run_oos_grid.py`` fit path (``scripts.calibrate_mirt.fit_m2pl_em`` with the
documented ``log_shrinkage_2pl_lambda16`` / 401-node normal_trapezoid config), and
exports the resulting (a, b) per criterion in the shared fitted-bank JSONL schema
(matching ``data/TutorEval/rubrics_qmatrix_final_unidim_fitted.jsonl``).

It then reproduces the grid's of-record ability estimates by recomputing full-bank
EAP theta per model from the exported params and correlating against the grid's
per-model ``theta_ref`` (``reports/infobench_oos_grid_unidim/oos_per_model_per_cell.csv``).

LOCAL ONLY. Does not touch the production engine (``tutor_cat/``,
``scripts/scenario_cat_lib.py``) or the CAT calibration inputs; writes exactly one new
file: ``eduLLM-Evals/infobench_calibration/bank/rubrics_qmatrix_instruction_following_unidim_fitted.jsonl``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EE = HERE.parents[1]  # eduLLM-Evals
sys.path.insert(0, str(EE))
import scripts.calibrate_mirt as cm  # noqa: E402
import scripts.scenario_cat_lib as scat  # noqa: E402

# --------------------------------------------------------------------------- config
DIM = "instruction_following"
BANK_DIR = EE / "infobench_calibration" / "bank"
MATRIX_PATH = BANK_DIR / "response_matrix.csv"
RUBRICS_PATH = EE / "data" / "InFoBench" / "rubrics.jsonl"
OUT_PATH = BANK_DIR / "rubrics_qmatrix_instruction_following_unidim_fitted.jsonl"
GRID_PERMODEL = EE / "reports" / "infobench_oos_grid_unidim" / "oos_per_model_per_cell.csv"

EXPECTED_MATRIX_SHA_PREFIX = "087948fc"

# of-record fit (configs/infobench_calibration_cat_2pl_only_v1.json, spec
# log_shrinkage_2pl_lambda16) -- identical to run_oos_grid.py's per-fold refit config.
FIT_MODEL = cm.LOG_SHRINKAGE_2PL
LOG_A_SHRINK = 16.0            # lambda16 (log_a_prior_sd = 0.25 => 1/0.25^2 = 16)
LOG_A_PRIOR_SD = 0.25
FIT_GRID = 401                 # numerical_lock.fit_grid
FIT_QUAD = "normal_trapezoid"
FIT_BOUND = 8.0
FIT_MAX_ITER = 200
FIT_TOL = 1e-4
FIT_PARAM_TOL = 5e-5
FIT_PASSES = 2
EXPORT_MAX_A = 6.0            # export policy: drop nonpositive / extreme discriminations

# reproduction EAP grid (of-record eap_grid = 801 normal_trapezoid, bound 8)
EAP_NODES = 801
EAP_BOUND = 8.0

# metadata fields carried over verbatim from rubrics.jsonl (the synthetic
# difficulty/discrimination/irt_params are dropped and replaced by the fit).
CARRY_FIELDS = (
    "scenario_id", "criterion", "expected_evidence", "score_anchors",
    "scoring_type", "question_label", "primary_skill", "q_mapping",
    "q_rationale", "criticality", "objectivity", "explicitness", "source",
    "status", "version",
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def variant_cols(Y, M):
    """Same zero-variance filter run_oos_grid.py uses before the refit."""
    succ = np.where(M, Y, 0.0).sum(0)
    cnt = M.sum(0)
    return (cnt >= 2) & (succ > 0) & (succ < cnt)


def main() -> None:
    # ---- load frozen calibration input matrix (verify provenance) ----
    matrix_sha = sha256_of(MATRIX_PATH)
    if not matrix_sha.startswith(EXPECTED_MATRIX_SHA_PREFIX):
        raise SystemExit(
            f"matrix sha {matrix_sha[:12]} != expected {EXPECTED_MATRIX_SHA_PREFIX}...; refusing to fit")
    matrix = pd.read_csv(MATRIX_PATH, index_col=0).apply(pd.to_numeric, errors="coerce")
    ids = list(matrix.columns)
    models = list(matrix.index)
    P = len(models)
    Yall = np.nan_to_num(matrix.to_numpy(float), nan=0.0)
    Mall = np.isfinite(matrix.to_numpy(float))
    print(f"[data] {P} models x {len(ids)} criteria; fill={Mall.mean():.6f}; "
          f"matrix_sha={matrix_sha[:16]}", flush=True)

    # ---- single FULL-DATA fit (all 52 models; NOT per-fold) ----
    keep = variant_cols(Yall, Mall)
    src = np.where(keep)[0]
    n_zero_variance = int((~keep).sum())
    Q = np.ones((int(keep.sum()), 1), dtype=int)
    print(f"[fit] full-data log-shrinkage-2pl lambda16, {FIT_GRID}-node {FIT_QUAD}; "
          f"{int(keep.sum())} variant of {len(ids)} criteria ...", flush=True)
    fit = cm.fit_m2pl_em(
        Yall[:, keep], Mall[:, keep], Q, FIT_GRID,
        estimate_corr=False, ridge=0.0, max_iter=FIT_MAX_ITER, tol=FIT_TOL,
        calibration_model=FIT_MODEL, log_a_shrinkage=LOG_A_SHRINK,
        quadrature_method=FIT_QUAD, linear_bound=FIT_BOUND,
        convergence_mode="returned_iterate", parameter_tol=FIT_PARAM_TOL,
        consecutive_convergence_passes=FIT_PASSES,
    )
    a_all = np.asarray(fit["A"], float).ravel()
    b_all = np.asarray(fit["b"], float).ravel()
    converged = bool(fit["converged"])
    n_iter = int(fit["n_iter"])

    # ---- export policy: drop a<=0 (nonpositive) or a>6 (extreme) ----
    nonpositive = a_all <= 0.0
    extreme = a_all > EXPORT_MAX_A
    exp_keep = (~nonpositive) & (~extreme)
    kept_cols = src[exp_keep]
    kept_ids = [ids[c] for c in kept_cols]
    a_kept = a_all[exp_keep]
    b_kept = b_all[exp_keep]
    n_kept = int(exp_keep.sum())
    n_drop_nonpos = int(nonpositive.sum())
    n_drop_extreme = int(extreme.sum())
    max_a = float(a_kept.max())
    print(f"[fit] converged={converged} n_iter={n_iter} | kept={n_kept} "
          f"(dropped: zero-variance={n_zero_variance}, a<=0={n_drop_nonpos}, "
          f"a>{EXPORT_MAX_A}={n_drop_extreme}) | max_a={max_a:.4f}", flush=True)

    # ---- reproduction: full-bank EAP theta per model vs grid theta_ref ----
    quad = scat.build_quadrature(1, EAP_NODES, np.eye(1), max_nodes=2000,
                                 method="normal_trapezoid", linear_bound=EAP_BOUND)
    A_kept = a_kept[:, None]
    theta_hat = {}
    for m in models:
        resp = pd.to_numeric(matrix.loc[m].reindex(kept_ids), errors="coerce").to_numpy(float)
        theta_hat[m] = float(scat.batch_eap(resp, A_kept, b_kept, quad).theta[0])

    grid_df = pd.read_csv(GRID_PERMODEL)
    theta_ref = grid_df.drop_duplicates("model").set_index("model")["theta_ref"].to_dict()
    common = [m for m in models if m in theta_ref]
    x = np.array([theta_ref[m] for m in common])
    y = np.array([theta_hat[m] for m in common])
    repro_r = float(np.corrcoef(x, y)[0, 1])
    repro_slope = float(np.polyfit(x, y, 1)[0])
    repro_mae = float(np.mean(np.abs(y - x)))
    print(f"[repro] full-bank EAP theta vs grid theta_ref over {len(common)} models: "
          f"r={repro_r:.6f} slope={repro_slope:.4f} MAE={repro_mae:.4f}", flush=True)

    # ---- provenance block ----
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    provenance = {
        "benchmark": "InFoBench",
        "exported_by": "eduLLM-Evals/infobench_calibration/scripts/build_infobench_unidim_bank.py",
        "date": now,
        "axis": "unidim",
        "matrix_csv": MATRIX_PATH.relative_to(EE).as_posix(),
        "matrix_sha256": matrix_sha,
        "fit_config": {
            "family": "log-shrinkage-2pl",
            "spec_id": "log_shrinkage_2pl_lambda16",
            "log_a_shrinkage": LOG_A_SHRINK,
            "log_a_prior_sd": LOG_A_PRIOR_SD,
            "fit_grid": FIT_GRID,
            "fit_quadrature": FIT_QUAD,
            "linear_bound": FIT_BOUND,
            "convergence_mode": "returned_iterate",
            "max_iter": FIT_MAX_ITER,
            "objective_tol": FIT_TOL,
            "parameter_tol": FIT_PARAM_TOL,
            "consecutive_convergence_passes": FIT_PASSES,
            "export_max_a": EXPORT_MAX_A,
            "negative_loading_policy": "drop nonpositive / a>6 at export",
        },
        "fit_result": {
            "converged": converged,
            "n_iter": n_iter,
            "n_source_criteria": len(ids),
            "n_zero_variance_dropped": n_zero_variance,
            "n_nonpositive_dropped": n_drop_nonpos,
            "n_extreme_dropped": n_drop_extreme,
            "n_kept": n_kept,
            "max_a": max_a,
        },
        "reproduction": {
            "reference": "run_oos_grid.py per-fold theta_ref (full fold-bank EAP; "
                         "reports/infobench_oos_grid_unidim/oos_per_model_per_cell.csv)",
            "method": "full-bank EAP (801-node normal_trapezoid, bound 8) on exported full-data params",
            "n_models": len(common),
            "pearson_r": repro_r,
            "slope": repro_slope,
            "theta_mae": repro_mae,
        },
        "low_n": True,
        "identifiability_caveat": (
            "N=52 persons < min_persons_identifiable=150; multi-dim loadings are unstable "
            "at this sample size, which is why the canonical InfoBench scale is UNIDIMENSIONAL "
            "(instruction_following)."),
        "note": (
            "Single full-data re-run of the of-record run_oos_grid.py in-process "
            "log-shrinkage-2PL bank (per-fold there; full-data here); item params not "
            "persisted by that grid so materialised here. LOCAL study; production engine untouched."),
    }

    # ---- load rubric metadata ----
    rub = {}
    with open(RUBRICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rub[r["criterion_id"]] = r

    # ---- write bank ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_missing_meta = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for cid, a, b in zip(kept_ids, a_kept.tolist(), b_kept.tolist()):
            meta = rub.get(cid)
            if meta is None:
                n_missing_meta += 1
                rec = {"criterion_id": cid, "scenario_id": cid.rsplit("_c", 1)[0]}
            else:
                rec = {"criterion_id": cid}
                for k in CARRY_FIELDS:
                    if k in meta:
                        rec[k] = meta[k]
            rec["difficulty"] = b
            rec["discrimination"] = {DIM: a}
            rec["q_modeled"] = {DIM: 1}
            rec["irt_params"] = {
                "source": "calibrated-2pl-unidim-infobench",
                "method": "log-shrinkage-2pl-mml-em",
                "calibrated": True,
                "fitted": True,
                "modeled_skills": [DIM],
                "n_persons": P,
                "flags": ["low_n"],
                "version": "1.0",
                "provenance": provenance,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_sha = sha256_of(OUT_PATH)
    print(f"[done] wrote {n_kept} criteria -> {OUT_PATH.relative_to(EE)}", flush=True)
    print(f"[done] out_sha256={out_sha}", flush=True)
    if n_missing_meta:
        print(f"[warn] {n_missing_meta} kept criteria had no rubric metadata", flush=True)

    print(json.dumps({
        "n_criteria_kept": n_kept,
        "n_dropped_total": len(ids) - n_kept,
        "n_zero_variance": n_zero_variance,
        "n_nonpositive": n_drop_nonpos,
        "n_extreme": n_drop_extreme,
        "max_a": max_a,
        "reproduction_r": repro_r,
        "reproduction_slope": repro_slope,
        "reproduction_theta_mae": repro_mae,
        "converged": converged,
        "n_iter": n_iter,
    }, indent=2))


if __name__ == "__main__":
    main()
