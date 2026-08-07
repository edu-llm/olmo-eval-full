"""Build a CORRECTNESS-ONLY UNIDIMENSIONAL 2PL bank for TutorBench (115 models).

STUDY-ONLY / LOCAL. Reuses (read-only) the exact fitter internals from
``scripts/calibrate_mirt.py`` (``fit_m2pl_em``) with the SAME config that produced
the collapse-all unidim bank (1 dim, fit_grid=7, ridge=1e-2). The production engine
(``tutor_cat/``, ``scenario_cat_lib.py``) is never modified.

Difference vs the collapse-all unidim bank: instead of keeping ALL 3666 fitted items
on one axis, we DROP every scaffolding-loading criterion and fit a unidimensional 2PL
on the correctness-loading criteria ONLY. Source of the modeled Q (which skill each
criterion loads on) is the of-record 2-skill fitted bank ``q_modeled``.

Outputs (into this report folder):
  * rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl  -- the 1-D bank (dim="correctness")
  * bank_fit_summary.json                                     -- kept/dropped, clamped-neg, max-a, loglik/AIC/BIC
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # eduLLM-Evals
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cp = cm.cp

MATRIX = ROOT / "staging" / "response_matrix_full_nonopt_115.csv"
SOURCE_BANK = ROOT / "data" / "TutorBench" / "rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl"

OUT_BANK = HERE / "rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl"
OUT_SUMMARY = HERE / "bank_fit_summary.json"

DIM = "correctness"
GRID = 7
RIDGE = 1e-2
MAX_ITER = 200
TOL = 1e-4


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def main() -> int:
    mat_sha = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    print(f"loading matrix {MATRIX.name} (sha256={mat_sha[:24]}...)")
    mat = pd.read_csv(MATRIX, index_col=0)
    n_models = mat.shape[0]
    print(f"matrix: {n_models} models x {mat.shape[1]} criteria")

    records = read_jsonl(SOURCE_BANK)
    n_source = len(records)

    # classify by MODELED Q (from the of-record 2-skill fit)
    keep_cids: list[str] = []
    n_scaff_only = n_both = n_corr_only = n_other = 0
    for r in records:
        qm = r["q_modeled"]
        qc = int(qm.get("correctness", 0))
        qs = int(qm.get("scaffolding", 0))
        if qc == 1 and qs == 0:
            n_corr_only += 1
            keep_cids.append(r["criterion_id"])
        elif qc == 0 and qs == 1:
            n_scaff_only += 1
        elif qc == 1 and qs == 1:
            n_both += 1
        else:
            n_other += 1
    n_dropped = n_source - n_corr_only
    print(f"source bank: {n_source} criteria")
    print(f"  correctness-only (kept)      : {n_corr_only}")
    print(f"  scaffolding-only (dropped)   : {n_scaff_only}")
    print(f"  correctness+scaffolding (dropped): {n_both}")
    print(f"  neither/other (dropped)      : {n_other}")
    print(f"  TOTAL dropped (any scaffolding load): {n_dropped}")

    meta = {r["criterion_id"]: r for r in records}

    # restrict matrix to kept criteria present in the matrix
    present = [c for c in keep_cids if c in mat.columns]
    n_missing_matrix = len(keep_cids) - len(present)
    sub = mat[present]

    # within-set zero-variance filter: >=2 observations AND both a pass and a fail
    kept_items: list[str] = []
    for c in present:
        col = sub[c].to_numpy(dtype=float)
        obs = ~np.isnan(col)
        nobs = int(obs.sum())
        if nobs >= 2:
            s = float(np.nansum(col[obs]))
            if 0.0 < s < nobs:
                kept_items.append(c)
    n_zero_var_dropped = len(present) - len(kept_items)
    sub = sub[kept_items]
    # drop persons with no observation across the kept set
    sub = sub[sub.notna().any(axis=1)]
    Y = np.nan_to_num(sub.to_numpy(dtype=float), nan=0.0)
    M = sub.notna().to_numpy()
    n_items = len(kept_items)
    n_persons = Y.shape[0]
    n_obs = int(M.sum())
    print(f"\nfit block: {n_items} items x {n_persons} persons, {n_obs} observed cells")
    print(f"  (dropped {n_missing_matrix} kept-cids absent from matrix, "
          f"{n_zero_var_dropped} zero-variance within correctness-only set)")

    print("\nfitting UNIDIMENSIONAL 2PL (grid 7, 1 dim, ridge 1e-2, clamp neg at runtime) ...",
          flush=True)
    Q_uni = np.ones((n_items, 1), dtype=int)
    fit = cm.fit_m2pl_em(Y, M, Q_uni, GRID, estimate_corr=False,
                         ridge=RIDGE, max_iter=MAX_ITER, tol=TOL)
    aic, bic = cm.aic_bic(fit["loglik"], fit["n_params"], n_obs)
    A = fit["A"][:, 0]
    b = fit["b"]
    n_persons_item = M.sum(axis=0).astype(int)

    n_neg = int(np.sum(A < 0))
    n_extreme = int(np.sum(np.abs(A) > cm.EXTREME_A))
    print(f"  loglik={fit['loglik']:.4f} k={fit['n_params']} AIC={aic:.2f} BIC={bic:.2f} "
          f"iters={fit['n_iter']} converged={fit['converged']}")
    print(f"  negative loadings (clamped to 0 at runtime): {n_neg} / {n_items}")
    print(f"  max discrimination a = {float(A.max()):.4f}  "
          f"(min a = {float(A.min()):.4f}, {n_extreme} extreme |a|>{cm.EXTREME_A})")

    provenance = {
        "matrix_csv": "staging/response_matrix_full_nonopt_115.csv",
        "matrix_sha256": mat_sha,
        "source_bank": str(SOURCE_BANK.relative_to(ROOT)),
        "q_source": "2-skill fitted bank q_modeled (correctness OR scaffolding)",
        "selection": "keep q_modeled.correctness==1 AND q_modeled.scaffolding==0; "
                     "drop every scaffolding-loading criterion",
        "ridge": RIDGE,
        "grid_nodes_per_dim": GRID,
        "n_persons": n_persons,
        "n_items_fit": n_items,
        "negative_loadings": "preserved in bank (NOT floored); clamp policy applied by consumer",
        "note": "CORRECTNESS-ONLY unidimensional 2PL; scaffolding criteria removed before the fit "
                "(cleaner variant of the collapse-all unidim bank).",
        "study_only": "LOCAL side-experiment; not committed; engine untouched.",
    }

    with OUT_BANK.open("w", encoding="utf-8") as fh:
        for j, cid in enumerate(kept_items):
            src = meta[cid]
            rec = {
                "criterion_id": cid,
                "scenario_id": src["scenario_id"],
                "criterion": src.get("criterion", ""),
                "primary_skill": src.get("primary_skill", ""),
                "q_mapping": src.get("q_mapping"),
                "scoring_type": src.get("scoring_type", "binary"),
                "criticality": src.get("criticality", "standard"),
                "status": src.get("status", "approved"),
                "difficulty": round(float(b[j]), 6),
                "discrimination": {DIM: round(float(A[j]), 6)},
                "q_modeled": {DIM: 1},
                "irt_params": {
                    "source": "calibrated-m2pl-correctness-only-unidim-115",
                    "method": "confirmatory-m2pl-mml-em",
                    "calibrated": True,
                    "fitted": True,
                    "modeled_skills": [DIM],
                    "n_persons": int(n_persons_item[j]),
                    "flags": (["extreme_a"] if abs(float(A[j])) > cm.EXTREME_A else []),
                    "version": "1.0",
                    "provenance": provenance,
                },
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote correctness-only unidim bank -> {OUT_BANK} ({n_items} criteria)")

    summary = {
        "generated_at": cp._utcnow(),
        "status": "STUDY ONLY - local; engine untouched; nothing committed.",
        "dim": DIM,
        "source_bank": str(SOURCE_BANK.relative_to(ROOT)),
        "matrix": "staging/response_matrix_full_nonopt_115.csv",
        "matrix_sha256": mat_sha,
        "n_models": n_models,
        "criteria_classification": {
            "n_source": n_source,
            "kept_correctness_only": n_corr_only,
            "dropped_scaffolding_only": n_scaff_only,
            "dropped_correctness_and_scaffolding": n_both,
            "dropped_other": n_other,
            "total_dropped_any_scaffolding_load": n_dropped,
        },
        "fit_block": {
            "n_items_fit": n_items,
            "n_persons_fit": n_persons,
            "n_observed_cells": n_obs,
            "dropped_absent_from_matrix": n_missing_matrix,
            "dropped_zero_variance_within_set": n_zero_var_dropped,
        },
        "fit_config": {"fitter": "calibrate_mirt.fit_m2pl_em (1 dim)",
                       "grid_nodes_per_dim": GRID, "ridge": RIDGE,
                       "max_iter": MAX_ITER, "tol": TOL,
                       "negative_policy": "clamp (at consume time)"},
        "fit_result": {
            "loglik": float(fit["loglik"]), "n_params": int(fit["n_params"]),
            "aic": float(aic), "bic": float(bic),
            "n_iter": int(fit["n_iter"]), "converged": bool(fit["converged"]),
            "n_negative_loadings_clamped": n_neg,
            "n_extreme_a": n_extreme,
            "max_discrimination": float(A.max()),
            "min_discrimination": float(A.min()),
            "median_discrimination": float(np.median(A)),
        },
        "compare_2skill_fit_negatives": {
            "2skill_clamped_negatives": 349,
            "note": "2-skill fit clamped 349 negative fitted loadings (per of-record bank provenance).",
        },
        "bank_out": str(OUT_BANK.relative_to(ROOT)),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote fit summary -> {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
