"""Phase-2 Step A: CURATE + REFIT the BiGGen gemini-3 UNIDIMENSIONAL bank.

LOCAL scratch driver (reuses ``recalibrate_gemini3.py`` machinery). It CALLS the frozen
production engine (calibrate_mirt, scenario_cat_lib, biggen_recovery_grid) -- it never edits
it -- reusing the EXACT locked production config that produced the recal bank:

  * unidimensional ("general"), Q = ones
  * fitter = calibrate_mirt.fit_m2pl_em (Bock-Aitkin MML-EM), 7 GH nodes/dim, ridge=0.01,
    max_iter=200
  * deployed POINT estimator for theta = FULL-BANK fine-grid EAP posterior mean (3201 nodes
    over [-8,8], standard-normal prior) -- scat.eap_all_models on the CURATED bank
  * SE_param via the observed-information parametric bootstrap (biggen_recovery_grid); SE_ability
    = full-bank EAP posterior SD; SE_total = sqrt(SE_ability^2 + SE_param^2)

CURATION (decided): exclude matrix columns whose OBSERVED pass_count <= 3 OR >= n_obs-3
(== ">=49 of 52" on the fully-observed matrix). This drops the pass-imbalance / near-separation
columns that drove the un-curated EM ridge-collapse (identical (a,b) atoms, |b| up to ~20). The
retained columns are fitted AND become the administrable pool. The two clean high-a spread items
(bgb_0202_c01 32/52, bgb_0047_c01 34/52) are well inside the band and are RETAINED.

Writes ONLY under eduLLM-Evals/reports/biggen_gemini3_recal/ (curated bank + scratch).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr

# eduLLM-Evals root = .../reports/biggen_gemini3_recal/scratch/this -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.scenario_cat_lib as scat  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
rg = _load("biggen_recovery_grid", ROOT / "biggen_calibration" / "scripts" / "biggen_recovery_grid.py")

DIM = "general"
JUDGE = "gemini-3-flash-preview"
FIT_GRID = 7
RIDGE = 1e-2
EAP_GRID = 3201
EAP_RANGE = 8.0
N_BOOT = 150
PU_SEED = 20260801
BOOT_GRID_N = 1201
IMBALANCE_MARGIN = 3          # exclude if pass_count <= MARGIN or fail_count <= MARGIN

MATRIX = ROOT / "api_judge_pilot" / "grading_biggen" / "response_matrix.csv"
RUBRICS = ROOT / "data" / "BiGGen" / "rubrics.jsonl"

OUT = ROOT / "reports" / "biggen_gemini3_recal"
BANK_UNCURATED = OUT / "bank" / "biggen_unidim_modeled_gemini3.jsonl"
BANK_OUT = OUT / "bank" / "biggen_unidim_modeled_gemini3_curated.jsonl"
SCRATCH = OUT / "scratch"
RECAL_PER_MODEL = SCRATCH / "per_model_gemini3.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    BANK_OUT.parent.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(MATRIX, index_col=0)
    models = list(matrix.index)
    all_ids = list(matrix.columns)
    Yall = matrix.to_numpy(float)
    Mall = ~np.isnan(Yall)
    Yall = np.nan_to_num(Yall, nan=0.0)
    print(f"matrix: {len(models)} models x {len(all_ids)} criteria (fill={Mall.mean():.4f})")

    meta = {r["criterion_id"]: r for r in read_jsonl(RUBRICS)}

    # ---- CURATION: pass-imbalance exclusion (observed counts) ----
    pass_count = np.where(Mall, Yall, 0.0).sum(axis=0).astype(int)
    n_obs = Mall.sum(axis=0).astype(int)
    fail_count = (n_obs - pass_count).astype(int)
    # keep only columns with enough spread in BOTH directions (and >=2 obs)
    keep_mask = (n_obs >= 2) & (pass_count > IMBALANCE_MARGIN) & (fail_count > IMBALANCE_MARGIN)
    keep_idx = np.where(keep_mask)[0]
    excluded = [all_ids[j] for j in np.where(~keep_mask)[0]]
    kept_ids = [all_ids[j] for j in keep_idx]
    Yk = Yall[:, keep_idx]
    Mk = Mall[:, keep_idx]
    Q = np.ones((len(kept_ids), 1), dtype=int)

    n_allfail = int((pass_count == 0).sum())
    n_allpass = int((fail_count == 0).sum())
    print(f"CURATION (pass_count<={IMBALANCE_MARGIN} or fail_count<={IMBALANCE_MARGIN}): "
          f"excluded {len(excluded)} / {len(all_ids)}  (all-fail={n_allfail}, all-pass={n_allpass})  "
          f"-> retained {len(kept_ids)}")
    for cid, want in (("bgb_0202_c01", 32), ("bgb_0047_c01", 34)):
        if cid in set(all_ids):
            j = all_ids.index(cid)
            print(f"  clean high-a check: {cid} pass={pass_count[j]}/{n_obs[j]} "
                  f"RETAINED={cid in set(kept_ids)}")

    # ---- unidim fit on the RETAINED columns (EXACT production config) ----
    print(f"fitting {len(kept_ids)} criteria x {len(models)} models, {FIT_GRID} GH nodes, "
          f"ridge={RIDGE}", flush=True)
    fit = cm.fit_m2pl_em(Yk, Mk, Q, FIT_GRID, estimate_corr=False, ridge=RIDGE, max_iter=200)
    A = fit["A"][:, 0]
    b = fit["b"]
    n_b6 = int((np.abs(b) > 6).sum())
    print(f"fit converged={fit['converged']} n_iter={fit['n_iter']} loglik={fit['loglik']:.1f} "
          f"max_a={A.max():.4f} min_a={A.min():.4f} |b|max={np.abs(b).max():.3f} |b|>6 count={n_b6}")

    matrix_sha = sha256(MATRIX)

    prov = {
        "source": "calibrate_mirt.fit_m2pl_em", "skill": DIM, "ridge": RIDGE, "grid": FIT_GRID,
        "matrix_csv": str(MATRIX.relative_to(ROOT)).replace("/", "\\"),
        "matrix_sha256": matrix_sha, "n_models": len(models), "judge": JUDGE,
        "recalibrated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "2A-curated",
        "curation_rule": (f"exclude observed pass_count<={IMBALANCE_MARGIN} OR "
                          f"fail_count<={IMBALANCE_MARGIN} (== '<=3 or >=49 of 52')"),
        "n_excluded": len(excluded), "n_retained": len(kept_ids),
        "note": "Phase-2 gemini-3 curated recalibration; pass-imbalance columns removed from "
                "fit AND administrable pool; observed labels (no SE_judge pre-correction).",
    }
    with open(BANK_OUT, "w", encoding="utf-8") as fh:
        for j, cid in enumerate(kept_ids):
            m = meta.get(cid, {})
            rec = {
                "criterion_id": cid,
                "scenario_id": m.get("scenario_id"),
                "criterion": m.get("criterion"),
                "expected_evidence": m.get("expected_evidence", []),
                "scoring_type": m.get("scoring_type", "binary"),
                "score_anchors": m.get("score_anchors"),
                "primary_skill": "general",
                "q_mapping": None, "q_rationale": None,
                "capability": m.get("capability"),
                "task": m.get("task"),
                "criticality": m.get("criticality", "not_critical"),
                "objectivity": m.get("objectivity"),
                "explicitness": m.get("explicitness"),
                "source": m.get("source"),
                "status": m.get("status", "approved"),
                "version": m.get("version", "1.0"),
                "exclude_from_fit": False,
                "discrimination": {DIM: float(A[j])},
                "difficulty": float(b[j]),
                "q_modeled": {DIM: 1},
                "irt_params": {"provenance": prov},
            }
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote curated bank -> {BANK_OUT}")

    # ---- deployed POINT estimator: full-bank fine-grid EAP posterior mean (curated) ----
    egrid, elog = scat.build_grid(1, EAP_GRID, EAP_RANGE)
    theta_full = scat.eap_all_models(Yk, Mk, A[:, None], b, egrid, elog)[:, 0]
    obs_pass = np.array([Yk[i][Mk[i]].mean() for i in range(len(models))])

    # ---- SE decomposition (full curated bank; observed-info parametric bootstrap) ----
    cov = rg.compute_item_cov(BANK_OUT, MATRIX, FIT_GRID, RIDGE)
    bgrid = np.linspace(-EAP_RANGE, EAP_RANGE, BOOT_GRID_N)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng = np.random.default_rng(PU_SEED)
    col = cov["col"]
    idx_full = np.array([col[c] for c in kept_ids], dtype=int)
    se_ability, se_param, se_total = [], [], []
    for i, mname in enumerate(models):
        y = cov["Ymat"][i]
        _, se_post, se_par = rg.bootstrap_theta(y, idx_full, cov["beta"], cov["chol"],
                                                bgrid, blp, N_BOOT, rng, batch=4)
        se_ability.append(float(se_post))
        se_param.append(float(se_par))
        se_total.append(float(np.sqrt(se_post ** 2 + se_par ** 2)))
    se_ability = np.array(se_ability); se_param = np.array(se_param); se_total = np.array(se_total)

    per = pd.DataFrame({
        "model": models, "theta": theta_full, "se_ability": se_ability,
        "se_param": se_param, "se_total": se_total, "observed_pass_rate": obs_pass,
    }).sort_values("theta", ascending=False).reset_index(drop=True)
    per.insert(0, "rank", np.arange(1, len(per) + 1))
    per.to_csv(SCRATCH / "per_model_gemini3_curated.csv", index=False)
    print(f"theta range [{theta_full.min():.3f}, {theta_full.max():.3f}]  "
          f"SE_ability med={np.median(se_ability):.4f}  SE_param med={np.median(se_param):.4f}  "
          f"SE_total med={np.median(se_total):.4f}")

    # ---- theta shift vs the UN-CURATED recal bank ----
    shift = shift_vs_uncurated(per)

    summary = {
        "phase": "2A-curated", "judge": JUDGE, "matrix": str(MATRIX.relative_to(ROOT)),
        "matrix_sha256": matrix_sha,
        "curation": {
            "rule": prov["curation_rule"], "imbalance_margin": IMBALANCE_MARGIN,
            "n_columns_total": len(all_ids), "n_excluded": len(excluded),
            "n_retained_fitted": len(kept_ids),
            "n_all_fail_excluded": n_allfail, "n_all_pass_excluded": n_allpass,
            "clean_high_a_retained": {
                "bgb_0202_c01": bool("bgb_0202_c01" in set(kept_ids)),
                "bgb_0047_c01": bool("bgb_0047_c01" in set(kept_ids)),
            },
            "excluded_sample": excluded[:20],
        },
        "config": {"dimensionality": "unidimensional", "fitter": "calibrate_mirt.fit_m2pl_em",
                   "method": "bock-aitkin-mml-em", "fit_grid_gh_nodes": FIT_GRID, "ridge": RIDGE,
                   "max_iter": 200,
                   "point_estimator": "full-bank fine-EAP posterior mean",
                   "eap_grid": EAP_GRID, "eap_range": EAP_RANGE, "eap_prior": "standard-normal",
                   "se_param": "observed-information parametric bootstrap (n_boot=%d)" % N_BOOT,
                   "boot_grid_nodes": BOOT_GRID_N, "pu_seed": PU_SEED},
        "n_criteria_fitted": len(kept_ids), "n_criteria_excluded": len(excluded),
        "n_models": len(models),
        "max_a": float(A.max()), "min_a": float(A.min()), "max_abs_b": float(np.abs(b).max()),
        "abs_b_gt6_count": n_b6,
        "fit_converged": bool(fit["converged"]), "fit_n_iter": int(fit["n_iter"]),
        "theta": {"min": float(theta_full.min()), "max": float(theta_full.max()),
                  "median": float(np.median(theta_full))},
        "se_medians": {"se_ability": float(np.median(se_ability)),
                       "se_param": float(np.median(se_param)),
                       "se_total": float(np.median(se_total))},
        "overall_observed_pass_rate": float(obs_pass.mean()),
        "shift_vs_uncurated": shift,
        "bank_out": str(BANK_OUT.relative_to(ROOT)),
        "bank_uncurated": str(BANK_UNCURATED.relative_to(ROOT)),
    }
    (SCRATCH / "summary_gemini3_curated.json").write_text(json.dumps(summary, indent=2),
                                                          encoding="utf-8")
    print(f"wrote summary -> {SCRATCH / 'summary_gemini3_curated.json'}")
    print("\n=== KEY NUMBERS (Step A) ===")
    print(json.dumps({k: summary[k] for k in
                      ("curation", "max_a", "abs_b_gt6_count", "theta", "se_medians",
                       "overall_observed_pass_rate", "shift_vs_uncurated")}, indent=2))
    return 0


def shift_vs_uncurated(per):
    prior = pd.read_csv(RECAL_PER_MODEL)[["model", "theta"]].rename(columns={"theta": "theta_uncurated"})
    g = per[["model", "theta", "observed_pass_rate"]].rename(columns={"theta": "theta_curated"})
    m = g.merge(prior, on="model", how="inner")
    m["dtheta"] = m["theta_curated"] - m["theta_uncurated"]
    m = m.sort_values("theta_uncurated", ascending=False).reset_index(drop=True)
    m.to_csv(SCRATCH / "shift_vs_uncurated.csv", index=False)
    pear = float(pearsonr(m["theta_curated"], m["theta_uncurated"])[0])
    spear = float(spearmanr(m["theta_curated"], m["theta_uncurated"])[0])
    j = int(m["dtheta"].abs().idxmax())
    tail = m.reindex(m["dtheta"].abs().sort_values(ascending=False).index).head(8)
    res = {"n_matched": int(len(m)), "theta_pearson_r": round(pear, 4),
           "theta_spearman_r": round(spear, 4),
           "max_abs_dtheta": round(float(m["dtheta"].abs().max()), 4),
           "max_abs_dtheta_model": m.loc[j, "model"],
           "mean_abs_dtheta": round(float(m["dtheta"].abs().mean()), 4),
           "median_dtheta": round(float(m["dtheta"].median()), 4),
           "largest_moves": [{"model": r["model"], "theta_uncurated": round(r["theta_uncurated"], 3),
                              "theta_curated": round(r["theta_curated"], 3),
                              "dtheta": round(r["dtheta"], 3)}
                             for _, r in tail.iterrows()]}
    print(f"  shift vs un-curated: Pearson={pear:.4f} Spearman={spear:.4f} "
          f"max|dtheta|={res['max_abs_dtheta']} ({res['max_abs_dtheta_model']}) "
          f"mean|dtheta|={res['mean_abs_dtheta']}")
    return res


if __name__ == "__main__":
    raise SystemExit(main())
