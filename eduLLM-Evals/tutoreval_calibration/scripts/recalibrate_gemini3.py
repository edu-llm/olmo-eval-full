"""TutorEval gemini-3 recalibration -- Phase 0 (lock config) + Phase 1 (recalibrate on
OBSERVED gemini-3 labels) + quantify-first extremes report.

LOCAL / STUDY ONLY. The production engine (``tutor_cat/``, ``scripts/scenario_cat_lib.py``,
``scripts/calibrate_mirt.py``) and every canonical study script are imported/called READ-ONLY
and are NEVER modified. Nothing is committed. Calibration is on the OBSERVED gemini-3 matrix
(Rule 0: NO pre-correction for judge over-strictness). This script STOPS after the quantify
report -- it does NOT curate, does NOT run any op-point grid, and does NOT lock anything.

What it does
------------
Phase 0 -- lock config + gold-pass stratum prep (report only):
  * Records the production config from the prior Qwen of-record (unidim ability, fit_m2pl_em,
    fit_grid 7, ridge 1e-2, clamp; CAT stop EAP-posterior 1-D marginal SD, 321-node,
    plateau delta 0.005/W 3, cap 70, MWLE-at-stop; op-point floor 15 / SE_ability 0.22).
  * Confirms the deployed of-record HEADLINE theta = op-point MWLE-at-stop @ f15/se22
    (leaderboard theta_mwle at n_scenarios=15), NOT full-bank.
  * Per-primary_skill gold 2x2 counts from the gold labels + the gemini-3 confusion; flags
    which strata carry enough gold-PASS cells for their OWN alpha vs pooled-to-overall.

Phase 1 -- recalibrate unidim ability on the gemini-3 matrix (production config):
  * Full-matrix EM fit (Q = ones) via calibrate_mirt.fit_m2pl_em.
  * Export the gemini-3-stamped fitted bank + provenance (matrix sha + config).
  * Per-model theta: full-bank (EAP + MWLE) AND op-point MWLE-at-stop @ f15/se22;
    SE_ability (EAP posterior SD at stop); SE_param (observed-info parametric bootstrap);
    SE_total. Light 1-D re-confirm (in-sample recovery of MWLE-at-stop vs full-bank EAP).
  * Qwen -> gemini-3 shift: pass-rate change, per-model theta shift, rank corr (Pearson+Spearman).

Quantify-first extremes (REPORT ONLY -- no curation applied):
  * zero-variance/all-fail column counts; per-criterion pass-rate distribution.
  * near-separation count (obs pass_count <=3 OR >=49 of 52); extreme (a,b): max_a, #|b|>6.
  * lighter CAT-selection check: administered share of near-sep/extreme items vs bank share.
  * over-strictness note (alpha=0.45) + a curation recommendation (NOT applied).

Usage
-----
    uv run python eduLLM-Evals/reports/tutoreval_gemini3_recal/recalibrate_gemini3.py --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

# eduLLM-Evals/reports/tutoreval_gemini3_recal/ -> reports/ -> eduLLM-Evals/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402  read-only shared engine lib

HERE = Path(__file__).resolve().parent

# -- locked production config (prior Qwen of-record) ---------------------------------------
FLOOR = 15
SE_TARGET = 0.22
CAP = 70
FIT_GRID = 7
RIDGE = 1e-2
DENSE_GRID = 321
DENSE_RANGE = 8.0
REF_GRID = 61
REF_RANGE = 6.0
PLATEAU_DELTA = 0.005
PLATEAU_W = 3
JUDGE_STAMP = "gemini-3-flash-preview"

DEFAULT_MATRIX = ROOT / "api_judge_pilot" / "grading_tutoreval" / "response_matrix.csv"
DEFAULT_RAW_BANK = ROOT / "data" / "TutorEval" / "rubrics_qmatrix_final.jsonl"
DEFAULT_SCENARIOS = ROOT / "data" / "TutorEval" / "scenarios_final.jsonl"
QWEN_LEADERBOARD = (ROOT / "tutoreval_calibration" / "experiments" / "08_leaderboard"
                    / "leaderboard_f15se22.csv")
QWEN_PIRT = (ROOT / "tutoreval_calibration" / "experiments" / "09_pirt_mae"
             / "pirt_per_model_f15se22.csv")
CONFUSION = (ROOT / "api_judge_pilot" / "gold" / "se_judge"
             / "tutoreval_gemini3_confusion.json")
GOLD_LABELS = ROOT / "api_judge_pilot" / "gold" / "tutoreval_core" / "gold_labels.jsonl"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
GRID = _load("run_oos_grid_unidim",
             ROOT / "tutoreval_calibration" / "scripts" / "run_oos_grid_unidim.py")
pu = _load("scenario_param_uncertainty", ROOT / "scripts" / "scenario_param_uncertainty.py")


def _read_jsonl(path: Path):
    for line in Path(path).open(encoding="utf-8-sig"):
        line = line.strip()
        if line:
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Phase 0 -- gold-pass stratum prep (report only)
# ---------------------------------------------------------------------------

def phase0(out: dict) -> dict:
    confusion = json.loads(Path(CONFUSION).read_text(encoding="utf-8"))
    gold = list(_read_jsonl(GOLD_LABELS))

    # per-primary_skill (stratum) 2x2 recomputed from the gold labels themselves
    strat: dict[str, dict] = {}
    for g in gold:
        s = g.get("stratum", "unknown")
        d = strat.setdefault(s, {"n": 0, "gold_pass": 0, "gold_fail": 0})
        d["n"] += 1
        if g.get("gold_label") == "pass":
            d["gold_pass"] += 1
        else:
            d["gold_fail"] += 1

    OWN_ALPHA_MIN = 8  # >=8-10 gold-PASS cells -> own alpha; else pool to overall
    conf_strat = confusion.get("by_stratum", {})
    strata_report = {}
    for s, d in sorted(strat.items()):
        cs = conf_strat.get(s, {})
        gp = d["gold_pass"]
        strata_report[s] = {
            "n": d["n"], "gold_pass": gp, "gold_fail": d["gold_fail"],
            "false_fail": cs.get("false_fail"),
            "alpha_own": cs.get("alpha_false_fail"),
            "beta_own": cs.get("beta_false_pass"),
            "own_alpha_supported": gp >= OWN_ALPHA_MIN,
            "recommendation": ("own-alpha" if gp >= OWN_ALPHA_MIN else "pool-to-overall"),
        }

    phase0 = {
        "production_config": {
            "scale": "unidimensional 2PL, single latent 'ability' (q_modeled ability:1)",
            "fitter": "calibrate_mirt.fit_m2pl_em (1 dim)",
            "fit_grid_nodes_per_dim": FIT_GRID, "ridge": RIDGE,
            "negative_policy": "preserved in-file; clamp at consume",
            "cat_stop": {
                "criterion": "EAP-posterior 1-D marginal SD",
                "dense_grid_nodes": DENSE_GRID, "dense_range": DENSE_RANGE,
                "plateau_delta": PLATEAU_DELTA, "plateau_w": PLATEAU_W, "cap": CAP,
                "theta_at_stop": "MWLE",
                "priority": ["precision", "plateau", "cap", "bank_exhausted"],
            },
            "op_point": {"floor_min_scenarios": FLOOR, "se_ability_target": SE_TARGET},
            "judge_stamp": JUDGE_STAMP,
        },
        "deployed_of_record_headline_theta": {
            "definition": "op-point MWLE-at-stop @ f15/se22 (leaderboard theta_mwle, n_scenarios=15)",
            "NOT": "full-bank theta",
            "source": str(QWEN_LEADERBOARD.relative_to(ROOT)),
        },
        "confusion_overall": confusion["overall"],
        "gold_pass_by_stratum": strata_report,
        "beta_pooled": confusion["overall"]["beta_false_pass"],
        "note_pooling": (
            "beta=0 pooled (no false-pass in 100 gold cells). Only strata with >=8 gold-PASS "
            "cells get their own alpha; the rest pool to the overall alpha (0.45)."
        ),
    }

    # confirm deployed headline theta definition against the leaderboard file
    lb = pd.read_csv(QWEN_LEADERBOARD)
    phase0["deployed_of_record_headline_theta"]["leaderboard_confirm"] = {
        "has_theta_mwle_column": "theta_mwle" in lb.columns,
        "all_n_scenarios_ge_floor15": bool((lb["n_scenarios"] >= FLOOR).all()),
        "n_stop_exactly_at_floor15": int((lb["n_scenarios"] == FLOOR).sum()),
        "n_models": int(len(lb)),
        "note": "headline = MWLE-at-stop at op-point; length is >= floor 15 (44/52 stop at the floor, rest run to precision/plateau)",
        "top_model": str(lb.sort_values("theta_mwle", ascending=False).iloc[0]["model"]),
        "top_theta_mwle": round(float(lb["theta_mwle"].max()), 4),
    }
    out["phase0"] = phase0
    return phase0


# ---------------------------------------------------------------------------
# Phase 1 -- recalibrate + deploy + shift
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--raw-bank", type=Path, default=DEFAULT_RAW_BANK)
    p.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--tmp-dir", type=Path, default=HERE / "_tmp")
    p.add_argument("--fit-grid", type=int, default=FIT_GRID)
    p.add_argument("--ridge", type=float, default=RIDGE)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--engine-seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": ("LOCAL / STUDY ONLY -- production engine untouched; nothing committed; "
                   "calibrated on OBSERVED gemini-3 labels (Rule 0, no pre-correction); "
                   "STOP after quantify (review checkpoint)."),
        "benchmark": "TutorEval", "judge": JUDGE_STAMP,
        "scale": "unidimensional (single 'ability' axis)",
    }

    bank_dir = args.out_dir / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 0 ----
    print("[Phase 0] config lock + gold-pass stratum prep ...", flush=True)
    phase0(out)

    # ---- load matrix + raw bank ----
    matrix = pd.read_csv(args.matrix, index_col=0)
    models = list(matrix.index)
    ids = list(matrix.columns)
    Y = np.nan_to_num(matrix.to_numpy(float), nan=0.0)
    M = ~np.isnan(matrix.to_numpy(float))
    n_models, n_cols = Y.shape
    matrix_sha = hashlib.sha256(Path(args.matrix).read_bytes()).hexdigest()
    print(f"[Phase 1] matrix {n_models} models x {n_cols} criteria; sha256={matrix_sha[:12]}",
          flush=True)

    raw_by = {r["criterion_id"]: r for r in _read_jsonl(args.raw_bank)}

    # ---- keep rule: >=2 observed AND not all-fail / all-pass among observed ----
    obs_count = M.sum(axis=0)
    pass_count = np.where(M, Y, 0.0).sum(axis=0)
    keep_mask = (obs_count >= 2) & (pass_count > 0) & (pass_count < obs_count)
    keep = np.where(keep_mask)[0]
    dropped = np.where(~keep_mask)[0]
    all_fail = [ids[j] for j in dropped if obs_count[j] >= 1 and pass_count[j] == 0]
    all_pass = [ids[j] for j in dropped if obs_count[j] >= 1 and pass_count[j] == obs_count[j]]
    too_sparse = [ids[j] for j in dropped if obs_count[j] < 2]
    print(f"  keep {len(keep)} / drop {len(dropped)} "
          f"(all_fail={len(all_fail)}, all_pass={len(all_pass)}, sparse<2={len(too_sparse)})",
          flush=True)

    # ---- fit unidim ability (Q = ones), production config ----
    Q_ones = np.ones((len(keep), 1), dtype=int)
    fit = cm.fit_m2pl_em(Y[:, keep], M[:, keep], Q_ones, args.fit_grid,
                         ridge=args.ridge, max_iter=args.max_iter)
    A_fit = fit["A"][:, 0]        # raw (negatives preserved)
    b_fit = fit["b"]
    kept_ids = [ids[j] for j in keep]
    print(f"  fit loglik={fit['loglik']:.2f} iters={fit['n_iter']} "
          f"converged={fit['converged']}", flush=True)

    # ---- export gemini-3-stamped fitted bank (negatives preserved in-file) ----
    provenance = {
        "benchmark": "TutorEval", "judge": JUDGE_STAMP, "axis": "unidim",
        "recalibrated_by": "reports/tutoreval_gemini3_recal/recalibrate_gemini3.py",
        "matrix_csv": str(args.matrix.relative_to(ROOT)),
        "matrix_sha256": matrix_sha,
        "fit_grid_nodes_per_dim": args.fit_grid, "ridge": args.ridge,
        "negative_loadings": "preserved (NOT floored to 0); clamp at consume",
        "reproduction_loglik": fit["loglik"],
        "n_persons": n_models, "n_items_fit": len(keep),
        "calibrated_on": "OBSERVED gemini-3 labels (Rule 0: no pre-correction)",
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    bank_path = bank_dir / "rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl"
    with bank_path.open("w", encoding="utf-8") as fh:
        for jj, cid in enumerate(kept_ids):
            raw = raw_by.get(cid, {})
            rec = {
                "criterion_id": cid,
                "scenario_id": raw.get("scenario_id", cid.rsplit("_c", 1)[0]),
                "criterion": raw.get("criterion", ""),
                "primary_skill": raw.get("primary_skill", ""),
                "scoring_type": raw.get("scoring_type", "binary"),
                "criticality": raw.get("criticality", "standard"),
                "status": raw.get("status", "approved"),
                "q_mapping": raw.get("q_mapping", {}),
                "difficulty": float(b_fit[jj]),
                "discrimination": {"ability": float(A_fit[jj])},
                "q_modeled": {"ability": 1},
                "irt_params": {
                    "source": "calibrated-2pl-unidim-tutoreval-gemini3",
                    "method": "unidimensional-2pl-mml-em",
                    "calibrated": True, "fitted": True, "modeled_skills": ["ability"],
                    "n_persons": n_models, "version": "1.0", "provenance": provenance,
                },
            }
            fh.write(json.dumps(rec) + "\n")
    (bank_dir / "bank_fit_summary.json").write_text(json.dumps({
        "generated_at": out["generated_at"], "status": out["status"],
        "bank_file": bank_path.name, "matrix": str(args.matrix.relative_to(ROOT)),
        "matrix_sha256": matrix_sha, "n_items_fit": len(keep), "n_persons_fit": n_models,
        "q_modeled": {"ability": 1},
        "fit_config": {"method": "unidimensional-2pl-mml-em",
                       "grid_nodes_per_dim": args.fit_grid, "ridge": args.ridge,
                       "negative_policy": "preserved in-file; clamp at consume"},
        "fit_result": {"loglik": fit["loglik"], "n_iter": fit["n_iter"],
                       "converged": fit["converged"]},
        "dropped": {"total": len(dropped), "all_fail": len(all_fail),
                    "all_pass": len(all_pass), "sparse_lt2": len(too_sparse)},
    }, indent=2), encoding="utf-8")
    print(f"  wrote fitted bank -> {bank_path.relative_to(ROOT)}", flush=True)

    # ---- reload the bank via production consume path (clamp negatives) ----
    records, dims, bank_stats = scat.load_fitted_bank(bank_path, "clamp")
    assert dims == ["ability"], dims
    r_ids, A_c, b_c = scat.assemble_arrays(records, dims)
    A_c = A_c[:, 0]
    col = {c: i for i, c in enumerate(r_ids)}
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    Q_c = np.array([[int(r["q_modeled"][d]) for d in dims] for r in records])

    # observed sub-matrix aligned to the fitted bank
    sub = matrix.reindex(columns=r_ids)
    Ysub = np.nan_to_num(sub.to_numpy(float), nan=0.0)
    Msub = ~np.isnan(sub.to_numpy(float))
    row_of = {m: i for i, m in enumerate(models)}

    # reference / stop grids
    egrid, elog = scat.build_grid(1, REF_GRID, REF_RANGE)
    A2d = A_c[:, None]
    gg = np.linspace(-DENSE_RANGE, DENSE_RANGE, DENSE_GRID)
    lp = -0.5 * gg ** 2
    lp = lp - np.log(np.exp(lp).sum())

    # ---- full-bank theta (EAP reference + MWLE) ----
    theta_full_eap = scat.eap_all_models(Ysub, Msub, A2d, b_c, egrid, elog)[:, 0]
    theta_full_mwle = np.full(n_models, np.nan)
    for i in range(n_models):
        idx = np.where(Msub[i])[0]
        if idx.size:
            th0 = scat.eap_subset(Ysub[i], idx, A2d, b_c, egrid, elog)
            thm, _ = scat.mwle_subset(Ysub[i], idx, A2d, b_c, th0)
            theta_full_mwle[i] = float(thm[0])

    # ---- op-point deploy: force full trace, then offline stop_point(15, 0.22) ----
    print("[Phase 1] forcing full CAT traces (production engine) ...", flush=True)
    res = GRID.run_forced(models, bank_path, args.matrix, args.scenarios, dims,
                          args.engine_seed, CAP, args.workers, args.tmp_dir / "runs")
    per_model = {}
    admin_counts = np.zeros(len(r_ids), dtype=int)  # op-point administered tallies
    for i, m in enumerate(models):
        walk = GRID.eap_walk(res[m]["order"], Ysub[i], col, scen_of, A_c, b_c, gg, lp)
        step, reason = GRID.stop_point(walk, FLOOR, SE_TARGET, cap=CAP)
        if step is None or step["idx"].size == 0:
            per_model[m] = {"idx": np.array([], int), "sd": float("nan"),
                            "n_scen": 0, "n_crit": 0, "reason": reason,
                            "theta_mwle": theta_full_mwle[i]}
            continue
        idx = step["idx"]
        admin_counts[idx] += 1
        th0 = scat.eap_subset(Ysub[i], idx, A2d, b_c, egrid, elog)
        thm, _ = scat.mwle_subset(Ysub[i], idx, A2d, b_c, th0)
        per_model[m] = {"idx": idx, "sd": float(step["eap_sd"]),
                        "n_scen": int(step["n_scen"]), "n_crit": int(step["n_crit"]),
                        "reason": reason, "theta_mwle": float(thm[0])}
    shutil.rmtree(args.tmp_dir / "runs", ignore_errors=True)

    # ---- SE_param: observed-info parametric bootstrap over the op-point administered set ----
    print("[Phase 1] SE_param parametric bootstrap (observed information) ...", flush=True)
    item_cov = pu.item_param_cov(Ysub, Msub, A2d, Q_c, b_c, 1, args.fit_grid, args.ridge)
    chol = []
    for _fd, _beta, cov in item_cov:
        try:
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            L = np.zeros_like(cov)
        chol.append(L)
    rng = np.random.default_rng(args.seed)
    A_draw = A2d.copy()
    b_draw = b_c.copy()
    se_param = {}
    for m in models:
        idx = per_model[m]["idx"]
        i = row_of[m]
        if idx.size == 0:
            se_param[m] = float("nan")
            continue
        boot = np.empty(args.n_boot)
        for t in range(args.n_boot):
            for j in idx:
                fd, beta, _ = item_cov[j]
                if fd.size == 0:
                    continue
                draw = beta + chol[j] @ rng.standard_normal(beta.shape[0])
                A_draw[j, 0] = draw[0]
                b_draw[j] = -draw[-1]
            boot[t] = float(pu.eap_mean(Ysub[i], idx, A_draw, b_draw, egrid, elog)[0])
        A_draw[idx] = A2d[idx]
        b_draw[idx] = b_c[idx]
        se_param[m] = float(np.std(boot, ddof=1))

    # ---- per-model leaderboard table ----
    qlb = pd.read_csv(QWEN_LEADERBOARD).set_index("model")
    qpirt = pd.read_csv(QWEN_PIRT).set_index("model")
    rows = []
    for m in models:
        i = row_of[m]
        pm = per_model[m]
        sd = pm["sd"]
        sp = se_param[m]
        st = float(np.sqrt(sd ** 2 + sp ** 2)) if np.isfinite(sd) and np.isfinite(sp) else float("nan")
        obs = Msub[i]
        rows.append({
            "model": m,
            "theta_full_bank_eap": round(float(theta_full_eap[i]), 4),
            "theta_full_bank_mwle": round(float(theta_full_mwle[i]), 4),
            "theta_mwle_f15se22": round(float(pm["theta_mwle"]), 4),
            "SE_ability": round(sd, 4) if np.isfinite(sd) else None,
            "SE_param": round(sp, 4) if np.isfinite(sp) else None,
            "SE_total": round(st, 4) if np.isfinite(st) else None,
            "n_scenarios": pm["n_scen"], "n_criteria": pm["n_crit"],
            "stop_reason": pm["reason"],
            "obs_pass_rate_g3": round(float(np.mean(Ysub[i, obs])) if obs.sum() else float("nan"), 4),
            "qwen_theta_mwle": round(float(qlb.loc[m, "theta_mwle"]), 4) if m in qlb.index else None,
            "qwen_obs_pass_rate": round(float(qpirt.loc[m, "actual_pass_rate"]), 4) if m in qpirt.index else None,
        })
    lb = pd.DataFrame(rows).sort_values("theta_mwle_f15se22", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))
    lb.to_csv(args.out_dir / "leaderboard_gemini3_f15se22.csv", index=False)

    # ---- light 1-D re-confirm (in-sample): MWLE-at-stop vs full-bank EAP ----
    x = lb["theta_full_bank_eap"].to_numpy(float)
    yv = lb["theta_mwle_f15se22"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(yv)
    r1 = float(np.corrcoef(x[ok], yv[ok])[0, 1])
    slope1, icpt1 = np.polyfit(x[ok], yv[ok], 1)
    mae1 = float(np.mean(np.abs(yv[ok] - x[ok])))

    # ---- Qwen -> gemini-3 shift ----
    j = lb.dropna(subset=["qwen_theta_mwle"])
    dtheta = j["theta_mwle_f15se22"].to_numpy(float) - j["qwen_theta_mwle"].to_numpy(float)
    pear = float(np.corrcoef(j["qwen_theta_mwle"], j["theta_mwle_f15se22"])[0, 1])
    spr = float(spearmanr(j["qwen_theta_mwle"], j["theta_mwle_f15se22"]).correlation)
    g3_overall_pr = float(np.nanmean(np.where(Msub, Ysub, np.nan)))
    g3_permodel_pr = float(lb["obs_pass_rate_g3"].mean())
    qwen_permodel_pr = float(qpirt["actual_pass_rate"].mean())
    shift = {
        "gemini3_overall_pass_rate": round(g3_overall_pr, 4),
        "gemini3_per_model_mean_pass_rate": round(g3_permodel_pr, 4),
        "qwen_per_model_mean_pass_rate": round(qwen_permodel_pr, 4),
        "per_model_pass_rate_delta_mean": round(g3_permodel_pr - qwen_permodel_pr, 4),
        "theta_shift_mean": round(float(np.mean(dtheta)), 4),
        "theta_shift_median": round(float(np.median(dtheta)), 4),
        "theta_shift_min": round(float(np.min(dtheta)), 4),
        "theta_shift_max": round(float(np.max(dtheta)), 4),
        "theta_rank_corr_pearson": round(pear, 4),
        "theta_rank_corr_spearman": round(spr, 4),
        "n_models_joined": int(len(j)),
    }

    out["phase1"] = {
        "bank_path": str(bank_path.relative_to(ROOT)),
        "n_items_fit": len(keep), "n_dropped": len(dropped),
        "dropped_breakdown": {"all_fail": len(all_fail), "all_pass": len(all_pass),
                              "sparse_lt2": len(too_sparse)},
        "fit_loglik": fit["loglik"], "fit_iters": fit["n_iter"],
        "fit_converged": bool(fit["converged"]),
        "bank_load_stats": bank_stats,
        "max_a_raw": round(float(np.max(np.abs(A_fit))), 4),
        "max_a_clamped": round(float(np.max(A_c)), 4),
        "median_SE_ability": round(float(lb["SE_ability"].median(skipna=True)), 4),
        "median_SE_param": round(float(lb["SE_param"].median(skipna=True)), 4),
        "median_SE_total": round(float(lb["SE_total"].median(skipna=True)), 4),
        "reconfirm_1d_insample": {"r": round(r1, 4), "slope": round(float(slope1), 4),
                                  "intercept": round(float(icpt1), 4), "theta_mae": round(mae1, 4)},
        "qwen_to_gemini3_shift": shift,
        "leaderboard_csv": str((args.out_dir / "leaderboard_gemini3_f15se22.csv").relative_to(ROOT)),
        "stop_reason_counts": lb["stop_reason"].value_counts().to_dict(),
    }

    # ---- quantify-first extremes report (REPORT ONLY) ----
    print("[quantify] extremes / near-separation / CAT-selection ...", flush=True)
    pr_all = np.where(obs_count > 0, pass_count / np.maximum(obs_count, 1), np.nan)
    qtiles = {str(q): round(float(np.nanpercentile(pr_all, q)), 4)
              for q in (0, 5, 10, 25, 50, 75, 90, 95, 100)}
    hist_edges = np.linspace(0, 1, 11)
    hist_counts, _ = np.histogram(pr_all[~np.isnan(pr_all)], bins=hist_edges)

    # near-separation among FITTED items (obs pass_count <=3 OR >=49 of 52)
    pc_fit = pass_count[keep]
    oc_fit = obs_count[keep]
    near_sep_mask = (pc_fit <= 3) | (pc_fit >= 49)
    extreme_a_mask = np.abs(A_fit) > cm.EXTREME_A
    extreme_b_mask = np.abs(b_fit) > 6.0
    flagged_mask = near_sep_mask | extreme_a_mask | extreme_b_mask

    # CAT-selection: administered share of flagged items vs bank share
    total_admin = int(admin_counts.sum())
    flagged_admin = int(admin_counts[flagged_mask].sum())
    near_sep_admin = int(admin_counts[near_sep_mask].sum())
    bank_share_flagged = float(np.mean(flagged_mask))
    admin_share_flagged = (flagged_admin / total_admin) if total_admin else float("nan")
    bank_share_nearsep = float(np.mean(near_sep_mask))
    admin_share_nearsep = (near_sep_admin / total_admin) if total_admin else float("nan")

    # over-strictness: all-fail columns that gemini plausibly wrongly failed (alpha=0.45)
    n_all_fail = len(all_fail)
    quant = {
        "n_columns_total": n_cols,
        "n_zero_variance_dropped": len(all_fail) + len(all_pass),
        "n_all_fail_dropped": n_all_fail,
        "n_all_pass_dropped": len(all_pass),
        "n_sparse_lt2_dropped": len(too_sparse),
        "per_criterion_pass_rate_quantiles": qtiles,
        "per_criterion_pass_rate_histogram": {
            "bin_edges": [round(float(e), 2) for e in hist_edges],
            "counts": [int(c) for c in hist_counts],
        },
        "near_separation_fitted": {
            "definition": "obs pass_count <=3 OR >=49 of 52",
            "count": int(near_sep_mask.sum()),
            "share_of_fitted": round(float(np.mean(near_sep_mask)), 4),
            "n_low_tail_le3": int((pc_fit <= 3).sum()),
            "n_high_tail_ge49": int((pc_fit >= 49).sum()),
        },
        "ridge_collapse_extreme": {
            "max_a_raw": round(float(np.max(np.abs(A_fit))), 4),
            "n_extreme_a_gt6": int(extreme_a_mask.sum()),
            "n_abs_b_gt6": int(extreme_b_mask.sum()),
            "n_negative_a_raw": int((A_fit < 0).sum()),
        },
        "cat_selection_check": {
            "op_point": {"floor": FLOOR, "se_target": SE_TARGET},
            "total_administered_criteria": total_admin,
            "flagged_definition": "near-sep OR |a|>6 OR |b|>6",
            "bank_share_flagged": round(bank_share_flagged, 4),
            "administered_share_flagged": round(admin_share_flagged, 4),
            "over_selection_ratio_flagged": round(admin_share_flagged / bank_share_flagged, 3)
            if bank_share_flagged > 0 else None,
            "bank_share_near_sep": round(bank_share_nearsep, 4),
            "administered_share_near_sep": round(admin_share_nearsep, 4),
            "over_selection_ratio_near_sep": round(admin_share_nearsep / bank_share_nearsep, 3)
            if bank_share_nearsep > 0 else None,
        },
        "over_strictness_note": {
            "judge_alpha_false_fail": 0.45,
            "interpretation": (
                f"With alpha=0.45 (P(judge=fail | gold=pass)), a sizeable share of the "
                f"{n_all_fail} all-fail columns are all-fail because gemini-3 WRONGLY failed "
                "truly-passable answers, not because the criterion is genuinely unpassable. "
                "Per Rule 0 the fit still uses OBSERVED labels (no pre-correction)."
            ),
            "expected_spurious_all_fail_order_of_magnitude": (
                "alpha applied to the passable share inflates all-fail counts; exact split "
                "needs the Beta-from-gold sampler (Phase 2, not run here)."
            ),
        },
        "curation_recommendation": (
            "DO NOT APPLY NOW (checkpoint). Recommended later: (1) keep Rule-0 observed fit as "
            "the of-record; (2) build a Beta-from-gold (alpha=0.45, beta=0) sampler to estimate, "
            "per all-fail column, P(truly unpassable) vs P(spurious all-fail from judge strictness); "
            "(3) consider an op-point/selection guard for near-separation & extreme-a items "
            "(they are the least trustworthy under a strict judge), evaluated on an op-point grid; "
            "(4) re-audit high-|b| / extreme-a items against gold before any curation."
        ),
    }
    out["quantify"] = quant

    (args.out_dir / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    # ---- console summary ----
    print("\n" + "=" * 78)
    print("TutorEval gemini-3 recal -- Phase 0/1 + quantify (LOCAL, observed labels)")
    print("=" * 78)
    ph0 = out["phase0"]
    print(f"[P0] deployed headline theta = {ph0['deployed_of_record_headline_theta']['definition']}")
    print(f"[P0] confusion overall: alpha={ph0['confusion_overall']['alpha_false_fail']} "
          f"beta={ph0['confusion_overall']['beta_false_pass']}")
    for s, d in ph0["gold_pass_by_stratum"].items():
        print(f"     stratum {s:26s} n={d['n']:3d} gold_pass={d['gold_pass']:2d} "
              f"alpha_own={d['alpha_own']} -> {d['recommendation']}")
    p1 = out["phase1"]
    print(f"[P1] fitted {p1['n_items_fit']} / dropped {p1['n_dropped']} "
          f"(all_fail={p1['dropped_breakdown']['all_fail']}, "
          f"all_pass={p1['dropped_breakdown']['all_pass']}, "
          f"sparse={p1['dropped_breakdown']['sparse_lt2']})")
    print(f"[P1] max_a raw={p1['max_a_raw']} clamped={p1['max_a_clamped']}  loglik={p1['fit_loglik']:.1f}")
    print(f"[P1] median SE_ability={p1['median_SE_ability']} SE_param={p1['median_SE_param']} "
          f"SE_total={p1['median_SE_total']}")
    print(f"[P1] 1-D reconfirm: r={p1['reconfirm_1d_insample']['r']} "
          f"slope={p1['reconfirm_1d_insample']['slope']} MAE={p1['reconfirm_1d_insample']['theta_mae']}")
    sh = p1["qwen_to_gemini3_shift"]
    print(f"[P1] Qwen->g3: passrate {sh['qwen_per_model_mean_pass_rate']}->"
          f"{sh['gemini3_per_model_mean_pass_rate']} (d={sh['per_model_pass_rate_delta_mean']}); "
          f"theta shift mean={sh['theta_shift_mean']} (min={sh['theta_shift_min']}); "
          f"rankcorr Pear={sh['theta_rank_corr_pearson']} Spear={sh['theta_rank_corr_spearman']}")
    q = out["quantify"]
    print(f"[Q]  zero-var dropped={q['n_zero_variance_dropped']} "
          f"(all_fail={q['n_all_fail_dropped']}, all_pass={q['n_all_pass_dropped']}); "
          f"near-sep fitted={q['near_separation_fitted']['count']}; "
          f"max_a={q['ridge_collapse_extreme']['max_a_raw']} #|b|>6={q['ridge_collapse_extreme']['n_abs_b_gt6']}")
    cs = q["cat_selection_check"]
    print(f"[Q]  CAT selection: near-sep admin share={cs['administered_share_near_sep']} vs "
          f"bank share={cs['bank_share_near_sep']} (x{cs['over_selection_ratio_near_sep']})")
    print(f"\nwrote -> {args.out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
