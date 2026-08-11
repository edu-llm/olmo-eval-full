"""Phase-3: SE_judge (judge-measurement-error) layer for the TutorEval gemini-3 recalibration.

LOCAL scratch driver. It CALLS the frozen production/study engine READ-ONLY
(``scenario_cat_lib``, ``calibrate_mirt``, and the tutoreval OOS-grid fold machinery
``run_oos_grid_unidim`` for the exact deployed CAT order + stop rule) and adds ONLY the
judge-error Monte-Carlo layer on top of the of-record theta. Nothing is committed; all writes
go under ``reports/tutoreval_gemini3_recal/se_judge/``.

Op-point (LOCKED): floor 15 / SE_ability target 0.25, dense-EAP posterior-SD stop (321-node grid
over [-8,8]), info-plateau delta 0.005 / W 3, cap 70, MWLE-at-stop. Two regimes per model:

  1. HEADLINE = op-point f15/se0.25 (deployed of-record CAT theta): freeze each model's deployed
     adaptive administration ORDER (observed-label max-info selection), then re-apply the dense-EAP
     f15/se0.25 stop on the RESAMPLED labels and take MWLE-at-stop on the stopped administered
     cells. SE_judge_op = SD(theta); theta_debiased_op = mean(theta). SE_param_op = the sound
     observed-info parametric-bootstrap offset from the Phase-1 deployed table (fixed per model).
  2. SECONDARY = full-bank cohort leaderboard: re-score the full-bank fine-EAP posterior-mean
     theta (and MWLE) on the resampled labels (frozen bank). SE_judge_full = SD(theta);
     theta_debiased_full = mean(theta). SE_param_full = vectorized observed-info parametric
     bootstrap over all observed cells; SE_ability_full = full-bank EAP posterior SD.

Resampling (per replicate): sample alpha per stratum (conceptual_understanding OWN alpha vs the
pooled OVERALL alpha) from Beta-from-gold-counts posteriors; beta pinned to 0 (no false-pass in
the 100 gold cells); de-bias per-criterion pi_c; correlated-within-criterion true-label resample
(one latent uniform per column) via the Bayes posterior. Tier A = frozen bank (B~500); Tier B =
refit calibrate_mirt each replicate (B~50), identical production config.

beta=0 nuance: the resampled TRUE matrix is elementwise >= the observed matrix (observed passes
never flip; only observed FAILs can flip UP), so a monotone estimator's theta_debiased is >=
observed theta by construction -- there is no shrinkage-crossover trap (unlike BiGGen where
beta>0). The driver still runs the direction/monotonicity GATE empirically and reports it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp
from scipy.stats import spearmanr

# reports/tutoreval_gemini3_recal/se_judge/scratch/this -> eduLLM-Evals/
ROOT = Path(__file__).resolve().parents[4]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.scenario_cat_lib as scat  # noqa: E402
import scripts.frq_total_se as fts       # noqa: E402  (quadrature helper -- the intended extension)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import se_judge_lib as L  # noqa: E402


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

DIM = "ability"
DIM_LIST = [DIM]
JUDGE = L.JUDGE

# locked config (Phase-0 production config + re-swept op-point)
FIT_GRID = 7
RIDGE = 1e-2
REF_GRID = 61
REF_RANGE = 6.0
DENSE_GRID = 321
DENSE_RANGE = 8.0
PLATEAU_DELTA = 0.005
PLATEAU_W = 3
OP_FLOOR = 15
OP_TARGET = 0.25
OP_CAP = 70
ENGINE_SEED = 42            # reproduce the deployed (Phase-1) administered ORDER
N_BOOT_SEPARAM_FULL = 200   # vectorized observed-info parametric bootstrap for full-bank SE_param

MATRIX = ROOT / "api_judge_pilot" / "grading_tutoreval" / "response_matrix.csv"
SCENARIOS = ROOT / "data" / "TutorEval" / "scenarios_final.jsonl"
BANK = (ROOT / "reports" / "tutoreval_gemini3_recal" / "bank"
        / "rubrics_qmatrix_final_unidim_gemini3_fitted.jsonl")
CONFUSION = ROOT / "api_judge_pilot" / "gold" / "se_judge" / "tutoreval_gemini3_confusion.json"
GOLD = ROOT / "api_judge_pilot" / "gold" / "tutoreval_core" / "gold_labels.jsonl"
PHASE1_LB = ROOT / "reports" / "tutoreval_gemini3_recal" / "leaderboard_gemini3_f15se22.csv"

OUT = ROOT / "reports" / "tutoreval_gemini3_recal" / "se_judge"


# ---------------------------------------------------------------------------
# op-point administration order (frozen) + fast per-replicate walk / stop
# ---------------------------------------------------------------------------


def deployed_order_blocks(models, A_c, b_c, r_ids, scen_of, workers, seed):
    """Run the deployed CAT once (observed labels) to the cap -> per-model scenario blocks.

    Returns (col, blocks): ``col`` maps criterion_id -> column index in the fitted bank;
    ``blocks[m]`` is an ordered list of np.int arrays (one per administered scenario, in
    administration order) capturing the deployed max-info order under the OBSERVED labels.
    """
    col = {c: i for i, c in enumerate(r_ids)}
    runs = OUT / "_op_runs"
    res = GRID.run_forced(models, BANK, MATRIX, SCENARIOS, DIM_LIST, seed, OP_CAP, workers, runs)
    blocks = {}
    for m in models:
        order = [c for c in res[m]["order"] if c in col]
        blk, cur, cur_sid = [], [], None
        for c in order:
            sid = scen_of.get(c)
            if sid != cur_sid and cur:
                blk.append(np.array(cur, dtype=int))
                cur = []
            cur_sid = sid
            cur.append(col[c])
        if cur:
            blk.append(np.array(cur, dtype=int))
        blocks[m] = blk
    shutil.rmtree(runs, ignore_errors=True)
    return col, blocks


def build_walk(y, blocks_m, SLP0_m, DIFF, gg, lp):
    """Per-scenario cumulative EAP (mean, SD) walk under labels ``y`` along a model's frozen
    scenario blocks -- the same structure ``GRID.eap_walk`` returns, so ``GRID.stop_point``
    (precision -> plateau -> cap -> bank_exhausted) applies unchanged.

    ``SLP0_m[si]`` = precomputed sum of log P(fail|theta) over scenario si's items (rep-
    independent); the y-dependent term is ``y[block] @ DIFF[block]`` with DIFF = logP1 - logP0.
    """
    ll = np.zeros(gg.size)
    walk = []
    cum = []
    n_crit = 0
    for si, block in enumerate(blocks_m):
        ll = ll + SLP0_m[si] + y[block] @ DIFF[block]
        cum.extend(block.tolist())
        n_crit += block.size
        post = np.exp(ll + lp - logsumexp(ll + lp))
        mean = float(post @ gg)
        sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
        walk.append({"n_scen": si + 1, "n_crit": n_crit, "eap_mean": mean,
                     "eap_sd": sd, "idx": np.array(cum, dtype=int)})
    return walk


def op_theta_at_stop(y, walk, A2d, b_c, egrid, elog):
    """MWLE-at-stop over the administered cells at the f15/se0.25 stop; returns
    (theta_mwle, eap_sd, n_scen, n_crit, reason)."""
    step, reason = GRID.stop_point(walk, OP_FLOOR, OP_TARGET, cap=OP_CAP,
                                   delta=PLATEAU_DELTA, w=PLATEAU_W)
    if step is None or step["idx"].size == 0:
        return float("nan"), float("nan"), 0, 0, reason
    idx = step["idx"]
    th0 = scat.eap_subset(y, idx, A2d, b_c, egrid, elog)
    thm, _ = scat.mwle_subset(y, idx, A2d, b_c, th0)
    return float(thm[0]), float(step["eap_sd"]), int(step["n_scen"]), int(step["n_crit"]), reason


# ---------------------------------------------------------------------------
# full-bank SE_param: vectorized observed-info parametric bootstrap
# ---------------------------------------------------------------------------


def full_bank_se_param(Ysub, Msub, A2d, b_c, egrid, elog, n_boot, seed):
    """SE_param per model over the FULL observed bank via a vectorized parametric bootstrap.

    One E-step at the frozen fit gives per-item 2x2 (a, -b) covariances (observed information);
    each replicate redraws ALL item params at once and re-scores full-bank EAP for every model.
    SE_param(model) = SD(theta across replicates).
    """
    Q = np.ones((A2d.shape[0], 1), dtype=int)
    item_cov = pu.item_param_cov(Ysub, Msub, A2d, Q, b_c, 1, FIT_GRID, RIDGE)
    n = A2d.shape[0]
    beta_all = np.zeros((n, 2))
    chol_all = np.zeros((n, 2, 2))
    for j, (fd, beta, cov) in enumerate(item_cov):
        if fd.size == 0:            # unidim Q=1 => never expected; degenerate guard
            beta_all[j] = [A2d[j, 0], -b_c[j]]
            continue
        beta_all[j] = beta          # [a, -b]
        try:
            chol_all[j] = np.linalg.cholesky(cov + 1e-10 * np.eye(2))
        except np.linalg.LinAlgError:
            chol_all[j] = np.zeros((2, 2))
    rng = np.random.default_rng(seed)
    n_models = Ysub.shape[0]
    boot = np.empty((n_boot, n_models))
    for t in range(n_boot):
        z = rng.standard_normal((n, 2))
        draw = beta_all + np.einsum("nij,nj->ni", chol_all, z)
        A_draw = draw[:, 0:1]
        b_draw = -draw[:, 1]
        boot[t] = scat.eap_all_models(Ysub, Msub, A_draw, b_draw, egrid, elog)[:, 0]
    return boot.std(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# bias band (alpha CI corners; beta=0), MC-averaged
# ---------------------------------------------------------------------------


def bias_band(Yobs, Mobs, p_obs_c, own_mask, A2d, b_c, egrid, elog, gg, lp,
              blocks, SLP0, DIFF, models, K, seed):
    """theta at the alpha CI corners (beta=0), MC-averaged over K flip draws (fixed rates).

    hi corner (max upward de-bias): alpha = CI high; lo corner: alpha = CI low. Applied
    uniformly across strata using the OVERALL confusion CI (the reported de-bias band).
    """
    n_models, n_items = Yobs.shape
    corners = {"hi": L.ALPHA_CI[1], "lo": L.ALPHA_CI[0]}
    res = {}
    for tag, a_val in corners.items():
        rng = np.random.default_rng(seed + (0 if tag == "hi" else 999))
        alpha_c = np.full(n_items, a_val)
        beta_c = np.zeros(n_items)
        pi_c = L.debias_pi(p_obs_c, alpha_c, beta_c)
        r_pass, r_fail = L.bayes_pass_probs(pi_c, alpha_c, beta_c)
        acc_full = np.zeros(n_models)
        acc_op = np.zeros(n_models)
        for _ in range(K):
            Ytrue = L.resample_true_labels(Yobs, Mobs, r_pass, r_fail, rng)
            acc_full += scat.eap_all_models(Ytrue, Mobs, A2d, b_c, egrid, elog)[:, 0]
            for i, m in enumerate(models):
                walk = build_walk(Ytrue[i], blocks[m], SLP0[m], DIFF, gg, lp)
                th, _, _, _, _ = op_theta_at_stop(Ytrue[i], walk, A2d, b_c, egrid, elog)
                acc_op[i] += th
        res[f"full_{tag}"] = acc_full / K
        res[f"op_{tag}"] = acc_op / K
    return res["full_lo"], res["full_hi"], res["op_lo"], res["op_hi"]


# ---------------------------------------------------------------------------
# Tier B: refit calibrate_mirt each replicate
# ---------------------------------------------------------------------------


def tier_b(Yobs, Mobs, p_obs_c, own_mask, post, A2d, b_c, egrid, elog, models, B, seed,
           se_judge_full_A):
    print(f"\nTier B: B={B} refits (calibrate_mirt each replicate) ...", flush=True)
    rng = np.random.default_rng(seed)
    n_models, n_items = Yobs.shape
    Q = np.ones((n_items, 1), dtype=int)
    th_B = np.empty((B, n_models))
    for t in range(B):
        alpha_c, beta_c = L.sample_alpha_beta(post, own_mask, rng)
        pi_c = L.debias_pi(p_obs_c, alpha_c, beta_c)
        r_pass, r_fail = L.bayes_pass_probs(pi_c, alpha_c, beta_c)
        Ytrue = L.resample_true_labels(Yobs, Mobs, r_pass, r_fail, rng)
        fit = cm.fit_m2pl_em(Ytrue, Mobs, Q, FIT_GRID, ridge=RIDGE, max_iter=200)
        th_B[t] = scat.eap_all_models(Ytrue, Mobs, fit["A"], fit["b"], egrid, elog)[:, 0]
        if (t + 1) % 10 == 0:
            print(f"  tierB {t + 1}/{B}", flush=True)
    return {"se_judge_full_B": th_B.std(axis=0, ddof=1),
            "theta_deb_full_B": th_B.mean(axis=0),
            "se_judge_full_A": se_judge_full_A}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b-tier-a", type=int, default=500)
    ap.add_argument("--b-tier-b", type=int, default=50)
    ap.add_argument("--b-band", type=int, default=100, help="MC draws per bias-band corner.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--skip-tier-b", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    # ---- frozen fitted bank + observed matrix ----
    records, dims, bank_stats = scat.load_fitted_bank(BANK, "clamp")
    assert dims == DIM_LIST, dims
    r_ids, A2d, b_c = scat.assemble_arrays(records, dims)
    A_c = A2d[:, 0]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    prim_of = {r["criterion_id"]: r.get("primary_skill") for r in records}

    matrix = pd.read_csv(MATRIX, index_col=0)
    models = list(matrix.index)
    sub = matrix.reindex(columns=r_ids)
    Ysub = np.nan_to_num(sub.to_numpy(float), nan=0.0)
    Msub = ~np.isnan(sub.to_numpy(float))
    n_models, n_items = Ysub.shape
    p_obs_c = np.array([Ysub[Msub[:, j], j].mean() if Msub[:, j].any() else 0.0
                        for j in range(n_items)])
    own_mask = L.stratum_own_mask([prim_of[c] for c in r_ids])
    print(f"bank={BANK.name} models={n_models} criteria={n_items} fill={Msub.mean():.4f}")
    print(f"stratum split: own(conceptual)={int(own_mask.sum())} pooled(overall)="
          f"{int((~own_mask).sum())}")

    # ---- Beta-from-gold-counts alpha posteriors (confusion.json), confirmed vs report ----
    confusion = L.load_confusion(CONFUSION)
    ov = confusion["overall"]
    assert (ov["n"], ov["gold_pass"], ov["false_fail"], ov["false_pass"]) == (100, 20, 9, 0), ov
    post = L.alpha_posteriors(confusion)
    print(f"alpha posteriors (Beta-from-gold, Jeffreys): "
          f"own[{post['own']['stratum']}] mean={post['own']['alpha_mean']:.3f} "
          f"CI{np.round(post['own']['alpha_ci95'], 3)}  "
          f"overall mean={post['overall']['alpha_mean']:.3f} "
          f"CI{np.round(post['overall']['alpha_ci95'], 3)}  beta=0 (pinned)")

    # ---- grids ----
    egrid, elog = scat.build_grid(1, REF_GRID, REF_RANGE)     # reference/subset EAP (Phase-1)
    gg = np.linspace(-DENSE_RANGE, DENSE_RANGE, DENSE_GRID)     # dense stop grid
    lp = -0.5 * gg ** 2
    lp = lp - logsumexp(lp)

    # ---- of-record estimators on OBSERVED labels ----
    theta_full_eap = scat.eap_all_models(Ysub, Msub, A2d, b_c, egrid, elog)[:, 0]
    theta_full_mwle = np.full(n_models, np.nan)
    se_ability_full = np.full(n_models, np.nan)
    for i in range(n_models):
        idx = np.where(Msub[i])[0]
        if idx.size:
            _, var = scat.eap_subset_mean_var(Ysub[i], idx, A2d, b_c, egrid, elog)
            se_ability_full[i] = float(np.sqrt(max(var[0], 0.0)))
            th0 = scat.eap_subset(Ysub[i], idx, A2d, b_c, egrid, elog)
            thm, _ = scat.mwle_subset(Ysub[i], idx, A2d, b_c, th0)
            theta_full_mwle[i] = float(thm[0])
    obs_pass = np.array([Ysub[i][Msub[i]].mean() if Msub[i].any() else np.nan
                         for i in range(n_models)])

    # ---- full-bank SE_param (vectorized parametric bootstrap over all observed cells) ----
    print("full-bank SE_param: observed-info parametric bootstrap ...", flush=True)
    se_param_full = full_bank_se_param(Ysub, Msub, A2d, b_c, egrid, elog,
                                       N_BOOT_SEPARAM_FULL, args.seed + 7)

    # ---- SE_param op-point: sound parametric-bootstrap offset from the Phase-1 deployed table ----
    p1 = pd.read_csv(PHASE1_LB).set_index("model")
    sp_col = "SE_param" if "SE_param" in p1.columns else "se_param"
    sp_by = {str(m): float(v) for m, v in zip(p1.index, p1[sp_col]) if np.isfinite(v)}
    sp_med = float(np.median(list(sp_by.values()))) if sp_by else 0.1013
    se_param_op = np.array([sp_by.get(m, sp_med) for m in models])
    print(f"op-point SE_param offset (Phase-1 parametric bootstrap): median={sp_med:.4f} "
          f"n_have={len(sp_by)}")

    # ---- deployed op-point order (observed labels) -> frozen blocks + rep-independent precompute ----
    print("running the deployed CAT once (observed labels) for the op-point order ...", flush=True)
    col, blocks = deployed_order_blocks(models, A_c, b_c, r_ids, scen_of, args.workers, ENGINE_SEED)
    ETA = A_c[:, None] * gg[None, :] - b_c[:, None]
    LP1 = log_expit(ETA)
    LP0 = log_expit(-ETA)
    DIFF = LP1 - LP0
    SLP0 = {m: [LP0[blk].sum(axis=0) for blk in blocks[m]] for m in models}

    # deployed observed-label op-point theta / SE_ability / lengths / stop reason
    op_theta_obs = np.full(n_models, np.nan)
    op_se_ab = np.full(n_models, np.nan)
    op_nscen = np.zeros(n_models, dtype=int)
    op_ncrit = np.zeros(n_models, dtype=int)
    op_reason = [""] * n_models
    for i, m in enumerate(models):
        walk = build_walk(Ysub[i], blocks[m], SLP0[m], DIFF, gg, lp)
        th, sd, nsc, ncr, reason = op_theta_at_stop(Ysub[i], walk, A2d, b_c, egrid, elog)
        op_theta_obs[i], op_se_ab[i] = th, sd
        op_nscen[i], op_ncrit[i], op_reason[i] = nsc, ncr, reason
    op_hitcap = [r == "cap" for r in op_reason]
    print(f"deployed op-point f15/se0.25: mean scen={op_nscen.mean():.1f} "
          f"mean crit={op_ncrit.mean():.1f}  theta range "
          f"[{np.nanmin(op_theta_obs):.3f},{np.nanmax(op_theta_obs):.3f}]  "
          f"precision={sum(r == 'precision' for r in op_reason)} "
          f"plateau={sum(r == 'plateau' for r in op_reason)} cap={sum(op_hitcap)}")

    # ---- Tier A resampling loop (frozen bank) ----
    print(f"\nTier A: B={args.b_tier_a} replicates (frozen bank) ...", flush=True)
    rng = np.random.default_rng(args.seed)
    B = args.b_tier_a
    th_full = np.empty((B, n_models))
    th_op = np.full((B, n_models), np.nan)
    for t in range(B):
        alpha_c, beta_c = L.sample_alpha_beta(post, own_mask, rng)
        pi_c = L.debias_pi(p_obs_c, alpha_c, beta_c)
        r_pass, r_fail = L.bayes_pass_probs(pi_c, alpha_c, beta_c)
        Ytrue = L.resample_true_labels(Ysub, Msub, r_pass, r_fail, rng)
        th_full[t] = scat.eap_all_models(Ytrue, Msub, A2d, b_c, egrid, elog)[:, 0]
        for i, m in enumerate(models):
            walk = build_walk(Ytrue[i], blocks[m], SLP0[m], DIFF, gg, lp)
            th, _, _, _, _ = op_theta_at_stop(Ytrue[i], walk, A2d, b_c, egrid, elog)
            th_op[t, i] = th
        if (t + 1) % 50 == 0:
            print(f"  tierA {t + 1}/{B}", flush=True)

    se_judge_full = th_full.std(axis=0, ddof=1)
    theta_deb_full = th_full.mean(axis=0)
    se_judge_op = np.nanstd(th_op, axis=0, ddof=1)
    theta_deb_op = np.nanmean(th_op, axis=0)

    # ---- bias band (alpha CI corners, beta=0) ----
    print("bias band at alpha CI corners (beta=0) ...", flush=True)
    band_full_lo, band_full_hi, band_op_lo, band_op_hi = bias_band(
        Ysub, Msub, p_obs_c, own_mask, A2d, b_c, egrid, elog, gg, lp,
        blocks, SLP0, DIFF, models, args.b_band, args.seed)

    # ---- assemble per-model CSVs (SE_total via the frq_total_se quadrature) ----
    full_df = pd.DataFrame({
        "model": models, "theta": theta_full_eap, "theta_mwle": theta_full_mwle,
        "theta_debiased": theta_deb_full,
        "se_ability": se_ability_full, "se_param": se_param_full, "se_judge": se_judge_full,
        "bias_band_low": band_full_lo, "bias_band_high": band_full_hi,
        "observed_pass_rate": obs_pass,
    })
    full_df["se_total"] = fts.se_quadrature(full_df["se_ability"], full_df["se_param"],
                                            full_df["se_judge"])
    full_df["judge"] = JUDGE
    full_df = full_df.sort_values("theta", ascending=False).reset_index(drop=True)
    full_df.insert(0, "rank", np.arange(1, len(full_df) + 1))
    full_df.to_csv(OUT / "per_model_full_bank.csv", index=False)

    op_df = pd.DataFrame({
        "model": models, "theta": op_theta_obs, "theta_debiased": theta_deb_op,
        "se_ability": op_se_ab, "se_param": se_param_op, "se_judge": se_judge_op,
        "bias_band_low": band_op_lo, "bias_band_high": band_op_hi,
        "n_admin_scenarios": op_nscen, "n_admin_criteria": op_ncrit,
        "stop_reason": op_reason, "hit_cap": op_hitcap, "observed_pass_rate": obs_pass,
    })
    op_df["se_total"] = fts.se_quadrature(op_df["se_ability"], op_df["se_param"],
                                          op_df["se_judge"])
    op_df["judge"] = JUDGE
    op_df = op_df.sort_values("theta", ascending=False).reset_index(drop=True)
    op_df.insert(0, "rank", np.arange(1, len(op_df) + 1))
    op_df.to_csv(OUT / "per_model_oppoint_f15se25.csv", index=False)

    # ---- Tier B ----
    tierB = None
    if not args.skip_tier_b:
        tierB = tier_b(Ysub, Msub, p_obs_c, own_mask, post, A2d, b_c, egrid, elog,
                       models, args.b_tier_b, args.seed + 1, se_judge_full)
        pd.DataFrame({
            "model": models, "theta_full": theta_full_eap,
            "se_judge_tierA_full": se_judge_full,
            "se_judge_tierB_full": tierB["se_judge_full_B"],
            "se_judge_diff_B_minus_A": tierB["se_judge_full_B"] - se_judge_full,
        }).to_csv(OUT / "tierB_sensitivity.csv", index=False)

    # ---- de-bias direction / monotonicity GATE + rank stability ----
    gate = _gate(full_df, op_df, post, args.seed)

    # ---- summary.json ----
    _write_summary(post, confusion, full_df, op_df, tierB, gate, args, bank_stats,
                   op_nscen, op_ncrit, op_reason, op_hitcap, t0)

    print(f"\nwrote -> {OUT}")
    print(f"  per_model_full_bank.csv       (SE_judge med={np.median(se_judge_full):.4f} "
          f"max={se_judge_full.max():.4f})")
    print(f"  per_model_oppoint_f15se25.csv (SE_judge med={np.nanmedian(se_judge_op):.4f} "
          f"max={np.nanmax(se_judge_op):.4f})")
    print(f"  de-bias direction gate: full pass={gate['full_bank']['frac_monotone']:.2f} "
          f"op pass={gate['oppoint']['frac_monotone']:.2f}  -> path={gate['recommended_path']}")
    print(f"elapsed {datetime.now() - t0}")
    return 0


def _gate(full_df, op_df, post, seed):
    """Direction/monotonicity gate + rank stability + aggregate pass-rate de-bias factor."""
    def _one(df, eps=1e-6):
        th = df["theta"].to_numpy(float)
        de = df["theta_debiased"].to_numpy(float)
        ok = np.isfinite(th) & np.isfinite(de)
        shift = de[ok] - th[ok]
        return {
            "n": int(ok.sum()),
            "frac_monotone": float(np.mean(shift >= -eps)),
            "n_violations": int(np.sum(shift < -eps)),
            "shift_min": float(shift.min()), "shift_median": float(np.median(shift)),
            "shift_mean": float(shift.mean()), "shift_max": float(shift.max()),
            "spearman_obs_vs_debiased": float(spearmanr(th[ok], de[ok]).correlation),
            "pearson_obs_vs_debiased": float(np.corrcoef(th[ok], de[ok])[0, 1]),
        }

    fb = _one(full_df)
    op = _one(op_df)

    # aggregate pass-rate de-bias factor 1/(1-alpha) (beta=0) from the OVERALL alpha posterior
    rng = np.random.default_rng(seed + 123)
    a_draw = rng.beta(post["overall"]["alpha_a"], post["overall"]["alpha_b"], size=20000)
    fac = 1.0 / np.clip(1.0 - a_draw, 1e-6, None)
    passed = fb["frac_monotone"] >= 0.98 and op["frac_monotone"] >= 0.95
    return {
        "definition": ("gate: per-model theta_debiased >= observed theta (>=98% full-bank, "
                       ">=95% op-point) AND no shrinkage-crossover (top models do not decrease)."),
        "full_bank": fb, "oppoint": op,
        "pass_rate_debias_factor": {
            "point_1_over_1_minus_alpha": 1.0 / (1.0 - post["overall"]["alpha_mean"]),
            "posterior_median": float(np.median(fac)),
            "posterior_ci95": [float(np.percentile(fac, 2.5)), float(np.percentile(fac, 97.5))],
            "ci_corners_from_alpha_ci": [1.0 / (1.0 - L.ALPHA_CI[0]), 1.0 / (1.0 - L.ALPHA_CI[1])],
            "note": "beta=0 => de-bias is purely UP: p_true = p_obs/(1-alpha). Direction firm; "
                    "magnitude band wide (only 20 gold-PASS cells).",
        },
        "gate_passed": bool(passed),
        "recommended_path": ("per_model_band" if passed else "cohort_fallback"),
    }


def _write_summary(post, confusion, full_df, op_df, tierB, gate, args, bank_stats,
                   op_nscen, op_ncrit, op_reason, op_hitcap, t0):
    fb = fts.regime_stats(full_df)
    op = fts.regime_stats(op_df)
    summary = {
        "phase": 3, "layer": "SE_judge + de-bias", "benchmark": "TutorEval", "judge": JUDGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": ("LOCAL / STUDY ONLY -- engine untouched (tutor_cat, scenario_cat_lib, "
                   "calibrate_mirt read-only); nothing committed/staged; calibrated on OBSERVED "
                   "gemini-3 labels (Rule 0). STOP at the SE_judge / de-bias interpretation "
                   "checkpoint (do NOT proceed to Phase 4)."),
        "op_point": {"floor": OP_FLOOR, "se_ability_target": OP_TARGET,
                     "stop": "dense-EAP posterior SD", "dense_grid_nodes": DENSE_GRID,
                     "dense_range": DENSE_RANGE, "plateau_delta": PLATEAU_DELTA,
                     "plateau_w": PLATEAU_W, "cap": OP_CAP, "estimator": "MWLE-at-stop",
                     "order": "frozen deployed adaptive order (observed labels, engine seed 42); "
                              "stop moves with the resample along that order"},
        "config": {"b_tier_a": args.b_tier_a, "b_tier_b": args.b_tier_b, "b_band": args.b_band,
                   "seed": args.seed, "fit_grid_gh": FIT_GRID, "ridge": RIDGE,
                   "ref_grid": REF_GRID, "ref_range": REF_RANGE, "dense_grid": DENSE_GRID,
                   "n_boot_separam_full": N_BOOT_SEPARAM_FULL,
                   "full_bank_estimator": "full-bank EAP posterior mean (REF_GRID)",
                   "se_param_full": "vectorized observed-info parametric bootstrap (all cells)",
                   "se_param_oppoint": "Phase-1 deployed parametric-bootstrap offset (fixed/model)",
                   "resample": "correlated-within-criterion (one latent uniform/column), single "
                               "de-bias, per-criterion pi_c, stratum alpha (own vs pooled), beta=0"},
        "paths": {
            "bank": str(BANK.relative_to(ROOT)).replace("/", "\\"),
            "matrix": str(MATRIX.relative_to(ROOT)).replace("/", "\\"),
            "confusion": str(CONFUSION.relative_to(ROOT)).replace("/", "\\"),
            "gold_labels": str(GOLD.relative_to(ROOT)).replace("/", "\\"),
            "phase1_leaderboard": str(PHASE1_LB.relative_to(ROOT)).replace("/", "\\"),
            "out_dir": str(OUT.relative_to(ROOT)).replace("/", "\\"),
        },
        "bank_load_stats": bank_stats,
        "confusion_model": {
            "overall": confusion["overall"],
            "alpha_posteriors_beta_from_gold_jeffreys": {
                "own_stratum": post["own"]["stratum"],
                "own_alpha_a": post["own"]["alpha_a"], "own_alpha_b": post["own"]["alpha_b"],
                "own_alpha_mean": post["own"]["alpha_mean"], "own_alpha_ci95": post["own"]["alpha_ci95"],
                "own_counts": post["own"]["counts"],
                "overall_alpha_a": post["overall"]["alpha_a"],
                "overall_alpha_b": post["overall"]["alpha_b"],
                "overall_alpha_mean": post["overall"]["alpha_mean"],
                "overall_alpha_ci95": post["overall"]["alpha_ci95"],
                "beta": 0.0, "beta_pinned_reason": "zero false-pass in 100 gold cells (CI [0,0])",
                "pooling": "conceptual_understanding=own alpha (15 gold-PASS>=8); "
                           "quantitative_procedural(3)+missing/unknown pool to overall alpha (0.45)",
            },
        },
        "regimes": {"full_bank": fb, "oppoint_f15se25": op},
        "op_vs_full_se_judge_ratio_median": (op["se_judge_median"] / fb["se_judge_median"]
                                             if fb["se_judge_median"] > 0 else None),
        "de_bias_gate": gate,
        "op_point_deployed": {
            "mean_scenarios": float(np.mean(op_nscen)),
            "mean_criteria": float(np.mean(op_ncrit)),
            "stop_reason_counts": {r: int(sum(x == r for x in op_reason))
                                   for r in ("precision", "plateau", "cap", "bank_exhausted")},
            "n_hit_cap": int(sum(op_hitcap)),
        },
        "elapsed_seconds": (datetime.now() - t0).total_seconds(),
    }
    if tierB is not None:
        A = tierB["se_judge_full_A"]
        Bv = tierB["se_judge_full_B"]
        mA, mB = float(np.median(A)), float(np.median(Bv))
        within = abs(mB - mA) <= 0.25 * mA
        if mB <= mA * 1.05:
            verdict = ("Tier B <= Tier A: refitting the bank each replicate absorbs judge noise "
                       "into the item parameters; frozen-bank Tier A is the conservative "
                       "of-record (item params do not inflate SE_judge).")
        elif within:
            verdict = "Tier B ~= Tier A (item params not dominating); report Tier A of-record."
        else:
            verdict = ("Tier B exceeds Tier A: item-parameter re-estimation amplifies SE_judge; "
                       "prefer Tier B as of-record.")
        summary["tier_b"] = {
            "b": args.b_tier_b, "se_judge_full_A_median": mA, "se_judge_full_B_median": mB,
            "se_judge_B_minus_A_median": float(np.median(Bv - A)),
            "within_25pct": bool(within),
            "tierA_is_conservative_of_record": bool(mB <= mA * 1.05),
            "verdict": verdict,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
