"""Phase-3: SE_judge (judge-measurement-error) layer for the BiGGen gemini-3 recalibration.

LOCAL scratch driver. It CALLS the frozen production engine (scenario_cat_lib, calibrate_mirt,
biggen_eap_stop_lib, biggen_recovery_grid) -- it NEVER edits it -- and adds only the
judge-error Monte-Carlo layer on top of the of-record theta. Nothing is committed; all writes
go under ``reports/biggen_gemini3_recal/se_judge/``.

Op-point: floor 10 / SE_post 0.12, dense EAP-posterior stop, MWLE-at-stop. Frozen curated
of-record bank (2015 fitted criteria). Two regimes per model:

  1. Full-bank cohort leaderboard: re-score the full-bank fine-EAP posterior-mean theta on the
     resampled labels (frozen bank).  SE_judge_full = SD(theta); theta_debiased_full = mean(theta).
  2. Op-point @10/0.12 (shipped CAT theta): along each model's deployed adaptive administration
     order (the real gemini-3 order), re-apply the dense-EAP floor-10/SE-0.12 stop on the
     resampled labels and take MWLE-at-stop on the stopped administered cells.  SE_judge_op =
     SD(theta); theta_debiased_op = mean(theta).

OF-RECORD scope (see README.md / summary.json): the of-record output of this driver is the
**SD** across replicates -- **SE_judge** -- plus the COHORT pass-rate de-bias factor
(1/(1-alpha-beta) = 1.33x [1.17, 1.57]). The per-model ``theta_debiased`` = mean(theta) is a
**shrinkage estimator** (the per-column-prevalence-prior resample compresses the ability scale)
and is **NOT of-record**; it is retained only as a labeled diagnostic. A proper per-model
noisy-label IRT de-bias is future work.

Resampling (per replicate): sample alpha_s,beta_s per stratum from their Jeffreys Beta
posteriors; de-bias per-criterion pi_c; correlated-within-criterion true-label resample (one
latent uniform per column) via the Bayes posterior.  Tier A = frozen bank (B~500); Tier B =
refit calibrate_mirt each replicate (B~50), identical production config.

Design note (op-point administration order): the deployed op-point administered SET is what the
real gemini-3 labels selected -- so we FREEZE each model's adaptive administration ORDER at the
deployed (observed-label) trace and let the judge-error resample move the dense-EAP STOP POINT
(and the MWLE-at-stop theta) along that order. This captures the dominant "fewer administered
cells => more per-label leverage" effect without 500 full engine re-selections; for a 1-D
max-Fisher-information CAT the re-selection is a second-order effect (the same high-info
scenarios stay on top under a few flipped cells).
"""

from __future__ import annotations

import argparse
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
from scipy.special import log_expit, logsumexp

# eduLLM-Evals root = .../reports/biggen_gemini3_recal/scratch/this -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.scenario_cat_lib as scat  # noqa: E402
import scripts.frq_total_se as fts  # noqa: E402 (quadrature helper -- the intended extension)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import se_judge_lib as L  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
rg = _load("biggen_recovery_grid", ROOT / "biggen_calibration" / "scripts" / "biggen_recovery_grid.py")
esl = _load("biggen_eap_stop_lib", ROOT / "biggen_calibration" / "scripts" / "biggen_eap_stop_lib.py")

DIM = "general"
JUDGE = L.JUDGE
FIT_GRID = 7
RIDGE = 1e-2
EAP_GRID = 3201
EAP_RANGE = 8.0
BOOT_GRID_N = 1201
N_BOOT_SEPARAM = 150     # observed-info parametric bootstrap draws for op-point SE_param
# Op-point: floor 10 / SE_post 0.12, dense EAP stop, forced-order cap.
OP_FLOOR = 10
OP_TARGET = 0.12
OP_LMAX = 40

MATRIX = ROOT / "api_judge_pilot" / "grading_biggen" / "response_matrix.csv"
SCENARIOS = ROOT / "data" / "BiGGen" / "scenarios.jsonl"
BANK = ROOT / "reports" / "biggen_gemini3_recal" / "bank" / "biggen_unidim_modeled_gemini3_curated.jsonl"
GOLD = ROOT / "api_judge_pilot" / "gold" / "biggen_core" / "gold_labels.jsonl"
VERDICTS = ROOT / "api_judge_pilot" / "gold" / "se_judge" / "biggen_gemini3_gold_verdicts_250.jsonl"
CRIT_TO_CAP = ROOT / "reports" / "biggen_gemini3_recal" / "scratch" / "criterion_to_capability.csv"
CURATED_PERMODEL = ROOT / "reports" / "biggen_gemini3_recal" / "scratch" / "per_model_gemini3_curated.csv"

OUT = ROOT / "reports" / "biggen_gemini3_recal" / "se_judge"


# ---------------------------------------------------------------------------
# op-point precompute + per-replicate evaluator (frozen order, moving stop)
# ---------------------------------------------------------------------------


def _engine_bank() -> Path:
    """Write a criticality-sanitized copy of the curated bank for the CAT engine ONLY.

    The engine's critical-failure branch expects a non-null ``criticality`` string; the
    curated bank leaves it null. Coercing null -> 'not_critical' affects ONLY the (unused)
    critical-failure log, never selection/theta/SE. Item params are byte-identical.
    """
    out = OUT / "_bank_engine.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(BANK, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("criticality") is None:
                rec["criticality"] = "not_critical"
            fout.write(json.dumps(rec) + "\n")
    return out


def deployed_order_blocks(models, row_of, a, b, kept_ids, scen_of, gg, lp, workers):
    """Run the deployed CAT once (observed labels) to the forced cap -> per-model scenario
    blocks (ordered lists of column indices, one np.array per administered scenario, capped
    at OP_LMAX) and the deployed observed-label stop (idx set + n_scen + eap_sd)."""
    col = {c: i for i, c in enumerate(kept_ids)}
    engine_bank = _engine_bank()
    spec = scat.RunSpec(seed=20260729, top_n=5, max_se=0.0, min_evals_per_skill=0,
                        min_scenarios=OP_LMAX, max_scenarios=OP_LMAX, selection="trace",
                        mode="cat", runs_dir=str(OUT / "_op_runs"))
    res = scat.run_models(models, engine_bank, MATRIX, SCENARIOS, "clamp", DIM_LIST, spec, workers=workers)
    res_by = {r["model"]: r for r in res}
    blocks = {}
    deployed = {}
    for m in models:
        order = [c for c in res_by[m]["order"] if c in col]
        # group consecutive criteria by scenario, preserving administration order
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
        blocks[m] = blk[:OP_LMAX]
    import shutil
    shutil.rmtree(OUT / "_op_runs", ignore_errors=True)
    return col, blocks


def op_stop_idx(y, blocks_m, SLP0_m, DIFF, gg, lp, floor=OP_FLOOR, target=OP_TARGET):
    """Dense-EAP floor/target stop along a model's frozen scenario blocks under labels ``y``.

    ``SLP0_m[si]`` = precomputed sum of log P(fail|theta) over scenario si's items (rep-
    independent); the y-dependent term is ``y[block] @ DIFF[block]`` where DIFF = logP1-logP0.
    Returns the cumulative administered column-index array at the stop (or the capped last).
    """
    ll = np.zeros(gg.size)
    cum = []
    n = len(blocks_m)
    n_scen_stop = 0
    for si in range(n):
        block = blocks_m[si]
        ll = ll + SLP0_m[si] + y[block] @ DIFF[block]
        cum.extend(block.tolist())
        n_scen = si + 1
        n_scen_stop = n_scen
        if n_scen >= floor:
            post = np.exp(ll + lp - logsumexp(ll + lp))
            mean = post @ gg
            sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
            if sd <= target:
                return np.array(cum, dtype=int), True, n_scen_stop
    return np.array(cum, dtype=int), False, n_scen_stop   # hit cap (precision not reached)


def op_theta_mwle(y, idx, A2, b, egrid, elog):
    """MWLE-at-stop theta over administered cells ``idx`` (EAP-warm-started)."""
    if idx.size == 0:
        return float("nan")
    th_ba = scat.eap_subset(y, idx, A2, b, egrid, elog)
    th_mw, _ = scat.mwle_subset(y, idx, A2, b, th_ba)
    return float(th_mw[0])


DIM_LIST = [DIM]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b-tier-a", type=int, default=500)
    ap.add_argument("--b-tier-b", type=int, default=50)
    ap.add_argument("--b-band", type=int, default=200, help="MC draws per bias-band corner.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--skip-tier-b", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()

    # ---- load frozen curated bank + matrix ----
    records, dims, _ = scat.load_fitted_bank(BANK, "clamp")
    assert dims == DIM_LIST, dims
    kept_ids, A2, bvec = scat.assemble_arrays(records, dims)   # A2 (n,1), b (n,)
    A = A2[:, 0]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}

    matrix = pd.read_csv(MATRIX, index_col=0)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    sub = matrix.reindex(columns=kept_ids)
    Yobs = np.nan_to_num(sub.to_numpy(float), nan=0.0)
    Mobs = ~np.isnan(sub.to_numpy(float))
    n_models, n_items = Yobs.shape
    p_obs_c = np.array([Yobs[Mobs[:, j], j].mean() if Mobs[:, j].any() else 0.0
                        for j in range(n_items)])
    print(f"bank={BANK.name} models={n_models} criteria={n_items} fill={Mobs.mean():.4f}")

    # ---- strata Beta posteriors + per-criterion stratum index ----
    post = L.strata_posteriors(GOLD, VERDICTS)
    sidx = L.stratum_index(kept_ids, CRIT_TO_CAP)
    print("per-stratum alpha/beta posteriors (mean [95%]):")
    for s in L.STRATA:
        p = post[s]
        print(f"  {s:22s} alpha={p['alpha_mean']:.3f} [{p['alpha_ci95'][0]:.3f},"
              f"{p['alpha_ci95'][1]:.3f}]  beta={p['beta_mean']:.3f} "
              f"[{p['beta_ci95'][0]:.3f},{p['beta_ci95'][1]:.3f}]")

    # ---- grids ----
    egrid, elog = scat.build_grid(1, EAP_GRID, EAP_RANGE)   # full-bank fine EAP (3201,[-8,8])
    gg, lp = esl.eap_grid(EAP_GRID, EAP_RANGE)              # dense EAP stop grid (same)

    # ---- of-record point estimators on OBSERVED labels ----
    theta_full_obs = scat.eap_all_models(Yobs, Mobs, A2, bvec, egrid, elog)[:, 0]
    obs_pass = np.array([Yobs[i][Mobs[i]].mean() for i in range(n_models)])

    # full-bank SE_ability / SE_param from the of-record curated decomposition
    cur = pd.read_csv(CURATED_PERMODEL).set_index("model")
    se_ab_full = np.array([float(cur.loc[m, "se_ability"]) for m in models])
    se_pa_full = np.array([float(cur.loc[m, "se_param"]) for m in models])

    # ---- deployed op-point administration (observed labels): order blocks + precompute ----
    print("running the deployed CAT once (observed labels) for the op-point order ...", flush=True)
    col, blocks = deployed_order_blocks(models, row_of, A, bvec, kept_ids, scen_of, gg, lp,
                                        args.workers)
    # rep-independent precompute over the fine stop grid
    ETA = A[:, None] * gg[None, :] - bvec[:, None]     # (n_items, n_grid)
    LP1 = log_expit(ETA)
    LP0 = log_expit(-ETA)
    DIFF = LP1 - LP0
    SLP0 = {m: [LP0[blk].sum(axis=0) for blk in blocks[m]] for m in models}

    # deployed observed-label op-point: stop idx + MWLE theta + SE_ability/SE_param
    cov = rg.compute_item_cov(BANK, MATRIX, FIT_GRID, RIDGE)
    bgrid = np.linspace(-EAP_RANGE, EAP_RANGE, BOOT_GRID_N)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(args.seed + 7)
    op_idx_obs, op_theta_obs, op_nscen, op_ncrit, op_hitcap = {}, {}, {}, {}, {}
    op_se_ab, op_se_pa = {}, {}
    for i, m in enumerate(models):
        y = Yobs[i]
        idx, reached, nsc = op_stop_idx(y, blocks[m], SLP0[m], DIFF, gg, lp)
        op_idx_obs[m] = idx
        op_theta_obs[m] = op_theta_mwle(y, idx, A2, bvec, egrid, elog)
        op_ncrit[m] = int(idx.size)
        op_nscen[m] = nsc
        op_hitcap[m] = bool(not reached)
        _, se_post, se_par = rg.bootstrap_theta(y, idx, cov["beta"], cov["chol"],
                                                bgrid, blp, N_BOOT_SEPARAM, rng_boot, batch=8)
        op_se_ab[m] = float(se_post)
        op_se_pa[m] = float(se_par)
    op_se_ab = np.array([op_se_ab[m] for m in models])
    op_se_pa = np.array([op_se_pa[m] for m in models])
    print(f"deployed op-point: mean scen={np.mean([op_nscen[m] for m in models]):.1f} "
          f"mean crit={np.mean([op_ncrit[m] for m in models]):.1f}  "
          f"theta range [{min(op_theta_obs.values()):.3f},{max(op_theta_obs.values()):.3f}]")

    # ---- Tier A resampling loop (frozen bank) ----
    print(f"\nTier A: B={args.b_tier_a} replicates (frozen bank) ...", flush=True)
    rng = np.random.default_rng(args.seed)
    B = args.b_tier_a
    th_full = np.empty((B, n_models))
    th_op = np.empty((B, n_models))
    for t in range(B):
        alpha_s, beta_s = L.sample_rates(post, rng)
        alpha_c, beta_c = alpha_s[sidx], beta_s[sidx]
        pi_c = L.debias_pi(p_obs_c, alpha_c, beta_c)
        r_pass, r_fail = L.bayes_pass_probs(pi_c, alpha_c, beta_c)
        Ytrue = L.resample_true_labels(Yobs, Mobs, r_pass, r_fail, rng)
        th_full[t] = scat.eap_all_models(Ytrue, Mobs, A2, bvec, egrid, elog)[:, 0]
        for i, m in enumerate(models):
            y = Ytrue[i]
            idx, _, _ = op_stop_idx(y, blocks[m], SLP0[m], DIFF, gg, lp)
            th_op[t, i] = op_theta_mwle(y, idx, A2, bvec, egrid, elog)
        if (t + 1) % 50 == 0:
            print(f"  tierA {t+1}/{B}", flush=True)

    se_judge_full = th_full.std(axis=0, ddof=1)
    theta_deb_full = th_full.mean(axis=0)
    se_judge_op = np.nanstd(th_op, axis=0, ddof=1)
    theta_deb_op = np.nanmean(th_op, axis=0)

    # ---- systematic bias band (alpha/beta CI corners, fixed rates, MC-averaged) ----
    print("bias band at alpha/beta CI corners ...", flush=True)
    band_full_lo, band_full_hi, band_op_lo, band_op_hi = _bias_band(
        Yobs, Mobs, p_obs_c, sidx, A2, bvec, egrid, elog, gg, lp, blocks, SLP0, DIFF,
        models, args.b_band, args.seed)

    # ---- assemble per-model CSVs (SE_total via the frq_total_se quadrature) ----
    full_df = pd.DataFrame({
        "model": models, "theta": theta_full_obs, "theta_debiased": theta_deb_full,
        "se_ability": se_ab_full, "se_param": se_pa_full, "se_judge": se_judge_full,
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
        "model": models,
        "theta": [op_theta_obs[m] for m in models], "theta_debiased": theta_deb_op,
        "se_ability": op_se_ab, "se_param": op_se_pa, "se_judge": se_judge_op,
        "bias_band_low": band_op_lo, "bias_band_high": band_op_hi,
        "n_admin_scenarios": [op_nscen[m] for m in models],
        "n_admin_criteria": [op_ncrit[m] for m in models],
        "hit_cap": [op_hitcap[m] for m in models],
        "observed_pass_rate": obs_pass,
    })
    op_df["se_total"] = fts.se_quadrature(op_df["se_ability"], op_df["se_param"],
                                          op_df["se_judge"])
    op_df["judge"] = JUDGE
    op_df = op_df.sort_values("theta", ascending=False).reset_index(drop=True)
    op_df.insert(0, "rank", np.arange(1, len(op_df) + 1))
    op_df.to_csv(OUT / "per_model_oppoint_f10se12.csv", index=False)

    # ---- Tier B (refit each replicate) ----
    tierB = None
    if not args.skip_tier_b:
        tierB = _tier_b(Yobs, Mobs, p_obs_c, sidx, post, A2, bvec, kept_ids, egrid, elog,
                        models, args.b_tier_b, args.seed + 1, se_judge_full)
        tb_df = pd.DataFrame({
            "model": models,
            "theta_full": theta_full_obs,
            "se_judge_tierA_full": se_judge_full,
            "se_judge_tierB_full": tierB["se_judge_full_B"],
            "se_judge_diff_B_minus_A": tierB["se_judge_full_B"] - se_judge_full,
        })
        tb_df.to_csv(OUT / "tierB_sensitivity.csv", index=False)

    # ---- summary.json ----
    _write_summary(post, full_df, op_df, tierB, args, models,
                   op_nscen, op_ncrit, op_hitcap, t0)

    print(f"\nwrote -> {OUT}")
    print(f"  per_model_full_bank.csv    (SE_judge med={np.median(se_judge_full):.4f} "
          f"max={se_judge_full.max():.4f})")
    print(f"  per_model_oppoint_f10se12.csv (SE_judge med={np.nanmedian(se_judge_op):.4f} "
          f"max={np.nanmax(se_judge_op):.4f})")
    print(f"  theta_debiased shift median: full={np.median(theta_deb_full-theta_full_obs):+.4f} "
          f"op={np.nanmedian(theta_deb_op-np.array([op_theta_obs[m] for m in models])):+.4f}")
    print(f"elapsed {datetime.now()-t0}")
    return 0


def _bias_band(Yobs, Mobs, p_obs_c, sidx, A2, bvec, egrid, elog, gg, lp, blocks, SLP0, DIFF,
               models, K, seed):
    """theta at the (alpha,beta) CI corners, MC-averaged over K flip draws (fixed rates).

    high corner (max upward de-bias): alpha=CI_hi, beta=CI_lo; low corner: alpha=CI_lo, beta=CI_hi.
    Applied uniformly across strata using the OVERALL confusion CI.
    """
    bvec_ = bvec
    A = A2[:, 0]
    n_models, n_items = Yobs.shape
    corners = {
        "hi": (L.ALPHA_CI[1], L.BETA_CI[0]),   # more false-fails corrected -> higher theta
        "lo": (L.ALPHA_CI[0], L.BETA_CI[1]),
    }
    res = {}
    for tag, (a_val, b_val) in corners.items():
        rng = np.random.default_rng(seed + (0 if tag == "hi" else 999))
        alpha_c = np.full(n_items, a_val)
        beta_c = np.full(n_items, b_val)
        pi_c = L.debias_pi(p_obs_c, alpha_c, beta_c)
        r_pass, r_fail = L.bayes_pass_probs(pi_c, alpha_c, beta_c)
        acc_full = np.zeros(n_models)
        acc_op = np.zeros(n_models)
        for _ in range(K):
            Ytrue = L.resample_true_labels(Yobs, Mobs, r_pass, r_fail, rng)
            acc_full += scat.eap_all_models(Ytrue, Mobs, A2, bvec_, egrid, elog)[:, 0]
            for i, m in enumerate(models):
                idx, _, _ = op_stop_idx(Ytrue[i], blocks[m], SLP0[m], DIFF, gg, lp)
                acc_op[i] += op_theta_mwle(Ytrue[i], idx, A2, bvec_, egrid, elog)
        res[f"full_{tag}"] = acc_full / K
        res[f"op_{tag}"] = acc_op / K
    return res["full_lo"], res["full_hi"], res["op_lo"], res["op_hi"]


def _tier_b(Yobs, Mobs, p_obs_c, sidx, post, A2, bvec, kept_ids, egrid, elog, models,
            B, seed, se_judge_full_A):
    """Tier B: refit calibrate_mirt each replicate (identical production config), re-score
    full-bank EAP theta. SE_judge_full_B = SD across refits. Confirms item params aren't
    dominating (if ~= Tier A, report Tier A as of-record).

    Same pass-imbalance exclusion == the frozen curated column set (production convention):
    we refit those 2015 columns on each replicate's resampled labels (unidim, 7 GH, ridge
    0.01, max_iter 200 -- identical to the production fit)."""
    print(f"\nTier B: B={B} refits (calibrate_mirt each replicate) ...", flush=True)
    rng = np.random.default_rng(seed)
    n_models, n_items = Yobs.shape
    Q = np.ones((n_items, 1), dtype=int)
    th_B = np.empty((B, n_models))
    for t in range(B):
        alpha_s, beta_s = L.sample_rates(post, rng)
        alpha_c, beta_c = alpha_s[sidx], beta_s[sidx]
        pi_c = L.debias_pi(p_obs_c, alpha_c, beta_c)
        r_pass, r_fail = L.bayes_pass_probs(pi_c, alpha_c, beta_c)
        Ytrue = L.resample_true_labels(Yobs, Mobs, r_pass, r_fail, rng)
        fit = cm.fit_m2pl_em(Ytrue, Mobs, Q, FIT_GRID, estimate_corr=False,
                             ridge=RIDGE, max_iter=200)
        Ak = fit["A"]
        bk = fit["b"]
        th_B[t] = scat.eap_all_models(Ytrue, Mobs, Ak, bk, egrid, elog)[:, 0]
        if (t + 1) % 10 == 0:
            print(f"  tierB {t+1}/{B}", flush=True)
    se_judge_full_B = th_B.std(axis=0, ddof=1)
    theta_deb_full_B = th_B.mean(axis=0)
    return {"se_judge_full_B": se_judge_full_B, "theta_deb_full_B": theta_deb_full_B,
            "se_judge_full_A": se_judge_full_A}


def _write_summary(post, full_df, op_df, tierB, args, models, op_nscen, op_ncrit, op_hitcap, t0):
    fb = fts._biggen_regime_stats(full_df)
    op = fts._biggen_regime_stats(op_df)
    strata = {}
    for s in L.STRATA:
        p = post[s]
        strata[s] = {"counts": p["counts"],
                     "alpha_mean": p["alpha_mean"], "alpha_ci95": p["alpha_ci95"],
                     "beta_mean": p["beta_mean"], "beta_ci95": p["beta_ci95"]}

    def _factor(a, b):
        return 1.0 / max(1.0 - a - b, 1e-9)

    summary = {
        "phase": 3, "layer": "SE_judge + de-bias", "judge": JUDGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_only": True, "engine_untouched": True, "committed": False,
        "op_point": {"floor": OP_FLOOR, "se_target": OP_TARGET, "stop": "dense-EAP posterior SD",
                     "estimator": "MWLE-at-stop", "forced_order_cap": OP_LMAX,
                     "order": "frozen deployed adaptive order (observed labels); stop moves with resample"},
        "config": {"b_tier_a": args.b_tier_a, "b_tier_b": args.b_tier_b, "b_band": args.b_band,
                   "seed": args.seed, "fit_grid_gh": FIT_GRID, "ridge": RIDGE,
                   "eap_grid": EAP_GRID, "eap_range": EAP_RANGE,
                   "full_bank_estimator": "full-bank fine-EAP posterior mean",
                   "resample": "correlated-within-criterion (one latent uniform/column), "
                               "single de-bias, per-criterion pi_c, stratum alpha/beta"},
        "paths": {
            "bank_curated": str(BANK.relative_to(ROOT)).replace("/", "\\"),
            "matrix": str(MATRIX.relative_to(ROOT)).replace("/", "\\"),
            "confusion": "api_judge_pilot\\gold\\se_judge\\biggen_gemini3_confusion.json",
            "gold_labels": str(GOLD.relative_to(ROOT)).replace("/", "\\"),
            "judge_verdicts": str(VERDICTS.relative_to(ROOT)).replace("/", "\\"),
            "stratum_map": str(CRIT_TO_CAP.relative_to(ROOT)).replace("/", "\\"),
            "out_dir": str(OUT.relative_to(ROOT)).replace("/", "\\"),
        },
        "strata_posteriors": strata,
        "strata_overall_counts": post["_overall_counts"],
        "regimes": {"full_bank": fb, "oppoint_f10se12": op},
        "op_vs_full_se_judge_ratio_median": (op["se_judge_median"] / fb["se_judge_median"]
                                             if fb["se_judge_median"] > 0 else None),
        "op_vs_full_se_judge_ratio_mean": (op["se_judge_mean"] / fb["se_judge_mean"]
                                           if fb["se_judge_mean"] > 0 else None),
        "debias": {
            "theta_shift_median_full": fb.get("theta_debias_shift_median"),
            "theta_shift_median_op": op.get("theta_debias_shift_median"),
            "pass_rate_debias_factor_point": _factor(L.ALPHA_OVERALL, L.BETA_OVERALL),
            # factor = 1/(1-alpha-beta): widest envelope over the CI box (both rates low / both
            # high). Matches the doc's ~[1.17, 1.57]x.
            "pass_rate_debias_factor_band": [_factor(L.ALPHA_CI[0], L.BETA_CI[0]),
                                             _factor(L.ALPHA_CI[1], L.BETA_CI[1])],
            "note": "p_true ~ (p_obs - beta)/(1 - alpha - beta); direction firm (alpha strict "
                    "=> theta biased down => de-bias UP), magnitude band wide (250 gold cells).",
        },
        "op_point_deployed": {
            "mean_scenarios": float(np.mean([op_nscen[m] for m in models])),
            "mean_criteria": float(np.mean([op_ncrit[m] for m in models])),
            "n_hit_cap": int(sum(op_hitcap[m] for m in models)),
        },
        "elapsed_seconds": (datetime.now() - t0).total_seconds(),
    }
    if tierB is not None:
        A = tierB["se_judge_full_A"]
        Bv = tierB["se_judge_full_B"]
        mA, mB = float(np.median(A)), float(np.median(Bv))
        within = abs(mB - mA) <= 0.25 * mA
        if mB <= mA * 1.05:
            # Refit absorbs the resampled judge noise into the item params => the frozen-bank
            # Tier A is the CONSERVATIVE (upper) of-record; item params do NOT amplify SE_judge.
            verdict = ("Tier B <= Tier A: refitting the bank each replicate absorbs judge noise "
                       "into the item parameters; frozen-bank Tier A is the conservative "
                       "of-record (item params do not inflate SE_judge).")
        elif within:
            verdict = "Tier B ~= Tier A (item params not dominating); report Tier A of-record."
        else:
            verdict = ("Tier B exceeds Tier A: item-parameter re-estimation amplifies SE_judge; "
                       "prefer Tier B as of-record.")
        summary["tier_b"] = {
            "b": args.b_tier_b,
            "se_judge_full_A_median": mA,
            "se_judge_full_B_median": mB,
            "se_judge_B_minus_A_median": float(np.median(Bv - A)),
            "se_judge_B_over_A_median": float(np.median(Bv / np.where(A > 0, A, np.nan))),
            "within_25pct": bool(within),
            "tierA_is_conservative_of_record": bool(mB <= mA * 1.05),
            "verdict": verdict,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
