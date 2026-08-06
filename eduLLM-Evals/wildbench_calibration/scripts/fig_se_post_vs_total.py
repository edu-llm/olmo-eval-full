"""Diagnostic: SE_posterior vs SE_total against the 0.12 CAT SE target (WildBench, 1-D, MWLE).

Two regimes on the locked catpool bank (fine uniform-EAP; observed-info parametric bootstrap):
  * FULL-BANK (leaderboard regime): each model scored on ALL its administrable CAT-pool
    criteria -> maximal info -> both SEs expected well below the target.
  * DEPLOYED CAT @ 8/0.12: each model scored on the engine's administered set at the locked
    stop rule -> SE_posterior sits ~at the 0.12 target by construction, SE_total = target (+)
    SE_param.

Per model, B parameter-bootstrap replicates yield, per replicate b, a posterior mean theta_b
and posterior SD s_b over the administered items. Point estimates:
    SE_posterior = mean_b(s_b);  SE_param = std_b(theta_b);  SE_total = sqrt(SE_post^2+SE_param^2).
95% CI whiskers = +/- 1.96 x the bootstrap SE-of-the-estimate, where the SE-of-the-estimate is
the std over an M-resample META-bootstrap of the B replicate indices (recomputing each
quantity per meta-resample). SE_posterior CI comes from the spread of the B replicate posterior
SEs; SE_param / SE_total CI from the meta-bootstrap of the SD statistic.

Outputs:
  experiments/07_parameter_uncertainty/figures/se_post_vs_total.png
  experiments/07_parameter_uncertainty/se_post_vs_total.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import wildbench_scenario_lib as L  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402

_pu_spec = importlib.util.spec_from_file_location(
    "wb_param_uncertainty", _HERE / "scenario_param_uncertainty.py")
PU = importlib.util.module_from_spec(_pu_spec)
sys.modules["wb_param_uncertainty"] = PU
_pu_spec.loader.exec_module(PU)

SE_TARGET = 0.12


def bootstrap_full(y, idx, beta, chol, grid, log_prior, n_boot, rng, batch=10):
    """Param bootstrap over administered items ``idx``. Returns (se_post0, thetas, sds).

    ``se_post0`` = posterior SD at the frozen params; ``thetas``/``sds`` = per-replicate
    posterior mean / SD (length n_boot). 1D only (single loading + intercept)."""
    idx = np.asarray(idx, dtype=int)
    k = idx.size
    if k == 0:
        return np.nan, np.array([]), np.array([])
    beta_sub = np.stack([beta[j] for j in idx])            # (k, 2): [a, -b]
    L_sub = np.stack([chol[j] for j in idx])               # (k, 2, 2)
    yk = y[idx]

    a0 = beta_sub[:, 0]
    mb0 = beta_sub[:, 1]
    eta0 = a0[:, None] * grid[None, :] + mb0[:, None]
    ll0 = yk @ log_expit(eta0) + (1.0 - yk) @ log_expit(-eta0)
    post0 = np.exp(ll0 + log_prior - logsumexp(ll0 + log_prior))
    mean0 = float(post0 @ grid)
    se_post0 = float(np.sqrt(max(post0 @ (grid ** 2) - mean0 ** 2, 0.0)))

    thetas = np.empty(n_boot)
    sds = np.empty(n_boot)
    done = 0
    while done < n_boot:
        bsz = min(batch, n_boot - done)
        z = rng.standard_normal((bsz, k, 2))
        draw = beta_sub[None] + np.einsum("kij,bkj->bki", L_sub, z)
        a_d = draw[:, :, 0]
        mb_d = draw[:, :, 1]
        eta = a_d[:, :, None] * grid[None, None, :] + mb_d[:, :, None]
        ll = (np.einsum("k,bkn->bn", yk, log_expit(eta))
              + np.einsum("k,bkn->bn", 1.0 - yk, log_expit(-eta)))
        post = np.exp(ll + log_prior[None, :]
                      - logsumexp(ll + log_prior[None, :], axis=1)[:, None])
        m = post @ grid
        v = post @ (grid ** 2) - m ** 2
        thetas[done:done + bsz] = m
        sds[done:done + bsz] = np.sqrt(np.clip(v, 0.0, None))
        done += bsz
    return se_post0, thetas, sds


def meta_ci(thetas, sds, se_post0, M, rng):
    """Point + SE-of-estimate (via M-resample meta-bootstrap of the B replicates)."""
    B = thetas.size
    se_post = float(np.mean(sds)) if B else np.nan
    se_param = float(np.std(thetas, ddof=1)) if B > 1 else 0.0
    se_total = float(np.sqrt(se_post ** 2 + se_param ** 2))
    if B < 2:
        return dict(se_post=se_post, se_post_se=0.0, se_param=se_param, se_param_se=0.0,
                    se_total=se_total, se_total_se=0.0)
    sp, spr, st = np.empty(M), np.empty(M), np.empty(M)
    for m_ in range(M):
        ix = rng.integers(0, B, B)
        sp[m_] = np.mean(sds[ix])
        spr[m_] = np.std(thetas[ix], ddof=1)
        st[m_] = np.sqrt(sp[m_] ** 2 + spr[m_] ** 2)
    return dict(se_post=se_post, se_post_se=float(np.std(sp, ddof=1)),
                se_param=se_param, se_param_se=float(np.std(spr, ddof=1)),
                se_total=se_total, se_total_se=float(np.std(st, ddof=1)))


def compute_regime(regime, idx_by_model, cov, grid, lp, n_boot, meta_m, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for i, m in enumerate(cov["matrix"].index):
        idx = idx_by_model.get(m, np.array([], dtype=int))
        se_post0, thetas, sds = bootstrap_full(
            cov["Ymat"][i], idx, cov["beta"], cov["chol"], grid, lp, n_boot, rng)
        if thetas.size == 0:
            continue
        ci = meta_ci(thetas, sds, se_post0, meta_m, rng)
        rows.append({
            "model": m, "regime": regime, "n_admin_criteria": int(np.asarray(idx).size),
            "theta": None,
            "SE_posterior": round(ci["se_post"], 5),
            "SE_post_ci_lo": round(max(ci["se_post"] - 1.96 * ci["se_post_se"], 0.0), 5),
            "SE_post_ci_hi": round(ci["se_post"] + 1.96 * ci["se_post_se"], 5),
            "SE_param": round(ci["se_param"], 5),
            "SE_total": round(ci["se_total"], 5),
            "SE_total_ci_lo": round(max(ci["se_total"] - 1.96 * ci["se_total_se"], 0.0), 5),
            "SE_total_ci_hi": round(ci["se_total"] + 1.96 * ci["se_total_se"], 5),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "wildbench_calibration"
    p.add_argument("--bank", type=Path,
                   default=base / "wildbench_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=L.DEFAULT_MATRIX)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--leaderboard", type=Path, default=base / "model_leaderboard.csv")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "07_parameter_uncertainty")
    p.add_argument("--tmp-dir", type=Path,
                   default=base / "experiments" / "07_parameter_uncertainty" / "_tmp")
    p.add_argument("--stop-rule", choices=["eap", "online"], default="eap")
    p.add_argument("--l-max", type=int, default=40)
    p.add_argument("--reuse-full-bank", type=Path, default=None,
                   help="reuse full_bank rows from this CSV (full-bank SE is stop-independent) "
                        "and only recompute the deployed regime.")
    p.add_argument("--min-scenarios", type=int, default=8)
    p.add_argument("--max-se", type=float, default=0.12)
    p.add_argument("--max-scenarios", type=int, default=50)
    p.add_argument("--min-evals-per-skill", type=int, default=10)
    p.add_argument("--fit-nodes", type=int, default=7)
    p.add_argument("--eap-grid", type=int, default=161)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--meta-m", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    cov = PU.compute_item_cov(args.bank, args.matrix, args.fit_nodes, args.ridge)
    d = cov["dims"][0]
    ids = cov["ids"]
    col = cov["col"]
    Mmat = cov["Mmat"]
    models = list(cov["matrix"].index)
    grid = np.linspace(-args.range, args.range, args.eap_grid)
    lp = -0.5 * grid ** 2
    lp = lp - logsumexp(lp)

    print("=" * 84)
    print("SE_posterior vs SE_total vs target (both regimes)")
    print("=" * 84)

    # --- FULL-BANK regime (stop-independent): reuse if provided, else recompute ---
    if args.reuse_full_bank and Path(args.reuse_full_bank).is_file():
        print(f"reusing full-bank rows from {args.reuse_full_bank.name} (stop-independent) ...",
              flush=True)
        _fb = pd.read_csv(args.reuse_full_bank)
        full_rows = _fb[_fb["regime"] == "full_bank"].drop(columns=["theta"], errors="ignore") \
            .to_dict("records")
    else:
        full_idx = {m: np.where(Mmat[i])[0] for i, m in enumerate(models)}
        print("full-bank bootstrap over all administrable criteria ...", flush=True)
        full_rows = compute_regime("full_bank", full_idx, cov, grid, lp,
                                   args.n_boot, args.meta_m, args.seed)

    # --- DEPLOYED CAT @ 8/0.12: idx = EAP-stop administered set per model (of-record) ---
    stop_rule = getattr(args, "stop_rule", "eap")
    l_max = getattr(args, "l_max", 40)
    print(f"administering deployed CAT @ 8/0.12 (stop_rule={stop_rule}) ...", flush=True)
    brecs = [__import__("json").loads(l) for l in Path(args.bank).open(encoding="utf-8") if l.strip()]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in brecs}
    row_of = {m: i for i, m in enumerate(models)}
    if stop_rule == "eap":
        admin = L.administer_eap(models, args.bank, args.matrix, args.scenarios, [d],
                                 cov["A"][:, 0], cov["b"], col, scen_of, cov["Ymat"], row_of,
                                 grid, lp, seed=20260729, floor=args.min_scenarios,
                                 target=args.max_se, l_max=l_max, workers=args.workers)
        dep_idx = {m: admin[m]["idx"] for m in models}
        dep_cap = {m: admin[m]["hit_cap"] for m in models}
    else:
        spec = scat.RunSpec(seed=42, top_n=5, max_se=args.max_se,
                            min_evals_per_skill=args.min_evals_per_skill,
                            min_scenarios=args.min_scenarios, max_scenarios=args.max_scenarios,
                            selection="trace", mode="cat", runs_dir=str(args.tmp_dir / "runs"))
        res = scat.run_models(models, args.bank, args.matrix, args.scenarios, "clamp",
                              [d], spec, workers=args.workers)
        dep_idx = {r["model"]: np.array([col[c] for c in r["order"] if c in col], dtype=int)
                   for r in res}
        dep_cap = {r["model"]: (r["stop_reason"] == "max_scenarios_reached") for r in res}
    print("deployed-CAT bootstrap over administered sets ...", flush=True)
    dep_rows = compute_regime("deployed_cat_8_0.12", dep_idx, cov, grid, lp,
                              args.n_boot, args.meta_m, args.seed + 1)

    # attach leaderboard theta for sorting
    lead = pd.read_csv(args.leaderboard).set_index("model")
    df = pd.DataFrame(full_rows + dep_rows)
    df["theta"] = df["model"].map(lambda m: float(lead.loc[m, "theta"]) if m in lead.index else np.nan)
    df["hit_cap"] = df.apply(
        lambda r: bool(dep_cap.get(r["model"], False)) if r["regime"] == "deployed_cat_8_0.12"
        else False, axis=1)
    df.to_csv(args.out_dir / "se_post_vs_total.csv", index=False)

    full = df[df["regime"] == "full_bank"].sort_values("theta").reset_index(drop=True)
    dep = df[df["regime"] == "deployed_cat_8_0.12"].sort_values("theta").reset_index(drop=True)

    def agg(sub):
        return {
            "post_mean": float(sub["SE_posterior"].mean()),
            "post_se": float(sub["SE_posterior"].std(ddof=1) / np.sqrt(len(sub))),
            "total_mean": float(sub["SE_total"].mean()),
            "total_se": float(sub["SE_total"].std(ddof=1) / np.sqrt(len(sub))),
        }
    a_full, a_dep = agg(full), agg(dep)
    import shutil
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    _figure(full, dep, a_full, a_dep, args.out_dir / "figures" / "se_post_vs_total.png")

    print("\n--- aggregate (mean +/- 1.96*SE) ---")
    print(f"FULL-BANK : SE_post={a_full['post_mean']:.4f} +/- {1.96*a_full['post_se']:.4f} | "
          f"SE_total={a_full['total_mean']:.4f} +/- {1.96*a_full['total_se']:.4f}")
    print(f"DEPLOYED  : SE_post={a_dep['post_mean']:.4f} +/- {1.96*a_dep['post_se']:.4f} | "
          f"SE_total={a_dep['total_mean']:.4f} +/- {1.96*a_dep['total_se']:.4f}")
    print(f"target = {SE_TARGET}")
    print(f"wrote -> {args.out_dir / 'figures' / 'se_post_vs_total.png'} and "
          f"{args.out_dir / 'se_post_vs_total.csv'}")
    return 0


def _bars(ax, sub, title, target=SE_TARGET):
    x = np.arange(len(sub))
    w = 0.42
    ax.bar(x - w / 2, sub["SE_posterior"], w, color="#4d648d", label="SE_posterior",
           yerr=[sub["SE_posterior"] - sub["SE_post_ci_lo"],
                 sub["SE_post_ci_hi"] - sub["SE_posterior"]], capsize=0, ecolor="#2b3a55",
           error_kw={"elinewidth": 0.6})
    ax.bar(x + w / 2, sub["SE_total"], w, color="#c1666b", label="SE_total",
           yerr=[sub["SE_total"] - sub["SE_total_ci_lo"],
                 sub["SE_total_ci_hi"] - sub["SE_total"]], capsize=0, ecolor="#7a2f34",
           error_kw={"elinewidth": 0.6})
    ax.axhline(target, ls=":", color="k", lw=1.4, label=f"CAT SE target {target}")
    ax.set_xticks([]); ax.set_xlabel("models (sorted by theta, low -> high)")
    ax.set_ylabel("SE"); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


def _agg_bars(ax, a, title, target=SE_TARGET):
    vals = [a["post_mean"], a["total_mean"]]
    err = [1.96 * a["post_se"], 1.96 * a["total_se"]]
    ax.bar([0, 1], vals, 0.6, color=["#4d648d", "#c1666b"], yerr=err, capsize=5,
           error_kw={"elinewidth": 1.2})
    ax.axhline(target, ls=":", color="k", lw=1.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SE_post", "SE_total"], fontsize=8)
    ax.set_ylabel("mean SE"); ax.set_title(title, fontsize=9)
    for i, (v, e) in enumerate(zip(vals, err)):
        ax.annotate(f"{v:.3f}", (i, v + e), ha="center", va="bottom", fontsize=7)


def _figure(full, dep, a_full, a_dep, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, width_ratios=[3.0, 1.0], hspace=0.32, wspace=0.22)
    axA = fig.add_subplot(gs[0, 0]); axAa = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 0]); axBa = fig.add_subplot(gs[1, 1])
    _bars(axA, full, "Panel A - FULL-BANK scoring (all ~8,345 criteria; leaderboard regime): "
                     "SE_posterior vs SE_total")
    _agg_bars(axAa, a_full, "A: aggregate")
    _bars(axB, dep, "Panel B - DEPLOYED CAT @ 8/0.12 (engine administered set): "
                    "SE_posterior vs SE_total")
    _agg_bars(axBa, a_dep, "B: aggregate")
    fig.suptitle("WildBench (1-D, MWLE) - SE_posterior vs SE_total vs CAT SE target 0.12 "
                 "(95% CI whiskers = +/-1.96 x bootstrap SE-of-estimate)", fontsize=11)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
