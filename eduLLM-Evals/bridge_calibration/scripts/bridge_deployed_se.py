"""Phase B exp-07 deployed-SE artifacts under the EAP-posterior stop @ 12/0.12.

Mirrors the WildBench ``fig_se_post_vs_total`` + ``precision_reached`` + ``fig_se_ability_vs_total``
adoption. Re-uses the frozen catpool bank (NO refit). Two regimes on the same bank:

  * FULL-BANK (leaderboard regime): each model scored on ALL its administrable CAT-pool
    criteria -> maximal info. Read from exp-07 ``leaderboard_se_components.csv`` (observed-info
    parametric bootstrap; SE_posterior / SE_param / SE_total) -- stop-independent, unchanged.
  * DEPLOYED CAT @ 12/0.12 (EAP stop): each model's administered set is the EAP-posterior-SD
    stop on a forced-long adaptive order (production selection, seed 20260729). SE_posterior =
    the fine-grid (321 / +/-8) EAP posterior SD at the stop; SE_param is RE-BOOTSTRAPPED on the
    EAP-administered set (observed-info parametric bootstrap); SE_total = sqrt(post^2 + param^2).
    hit_cap = the model never reached SE-ability <= 0.12 within the forced cap (EAP-native cap).

Writes (of-record, into experiments/07_parameter_uncertainty/):
  se_post_vs_total.csv            per-model both regimes (+ hit_cap on deployed)
  precision_reached.{csv,json}    deployed: SE_ability <= 0.12 reached-rate + missed models
  se_ability_vs_total_summary.csv aggregate two-bar summary (deployed)
  figures/se_post_vs_total.png    full-bank vs deployed bars
  figures/se_ability_vs_total.png deployed aggregate two-bar (BiGGen-parity)

weakly_identified (EAP-native cap) = deployed hit_cap; consumed by scenario_leaderboard.py.
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
from scipy.special import logsumexp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import bridge_scenario_lib as L  # noqa: E402
import bridge_eap_stop_lib as E  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402

_pu_spec = importlib.util.spec_from_file_location(
    "bridge_scenario_param_uncertainty", _HERE / "scenario_param_uncertainty.py")
PU = importlib.util.module_from_spec(_pu_spec)
sys.modules["bridge_scenario_param_uncertainty"] = PU
_pu_spec.loader.exec_module(PU)

SE_TARGET = 0.12
REGIME = "deployed_cat_12_0.12"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "07_parameter_uncertainty")
    p.add_argument("--full-bank-se", type=Path,
                   default=base / "experiments" / "07_parameter_uncertainty"
                   / "leaderboard_se_components.csv")
    p.add_argument("--leaderboard", type=Path, default=base / "model_leaderboard.csv")
    p.add_argument("--tmp-dir", type=Path,
                   default=base / "experiments" / "07_parameter_uncertainty" / "_dep_tmp")
    p.add_argument("--stop-rule", choices=["eap", "online"], default="eap")
    p.add_argument("--min-scenarios", type=int, default=12)
    p.add_argument("--max-se", type=float, default=0.12)
    p.add_argument("--max-scenarios", type=int, default=228)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--l-max", type=int, default=60)
    p.add_argument("--eap-grid", type=int, default=321)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--se-param-grid", type=int, default=61)
    p.add_argument("--se-param-range", type=float, default=6.0)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--pu-seed", type=int, default=20260801)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    d = dims[0]
    ids, A, b = scat.assemble_arrays(records, dims)
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(args.matrix, index_col=0)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col = {c: i for i, c in enumerate(ids)}
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)
    gg, lp = E.eap_grid(args.eap_grid, args.range)

    cov = PU.compute_item_cov(args.bank, args.matrix, args.fit_grid, args.ridge)
    Yfull = cov["Ymat"]
    bgrid = np.linspace(-args.se_param_range, args.se_param_range, args.se_param_grid)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(args.pu_seed)

    print("=" * 88)
    print(f"EXP 07 deployed-SE @ {args.min_scenarios}/{args.max_se} (stop_rule={args.stop_rule})")
    print("=" * 88)

    # --- FULL-BANK regime: reuse exp-07 leaderboard_se_components (stop-independent) ---
    full_rows = []
    fb = pd.read_csv(args.full_bank_se).set_index("model")
    for m in models:
        if m in fb.index:
            full_rows.append({
                "model": m, "regime": "full_bank",
                "SE_posterior": float(fb.loc[m, f"se_posterior_{d}"]),
                "SE_param": float(fb.loc[m, f"se_param_{d}"]),
                "SE_total": float(fb.loc[m, f"se_total_{d}"]),
                "n_admin_criteria": int(fb.loc[m, "n_admin"]) if "n_admin" in fb.columns else len(ids),
                "hit_cap": False})

    # --- DEPLOYED regime: administer the EAP stop, re-bootstrap SE_param on admin sets ---
    print(f"administering deployed CAT @ {args.min_scenarios}/{args.max_se} "
          f"(stop_rule={args.stop_rule}) ...", flush=True)
    if args.stop_rule == "eap":
        admin = E.administer_eap(models, args.bank, args.matrix, args.scenarios, dims,
                                 A[:, 0], b, col, scen_of, Yfull, row_of, gg, lp,
                                 seed=args.seed, floor=args.min_scenarios, target=args.max_se,
                                 l_max=args.l_max, workers=args.workers,
                                 runs_dir=str(args.tmp_dir / "eap_runs"))
        dep_idx = {m: admin[m]["idx"] for m in models}
        dep_cap = {m: admin[m]["hit_cap"] for m in models}
        dep_post = {m: admin[m]["eap_sd"] for m in models}
    else:
        spec = scat.RunSpec(seed=42, top_n=5, max_se=args.max_se,
                            min_evals_per_skill=args.min_evals_per_skill,
                            min_scenarios=args.min_scenarios, max_scenarios=args.max_scenarios,
                            selection="trace", mode="cat", runs_dir=str(args.tmp_dir / "runs"))
        res = scat.run_models(models, args.bank, args.matrix, args.scenarios, "clamp", dims,
                              spec, workers=args.workers)
        dep_idx, dep_cap, dep_post = {}, {}, {}
        for r in res:
            m = r["model"]
            idx = np.array([col[c] for c in r["order"] if c in col], dtype=int)
            dep_idx[m] = idx
            dep_cap[m] = (r["stop_reason"] == "max_scenarios_reached")
            _, var = scat.eap_subset_mean_var(Yfull[row_of[m]], idx, A, b, egrid, elog)
            dep_post[m] = float(np.sqrt(max(float(var[0]), 0.0))) if idx.size else np.nan

    print("re-bootstrapping SE_param over the EAP-administered sets ...", flush=True)
    dep_rows = []
    for m in models:
        idx = np.asarray(dep_idx[m], dtype=int)
        se_post = float(dep_post[m])
        _, _, se_param = PU.bootstrap_theta(Yfull[row_of[m]], idx, cov["beta"], cov["chol"],
                                            bgrid, blp, args.n_boot, rng_boot)
        se_param = float(se_param) if np.isfinite(se_param) else np.nan
        se_total = float(np.sqrt(se_post ** 2 + se_param ** 2)) if np.isfinite(se_param) else se_post
        dep_rows.append({"model": m, "regime": REGIME, "SE_posterior": round(se_post, 6),
                         "SE_param": round(se_param, 6), "SE_total": round(se_total, 6),
                         "n_admin_criteria": int(idx.size), "hit_cap": bool(dep_cap[m])})

    lead = pd.read_csv(args.leaderboard).set_index("model") if args.leaderboard.is_file() else None
    df = pd.DataFrame(full_rows + dep_rows)
    df["theta"] = df["model"].map(
        lambda m: float(lead.loc[m, "theta"]) if lead is not None and m in lead.index else np.nan)
    df.to_csv(args.out_dir / "se_post_vs_total.csv", index=False)

    dep = df[df["regime"] == REGIME].copy()
    n = len(dep)
    reached = int((dep["SE_posterior"] <= SE_TARGET).sum())
    n_gt15 = int((dep["SE_total"] > 0.15).sum())
    n_gt20 = int((dep["SE_total"] > 0.20).sum())
    dep_sorted = dep.sort_values("SE_posterior").reset_index(drop=True)
    dep_sorted["precision_reached"] = dep_sorted["SE_posterior"] <= SE_TARGET
    dep_sorted["weakly_identified"] = dep_sorted["hit_cap"]
    dep_sorted[["model", "theta", "SE_posterior", "SE_total", "n_admin_criteria",
                "hit_cap", "weakly_identified", "precision_reached"]].to_csv(
        args.out_dir / "precision_reached.csv", index=False)

    missed = dep_sorted[~dep_sorted["precision_reached"]].sort_values("SE_posterior",
                                                                      ascending=False)

    def q(s):
        return {"mean": float(s.mean()), "sd": float(s.std(ddof=1)), "median": float(s.median()),
                "q3": float(s.quantile(0.75)), "max": float(s.max())}

    pr = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "deployed-CAT precision-reached + SE tail at the locked op-point under the "
                   "EAP-posterior stop",
        "operating_point": {"min_scenarios": args.min_scenarios, "se_target_ability": SE_TARGET,
                            "stop_rule": args.stop_rule},
        "n_models": n,
        "precision_reached_posterior_metric": {
            "definition": "fine-grid EAP posterior SD (SE_ability) <= 0.12",
            "n_reached": reached, "pct_reached": round(100.0 * reached / n, 1),
            "n_missed": n - reached, "pct_missed": round(100.0 * (n - reached) / n, 1),
        },
        "se_total_tail": {"n_gt_0.15": n_gt15, "n_gt_0.20": n_gt20},
        "se_ability_distribution": q(dep["SE_posterior"]),
        "se_total_distribution": q(dep["SE_total"]),
        "deployed_length": {
            "mean_criteria": round(float(dep["n_admin_criteria"].mean()), 1),
            "median_criteria": int(dep["n_admin_criteria"].median()),
        },
        "weakly_identified_eap_cap": {
            "n": int(dep["hit_cap"].sum()),
            "models": dep[dep["hit_cap"]]["model"].tolist(),
        },
        "missed_target_models": missed[["model", "theta", "SE_posterior", "SE_total",
                                        "weakly_identified"]].round(4).to_dict("records"),
    }
    (args.out_dir / "precision_reached.json").write_text(json.dumps(pr, indent=2),
                                                         encoding="utf-8")

    ability = dep["SE_posterior"].to_numpy()
    total = dep["SE_total"].to_numpy()
    pd.DataFrame([
        {"metric": "SE_ability_only", "mean": float(ability.mean()),
         "sd_across_models": float(ability.std(ddof=1)), "median": float(np.median(ability)),
         "n_models": n, "se_target": SE_TARGET, "regime": REGIME},
        {"metric": "SE_total", "mean": float(total.mean()),
         "sd_across_models": float(total.std(ddof=1)), "median": float(np.median(total)),
         "n_models": n, "se_target": SE_TARGET, "regime": REGIME},
    ]).to_csv(args.out_dir / "se_ability_vs_total_summary.csv", index=False)

    import shutil
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    _fig_two_bar(ability, total, n, args.out_dir / "figures" / "se_ability_vs_total.png")
    _fig_regimes(df[df["regime"] == "full_bank"], dep, args.out_dir / "figures" / "se_post_vs_total.png")

    print(f"precision reached (SE_ability<=0.12): {reached}/{n} "
          f"({100.0*reached/n:.1f}%)   missed: {n-reached}")
    print(f"SE_total tail: #>0.15={n_gt15}  #>0.20={n_gt20}")
    print(f"SE_ability: mean={ability.mean():.4f} median={np.median(ability):.4f} "
          f"max={ability.max():.4f}")
    print(f"SE_total  : mean={total.mean():.4f} median={np.median(total):.4f} "
          f"max={total.max():.4f}")
    print(f"weakly_identified (EAP cap): {int(dep['hit_cap'].sum())} "
          f"{dep[dep['hit_cap']]['model'].tolist()}")
    print(f"wrote -> {args.out_dir}")
    return 0


def _fig_two_bar(ability, total, n, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = [float(ability.mean()), float(total.mean())]
    sds = [float(ability.std(ddof=1)), float(total.std(ddof=1))]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.bar([0, 1], means, 0.58, color=["#4d648d", "#c1666b"], yerr=sds, capsize=6,
           error_kw={"elinewidth": 1.6, "ecolor": "#333333"})
    ax.axhline(SE_TARGET, ls=":", color="k", lw=1.6)
    ax.text(1.45, SE_TARGET + 0.004, "SE target=0.12", ha="right", va="bottom", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SE ability-only", "SE total (+calibration)"],
                                              fontsize=10)
    ax.set_ylabel("SE (overall-ability theta, 1-D MWLE)")
    ax.set_title(f"Bridge deployed-CAT SE: ability-only vs total\n"
                 f"(EAP stop, locked 12/0.12; N={n})", fontsize=11)
    for i, (mn, sd) in enumerate(zip(means, sds)):
        ax.annotate(f"{mn:.3f}\n(SD {sd:.3f})", (i, mn), ha="center", va="bottom",
                    fontsize=9, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, max(means[1] + sds[1], SE_TARGET) * 1.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _fig_regimes(full, dep, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, sub, ttl in ((axes[0], full.sort_values("theta"), "FULL-BANK (leaderboard)"),
                         (axes[1], dep.sort_values("theta"), "DEPLOYED CAT @ 12/0.12 (EAP)")):
        x = np.arange(len(sub)); w = 0.42
        ax.bar(x - w / 2, sub["SE_posterior"], w, color="#4d648d", label="SE_posterior")
        ax.bar(x + w / 2, sub["SE_total"], w, color="#c1666b", label="SE_total")
        ax.axhline(SE_TARGET, ls=":", color="k", lw=1.4, label="SE target 0.12")
        ax.set_xticks([]); ax.set_xlabel("models (sorted by theta)")
        ax.set_ylabel("SE"); ax.set_title(ttl, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("Bridge (1-D, MWLE) SE_posterior vs SE_total vs CAT SE target 0.12", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
