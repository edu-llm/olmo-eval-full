"""Regenerate BiGGen's OF-RECORD exp-05 recovery + exp-08 leaderboard.

The reference/leaderboard EAP defaults to a FINE uniform grid (3201 nodes over [-8,8],
standard-normal prior) -- the de-quantized reference that fixes the coarse-grid banding of
the razor-sharp full-bank posterior. The legacy COARSE mode (Gauss-Hermite-scale 61 nodes
over [-6,6], node spacing 0.2) stays reachable behind ``--eap-mode gh``.

Pipeline (both stages drive the SAME production engine as the committed study, via the
shared ``scripts/scenario_kfold_estimator_cv.py`` and ``scripts/scenario_param_uncertainty.py``
at the LOCKED config):
  * exp-05: OOS k-fold MWLE recovery vs the full-bank EAP reference -> metrics.json,
    oos_per_model.csv, figures/{oos_recovery_general.png (3-panel), oos_recovery_scatter_general.png}.
  * exp-08: full-data leaderboard theta + SE_total (ranked) -> results.csv, figures/leaderboard_general.png.

Before overwriting, the current (coarse) of-record files are archived as ``*_coarse.*`` for
provenance (mirrors how Bridge archived its jackknife as ``*_jackknife.*``). Archiving is
skipped if a ``*_coarse.*`` already exists, so re-runs never clobber the archived coarse copy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EAP_MODES = {"fine": (3201, 8.0), "gh": (61, 6.0)}


def archive_coarse(path: Path, suffix: str = "_coarse") -> str | None:
    """Copy ``path`` -> ``stem+suffix+ext`` once (skip if the archive already exists)."""
    if not path.is_file():
        return None
    arch = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if arch.exists():
        return f"(kept existing {arch.name})"
    shutil.copy2(path, arch)
    return arch.name


def run_script(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    env = dict(os.environ)
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[v] = "1"
    print(f"  $ {script} {' '.join(args)}", flush=True)
    res = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
        raise SystemExit(f"{script} failed (exit {res.returncode})")


def scatter_figure(df: pd.DataFrame, d: str, out_png: Path, mode: str, gridn: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = df[f"theta_ref_{d}"].to_numpy()
    y = df[f"theta_mwle_{d}"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(np.abs(y - x)))
    xs = np.linspace(x.min(), x.max(), 60)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.plot(xs, slope * xs + intercept, color="#d95f0e", lw=1.5, label=f"OLS slope={slope:.3f}")
    ax.scatter(x, y, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ref_lbl = ("fine uniform grid" if mode == "fine" else "coarse GH-scale grid")
    ax.set_xlabel(f"full-bank EAP reference theta ({ref_lbl}, {gridn} nodes)")
    ax.set_ylabel("CAT MWLE theta")
    ax.set_title(f"BiGGen scenario OOS recovery @ LOCKED (of-record)\n"
                 f"r={r:.4f}, slope={slope:.3f}, theta-MAE={mae:.3f} (n={len(df)})", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    return {"r": round(r, 4), "slope": round(slope, 4), "theta_mae": round(mae, 4)}


def leaderboard_figure(rank_df: pd.DataFrame, out_png: Path, mode: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = rank_df.sort_values("theta").reset_index(drop=True)
    ypos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, max(9, 0.22 * len(order))))
    ax.errorbar(order["theta"], ypos, xerr=1.96 * order["se_total"], fmt="none",
                ecolor="#c1666b", elinewidth=2.4, alpha=0.6,
                label="+/-1.96 SE_total (params uncertain)")
    ax.errorbar(order["theta"], ypos, xerr=1.96 * order["se_ability"], fmt="none",
                ecolor="#4d648d", elinewidth=1.2, label="+/-1.96 SE_ability (params known)")
    ax.scatter(order["theta"], ypos, s=10, color="k", zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels(order["model"], fontsize=5)
    ax.set_xlabel("general ability (theta)")
    ax.set_title(f"BiGGen general-ability leaderboard (of-record; {mode} EAP grid)\n"
                 "adjacent +/-1.96 SE_total bars overlap -> only coarse ability bands distinguishable",
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "biggen_calibration"
    p.add_argument("--eap-mode", choices=list(EAP_MODES), default="fine",
                   help="reference EAP grid: 'fine' (3201/[-8,8], DEFAULT, de-quantized) "
                        "or 'gh' (legacy coarse 61/[-6,6], node spacing 0.2).")
    p.add_argument("--bank", type=Path, default=ROOT / "staging" / "biggen_unidim_modeled.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "staging" / "biggen_response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "BiGGen" / "scenarios.jsonl")
    p.add_argument("--dim", default="general")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--pu-seed", type=int, default=20260801)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-se", type=float, default=0.12)
    p.add_argument("--min-scenarios", type=int, default=8)
    p.add_argument("--max-scenarios", type=int, default=50)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-archive", action="store_true")
    args = p.parse_args()

    d = args.dim
    gridn, grange = EAP_MODES[args.eap_mode]
    exp05 = base / "experiments" / "05_oos_recovery"
    exp08 = base / "experiments" / "08_leaderboard"
    (exp05 / "figures").mkdir(parents=True, exist_ok=True)
    (exp08 / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"REGEN OF-RECORD | eap-mode={args.eap_mode} (grid={gridn} over +/-{grange}) | "
          f"LOCKED SE={args.max_se} floor={args.min_scenarios} ridge={args.ridge}")
    print("=" * 88)

    archived = {}
    if not args.skip_archive:
        for f in (exp05 / "metrics.json", exp05 / "oos_per_model.csv",
                  exp05 / "figures" / "oos_recovery_general.png",
                  exp05 / "figures" / "oos_recovery_scatter_general.png",
                  exp08 / "results.csv", exp08 / "figures" / "leaderboard_general.png"):
            a = archive_coarse(f)
            if a:
                archived[str(f.relative_to(base))] = a
        print("archived coarse:", json.dumps(archived, indent=2))

    with tempfile.TemporaryDirectory(prefix="biggen_regen_") as td:
        td = Path(td)
        # ---- exp-05: OOS k-fold recovery at the chosen grid ----
        k5_out = td / "kfold"
        run_script("scenario_kfold_estimator_cv.py", [
            "--bank", str(args.bank), "--matrix", str(args.matrix),
            "--scenarios", str(args.scenarios), "--out-dir", str(k5_out),
            "--tmp-dir", str(td / "kfold_tmp"), "--k", str(args.k), "--seed", str(args.seed),
            "--fit-grid", str(args.fit_grid), "--ridge", str(args.ridge),
            "--eap-grid", str(gridn), "--range", str(grange), "--max-se", str(args.max_se),
            "--min-scenarios", str(args.min_scenarios), "--max-scenarios", str(args.max_scenarios),
            "--min-evals-per-skill", str(args.min_evals_per_skill), "--workers", str(args.workers),
        ])
        shutil.copy2(k5_out / "metrics.json", exp05 / "metrics.json")
        shutil.copy2(k5_out / "oos_per_model.csv", exp05 / "oos_per_model.csv")
        shutil.copy2(k5_out / "figures" / f"oos_recovery_{d}.png",
                     exp05 / "figures" / "oos_recovery_general.png")
        df05 = pd.read_csv(exp05 / "oos_per_model.csv")
        head = scatter_figure(df05, d, exp05 / "figures" / "oos_recovery_scatter_general.png",
                              args.eap_mode, gridn)

        # ---- exp-08: full-data leaderboard theta + SE_total at the chosen grid ----
        pu_out = td / "pu"
        run_script("scenario_param_uncertainty.py", [
            "--bank", str(args.bank), "--matrix", str(args.matrix),
            "--scenarios", str(args.scenarios), "--out-dir", str(pu_out),
            "--runs-dir", str(td / "pu_runs"), "--fit-nodes", str(args.fit_grid),
            "--eap-grid", str(gridn), "--range", str(grange), "--max-se", str(args.max_se),
            "--min-scenarios", str(args.min_scenarios), "--max-scenarios", str(args.max_scenarios),
            "--min-evals-per-skill", str(args.min_evals_per_skill), "--n-boot", str(args.n_boot),
            "--ridge", str(args.ridge), "--seed", str(args.pu_seed), "--workers", str(args.workers),
        ])
        pu = pd.read_csv(pu_out / "leaderboard_se_components.csv")
        rank = pu[["model", f"theta_{d}", f"se_posterior_{d}", f"se_total_{d}"]].copy()
        rank.columns = ["model", "theta", "se_ability", "se_total"]
        rank = rank.sort_values("theta", ascending=False).reset_index(drop=True)
        rank["rank"] = np.arange(1, len(rank) + 1)
        rank.to_csv(exp08 / "results.csv", index=False)
        leaderboard_figure(rank, exp08 / "figures" / "leaderboard_general.png", args.eap_mode)

    # ---- Spearman vs archived coarse ranking (if archive exists) ----
    coarse_res = exp08 / "results_coarse.csv"
    rank_check = None
    if coarse_res.is_file():
        c = pd.read_csv(coarse_res)[["model", "rank"]].rename(columns={"rank": "rank_coarse"})
        m = rank.merge(c, on="model")
        rank_check = {
            "spearman_vs_coarse": round(float(pd.Series(m["rank"]).corr(
                pd.Series(m["rank_coarse"]), method="spearman")), 5),
            "n_ranks_changed": int((m["rank"] != m["rank_coarse"]).sum()),
            "max_rank_move": int((m["rank"] - m["rank_coarse"]).abs().max()),
        }

    prov = {
        "of_record": True,
        "eap_mode": args.eap_mode,
        "eap_grid": gridn, "eap_range": grange,
        "eap_prior": "standard-normal (uniform product grid)",
        "reason": ("fine uniform grid is the de-quantized reference; the coarse GH-scale grid "
                   "(61/[-6,6], spacing 0.2) quantized the razor-sharp full-bank posterior -> "
                   "0.2 banding on the recovery reference. Coarse mode kept behind --eap-mode gh."),
        "locked_config": {"max_se": args.max_se, "min_scenarios": args.min_scenarios,
                          "ridge": args.ridge, "k": args.k, "seed": args.seed,
                          "estimator": "mwle"},
        "recovery_headline_mwle": head,
        "leaderboard_rank_vs_coarse": rank_check,
        "archived_coarse": archived or "(pre-existing archives kept)",
        "generated_by": "biggen_calibration/scripts/biggen_regen_of_record.py",
    }
    (exp05 / "of_record_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (exp08 / "of_record_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"OF-RECORD recovery (MWLE): r={head['r']} slope={head['slope']} MAE={head['theta_mae']}")
    if rank_check:
        print(f"leaderboard vs coarse: Spearman={rank_check['spearman_vs_coarse']} "
              f"changed={rank_check['n_ranks_changed']} maxMove={rank_check['max_rank_move']}")
    print(f"wrote of-record -> {exp05}  and  {exp08}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
