"""Phase 2b exp 04 addendum: measurement-SE-vs-length, adaptive vs random.

The recovery-r-vs-length view saturates fast, hiding the adaptive advantage. This adds the
clearer view: mean online theta-SE at each administered length L = 1..N, adaptive (CAT) vs
random (baseline), with an across-model spread band, the SE target (0.15) marked, and the
scenarios-to-target for each arm annotated. ADDITIVE only -- does not touch the existing
``efficiency_adaptive_vs_random.png`` / ``results.csv`` / ``summary.json``.

Outputs (new):
  experiments/04_efficiency_vs_random/se_by_length.csv
  experiments/04_efficiency_vs_random/figures/se_vs_length.png
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import bridge_scenario_lib as L  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "bridge_calibration"
    p.add_argument("--bank", type=Path, default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "bridgegrade" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "Bridge" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, default=base / "experiments" / "04_efficiency_vs_random")
    p.add_argument("--tmp-dir", type=Path, default=base / "experiments" / "04_efficiency_vs_random" / "_tmp_se")
    p.add_argument("--max-length", type=int, default=20)
    p.add_argument("--se-target", type=float, default=0.15)
    p.add_argument("--eap-grid", type=int, default=321)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--band", type=str, default="10,90", help="percentile band across models.")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    ids, A, b = scat.assemble_arrays(records, dims)
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    mask = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col = {c: i for i, c in enumerate(ids)}
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)
    theta_ref = scat.eap_all_models(Y, mask, A, b, egrid, elog)[:, 0]
    lengths = list(range(1, args.max_length + 1))
    plo, phi = [float(x) for x in args.band.split(",")]

    print("=" * 84)
    print("EXP 04 addendum: online theta-SE vs length (adaptive vs random)")
    print("=" * 84)

    rows = []
    curve = {"cat": {}, "baseline": {}}
    for mode in ("cat", "baseline"):
        for Ln in lengths:
            spec = scat.RunSpec(seed=args.seed, top_n=5, max_se=0.0, min_evals_per_skill=0,
                                min_scenarios=Ln, max_scenarios=Ln, selection="trace",
                                mode=mode, runs_dir=str(args.tmp_dir / f"{mode}_L{Ln}"))
            res = scat.run_models(models, args.bank, args.matrix, args.scenarios,
                                  "clamp", dims, spec, workers=args.workers)
            ses = np.array([r0["se_online"][0] for r0 in res])
            # recovery r of MWLE vs full-bank reference (handy companion column)
            cat_theta = np.full(len(models), np.nan)
            for r0 in res:
                r = row_of[r0["model"]]
                idx = np.array([col[c] for c in r0["order"] if c in col], dtype=int)
                if idx.size:
                    th0 = scat.eap_subset(Y[r], idx, A, b, egrid, elog)
                    thm, _ = scat.mwle_subset(Y[r], idx, A, b, th0)
                    cat_theta[r] = thm[0]
                else:
                    cat_theta[r] = 0.0
            rr = float(np.corrcoef(theta_ref, cat_theta)[0, 1])
            mean_se = float(ses.mean())
            row = {"length": Ln, "arm": ("adaptive" if mode == "cat" else "random"),
                   "mean_se": round(mean_se, 4),
                   "se_lo": round(float(np.percentile(ses, plo)), 4),
                   "se_hi": round(float(np.percentile(ses, phi)), 4),
                   "median_se": round(float(np.median(ses)), 4),
                   "recovery_r": round(rr, 4)}
            rows.append(row)
            curve[mode][Ln] = mean_se
            print(f"  {row['arm']:8s} L={Ln:2d}: mean_se={mean_se:.3f} "
                  f"[{row['se_lo']:.3f},{row['se_hi']:.3f}] r={rr:.3f}", flush=True)

    with (args.out_dir / "se_by_length.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def scen_to_target(mode):
        for Ln in lengths:
            if curve[mode][Ln] <= args.se_target:
                return Ln
        return None
    cross = {"adaptive": scen_to_target("cat"), "random": scen_to_target("baseline")}

    _figure(rows, lengths, args.se_target, cross, args.out_dir / "figures")

    summary_add = {
        "purpose": "measurement SE vs administered length, adaptive vs random",
        "se_target": args.se_target,
        "scenarios_to_reach_se_target": cross,
        "efficiency_gain_scenarios": (
            None if None in cross.values() else cross["random"] - cross["adaptive"]),
        "band_percentiles": [plo, phi], "n_models": len(models),
        "se_by_length": rows,
    }
    (args.out_dir / "se_by_length_summary.json").write_text(json.dumps(summary_add, indent=2),
                                                            encoding="utf-8")
    shutil.rmtree(args.tmp_dir, ignore_errors=True)

    print("\nscenarios to reach SE<=%.2f: adaptive=%s random=%s" %
          (args.se_target, cross["adaptive"], cross["random"]))
    print(f"wrote -> {args.out_dir / 'se_by_length.csv'} and figures/se_vs_length.png")
    return 0


def _figure(rows, lengths, se_target, cross, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"adaptive": "#1b7837", "random": "#762a83"}
    for arm in ("adaptive", "random"):
        dd = df[df["arm"] == arm].sort_values("length")
        ax.plot(dd["length"], dd["mean_se"], "o-", color=colors[arm], label=f"{arm} (mean)")
        ax.fill_between(dd["length"], dd["se_lo"], dd["se_hi"], color=colors[arm], alpha=0.15)
    ax.axhline(se_target, ls="--", color="k", alpha=0.8, label=f"SE target = {se_target}")
    for arm in ("adaptive", "random"):
        Lc = cross[arm]
        if Lc is not None:
            ax.axvline(Lc, ls=":", color=colors[arm], alpha=0.8)
            ax.annotate(f"{arm}: {Lc} scen", xy=(Lc, se_target),
                        xytext=(Lc + 0.3, se_target + 0.03 + (0.03 if arm == "random" else 0)),
                        color=colors[arm], fontsize=9)
    ax.set_xlabel("scenarios administered (L)")
    ax.set_ylabel("mean online theta-SE (10-90 pct band across models)")
    ttl = "Measurement SE vs test length (adaptive vs random)"
    if cross["adaptive"] and cross["random"]:
        ttl += f"\nscenarios to SE<={se_target}: adaptive={cross['adaptive']}, random={cross['random']}"
    ax.set_title(ttl, fontsize=10)
    ax.set_xticks(lengths)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "se_vs_length.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
