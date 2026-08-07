"""FULL-AXES OOS floor x SE_ability GRID study for the CORRECTNESS-ONLY UNIDIMENSIONAL
EAP-posterior stop CAT (TutorBench, 115 models).

STUDY-ONLY / LOCAL. Production engine untouched. Nothing committed. This is the full-grid
extension of ``oos_grid_unidim_correctness_only.py`` (the compact 15-cell run): it reuses that
module's fold/refit/trace machinery verbatim (same fitter/config, same EAP-posterior stop,
same dense 321-node 1-D grid, same seed) and only widens the axes to match the 2-skill OOS
grid for an apples-to-apples comparison.

Full grid: floors {10,12,15,20,25} x SE_ability {0.15,0.20,0.22,0.25,0.27,0.30,0.32} = 35 cells.

Headline pool EXCLUDES only ``Qwen/Qwen1.5-1.8B`` (genuine low-ability info-limited outlier).
``salamandra-7b-instruct`` is rehabilitated under the correctness-only unidim scale and is kept
in the headline. An all-115 pool is also reported.

%reach = SE_ability<=target AND SE_total<=0.30. SE_total = sqrt(SD^2 + SE_param^2), SE_param the
FIXED per-model offset from the correctness-only param_uncertainty study (median 0.113).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # eduLLM-Evals
for _p in (str(ROOT), str(ROOT / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scenario_cat_lib as scl  # noqa: E402
import eap_stop_grid_study as G  # noqa: E402
import calibrate_mirt as cm  # noqa: E402 (referenced indirectly via base)
import oos_grid_unidim_correctness_only as base  # noqa: E402

# ---- FULL axes (match the 2-skill OOS grid exactly) ----
FLOORS = [10, 12, 15, 20, 25]
SE_TARGETS = [0.15, 0.20, 0.22, 0.25, 0.27, 0.30, 0.32]
CAP = base.CAP
PLATEAU_W = base.PLATEAU_W
PLATEAU_DELTA = base.PLATEAU_DELTA
SE_TOTAL_TARGETS = (0.25, 0.30)
SE_TOTAL_REACH = 0.30  # gate used in the composite %reach

# Headline excludes ONLY the genuine outlier; salamandra rehabilitated under this scale.
HEADLINE_EXCLUDE = ("Qwen/Qwen1.5-1.8B",)
FLAGGED_NOTE = {"Qwen/Qwen1.5-1.8B": "headline-excluded (info-limited outlier)",
                "BSC-LT/salamandra-7b-instruct": "rehabilitated under correctness-only unidim"}


def _cell_metrics(traces, dims, floor, se_t, se_param, cap, exclude_models):
    excl = set(exclude_models)
    d = dims[0]
    x, y, sd, stot, lens, capf, platf, reachf = [], [], [], [], [], [], [], []
    for tr in traces:
        if tr["model"] in excl:
            continue
        i, reason = G.resolve_stop(tr["sd"], tr["nscen"], floor, se_t,
                                   PLATEAU_DELTA, PLATEAU_W, cap)
        sd_stop = tr["sd"][i]
        mwle_stop = tr["mwle"][i]
        sp = se_param[tr["model"]]
        st = float(np.sqrt(sd_stop[0] ** 2 + sp[d] ** 2))
        x.append(float(tr["theta_ref"][0]))
        y.append(float(mwle_stop[0]))
        sd.append(float(sd_stop[0]))
        stot.append(st)
        lens.append(tr["nscen"][i])
        capf.append(reason == "cap")
        platf.append(reason == "info_plateau")
        reachf.append(bool(sd_stop[0] <= se_t and st <= SE_TOTAL_REACH))
    L = np.array(lens, float)
    xa, ya, sda, stota = map(lambda z: np.array(z, float), (x, y, sd, stot))
    row = {
        "floor": floor, "se_target": se_t, "n_models": int(L.size),
        "median_len": float(np.median(L)), "mean_len": float(L.mean()),
        "pct_cap": float(np.mean(capf)), "pct_info_plateau": float(np.mean(platf)),
        "pct_reach": float(np.mean(reachf)),
        f"r_{d}": float(np.corrcoef(xa, ya)[0, 1]),
        f"slope_{d}": float(np.polyfit(xa, ya, 1)[0]),
        f"theta_mae_{d}": float(np.mean(np.abs(ya - xa))),
        f"median_sd_{d}": float(np.median(sda)),
        f"median_se_total_{d}": float(np.median(stota)),
        f"pct_reach_se_ability_{d}": float(np.mean(sda <= se_t)),
    }
    for thr in SE_TOTAL_TARGETS:
        row[f"pct_se_total_le_{thr}_{d}"] = float(np.mean(stota <= thr))
    return row


def _heatmaps(rows, dims, path: Path, n_excl: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = dims[0]
    floors = sorted({r["floor"] for r in rows})
    ses = sorted({r["se_target"] for r in rows})
    by = {(r["floor"], r["se_target"]): r for r in rows}

    def grid_of(key):
        return np.array([[by[(fl, se)][key] for se in ses] for fl in floors], float)

    panels = [(f"r_{d}", "OOS recovery r (correctness)", "viridis", "{:.3f}"),
              (f"median_se_total_{d}", "median SE_total", "magma_r", "{:.3f}"),
              ("median_len", "median length (scenarios)", "cividis", "{:.0f}"),
              ("pct_reach", "%reach (SE_ability<=tgt AND SE_total<=0.30)", "YlGnBu", "{:.0%}")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (key, title, cmap, fmt) in zip(axes.ravel(), panels):
        M = grid_of(key)
        im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(ses))); ax.set_xticklabels(ses)
        ax.set_yticks(range(len(floors))); ax.set_yticklabels(floors)
        ax.set_xlabel("SE_ability target"); ax.set_ylabel("min_scenarios floor")
        ax.set_title(title, fontsize=10)
        for ii in range(len(floors)):
            for jj in range(len(ses)):
                ax.text(jj, ii, fmt.format(M[ii, jj]), ha="center", va="center",
                        fontsize=8, color="w" if cmap in ("viridis", "cividis") else "k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"TutorBench CORRECTNESS-ONLY unidim EAP stop: FULL OOS floor x SE grid "
                 f"(N={n_excl} headline, excl Qwen1.5-1.8B; k=5)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def _decision_table(rows_excl, dims, path: Path, n_excl: int):
    d = dims[0]
    by = {(r["floor"], r["se_target"]): r for r in rows_excl}
    floors = sorted({r["floor"] for r in rows_excl})
    ses = sorted({r["se_target"] for r in rows_excl})

    def line(r):
        return (f"| {r['floor']} | {r['se_target']:.2f} | {r['median_len']:.0f} | "
                f"{r['mean_len']:.1f} | {r['pct_reach']*100:.0f}% | "
                f"{r[f'median_se_total_{d}']:.3f} | {r[f'median_sd_{d}']:.3f} | "
                f"{r[f'r_{d}']:.3f} | {r[f'slope_{d}']:.3f} | {r[f'theta_mae_{d}']:.3f} | "
                f"{r['pct_info_plateau']*100:.0f}% | {r['pct_cap']*100:.0f}% |")

    hdr = ("| floor | SE_tgt | med len | mean len | %reach | med SE_total | med SD | "
           "r | slope | theta MAE | %plateau | %cap |")
    sep = "|" + "---|" * 12

    lines = ["# Correctness-only unidim OOS EAP grid - DECISION TABLE",
             "",
             f"STUDY-ONLY / LOCAL. Headline pool N={n_excl} (excludes Qwen/Qwen1.5-1.8B only; "
             "salamandra-7b-instruct kept/rehabilitated). k=5 model-fold refit-per-fold, "
             "seed 20260729, dense 321-node 1-D EAP stop, cap 70, plateau delta=0.005/W=3. "
             "SE_total = sqrt(SD^2 + SE_param^2), SE_param FIXED per-model offset (median 0.113).",
             "",
             "No operating point is locked here - these are the candidate cells for the user to choose from.",
             ""]

    # Shortlist: SE sweep at a mid floor + floor sweep at a mid SE target.
    mid_floor = 15 if 15 in floors else floors[len(floors) // 2]
    mid_se = 0.27 if 0.27 in ses else ses[len(ses) // 2]
    lines += [f"## SE_ability sweep at floor={mid_floor} (floor effectively inert here)", "",
              hdr, sep]
    lines += [line(by[(mid_floor, se)]) for se in ses]
    lines += ["", f"## Floor sweep at SE_ability={mid_se:.2f}", "", hdr, sep]
    lines += [line(by[(fl, mid_se)]) for fl in floors]

    # Curated operating-point candidates spanning the length/precision tradeoff.
    cand = [(15, 0.20), (15, 0.25), (15, 0.27), (20, 0.25), (20, 0.27), (20, 0.30)]
    cand = [c for c in cand if c in by]
    lines += ["", "## Curated operating-point candidates", "", hdr, sep]
    lines += [line(by[c]) for c in cand]

    lines += ["", "## Full 35-cell grid (headline pool)", "", hdr, sep]
    for fl in floors:
        for se in ses:
            lines.append(line(by[(fl, se)]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path,
                    default=HERE / "rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl")
    ap.add_argument("--matrix", type=Path,
                    default=ROOT / "staging/response_matrix_full_nonopt_115.csv")
    ap.add_argument("--scenarios", type=Path, default=ROOT / "data/TutorBench/scenarios.jsonl")
    ap.add_argument("--se-components", type=Path,
                    default=HERE / "param_uncertainty" / "leaderboard_se_components.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "oos_grid_full")
    ap.add_argument("--tmp-dir", type=Path,
                    default=ROOT / "staging" / "_eap_oos_grid_corr_only_full")
    ap.add_argument("--stop-nodes", type=int, default=321)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--fit-grid", type=int, default=7)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--negative-policy", default="clamp")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-test", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    records, dims, stats = scl.load_fitted_bank(args.bank, args.negative_policy)
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    crit_of = {r["criterion_id"]: r.get("criterion", "") for r in records}
    Q_all = np.array([[int(r["q_modeled"][dd]) for dd in dims] for r in records])
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    folds = base.make_folds(models, args.k, args.seed)

    n_joint = args.stop_nodes ** len(dims)
    print(f"[data] bank={args.bank.name} dims={dims} models={len(models)} items={len(ids)} "
          f"k={args.k} seed={args.seed}")
    print(f"[grid] FULL floors={FLOORS} se={SE_TARGETS} ({len(FLOORS)*len(SE_TARGETS)} cells)")
    print(f"[grid] stop/ref {args.stop_nodes} nodes/dim ({n_joint} joint); cap={args.cap}; "
          f"fit_grid={args.fit_grid} ridge={args.ridge} neg={args.negative_policy}")
    print(f"[folds] sizes={[len(f) for f in folds]}")

    se_df = pd.read_csv(args.se_components, index_col=0)
    se_df = se_df[~se_df.index.duplicated(keep="first")]
    med_param = {dd: float(np.median(se_df[f"se_param_{dd}"])) for dd in dims}
    se_param, n_fallback = {}, 0
    for m in models:
        if m in se_df.index:
            se_param[m] = {dd: float(se_df.loc[m, f"se_param_{dd}"]) for dd in dims}
        else:
            se_param[m] = dict(med_param)
            n_fallback += 1

    from concurrent.futures import ProcessPoolExecutor

    spec = scl.RunSpec(seed=args.seed, max_se=1e-9, min_evals_per_skill=0, min_scenarios=0,
                       max_scenarios=1_000_000, selection="trace", mode="cat", write_logs=False,
                       runs_dir=str(args.tmp_dir / "engine_runs"))
    all_traces: list[dict] = []
    fold_diag = []
    for f in range(args.k):
        test = folds[f]
        if args.limit_test:
            test = test[: args.limit_test]
        train = [m for m in models if m not in set(folds[f])]
        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        diag = base.refit_fold(train, models, row_of, Yraw, Mall, ids, Q_all, scen_of, crit_of,
                               dims, fold_bank, args.fit_grid, args.ridge, args.max_iter)
        fold_diag.append({"fold": f, "n_train": len(train), "n_test": len(test), **diag})
        print(f"  fold {f}: TRAIN={len(train)} TEST={len(test)} fit {diag['n_items']} items "
              f"(loglik={diag['loglik']:.0f} iters={diag['n_iter']} conv={diag['converged']})",
              flush=True)
        initargs = (str(fold_bank), str(args.matrix), str(args.scenarios), args.negative_policy,
                    args.stop_nodes, args.cap, asdict(spec))
        if args.workers and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=base._init_worker,
                                     initargs=initargs) as ex:
                for tr in ex.map(base._run_model, test):
                    if tr is not None:
                        tr["fold"] = f
                        all_traces.append(tr)
        else:
            base._init_worker(*initargs)
            for m in test:
                tr = base._run_model(m)
                if tr is not None:
                    tr["fold"] = f
                    all_traces.append(tr)
        print(f"    -> {len(test)} test traces done", flush=True)

    print(f"[traces] total held-out model traces: {len(all_traces)}")

    rows_all, rows_excl = [], []
    for floor in FLOORS:
        for se_t in SE_TARGETS:
            rows_all.append(_cell_metrics(all_traces, dims, floor, se_t, se_param, args.cap, ()))
            rows_excl.append(_cell_metrics(all_traces, dims, floor, se_t, se_param, args.cap,
                                           HEADLINE_EXCLUDE))

    excl_set = set(HEADLINE_EXCLUDE)
    per_model_cell_rows = []
    for floor in FLOORS:
        for se_t in SE_TARGETS:
            for tr in all_traces:
                i, reason = G.resolve_stop(tr["sd"], tr["nscen"], floor, se_t,
                                           PLATEAU_DELTA, PLATEAU_W, args.cap)
                sd_stop = tr["sd"][i]
                mwle_stop = tr["mwle"][i]
                sp = se_param[tr["model"]]
                stot = {dd: float(np.sqrt(sd_stop[k] ** 2 + sp[dd] ** 2))
                        for k, dd in enumerate(dims)}
                per_model_cell_rows.append({
                    "floor": floor, "se_target": se_t, "model": tr["model"], "fold": tr["fold"],
                    "is_headline_excluded": tr["model"] in excl_set,
                    "flag_note": FLAGGED_NOTE.get(tr["model"], ""),
                    "length": tr["nscen"][i], "stop_reason": reason,
                    **{f"theta_ref_{dd}": float(tr["theta_ref"][k]) for k, dd in enumerate(dims)},
                    **{f"theta_mwle_{dd}": float(mwle_stop[k]) for k, dd in enumerate(dims)},
                    **{f"sd_stop_{dd}": float(sd_stop[k]) for k, dd in enumerate(dims)},
                    **{f"se_total_{dd}": stot[dd] for dd in dims},
                })

    def _write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    _write_csv(args.out_dir / "oos_per_cell_grid_all115.csv", rows_all)
    _write_csv(args.out_dir / "oos_per_cell_grid_excl.csv", rows_excl)
    _write_csv(args.out_dir / "oos_per_model_per_cell.csv", per_model_cell_rows)

    n_excl = rows_excl[0]["n_models"]
    _heatmaps(rows_excl, dims, args.out_dir / "figures" / "oos_grid_full_heatmaps.png", n_excl)
    _decision_table(rows_excl, dims, args.out_dir / "DECISION_TABLE.md", n_excl)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STUDY ONLY - local; production engine untouched; nothing committed.",
        "benchmark": "TutorBench", "scale": "correctness-only unidimensional", "dims": dims,
        "protocol": "OOS k-fold refit-per-fold, offline FULL 35-cell replay",
        "model_set": "full-115", "n_models": len(models), "n_headline": n_excl,
        "matrix": str(args.matrix.relative_to(ROOT)),
        "k": args.k, "seed": args.seed, "fold_sizes": [len(f) for f in folds],
        "fit_config": {"fitter": "calibrate_mirt.fit_m2pl_em (unidim)",
                       "fit_grid_nodes_per_dim": args.fit_grid, "ridge": args.ridge,
                       "max_iter": args.max_iter, "negative_policy": args.negative_policy,
                       "within_fold_zero_variance_filter": True, "q_source": "unidim ones"},
        "stop_rule": {"metric": "EAP-posterior marginal SD (1-D)",
                      "stop_nodes_per_dim": args.stop_nodes, "grid_joint_nodes": int(n_joint),
                      "plateau_delta": PLATEAU_DELTA, "plateau_W": PLATEAU_W, "cap": args.cap,
                      "priority": "precision -> info_plateau -> cap -> bank_exhausted",
                      "estimator_at_stop": "MWLE (scenario_cat_lib.mwle_subset), start=dense EAP mean"},
        "reference": "full-bank dense-grid EAP theta on fold-refit clamped params",
        "reach_definition": f"SE_ability<=target AND SE_total<={SE_TOTAL_REACH}",
        "se_total_thresholds_reported": list(SE_TOTAL_TARGETS),
        "se_param_source": str(args.se_components),
        "se_param_note": "correctness-only per-model SE_param used as FIXED offset (NOT per-fold)",
        "se_param_median": med_param, "n_se_param_fallback": n_fallback,
        "headline_exclude": list(HEADLINE_EXCLUDE), "flag_notes": FLAGGED_NOTE,
        "floors": FLOORS, "se_targets": SE_TARGETS,
        "bank_stats": stats, "fold_diagnostics": fold_diag, "n_held_out_traces": len(all_traces),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[write] {args.out_dir}")
    d = dims[0]
    print("\n[excl-Qwen] floor SE   len  r      slope  MAE    medSDa medSEtot %reach %plat %cap")
    for r in rows_excl:
        print(f"  {r['floor']:<4} {r['se_target']:<4} {r['median_len']:<4.0f} "
              f"{r[f'r_{d}']:<6.3f} {r[f'slope_{d}']:<6.3f} {r[f'theta_mae_{d}']:<6.3f} "
              f"{r[f'median_sd_{d}']:<6.3f} {r[f'median_se_total_{d}']:<8.3f} "
              f"{r['pct_reach']*100:<6.0f} {r['pct_info_plateau']*100:<5.0f} {r['pct_cap']*100:<4.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
