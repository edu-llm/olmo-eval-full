"""Task C: BiGGen recovery x (floor x SE) operating-point grid + heatmaps.

Mirrors Bridge ``scenario_recovery_grid.py`` (06b_operating_point): for a rectangular grid
of operating points (min_scenarios floor x SE target) it reports, per cell,

  * OOS k-fold (k=5) theta-recovery r + slope (held-out models, MWLE vs the full-bank EAP
    reference) -- REUSED from the committed exp-06 grid CSVs where they already cover a
    cell, and re-run ONLY for the cells needed to complete the rectangle;
  * DEPLOYED (full-data frozen-bank) mean test length + median SE_total, where
    SE_total = sqrt(SE_posterior^2 + SE_param^2), SE_param from the observed-information
    parametric bootstrap over the engine's administered set (exp-07 machinery).

Convention note (mirrors Bridge + the BiGGen README reconciliation): ``recovery_r`` is the
OOS k-fold measure (fold-refit params); ``scenarios_mean`` / ``se_total_median`` are the
DEPLOYED frozen-bank measures (the true in-deployment length ~13.6 at floor 8 / SE 0.12).
The two conventions legitimately differ; each column is labelled by its convention.

To stay consistent with the committed exp-06 recovery numbers, the OOS reference EAP here
uses the SAME coarse uniform grid the exp-06 grid used (eap_grid=61 over [-6,6]); Task B
showed the fine grid barely moves r/slope (r 0.9703 -> 0.9715).

Writes ``experiments/06_floor_se_grid/06b_operating_point/{recovery_grid.csv,
figures/heatmaps_floor_x_se.png}`` (a NEW subdir; committed exp-06 figures untouched).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.scenario_cat_lib as scat  # noqa: E402
import scripts.scenario_param_uncertainty as refpu  # noqa: E402 (item_param_cov)

import importlib.util  # noqa: E402
_cm_spec = importlib.util.spec_from_file_location("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cm = importlib.util.module_from_spec(_cm_spec)
sys.modules["calibrate_mirt"] = cm
_cm_spec.loader.exec_module(cm)


def compute_item_cov(bank_path, matrix_path, fit_nodes, ridge):
    """Per-criterion (beta=[a,-b], chol) from observed information at the frozen fit."""
    records, dims, _ = scat.load_fitted_bank(bank_path, "clamp")
    ids, A, b = scat.assemble_arrays(records, dims)
    Q = np.array([[int(r["q_modeled"][d]) for d in dims] for r in records])
    matrix = pd.read_csv(matrix_path, index_col=0)
    sub = matrix.reindex(columns=ids)
    Ymat = np.nan_to_num(sub.to_numpy(float), nan=0.0)
    Mmat = ~np.isnan(sub.to_numpy(float))
    col = {c: i for i, c in enumerate(ids)}
    item_cov = refpu.item_param_cov(Ymat, Mmat, A, Q, b, len(dims), fit_nodes, ridge)
    beta, chol = [], []
    for fd, bta, cov in item_cov:
        beta.append(bta)
        try:
            Lc = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            Lc = np.zeros_like(cov)
        chol.append(Lc)
    return {"ids": ids, "dims": dims, "A": A, "b": b, "Ymat": Ymat, "Mmat": Mmat,
            "col": col, "beta": beta, "chol": chol, "matrix": matrix}


def bootstrap_theta(y, idx, beta, chol, grid, log_prior, n_boot, rng, batch=40):
    """Vectorised 1D parametric bootstrap of theta over administered items ``idx``."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.nan, np.nan, np.nan
    beta_sub = np.stack([beta[j] for j in idx])
    L_sub = np.stack([chol[j] for j in idx])
    yk = y[idx]
    a0, mb0 = beta_sub[:, 0], beta_sub[:, 1]
    eta0 = a0[:, None] * grid[None, :] + mb0[:, None]
    ll0 = yk @ log_expit(eta0) + (1.0 - yk) @ log_expit(-eta0)
    post0 = np.exp(ll0 + log_prior - logsumexp(ll0 + log_prior))
    mean0 = float(post0 @ grid)
    var0 = float(post0 @ (grid ** 2) - mean0 ** 2)
    se_post = float(np.sqrt(max(var0, 0.0)))
    thetas = np.empty(n_boot)
    done = 0
    while done < n_boot:
        bsz = min(batch, n_boot - done)
        z = rng.standard_normal((bsz, idx.size, 2))
        draw = beta_sub[None] + np.einsum("kij,bkj->bki", L_sub, z)
        eta = draw[:, :, 0][:, :, None] * grid[None, None, :] + draw[:, :, 1][:, :, None]
        ll = (np.einsum("k,bkn->bn", yk, log_expit(eta))
              + np.einsum("k,bkn->bn", 1.0 - yk, log_expit(-eta)))
        post = np.exp(ll + log_prior[None, :] - logsumexp(ll + log_prior[None, :], axis=1)[:, None])
        thetas[done:done + bsz] = post @ grid
        done += bsz
    return mean0, se_post, float(thetas.std(ddof=1))


def make_folds(models, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def load_existing_grid():
    """(floor, se_round2) -> {oos_r, oos_slope, mean_scen} from committed exp-06 CSVs."""
    base = ROOT / "biggen_calibration" / "experiments" / "06_floor_se_grid"
    out = {}
    for name in ("results.csv", "results_fine_se.csv", "results_tight_se.csv"):
        pth = base / name
        if not pth.is_file():
            continue
        df = pd.read_csv(pth)
        for _, r in df.iterrows():
            key = (int(r["min_scenarios"]), round(float(r["se_target"]), 2))
            out.setdefault(key, {"oos_r": float(r["oos_r"]), "oos_slope": float(r["oos_slope"]),
                                 "mean_scen_oos": float(r["mean_scen"])})
    return out


def recovery_stats(ref, cat):
    ref, cat = np.asarray(ref), np.asarray(cat)
    slope, intercept = np.polyfit(ref, cat, 1)
    return (round(float(np.corrcoef(ref, cat)[0, 1]), 4), round(float(slope), 4),
            round(float(np.mean(np.abs(cat - ref))), 4))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "biggen_calibration"
    p.add_argument("--bank", type=Path, default=ROOT / "staging" / "biggen_unidim_modeled.jsonl")
    p.add_argument("--matrix", type=Path, default=ROOT / "staging" / "biggen_response_matrix.csv")
    p.add_argument("--scenarios", type=Path, default=ROOT / "data" / "BiGGen" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path,
                   default=base / "experiments" / "06_floor_se_grid" / "06b_operating_point")
    p.add_argument("--tmp-dir", type=Path, default=ROOT / "staging" / "_biggen_recovery_grid")
    p.add_argument("--floors", type=str, default="6,8,10,12,15")
    p.add_argument("--se-list", type=str, default="0.10,0.11,0.12,0.13,0.14,0.15")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-scenarios", type=int, default=50)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--from-csv", action="store_true",
                   help="redraw heatmaps from an existing recovery_grid.csv (no engine/CV re-run).")
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    floors = [int(x) for x in args.floors.split(",") if x.strip()]
    se_list = [round(float(x), 2) for x in args.se_list.split(",") if x.strip()]

    if args.from_csv:
        rows = pd.read_csv(args.out_dir / "recovery_grid.csv").to_dict("records")
        _figures(rows, floors, se_list, args.out_dir / "figures")
        print(f"(--from-csv) redrew -> {args.out_dir / 'figures' / 'heatmaps_floor_x_se.png'}")
        return 0

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    assert len(dims) == 1
    d = dims[0]
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(args.matrix, index_col=0)
    Yraw = matrix.reindex(columns=ids).to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    col_full = {c: i for i, c in enumerate(ids)}
    Q_all = np.ones((len(ids), 1), dtype=int)
    egrid, elog = scat.build_grid(1, args.eap_grid, args.range)

    print("=" * 90)
    print(f"Task C: recovery x (floor x SE) grid | floors={floors} SE={se_list}")
    print("=" * 90)

    existing = load_existing_grid()
    grid_cells = [(fl, se) for fl in floors for se in se_list]
    missing_r = [(fl, se) for (fl, se) in grid_cells if (fl, se) not in existing]
    print(f"cells={len(grid_cells)}  reused recovery_r={len(grid_cells)-len(missing_r)}  "
          f"re-run recovery_r={len(missing_r)} -> {missing_r}")

    # SE_param machinery (full-data observed info) + 1D EAP bootstrap grid
    cov = compute_item_cov(args.bank, args.matrix, args.fit_grid, args.ridge)
    bgrid = np.linspace(-args.range, args.range, args.eap_grid)
    blp = -0.5 * bgrid ** 2
    blp = blp - logsumexp(blp)
    rng_boot = np.random.default_rng(20260801)

    # Fit the k folds ONCE (op-point-independent) for the re-run recovery cells.
    fold_data = []
    if missing_r:
        print("\nfitting 5 CV folds once (for the re-run recovery cells) ...", flush=True)
        folds = make_folds(models, args.k, args.seed)
        for f in range(args.k):
            test = folds[f]
            train = [m for m in models if m not in set(test)]
            tr = [row_of[m] for m in train]
            Ytr, Mtr = Yraw[tr], Mall[tr]
            keep = [j for j in range(len(ids)) if Mtr[:, j].sum() >= 2
                    and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
            keep = np.array(keep, dtype=int)
            fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep],
                                 Q_all[keep], args.fit_grid, ridge=args.ridge, max_iter=200)
            Ak, bk = fit["A"], fit["b"]
            kept_ids = [ids[j] for j in keep]
            colk = {c: i for i, c in enumerate(kept_ids)}
            fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
            with fold_bank.open("w", encoding="utf-8") as fh:
                for jj, cid in enumerate(kept_ids):
                    fh.write(json.dumps({"criterion_id": cid, "scenario_id": scen_of[cid],
                                         "criterion": "", "discrimination": {d: float(Ak[jj, 0])},
                                         "q_modeled": {d: 1}, "difficulty": float(bk[jj])}) + "\n")
            subte = matrix.loc[test].reindex(columns=kept_ids)
            Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
            Mte = ~np.isnan(subte.to_numpy(float))
            theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)
            fold_data.append({"test": test, "bank": fold_bank, "colk": colk,
                              "Ak": Ak, "bk": bk, "Yte": Yte, "theta_ref": theta_ref})
            print(f"  fold {f}: train={len(train)} test={len(test)} kept={len(kept_ids)}", flush=True)

    rows = []
    for fl, se in grid_cells:
        # (A) DEPLOYED full-data frozen-bank engine run -> length + SE_total
        spec = scat.RunSpec(seed=42, top_n=5, max_se=se,
                            min_evals_per_skill=args.min_evals_per_skill,
                            min_scenarios=fl, max_scenarios=args.max_scenarios,
                            selection="trace", mode="cat",
                            runs_dir=str(args.tmp_dir / "full_runs"))
        full = scat.run_models(models, args.bank, args.matrix, args.scenarios,
                               "clamp", dims, spec, workers=args.workers)
        scen_admin = np.array([r["scenarios_administered"] for r in full])
        n_conv = sum(1 for r in full if r["precision_reached"])
        se_tot_list, se_post_list, se_par_list = [], [], []
        for r0 in full:
            y = Yraw[row_of[r0["model"]]]
            idx = np.array([col_full[c] for c in r0["order"] if c in col_full], dtype=int)
            _, se_post, se_par = bootstrap_theta(y, idx, cov["beta"], cov["chol"],
                                                 bgrid, blp, args.n_boot, rng_boot)
            if np.isfinite(se_post):
                se_post_list.append(se_post)
                se_par_list.append(se_par)
                se_tot_list.append(float(np.sqrt(se_post ** 2 + se_par ** 2)))

        # (B) OOS recovery r/slope: reuse committed exp-06 where present, else re-run
        if (fl, se) in existing:
            oos_r = round(existing[(fl, se)]["oos_r"], 4)
            oos_slope = round(existing[(fl, se)]["oos_slope"], 4)
            theta_mae = np.nan
            r_source = "reused_exp06"
        else:
            ref_all, cat_all = [], []
            for fd in fold_data:
                spec_f = scat.RunSpec(seed=args.seed, top_n=5, max_se=se,
                                      min_evals_per_skill=args.min_evals_per_skill,
                                      min_scenarios=fl, max_scenarios=args.max_scenarios,
                                      selection="trace", mode="cat",
                                      runs_dir=str(args.tmp_dir / "runs_f"))
                res = scat.run_models(fd["test"], fd["bank"], args.matrix, args.scenarios,
                                      "clamp", dims, spec_f, workers=args.workers)
                res_by = {r["model"]: r for r in res}
                Ak, bk, colk = fd["Ak"], fd["bk"], fd["colk"]
                for ti, m in enumerate(fd["test"]):
                    r0 = res_by[m]
                    y = fd["Yte"][ti]
                    idx = np.array([colk[c] for c in r0["order"] if c in colk], dtype=int)
                    if idx.size == 0:
                        th = fd["theta_ref"][ti].copy()
                    else:
                        th_ba = scat.eap_subset(y, idx, Ak, bk, egrid, elog)
                        th, _ = scat.mwle_subset(y, idx, Ak, bk, th_ba)
                    ref_all.append(float(fd["theta_ref"][ti][0]))
                    cat_all.append(float(th[0]))
            oos_r, oos_slope, theta_mae = recovery_stats(ref_all, cat_all)
            r_source = "rerun_kfold"

        row = {
            "op_point": f"f{fl}_se{se:.2f}", "min_scenarios": fl, "se_target": se,
            "recovery_r": oos_r, "recovery_slope": oos_slope, "theta_mae": theta_mae,
            "recovery_source": r_source,
            "scenarios_mean_deployed": round(float(scen_admin.mean()), 2),
            "scenarios_median_deployed": int(np.median(scen_admin)),
            "convergence_rate_deployed": round(n_conv / len(full), 3),
            "ability_se_median_deployed": round(float(np.median(se_post_list)), 4),
            "se_param_median_deployed": round(float(np.median(se_par_list)), 4),
            "se_total_median_deployed": round(float(np.median(se_tot_list)), 4),
        }
        rows.append(row)
        print(f"  {row['op_point']:11s} [{r_source:12s}] r={oos_r:.3f} "
              f"scen_dep={row['scenarios_mean_deployed']:5.1f} "
              f"SEtot_dep={row['se_total_median_deployed']:.3f} "
              f"conv={row['convergence_rate_deployed']:.2f}", flush=True)

    with (args.out_dir / "recovery_grid.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _figures(rows, floors, se_list, args.out_dir / "figures")

    # tidy temp run dirs (keep fold banks under tmp for provenance)
    for sub_ in ("full_runs", "runs_f"):
        shutil.rmtree(args.tmp_dir / sub_, ignore_errors=True)
    print(f"\nwrote -> {args.out_dir / 'recovery_grid.csv'}")
    print(f"wrote -> {args.out_dir / 'figures' / 'heatmaps_floor_x_se.png'}")
    return 0


def _pivot(df, value, floors, se_list):
    M = np.full((len(floors), len(se_list)), np.nan)
    fi = {f: i for i, f in enumerate(floors)}
    si = {s: j for j, s in enumerate(se_list)}
    for _, r in df.iterrows():
        M[fi[int(r["min_scenarios"])], si[round(float(r["se_target"]), 2)]] = r[value]
    return M


def _heatmap(ax, M, floors, se_list, title, cmap, fmt):
    im = ax.imshow(M, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(se_list))); ax.set_xticklabels([f"{s:.2f}" for s in se_list], fontsize=8)
    ax.set_yticks(range(len(floors))); ax.set_yticklabels([str(f) for f in floors], fontsize=8)
    ax.set_xlabel("SE target"); ax.set_ylabel("min_scenarios (floor)")
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5)
    return im


def _figures(rows, floors, se_list, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    floors = sorted(set(int(x) for x in floors))
    se_list = sorted(set(round(float(x), 2) for x in se_list))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = _heatmap(axes[0], _pivot(df, "recovery_r", floors, se_list), floors, se_list,
                   "OOS recovery r (k-fold)", "viridis", "{:.3f}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = _heatmap(axes[1], _pivot(df, "se_total_median_deployed", floors, se_list), floors,
                   se_list, "median SE_total (deployed)", "magma_r", "{:.3f}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = _heatmap(axes[2], _pivot(df, "scenarios_mean_deployed", floors, se_list), floors,
                   se_list, "mean scenarios administered (DEPLOYED)", "cividis", "{:.1f}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[0].set_title("OOS recovery r  (OOS k-fold, fold-refit params)", fontsize=10)
    axes[1].set_title("median SE_total  (DEPLOYED frozen-bank)", fontsize=10)
    fig.suptitle("BiGGen recovery x (floor x SE) operating-point grid", fontsize=13)
    fig.text(0.5, 0.005,
             "REGIME SPLIT (per the README 9.4-vs-13.6 reconciliation): recovery r is the "
             "OOS k-fold measure (fold-refit params); SE_total and test length are the "
             "DEPLOYED frozen-bank measures (longer than OOS). The two conventions differ by "
             "design and are NOT directly comparable across panels.",
             ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(fig_dir / "heatmaps_floor_x_se.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
