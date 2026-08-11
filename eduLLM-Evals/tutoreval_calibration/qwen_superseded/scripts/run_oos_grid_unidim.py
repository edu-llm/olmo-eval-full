"""OOS floor x SE EAP-posterior-stop grid for TutorEval on its CANONICAL UNIDIM scale.

Remediation of a teammate prototype that wrongly ran the 2-skill bank. The of-record
TutorEval instrument is *unidimensional* (single ``ability`` axis), so this study runs the
whole floor x SE_ability grid on the unidim fitted bank.

It is a STUDY harness only. It does NOT modify the production engine (``tutor_cat/``,
``scripts/scenario_cat_lib.py``) -- it imports and calls them read-only. All outputs land in
this folder (``reports/eap_oos_grid_tutoreval_unidim/``).

Design (mirrors the of-record TutorEval unidim k-fold + the validated 1-D EAP-stop prototype):
  * Assets: unidim fitted bank ``data/TutorEval/rubrics_qmatrix_final_unidim_fitted.jsonl``
    (single ``ability`` discrimination) + the of-record response matrix
    ``runs/judge/TutorEval/response_matrix.csv`` (sha256 f7830280...; an S3-only calibration
    input -- pass it with --matrix). scenarios: ``data/TutorEval/scenarios_final.jsonl``.
  * k=5 model-fold OOS, seed 20260729, refit (a,b) per fold via ``calibrate_mirt.fit_m2pl_em``
    (1 dim, ridge 0.01, fit_grid 7, within-fold zero-variance filtering). Q held fixed = ones.
  * Full-length forced trace per held-out model ONCE per fold (engine cap = 70 scenarios,
    production max-info scenario selection), then OFFLINE-REPLAY all 35 grid cells on that
    single trace.
  * Stop: EAP-posterior SD on a DENSE 1-D grid (321 nodes, spacing 0.05 over [-8,8]);
    info-plateau delta=0.005 / W=3; cap 70; priority precision -> plateau -> cap ->
    bank_exhausted. Ability at stop = deployed MWLE over the administered items.
  * Grid: floors {10,12,15,20,25} x SE_ability {0.15,0.20,0.22,0.25,0.27,0.30,0.32} = 35 cells.
  * SE_total = sqrt(SD^2 + SE_param^2), SE_param per model from the of-record unidim bootstrap
    (``param_uncertainty/tutoreval_unidim/leaderboard_se_components.csv``), else the median.
  * %reach = SE_ability(=posterior SD) <= target AND SE_total <= 0.30.
  * Weakly-identified models are IDENTIFIED from the run (best-achievable SE_total at the cap
    still > 0.30) -> an excl-weak grid variant is emitted alongside the all-52 headline.

Usage
-----
    python reports/eap_oos_grid_tutoreval_unidim/run_oos_grid_unidim.py --workers 6
    # (optionally) --matrix <path-to-response_matrix.csv>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp

# reports/eap_oos_grid_tutoreval_unidim/ -> reports/ -> eduLLM-Evals/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402  (read-only import of the shared engine lib)

HERE = Path(__file__).resolve().parent

# of-record unidim median SE_param fallback (param_uncertainty/tutoreval_unidim/metrics.json)
SE_PARAM_MEDIAN = 0.1166732479300126
SE_TOTAL_GATE = 0.30
CAP = 70
FLOORS = [10, 12, 15, 20, 25]
SE_TARGETS = [0.15, 0.20, 0.22, 0.25, 0.27, 0.30, 0.32]
PLATEAU_DELTA = 0.005
PLATEAU_W = 3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")


def make_folds(models, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def eap_walk(order, yrow, col, scen_of, a, b, gg, lp):
    """Per-scenario cumulative EAP (mean, SD) along an administration order (1-D dense grid).

    Returns a list of steps: {n_scen, n_crit, eap_mean, eap_sd, idx}. Mirrors the validated
    prototype ``eap_walk`` -- the posterior is L(administered|theta) * N(0,1) on ``gg``.
    """
    ll_data = np.zeros(gg.size)
    out = []
    cur = None
    n_scen = 0
    n_crit = 0
    cum_idx: list[int] = []

    def _snap():
        post = np.exp(ll_data + lp - logsumexp(ll_data + lp))
        mean = float(post @ gg)
        sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
        out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                    "eap_sd": sd, "idx": np.array(cum_idx, dtype=int)})

    for cid in order:
        j = col.get(cid)
        if j is None:
            continue
        sid = scen_of.get(cid)
        if sid != cur:
            if cur is not None:
                _snap()
            cur = sid
            n_scen += 1
        eta = a[j] * gg - b[j]
        ll_data = ll_data + yrow[j] * log_expit(eta) + (1.0 - yrow[j]) * log_expit(-eta)
        n_crit += 1
        cum_idx.append(j)
    if cur is not None:
        _snap()
    return out


def stop_point(walk, floor, target, cap=CAP, delta=PLATEAU_DELTA, w=PLATEAU_W):
    """Apply the stop rule to one model's per-scenario walk.

    Priority precision -> plateau -> cap -> bank_exhausted. Precision: n_scen>=floor AND
    posterior SD<=target. Plateau: n_scen>=floor AND SD improved by <delta over the last w
    scenarios. Otherwise stop at the last step (cap if it reached the forced cap, else
    bank_exhausted). Returns (step_dict, reason).
    """
    if not walk:
        return None, "bank_exhausted"
    sds = [s["eap_sd"] for s in walk]
    for i, step in enumerate(walk):
        if step["n_scen"] < floor:
            continue
        if sds[i] <= target:
            return step, "precision"
        if i >= w and (sds[i - w] - sds[i]) < delta:
            return step, "plateau"
    last = walk[-1]
    return last, ("cap" if last["n_scen"] >= cap else "bank_exhausted")


def run_forced(models, bank, matrix_path, scenarios_path, dims, seed, cap, workers, tmp_dir):
    """Force the production engine to the cap so we capture each model's full admin ORDER."""
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=0.0, min_evals_per_skill=0,
                        min_scenarios=cap, max_scenarios=cap, selection="trace",
                        mode="cat", runs_dir=str(tmp_dir))
    return {r["model"]: r for r in scat.run_models(models, bank, matrix_path, scenarios_path,
                                                    "clamp", dims, spec, workers=workers)}


def recovery(ref, cat):
    ref = np.asarray(ref, float)
    cat = np.asarray(cat, float)
    ok = np.isfinite(ref) & np.isfinite(cat)
    ref, cat = ref[ok], cat[ok]
    if ref.size < 3:
        return {"r": float("nan"), "slope": float("nan"), "theta_mae": float("nan"),
                "n": int(ref.size)}
    return {"r": float(np.corrcoef(ref, cat)[0, 1]),
            "slope": float(np.polyfit(ref, cat, 1)[0]),
            "theta_mae": float(np.mean(np.abs(cat - ref))),
            "n": int(ref.size)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path,
                   default=ROOT / "data" / "TutorEval" / "rubrics_qmatrix_final_unidim_fitted.jsonl")
    p.add_argument("--matrix", type=Path,
                   default=ROOT / "runs" / "judge" / "TutorEval" / "response_matrix.csv")
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "data" / "TutorEval" / "scenarios_final.jsonl")
    p.add_argument("--se-components", type=Path,
                   default=ROOT / "regenerated_figures" / "scenario_level" / "param_uncertainty"
                   / "tutoreval_unidim" / "leaderboard_se_components.csv")
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--tmp-dir", type=Path, default=HERE / "_tmp")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--ref-grid", type=int, default=61, help="reference/subset EAP grid (of-record).")
    p.add_argument("--ref-range", type=float, default=6.0)
    p.add_argument("--dense-grid", type=int, default=321, help="dense 1-D stop-rule SD grid.")
    p.add_argument("--dense-range", type=float, default=8.0)
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    if not args.matrix.is_file():
        sys.stderr.write(
            "\nERROR: TutorEval response matrix not found:\n"
            f"    {args.matrix}\n\n"
            "This file is the of-record calibration input (sha256 f7830280a40f6bb774116bb6e7a44f9840073c88d7ce8fbd538b357cdcbd4781).\n"
            "It is NOT tracked in git (.gitignore: runs/) and lives on S3. Provide it, e.g.:\n"
            "    --matrix path/to/response_matrix.csv\n"
            "Rows = 52 models, columns = te_* criterion ids. Nothing else is missing.\n")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- load unidim bank (clamp negatives at load, as the of-record CAT pipeline does) ------
    records, dims, bank_stats = scat.load_fitted_bank(args.bank, "clamp")
    if dims != ["ability"]:
        sys.stderr.write(f"ERROR: expected unidim bank with dims=['ability'], got {dims}\n")
        return 2
    d = dims[0]
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    crit_of = {r["criterion_id"]: r.get("criterion", "") for r in records}
    Q_all = np.array([[int(r["q_modeled"][dd]) for dd in dims] for r in records])

    prov = scat.verify_provenance(args.bank, args.matrix)

    matrix = pd.read_csv(args.matrix, index_col=0)
    sub = matrix.reindex(columns=ids)
    Yraw = sub.to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    n_models = len(models)
    print(f"bank={args.bank.name} dims={dims} models={n_models} items={len(ids)} "
          f"k={args.k} seed={args.seed} matrix_aligned={prov['aligned']}")

    # SE_param per model (of-record unidim bootstrap), else median fallback
    se_param_by: dict[str, float] = {}
    if args.se_components.is_file():
        sc = pd.read_csv(args.se_components)
        se_param_by = dict(zip(sc["model"], sc["se_param_ability"]))
    se_param = np.array([float(se_param_by.get(m, SE_PARAM_MEDIAN)) for m in models])

    # dense stop grid + reference EAP grid
    gg = np.linspace(-args.dense_range, args.dense_range, args.dense_grid)
    lp = -0.5 * gg ** 2
    lp = lp - logsumexp(lp)
    egrid, elog = scat.build_grid(1, args.ref_grid, args.ref_range)

    folds = make_folds(models, args.k, args.seed)

    # ------- ONE forced full-length trace per held-out model (per fold) -----------------------
    # store per-model: walk (dense-grid), theta_ref, theta_mwle_at_each_step is computed offline
    per_model: dict[str, dict] = {}
    for f in range(args.k):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr_idx = [row_of[m] for m in train]
        Ytr, Mtr = Yraw[tr_idx], Mall[tr_idx]
        # within-fold zero-variance filtering (>=2 observed TRAIN, both a pass and a fail)
        keep = [j for j in range(len(ids))
                if Mtr[:, j].sum() >= 2 and 0 < Ytr[Mtr[:, j], j].sum() < Mtr[:, j].sum()]
        keep = np.array(keep, dtype=int)
        fit = cm.fit_m2pl_em(np.nan_to_num(Ytr[:, keep], nan=0.0), Mtr[:, keep],
                             Q_all[keep], args.fit_grid, ridge=args.ridge, max_iter=args.max_iter)
        Ak, bk = fit["A"], fit["b"]
        ak = Ak[:, 0]
        kept_ids = [ids[j] for j in keep]
        colk = {c: i for i, c in enumerate(kept_ids)}
        print(f"  fold {f}: TRAIN={len(train)} TEST={len(test)} fit {len(kept_ids)} items "
              f"(loglik={fit['loglik']:.0f}, iters={fit['n_iter']})", flush=True)

        # temp fold bank in the fitted-bank schema (single 'ability' loading)
        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                fh.write(json.dumps({"criterion_id": cid, "scenario_id": scen_of[cid],
                                     "criterion": crit_of[cid],
                                     "discrimination": {d: float(Ak[jj, 0])},
                                     "q_modeled": {d: 1}, "difficulty": float(bk[jj])}) + "\n")

        # reference full-info ability for TEST under fold params
        subte = matrix.loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, egrid, elog)

        res = run_forced(test, fold_bank, args.matrix, args.scenarios, dims, args.seed,
                         args.cap, args.workers, args.tmp_dir / f"runs_f{f}")
        for ti, m in enumerate(test):
            walk = eap_walk(res[m]["order"], Yte[ti], colk, scen_of, ak, bk, gg, lp)
            per_model[m] = {"fold": f, "walk": walk, "theta_ref": float(theta_ref[ti][0]),
                            "y": Yte[ti], "Ak": Ak, "bk": bk, "colk": colk}
        shutil.rmtree(args.tmp_dir / f"runs_f{f}", ignore_errors=True)

    # ------- identify weakly-identified models (data-driven) ----------------------------------
    # best achievable SE_total at the forced cap (min over the walk) still exceeds the gate.
    weak = []
    best_se_total = {}
    for m in models:
        walk = per_model[m]["walk"]
        sp = float(se_param_by.get(m, SE_PARAM_MEDIAN))
        if walk:
            se_totals = [float(np.sqrt(s["eap_sd"] ** 2 + sp ** 2)) for s in walk]
            best_se_total[m] = min(se_totals)
        else:
            best_se_total[m] = float("nan")
        if not (best_se_total[m] <= SE_TOTAL_GATE):
            weak.append(m)

    # ------- OFFLINE replay: all 35 cells on the single trace per model -----------------------
    per_cell_rows: list[dict] = []
    per_model_cell_rows: list[dict] = []
    for floor in FLOORS:
        for tgt in SE_TARGETS:
            recs = []
            for m in models:
                info = per_model[m]
                walk = info["walk"]
                step, reason = stop_point(walk, floor, tgt, cap=args.cap)
                sp = float(se_param_by.get(m, SE_PARAM_MEDIAN))
                if step is None or step["idx"].size == 0:
                    n_scen = n_crit = 0
                    sd = float("nan")
                    th_mw = info["theta_ref"]
                else:
                    n_scen = step["n_scen"]
                    n_crit = step["n_crit"]
                    sd = step["eap_sd"]
                    idx = step["idx"]
                    th_ba = scat.eap_subset(info["y"], idx, info["Ak"], info["bk"], egrid, elog)
                    th_mw_v, _ = scat.mwle_subset(info["y"], idx, info["Ak"], info["bk"], th_ba)
                    th_mw = float(th_mw_v[0])
                se_total = float(np.sqrt(sd ** 2 + sp ** 2)) if np.isfinite(sd) else float("nan")
                reach = bool(np.isfinite(sd) and sd <= tgt and se_total <= SE_TOTAL_GATE)
                rec = {"model": m, "floor": floor, "se_target": tgt,
                       "n_scenarios": n_scen, "n_criteria": n_crit, "stop_reason": reason,
                       "sd": sd, "se_param": sp, "se_total": se_total,
                       "theta_ref": info["theta_ref"], "theta_mwle": th_mw,
                       "reach": reach, "weak": m in weak}
                recs.append(rec)
                per_model_cell_rows.append(rec)

            def _agg(rows):
                n = len(rows)
                if n == 0:
                    return {}
                ls = np.array([r["n_scenarios"] for r in rows], float)
                lc = np.array([r["n_criteria"] for r in rows], float)
                sd = np.array([r["sd"] for r in rows], float)
                st = np.array([r["se_total"] for r in rows], float)
                rr = np.array([1.0 if r["reach"] else 0.0 for r in rows])
                reasons = [r["stop_reason"] for r in rows]
                rc = recovery([r["theta_ref"] for r in rows], [r["theta_mwle"] for r in rows])
                return {"floor": floor, "se_target": tgt, "n_models": n,
                        "median_len_scenarios": float(np.median(ls)),
                        "mean_len_scenarios": float(np.mean(ls)),
                        "median_len_criteria": float(np.median(lc)),
                        "mean_len_criteria": float(np.mean(lc)),
                        "pct_reach": round(100.0 * rr.mean(), 1),
                        "pct_precision": round(100.0 * np.mean([x == "precision" for x in reasons]), 1),
                        "pct_plateau": round(100.0 * np.mean([x == "plateau" for x in reasons]), 1),
                        "pct_cap": round(100.0 * np.mean([x == "cap" for x in reasons]), 1),
                        "pct_bank_exhausted": round(100.0 * np.mean([x == "bank_exhausted" for x in reasons]), 1),
                        "recovery_r": round(rc["r"], 4), "recovery_slope": round(rc["slope"], 4),
                        "theta_mae": round(rc["theta_mae"], 4),
                        "median_sd": round(float(np.nanmedian(sd)), 4),
                        "median_se_total": round(float(np.nanmedian(st)), 4)}

            per_cell_rows.append(_agg(recs))

    grid_all = pd.DataFrame(per_cell_rows)
    grid_all.to_csv(args.out_dir / "oos_per_cell_grid.csv", index=False)
    pm = pd.DataFrame(per_model_cell_rows)
    pm.to_csv(args.out_dir / "oos_per_model_per_cell.csv", index=False)

    # excl-weak variant (recompute cell aggregates dropping weak models)
    grid_excl = None
    if weak:
        rows_excl = []
        for floor in FLOORS:
            for tgt in SE_TARGETS:
                rows = pm[(pm.floor == floor) & (pm.se_target == tgt) & (~pm.weak)]
                if len(rows) == 0:
                    continue
                rc = recovery(rows["theta_ref"].to_numpy(), rows["theta_mwle"].to_numpy())
                reasons = rows["stop_reason"]
                rows_excl.append({
                    "floor": floor, "se_target": tgt, "n_models": int(len(rows)),
                    "median_len_scenarios": float(rows["n_scenarios"].median()),
                    "mean_len_scenarios": float(rows["n_scenarios"].mean()),
                    "median_len_criteria": float(rows["n_criteria"].median()),
                    "mean_len_criteria": float(rows["n_criteria"].mean()),
                    "pct_reach": round(100.0 * rows["reach"].mean(), 1),
                    "pct_precision": round(100.0 * (reasons == "precision").mean(), 1),
                    "pct_plateau": round(100.0 * (reasons == "plateau").mean(), 1),
                    "pct_cap": round(100.0 * (reasons == "cap").mean(), 1),
                    "pct_bank_exhausted": round(100.0 * (reasons == "bank_exhausted").mean(), 1),
                    "recovery_r": round(rc["r"], 4), "recovery_slope": round(rc["slope"], 4),
                    "theta_mae": round(rc["theta_mae"], 4),
                    "median_sd": round(float(rows["sd"].median()), 4),
                    "median_se_total": round(float(rows["se_total"].median()), 4)})
        grid_excl = pd.DataFrame(rows_excl)
        grid_excl.to_csv(args.out_dir / "oos_per_cell_grid_excl_weak.csv", index=False)

    _heatmaps(grid_all, args.out_dir / "figures" / "oos_grid_heatmaps.png",
              title=f"TutorEval unidim OOS grid (all {n_models} models)")

    _decision_table(grid_all, grid_excl, weak, args.out_dir / "DECISION_TABLE.md", prov, n_models)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": "OOS floor x SE EAP-posterior-stop grid, TutorEval UNIDIM (single 'ability' axis)",
        "engine": "read-only: tutor_cat + scripts/scenario_cat_lib (production); refit via calibrate_mirt.fit_m2pl_em",
        "bank": str(args.bank), "matrix": str(args.matrix), "scenarios": str(args.scenarios),
        "matrix_provenance": prov, "bank_load_stats": bank_stats,
        "dims": dims, "n_models": n_models, "k": args.k, "seed": args.seed,
        "fit_config": {"fit_grid": args.fit_grid, "ridge": args.ridge, "max_iter": args.max_iter,
                       "negative_policy_at_load": "clamp"},
        "stop_rule": {"dense_grid_nodes": args.dense_grid, "dense_range": args.dense_range,
                      "plateau_delta": PLATEAU_DELTA, "plateau_w": PLATEAU_W, "cap": args.cap,
                      "priority": ["precision", "plateau", "cap", "bank_exhausted"],
                      "ability_at_stop": "MWLE"},
        "grid": {"floors": FLOORS, "se_targets": SE_TARGETS, "n_cells": len(FLOORS) * len(SE_TARGETS)},
        "reach_definition": "SE_ability(posterior SD) <= target AND SE_total <= 0.30",
        "se_param_source": (str(args.se_components) if args.se_components.is_file()
                            else f"median fallback {SE_PARAM_MEDIAN}"),
        "weak_models": {"criterion": "best achievable SE_total at cap > 0.30 (data-driven)",
                        "n": len(weak), "models": weak,
                        "best_se_total_at_cap": {m: round(best_se_total[m], 4) for m in weak}},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    print(f"\nwrote -> {args.out_dir}")
    print(f"weakly-identified (SE_total>0.30 even at cap): {len(weak)} models")
    return 0


def _heatmaps(grid: pd.DataFrame, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floors = sorted(grid["floor"].unique())
    tgts = sorted(grid["se_target"].unique())

    def mat(col):
        M = np.full((len(floors), len(tgts)), np.nan)
        for _, r in grid.iterrows():
            M[floors.index(r["floor"]), tgts.index(r["se_target"])] = r[col]
        return M

    panels = [("recovery_r", "OOS recovery r (ability)", "viridis"),
              ("median_se_total", "median SE_total", "viridis_r"),
              ("median_len_scenarios", "median length (scenarios)", "magma"),
              ("pct_reach", "% reach (SE<=tgt & SE_total<=0.30)", "viridis")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (col, ttl, cmap) in zip(axes.ravel(), panels):
        M = mat(col)
        im = ax.imshow(M, aspect="auto", cmap=cmap, origin="upper")
        ax.set_xticks(range(len(tgts)), [f"{t:g}" for t in tgts])
        ax.set_yticks(range(len(floors)), [str(f) for f in floors])
        ax.set_xlabel("SE_ability target"); ax.set_ylabel("floor (min scenarios)")
        ax.set_title(ttl, fontsize=10)
        for i in range(len(floors)):
            for j in range(len(tgts)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="w")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _decision_table(grid: pd.DataFrame, grid_excl, weak, path: Path, prov, n_models):
    lines = ["# TutorEval unidim -- OOS EAP-stop grid: decision table", "",
             f"- Bank: `data/TutorEval/rubrics_qmatrix_final_unidim_fitted.jsonl` (single `ability`).",
             f"- Matrix: `runs/judge/TutorEval/response_matrix.csv` (aligned={prov['aligned']}).",
             f"- N models: {n_models}. Fit: 1-D 2PL, ridge 0.01, fit_grid 7. Stop: dense-grid EAP SD, "
             "plateau delta=0.005/W=3, cap 70. Ability at stop = MWLE.",
             f"- Weakly-identified (SE_total>0.30 even at cap): {len(weak)} -> "
             f"`{', '.join(weak) if weak else 'none'}`.", "",
             "## All-model grid (key columns)", "",
             "| floor | SE target | median len (scen) | %reach | %cap | recovery r | median SE_total |",
             "|---|---|---|---|---|---|---|"]
    for _, r in grid.sort_values(["floor", "se_target"]).iterrows():
        lines.append(f"| {int(r['floor'])} | {r['se_target']:g} | {r['median_len_scenarios']:g} | "
                     f"{r['pct_reach']:g}% | {r['pct_cap']:g}% | {r['recovery_r']:.3f} | "
                     f"{r['median_se_total']:.3f} |")
    if grid_excl is not None and len(grid_excl):
        lines += ["", "## Excl-weak grid (key columns)", "",
                  "| floor | SE target | median len (scen) | %reach | %cap | recovery r | median SE_total |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in grid_excl.sort_values(["floor", "se_target"]).iterrows():
            lines.append(f"| {int(r['floor'])} | {r['se_target']:g} | {r['median_len_scenarios']:g} | "
                         f"{r['pct_reach']:g}% | {r['pct_cap']:g}% | {r['recovery_r']:.3f} | "
                         f"{r['median_se_total']:.3f} |")
    lines += ["", "> Operating point is left OPEN: trade test length against %reach / recovery r / "
              "SE_total. Lower SE targets + higher floors buy precision at the cost of length and "
              "(for the weak tail) more caps."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
