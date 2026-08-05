"""Bridge calibration — 'main info' package.

Produces the essentials of a calibration study (not the full InfoBench sprawl):
  * item parameters a (discrimination) + b (difficulty)  [unidimensional 2PL]
  * per-model ability theta leaderboard (EAP + posterior SE)
  * honest held-out k-fold recovery (r / slope / MAE): each model is scored with
    item params fit on the OTHER models
  * calibrated rubric copy, coverage, a human summary, and a manifest
Then zips the folder like `Bridge_calibration_results_<date>.zip`.

Uses the stable pure-numpy EM from calibrate_mirt (Q = 1 -> unidimensional). Applies
Bridge's source_id grouping (collapse duplicate-source scenarios). Preliminary: N~51.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import calibrate_partial as cp   # noqa: E402
import calibrate_mirt as cm      # noqa: E402


def _load_map(path: Path, key: str, val: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in cp.read_jsonl(path):
        k = rec.get(key)
        if k is None:
            continue
        v = rec.get(val)
        out[str(k)] = str(v) if v is not None else str(k)
    return out


def dedupe_by_source(columns, crit_to_scen, scen_to_source):
    src_to_scens: dict[str, set] = {}
    for c in columns:
        sid = crit_to_scen.get(c, c)
        src_to_scens.setdefault(scen_to_source.get(sid, sid), set()).add(sid)
    keep = {sorted(v)[0] for v in src_to_scens.values()}
    kept = [c for c in columns if crit_to_scen.get(c, c) in keep]
    return kept, len(columns) - len(kept), len(src_to_scens)


def eap_theta(Y, M, a, b, nodes=61):
    """EAP ability + posterior SE per person under the fitted unidimensional 2PL."""
    grid = cm.build_grid(1, nodes)[:, 0]
    logw = cm.base_log_weights(1, nodes)
    logw = logw - logsumexp(logw)
    eta = a[:, None] * grid[None, :] - b[:, None]      # (items, nodes)
    logP, log1mP = log_expit(eta), log_expit(-eta)
    YM = np.where(M, Y, 0.0)
    NM = np.where(M, 1.0 - Y, 0.0)
    LL = YM @ logP + NM @ log1mP                        # (persons, nodes)
    joint = LL + logw[None, :]
    post = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    theta = post @ grid
    var = post @ (grid ** 2) - theta ** 2
    return theta, np.sqrt(np.clip(var, 0.0, None))


def fit_items(Y, M, grid, ridge, max_iter, tol):
    Q = np.ones((Y.shape[1], 1), dtype=int)
    fit = cm.fit_m2pl_em(Y, M, Q, grid, estimate_corr=False,
                         ridge=ridge, max_iter=max_iter, tol=tol)
    return fit["A"][:, 0], fit["b"], fit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--rubrics", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="package directory (a .zip is written alongside it).")
    p.add_argument("--benchmark", default="Bridge")
    p.add_argument("--group-source-id", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--k", type=int, default=5, help="recovery folds over models (default 5).")
    p.add_argument("--grid", type=int, default=41)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    mat = cp.load_matrix(args.matrix)
    n_cols_raw = mat.shape[1]
    grouping = {"applied": False}
    if args.group_source_id:
        c2s = _load_map(args.rubrics, "criterion_id", "scenario_id")
        s2src = _load_map(args.scenarios, "scenario_id", "source_id")
        kept, dropped, n_src = dedupe_by_source(list(mat.columns), c2s, s2src)
        mat = mat[kept]
        grouping = {"applied": True, "kept_columns": len(kept),
                    "dropped_duplicate_source": dropped, "n_source_groups": n_src}
        print(f"source_id grouping: kept {len(kept)}/{n_cols_raw} cols "
              f"({dropped} dropped; {n_src} source groups)")

    sub, _ = cp.select_sparse(mat)
    kept, all_fail, all_pass = cp.split_zero_variance(sub)
    sub = sub[kept]
    sub = sub[sub.notna().any(axis=1)]
    items = list(sub.columns)
    models = list(sub.index)
    Y = np.nan_to_num(sub.to_numpy(dtype=float), nan=0.0)
    M = sub.notna().to_numpy()
    n_items, n_persons = len(items), len(models)
    print(f"fitting {n_items} items x {n_persons} models "
          f"(dropped {len(all_fail)} all-fail + {len(all_pass)} all-pass)")

    # --- full fit: item params + abilities -----------------------------------
    a, b, fit = fit_items(Y, M, args.grid, args.ridge, args.max_iter, args.tol)
    nper = cm.per_item_n_persons(M)
    flags = ["extreme_a" if (not np.isfinite(x) or abs(x) > cm.EXTREME_A) else "" for x in a]
    theta, _ = eap_theta(Y, M, a, b)
    # Ability SE from full-bank Fisher information at theta-hat. The coarse EAP grid
    # underflows to ~0 for extreme models given ~2.7k items, so use the analytic SE
    # 1/sqrt(sum a^2 P Q). NOTE: this is measurement/ability SE only; the *total* SE
    # (adding item-parameter uncertainty, which dominates at N=51) is experiment
    # 07_parameter_uncertainty, not this bar.
    _P = expit(a[None, :] * theta[:, None] - b[None, :])
    theta_se = 1.0 / np.sqrt(((a[None, :] ** 2 * _P * (1.0 - _P)) * M).sum(axis=1) + 1e-9)
    obs = np.array([np.nanmean(np.where(M[i], Y[i], np.nan)) for i in range(n_persons)])

    # --- held-out k-fold recovery (models excluded from their own scoring) ----
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_persons)
    folds = np.array_split(order, args.k)
    est = np.full(n_persons, np.nan)
    for f in folds:
        te = np.zeros(n_persons, bool); te[f] = True; tr = ~te
        a_tr, b_tr, _ = fit_items(Y[tr], M[tr], args.grid, args.ridge, args.max_iter, args.tol)
        th, _se = eap_theta(Y[te], M[te], a_tr, b_tr)
        est[f] = th
    ok = np.isfinite(est) & np.isfinite(theta)
    r = float(np.corrcoef(est[ok], theta[ok])[0, 1])
    slope = float(np.polyfit(theta[ok], est[ok], 1)[0])
    mae = float(np.mean(np.abs(est[ok] - theta[ok])))
    print(f"held-out recovery ({args.k}-fold): r={r:.3f} slope={slope:.3f} mae={mae:.3f}")

    # --- write package files --------------------------------------------------
    with (out / "item_params.csv").open("w", encoding="utf-8") as fh:
        fh.write("criterion_id,a,b,n_persons,flags\n")
        for cid, av, bv, nv, fl in zip(items, a, b, nper, flags):
            fh.write(f"{cid},{av:.6f},{bv:.6f},{int(nv)},{fl}\n")

    lb = sorted(zip(models, theta, theta_se, obs), key=lambda t: -t[1])
    with (out / "model_leaderboard.csv").open("w", encoding="utf-8") as fh:
        fh.write("rank,model,theta,theta_se,observed_pass_rate\n")
        for i, (m, th, se, ob) in enumerate(lb, 1):
            fh.write(f"{i},{m},{th:.4f},{se:.4f},{ob:.4f}\n")

    (out / "recovery.json").write_text(json.dumps(
        {"method": "held-out k-fold over models (EAP)", "k": args.k,
         "recovery_r": r, "slope": slope, "mae": mae, "n_models": n_persons},
        indent=2), encoding="utf-8")

    (out / "coverage.json").write_text(json.dumps(
        {"n_models": n_persons, "n_criteria_fit": n_items,
         "dropped_all_fail": len(all_fail), "dropped_all_pass": len(all_pass),
         "source_id_grouping": grouping}, indent=2), encoding="utf-8")

    # calibrated rubric copy (discrimination broadcast onto the criterion's q skills)
    a_by = {c: float(x) for c, x in zip(items, a)}
    b_by = {c: float(x) for c, x in zip(items, b)}
    recs = list(cp.read_jsonl(args.rubrics))
    n_up = 0
    for rec in recs:
        cid = rec.get("criterion_id")
        if cid in a_by and np.isfinite(a_by[cid]) and np.isfinite(b_by[cid]):
            qm = rec.get("q_mapping") or {}
            rec["discrimination"] = ({s: round(a_by[cid], 4) if int(qm.get(s, 0)) == 1 else 0.0
                                      for s in qm} if qm else round(a_by[cid], 4))
            rec["difficulty"] = round(b_by[cid], 4)
            rec["irt_params"] = {"source": "calibrated-2pl-unidim-preliminary"}
            n_up += 1
    cp.write_jsonl(out / "rubrics_calibrated.jsonl", recs)

    manifest = {
        "benchmark": args.benchmark, "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "unidimensional-2pl-mml-em (calibrate_mirt.fit_m2pl_em, Q=1)",
        "preliminary": True, "inputs": {"matrix": str(args.matrix),
        "rubrics": str(args.rubrics), "scenarios": str(args.scenarios)},
        "fit": {"grid": args.grid, "ridge": args.ridge, "loglik": fit["loglik"],
                "converged": fit["converged"], "n_iter": fit["n_iter"]},
        "n_models": n_persons, "n_criteria_fit": n_items,
        "n_criteria_updated_in_bank": n_up, "source_id_grouping": grouping,
        "recovery": {"k": args.k, "r": r, "slope": slope, "mae": mae},
    }
    (out / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    au = a[np.isfinite(a)]
    top = lb[:5]; bot = lb[-5:]
    summary = f"""# {args.benchmark} calibration study summary (preliminary)

**Model:** unidimensional 2PL (single tutoring-ability axis), fit with the pure-numpy
Bock-Aitkin EM. Preliminary: {n_persons} models (below the ~150 needed for a stable
5-skill MIRT). Bridge `source_id` grouping applied (duplicate-source scenarios collapsed).

## Item parameters
- Criteria calibrated: **{n_items}** (dropped {len(all_fail)} all-fail + {len(all_pass)} all-pass constants).
- Discrimination `a`: median **{np.median(au):.2f}**, mean {np.mean(au):.2f}, range [{np.min(au):.2f}, {np.max(au):.2f}].
- {sum(1 for x in au if x >= 0.3)} of {n_items} criteria discriminate usefully (a >= 0.3); {sum(1 for f in flags if f)} flagged `extreme_a`.
- Full per-criterion table: `item_params.csv`. Calibrated bank: `rubrics_calibrated.jsonl`.

## Model leaderboard (ability theta)
- theta range [{theta.min():.2f}, {theta.max():.2f}]. Full table: `model_leaderboard.csv`.
- Top: {", ".join(f"{m} ({th:.2f})" for m, th, _, _ in top)}
- Bottom: {", ".join(f"{m} ({th:.2f})" for m, th, _, _ in bot)}

## Held-out recovery ({args.k}-fold, models excluded from their own scoring)
- **r = {r:.3f}**, slope = {slope:.3f}, MAE = {mae:.3f} (theta recovered on unseen models).
- This is the validity check: how well the calibrated items recover a model's ability
  when that model was NOT used to fit the items.

## Caveats
- Preliminary (N={n_persons}, unidimensional). The committed 5-skill numbers come from the
  200-model run. Exclude the `extreme_a` items and A3 (safety tripwire) from CAT use.
"""
    (out / "CALIBRATION_STUDY_SUMMARY.md").write_text(summary, encoding="utf-8")

    stamp = datetime.now().strftime("%Y%m%d")
    zip_base = out.parent / f"{args.benchmark}_calibration_results_{stamp}"
    shutil.make_archive(str(zip_base), "zip", root_dir=out.parent, base_dir=out.name)
    print(f"\nwrote package -> {out}/")
    print(f"wrote zip     -> {zip_base}.zip")
    print("main files: CALIBRATION_STUDY_SUMMARY.md, item_params.csv, model_leaderboard.csv, "
          "recovery.json, rubrics_calibrated.jsonl, study_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
