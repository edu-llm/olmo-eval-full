"""Bridge parameter-uncertainty / total-SE (experiment 07) + fixed leaderboard (08).

At N=51 the item-parameter (calibration) error dominates the ability (measurement)
error, so ability-only SE badly understates uncertainty (and the coarse EAP grid even
underflows it to 0 for extreme models). This computes:

  SE_ability  = 1/sqrt(sum a^2 P Q)            (Fisher measurement SE at theta-hat)
  SE_param    = delete-1 jackknife over models (refit item params leaving each model
                out, rescore every model; SD of theta across the n leave-one-out fits)
  SE_total    = sqrt(SE_ability^2 + SE_param^2)

Writes leaderboard_se_components.csv, a rebuilt model_leaderboard.csv with se_total, and
the leaderboard figure with honest total-SE error bars.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import calibrate_partial as cp   # noqa: E402
import calibrate_mirt as cm      # noqa: E402


def _load_map(path, key, val):
    out = {}
    for rec in cp.read_jsonl(path):
        k = rec.get(key)
        if k is not None:
            v = rec.get(val); out[str(k)] = str(v) if v is not None else str(k)
    return out


def dedupe_by_source(cols, c2s, s2src):
    grp = {}
    for c in cols:
        grp.setdefault(s2src.get(c2s.get(c, c), c2s.get(c, c)), set()).add(c2s.get(c, c))
    keep = {sorted(v)[0] for v in grp.values()}
    return [c for c in cols if c2s.get(c, c) in keep]


def eap(Y, M, a, b, grid, logw):
    eta = a[:, None] * grid[None, :] - b[:, None]
    LL = np.where(M, Y, 0.0) @ log_expit(eta) + np.where(M, 1.0 - Y, 0.0) @ log_expit(-eta)
    post = np.exp((LL + logw[None, :]) - logsumexp(LL + logw[None, :], axis=1, keepdims=True))
    return post @ grid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--rubrics", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--leaderboard-out", type=Path, default=None,
                   help="also rewrite this model_leaderboard.csv with se_total.")
    p.add_argument("--grid", type=int, default=41)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--jack-max-iter", type=int, default=80)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    mat = cp.load_matrix(args.matrix)
    c2s = _load_map(args.rubrics, "criterion_id", "scenario_id")
    s2src = _load_map(args.scenarios, "scenario_id", "source_id")
    mat = mat[dedupe_by_source(list(mat.columns), c2s, s2src)]
    sub, _ = cp.select_sparse(mat)
    kept, _af, _ap = cp.split_zero_variance(sub)
    sub = sub[kept]; sub = sub[sub.notna().any(axis=1)]
    models = list(sub.index)
    Y = np.nan_to_num(sub.to_numpy(float), nan=0.0); M = sub.notna().to_numpy()
    n, J = Y.shape
    egrid = cm.build_grid(1, args.eap_grid)[:, 0]
    elogw = cm.base_log_weights(1, args.eap_grid); elogw -= logsumexp(elogw)
    print(f"param-uncertainty: {J} items x {n} models; delete-1 jackknife ({n} refits)")

    # full fit -> theta, SE_ability (Fisher)
    fit = cm.fit_m2pl_em(Y, M, np.ones((J, 1), int), args.grid, ridge=args.ridge, max_iter=150, tol=1e-4)
    a, b = fit["A"][:, 0], fit["b"]
    theta = eap(Y, M, a, b, egrid, elogw)
    P = expit(a[None, :] * theta[:, None] - b[None, :])
    se_ability = 1.0 / np.sqrt(((a[None, :] ** 2 * P * (1.0 - P)) * M).sum(axis=1) + 1e-9)

    # delete-1 jackknife over models -> SE_param
    theta_jack = np.zeros((n, n))
    for j in range(n):
        tr = np.array([i for i in range(n) if i != j])
        ft = cm.fit_m2pl_em(Y[tr], M[tr], np.ones((J, 1), int), args.grid,
                            ridge=args.ridge, max_iter=args.jack_max_iter, tol=1e-4)
        theta_jack[:, j] = eap(Y, M, ft["A"][:, 0], ft["b"], egrid, elogw)
        if (j + 1) % 10 == 0:
            print(f"  jackknife {j+1}/{n}", flush=True)
    jbar = theta_jack.mean(axis=1)
    se_param = np.sqrt((n - 1) / n * ((theta_jack - jbar[:, None]) ** 2).sum(axis=1))
    se_total = np.sqrt(se_ability ** 2 + se_param ** 2)
    obs = np.array([np.nanmean(np.where(M[i], Y[i], np.nan)) for i in range(n)])

    rows = sorted(zip(models, theta, se_ability, se_param, se_total, obs),
                  key=lambda t: -t[1])
    with (args.out_dir / "leaderboard_se_components.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rank", "model", "theta", "se_ability", "se_param", "se_total", "obs_pass_rate"])
        for i, (m, th, sa, sp, stot, ob) in enumerate(rows, 1):
            w.writerow([i, m, f"{th:.4f}", f"{sa:.4f}", f"{sp:.4f}", f"{stot:.4f}", f"{ob:.4f}"])

    (args.out_dir / "metrics.json").write_text(json.dumps({
        "n_models": n, "n_items": J,
        "mean_se_ability": float(se_ability.mean()), "mean_se_param": float(se_param.mean()),
        "mean_se_total": float(se_total.mean()),
        "note": "SE_param (calibration) dominates SE_ability at this N; total-SE bars are the honest ones."
    }, indent=2))

    if args.leaderboard_out:
        with args.leaderboard_out.open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["rank", "model", "theta", "theta_se", "se_ability", "se_param", "observed_pass_rate"])
            for i, (m, th, sa, sp, stot, ob) in enumerate(rows, 1):
                w.writerow([i, m, f"{th:.4f}", f"{stot:.4f}", f"{sa:.4f}", f"{sp:.4f}", f"{ob:.4f}"])
        print(f"rewrote leaderboard with total SE -> {args.leaderboard_out}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # SE components
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(se_ability, bins=20, alpha=0.6, label=f"SE_ability (mean {se_ability.mean():.3f})")
    ax.hist(se_param, bins=20, alpha=0.6, label=f"SE_param (mean {se_param.mean():.3f})")
    ax.set_xlabel("theta SE"); ax.set_ylabel("models"); ax.legend()
    ax.set_title(f"Bridge SE components (N={n}); calibration error dominates")
    fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "se_components.png", dpi=140); plt.close(fig)
    # leaderboard with total-SE bars (08)
    top = rows[:25]
    fig, ax = plt.subplots(figsize=(7, 8))
    yy = np.arange(len(top))[::-1]
    ax.barh(yy, [t[1] for t in top], xerr=[t[4] for t in top], color="#4C86C6",
            error_kw={"elinewidth": 1, "capsize": 2})
    ax.set_yticks(yy); ax.set_yticklabels([t[0] for t in top], fontsize=6)
    ax.set_xlabel("ability theta (bars = total SE)")
    ax.set_title(f"Bridge leaderboard (top 25 of {n}, total-SE bars)")
    fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "leaderboard_general.png", dpi=140); plt.close(fig)

    print(f"mean SE_ability={se_ability.mean():.3f}  SE_param={se_param.mean():.3f}  "
          f"SE_total={se_total.mean():.3f}")
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
