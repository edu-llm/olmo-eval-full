"""Bridge CAT-vs-random efficiency (experiment 04).

Held-out k-fold. For each test model, estimate ability from (a) adaptive Fisher-max item
selection and (b) random item order, as a function of test length L. Recovery r against
the full-bank held-out EAP reference at each L shows how many criteria adaptive selection
saves to reach a target accuracy -- the core "adaptive saves items" claim.
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


def eap_one(y, a, b, egrid, elogw):
    eta = a[:, None] * egrid[None, :] - b[:, None]
    ll = y @ log_expit(eta) + (1.0 - y) @ log_expit(-eta)
    w = ll + elogw
    return float(np.exp(w - logsumexp(w)) @ egrid)


def eap_all(Y, M, a, b, egrid, elogw):
    eta = a[:, None] * egrid[None, :] - b[:, None]
    LL = np.where(M, Y, 0.0) @ log_expit(eta) + np.where(M, 1.0 - Y, 0.0) @ log_expit(-eta)
    post = np.exp((LL + elogw[None, :]) - logsumexp(LL + elogw[None, :], axis=1, keepdims=True))
    return post @ egrid


def adaptive_theta(y, a, b, obs_idx, L, egrid, elogw):
    theta = 0.0; chosen = []
    pool = list(obs_idx)
    for _ in range(min(L, len(pool))):
        Pj = expit(a[pool] * theta - b[pool])
        info = a[pool] ** 2 * Pj * (1.0 - Pj)
        k = pool[int(np.argmax(info))]
        chosen.append(k); pool.remove(k)
        theta = eap_one(y[chosen], a[chosen], b[chosen], egrid, elogw)
    return theta, chosen


def random_theta(y, a, b, perm_obs, L, egrid, elogw):
    chosen = perm_obs[:L]
    return eap_one(y[chosen], a[chosen], b[chosen], egrid, elogw), chosen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--rubrics", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--grid", type=int, default=41)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--se-target", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    mat = cp.load_matrix(args.matrix)
    c2s = _load_map(args.rubrics, "criterion_id", "scenario_id")
    s2src = _load_map(args.scenarios, "scenario_id", "source_id")
    mat = mat[dedupe_by_source(list(mat.columns), c2s, s2src)]
    sub, _ = cp.select_sparse(mat)
    kept, _af, _ap = cp.split_zero_variance(sub)
    sub = sub[kept]; sub = sub[sub.notna().any(axis=1)]
    Y = np.nan_to_num(sub.to_numpy(float), nan=0.0); M = sub.notna().to_numpy()
    n, J = Y.shape
    egrid = cm.build_grid(1, args.eap_grid)[:, 0]
    elogw = cm.base_log_weights(1, args.eap_grid); elogw -= logsumexp(elogw)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n); folds = [order[i::args.k] for i in range(args.k)]
    lengths = [4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 80, 100]
    lengths = [L for L in lengths if L <= J]
    ada = {L: [] for L in lengths}; rnd = {L: [] for L in lengths}
    ref_all, ada_full = [], {L: [] for L in lengths}

    print(f"cat-vs-random: {J} items x {n} models; {args.k}-fold")
    for fi, te in enumerate(folds):
        tr = np.array([i for i in range(n) if i not in set(te.tolist())])
        ft = cm.fit_m2pl_em(Y[tr], M[tr], np.ones((J, 1), int), args.grid, ridge=args.ridge,
                            max_iter=150, tol=1e-4)
        a, b = ft["A"][:, 0], ft["b"]
        ref = eap_all(Y[te], M[te], a, b, egrid, elogw)     # held-out full-bank reference
        for r_i, mi in enumerate(te):
            obs = np.where(M[mi])[0]
            perm = rng.permutation(obs).tolist()
            for L in lengths:
                ta, _ = adaptive_theta(Y[mi], a, b, obs, L, egrid, elogw)
                tr_, _ = random_theta(Y[mi], a, b, perm, min(L, len(perm)), egrid, elogw)
                ada[L].append((ref[r_i], ta)); rnd[L].append((ref[r_i], tr_))
        print(f"  fold {fi+1}/{args.k} done", flush=True)

    def rr(pairs):
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")

    rows = [{"n_items": L, "r_adaptive": round(rr(ada[L]), 4), "r_random": round(rr(rnd[L]), 4)}
            for L in lengths]
    with (args.out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n_items", "r_adaptive", "r_random"]); w.writeheader(); w.writerows(rows)

    # items each arm needs to reach r >= 0.9
    def items_to(target, key):
        for r in rows:
            if r[key] >= target:
                return r["n_items"]
        return None
    summary = {"r_target": 0.9, "adaptive_items_to_r": items_to(0.9, "r_adaptive"),
               "random_items_to_r": items_to(0.9, "r_random"),
               "full_bank_items": J, "se_target": args.se_target}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot([r["n_items"] for r in rows], [r["r_adaptive"] for r in rows], "o-", label="adaptive (Fisher-max)")
    ax.plot([r["n_items"] for r in rows], [r["r_random"] for r in rows], "s--", label="random order")
    ax.axhline(0.9, color="gray", ls=":", lw=1)
    ax.set_xlabel("criteria administered"); ax.set_ylabel("recovery r vs full-bank theta")
    ax.set_title(f"Bridge CAT vs random (N={n}, {args.k}-fold)")
    ax.legend(); fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "efficiency_adaptive_vs_random.png", dpi=140)
    print(f"adaptive reaches r>=0.9 at {summary['adaptive_items_to_r']} items; "
          f"random at {summary['random_items_to_r']}. wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
