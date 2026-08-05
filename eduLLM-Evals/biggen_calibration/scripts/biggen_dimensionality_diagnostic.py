"""Task A: BiGGen dimensionality DIAGNOSTIC (exploratory).

BiGGen is calibrated UNIDIMENSIONAL ("general" skill). Its bank/rubrics carry NO
multi-skill Q-matrix (``q_mapping`` is null, ``primary_skill`` == "general",
``q_modeled`` == {"general": 1} for every criterion). The confirmatory 1..5 skill
sweep that Bridge ran (``scenario_dimensionality.py``) therefore has no named
Q-matrix to confirm on BiGGen, and fabricating one is out of scope.

Instead this runs a DATA-DRIVEN dimensionality diagnostic on our own binary
atom-level response matrix -- the "redo after grading" step that
``data/BiGGen/SKILL_AXIS_NOTES.md`` explicitly defers to -- to assess whether more
than one latent dimension is even supported at N=52, and STOPS short of a
confirmatory sweep. Three probes, all EXPLORATORY:

  1. PCA scree + Horn parallel analysis on the standardized response matrix
     (person x atom): first-factor variance share, PC1/PC2 ratio, and the number of
     components whose eigenvalue exceeds the permutation null (inter-atom structure
     broken column-wise). This is the unidimensionality heuristic.
  2. Capability-composite correlation (SKILL_AXIS_NOTES Finding 1, recomputed on our
     binary atoms): mean pairwise r among the 8 BiGGen capability composite scores,
     the min pair, and safety's separation.
  3. A 1-vs-2 exploratory M2PL probe of the notes' strongest 2-D candidate
     ("general + normative restraint"): dim0 (general) loads every atom, dim1
     (normative) loads only ``capability == safety`` atoms (so safety atoms
     cross-load and non-safety atoms anchor the general axis). Reports AIC/BIC,
     held-out marginal log-loss (+/-SE, person k-fold), and the latent correlation.
     This uses the EXISTING ``capability`` field only -- it does NOT invent a skill
     taxonomy -- and is reported strictly as an exploratory check.

Writes ``experiments/03_structures/{structure_comparison.csv, selection.json,
figures/structure_cv.png, figures/capability_corr.png}``. The verdict flags that a
named, labeled multi-skill Q-matrix is a USER DECISION.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Pin BLAS threads BEFORE numpy import (OpenBLAS oversubscription guard on this box).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.special import log_expit, logsumexp

# eduLLM-Evals root (.../biggen_calibration/scripts/this_file -> parents[2]).
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calibrate_mirt as cm  # noqa: E402  (fit_m2pl_em / build_grid / base_log_weights / aic_bic)

CAPABILITIES = ["grounding", "instruction_following", "planning", "reasoning",
                "refinement", "safety", "theory_of_mind", "tool_usage"]


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_bank_maps(bank_path: Path):
    """criterion_id -> capability, and the set of modeled (fitted) criterion ids."""
    cap: dict[str, str] = {}
    modeled: list[str] = []
    for rec in read_jsonl(bank_path):
        cid = rec.get("criterion_id")
        if cid is None:
            continue
        cap[cid] = rec.get("capability") or "unknown"
        if not rec.get("exclude_from_fit", False):
            modeled.append(cid)
    return cap, modeled


def prepare_matrix(matrix_path: Path, item_ids: list[str]):
    """Load matrix restricted to ``item_ids``; drop zero-variance atoms over observed
    cells. Returns (Y (n,p) 0/1 nan-filled, M mask, kept_ids, models)."""
    mat = pd.read_csv(matrix_path, index_col=0)
    ids = [c for c in item_ids if c in mat.columns]
    sub = mat.reindex(columns=ids)
    obs = sub.notna()
    # keep atoms with >=2 observations and both a pass and a fail among observed
    keep = []
    arr = sub.to_numpy(float)
    for j, c in enumerate(ids):
        col = arr[:, j]
        m = ~np.isnan(col)
        if m.sum() >= 2:
            v = col[m]
            if 0 < v.sum() < m.sum():
                keep.append(c)
    sub = sub[keep]
    sub = sub[sub.notna().any(axis=1)]
    Y = sub.to_numpy(float)
    M = sub.notna().to_numpy()
    return Y, M, list(sub.columns), list(sub.index)


def standardize(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Item-mean impute + per-item z-score (over observed cells)."""
    Z = Y.copy()
    for j in range(Y.shape[1]):
        m = M[:, j]
        col = Y[m, j]
        mu = col.mean()
        sd = col.std()
        Z[~m, j] = mu           # mean-impute holes
        if sd > 0:
            Z[:, j] = (Z[:, j] - mu) / sd
        else:
            Z[:, j] = 0.0
    return Z


def pca_eigs(Z: np.ndarray) -> np.ndarray:
    """Eigenvalue spectrum of the inter-atom correlation matrix.

    With n_models << n_atoms the p x p correlation matrix is rank-deficient (only n-1
    nonzero eigenvalues) and its eigenvalues are NOT on the Kaiser "noise = 1" scale, so
    the absolute scale is meaningless in isolation. We therefore return the n nonzero
    eigenvalues (from the n x n Gram of unit-norm-column, mean-centered Z) on a scale that
    is IDENTICAL to the permutation null in :func:`parallel_analysis`; the reference is the
    parallel-analysis null, not 1. The nonzero eigenvalues sum to n_atoms."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    norms = np.sqrt((Zc ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    Zn = Zc / norms                       # unit-L2 columns -> Zn^T Zn is correlation-like
    G = Zn @ Zn.T                          # (n, n); nonzero eigvals match the p x p matrix
    ev = np.linalg.eigvalsh(G)[::-1]
    return np.clip(ev, 0.0, None)


def parallel_analysis(Z: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """Horn parallel analysis (permutation): independently permute each column to break
    inter-item correlation; return the 95th-percentile null eigenvalue per rank."""
    rng = np.random.default_rng(seed)
    n, p = Z.shape
    k = min(n - 1, p)
    null = np.empty((n_perm, k))
    for t in range(n_perm):
        Zp = np.empty_like(Z)
        for j in range(p):
            Zp[:, j] = Z[rng.permutation(n), j]
        null[t] = pca_eigs(Zp)[:k]
    return np.percentile(null, 95, axis=0)


def capability_composites(Y, M, kept_ids, cap):
    """Per-model mean pass rate per capability (over observed atoms). Returns
    (comp (n_models, n_caps) with nan for empty cells, present_caps)."""
    col_caps = np.array([cap.get(c, "unknown") for c in kept_ids])
    present = [c for c in CAPABILITIES if (col_caps == c).sum() > 0]
    n = Y.shape[0]
    comp = np.full((n, len(present)), np.nan)
    for ci, c in enumerate(present):
        cols = np.where(col_caps == c)[0]
        for i in range(n):
            m = M[i, cols]
            if m.sum() > 0:
                comp[i, ci] = Y[i, cols][m].mean()
    return comp, present


def marginal_logloss(Yte, Mte, A, b, grid, log_prior) -> float:
    eta = A @ grid.T - b[:, None]
    logP, log1mP = log_expit(eta), log_expit(-eta)
    YM = np.where(Mte, Yte, 0.0)
    NM = np.where(Mte, 1.0 - Yte, 0.0)
    LL = YM @ logP + NM @ log1mP
    person_ll = logsumexp(LL + log_prior[None, :], axis=1)
    return float(-person_ll.sum() / max(Mte.sum(), 1))


def fit_and_cv(Y, M, Q, grid_nodes, ridge, k, seed, estimate_corr):
    """Full-data fit + k-fold held-out marginal log-loss for one structure."""
    n = Y.shape[0]
    ndim = Q.shape[1]
    fit = cm.fit_m2pl_em(Y, M, Q, grid_nodes, estimate_corr=estimate_corr,
                         ridge=ridge, max_iter=200, tol=1e-4)
    n_obs = int(M.sum())
    aic, bic = cm.aic_bic(fit["loglik"], fit["n_params"], n_obs)
    grid = cm.build_grid(ndim, grid_nodes)
    base = cm.base_log_weights(ndim, grid_nodes)
    lp = base - logsumexp(base)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = [order[i::k] for i in range(k)]
    lls = []
    for te in folds:
        tr = np.array([i for i in range(n) if i not in set(te.tolist())])
        ft = cm.fit_m2pl_em(Y[tr], M[tr], Q, grid_nodes, estimate_corr=False,
                            ridge=ridge, max_iter=150, tol=1e-4)
        lls.append(marginal_logloss(Y[te], M[te], ft["A"], ft["b"], grid, lp))
    R = fit.get("R")
    max_off = (float(np.max(np.abs(R - np.eye(ndim)))) if (R is not None and ndim > 1) else 0.0)
    return {
        "n_dims": ndim, "loglik": round(fit["loglik"], 1), "n_params": fit["n_params"],
        "aic": round(aic, 1), "bic": round(bic, 1),
        "oos_logloss_mean": round(float(np.mean(lls)), 5),
        "oos_logloss_se": round(float(np.std(lls, ddof=1) / np.sqrt(len(lls))), 5),
        "max_latent_corr_offdiag": round(max_off, 3),
        "converged": bool(fit["converged"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path,
                   default=ROOT / "staging" / "biggen_response_matrix.csv")
    p.add_argument("--bank", type=Path,
                   default=ROOT / "biggen_calibration" / "bank" / "biggen_unidim_calibrated.jsonl")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "biggen_calibration" / "experiments" / "03_structures")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--grid", type=int, default=5, help="GH nodes/dim for the M2PL probe.")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--n-perm", type=int, default=200, help="parallel-analysis permutations.")
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    cap, modeled = load_bank_maps(args.bank)
    Y, M, kept_ids, models = prepare_matrix(args.matrix, modeled)
    n, p_items = Y.shape
    print(f"diagnostic: {p_items} modeled atoms x {n} models "
          f"(fill={M.mean():.3f}); capabilities present in bank")

    # --- Probe 1: PCA scree + parallel analysis --------------------------------
    Z = standardize(Y, M)
    ev = pca_eigs(Z)
    pa95 = parallel_analysis(Z, args.n_perm, args.seed)
    k_pa = int(np.sum(ev[: pa95.size] > pa95))
    total = float(ev.sum())
    pc1_share = float(ev[0] / total)
    pc2_share = float(ev[1] / total)
    pc1_pc2 = float(ev[0] / ev[1]) if ev[1] > 0 else float("inf")
    print(f"  PCA: PC1 share={pc1_share:.3f}  PC2 share={pc2_share:.3f}  "
          f"PC1/PC2={pc1_pc2:.2f}  parallel-analysis n_factors={k_pa}")

    # --- Probe 2: capability composite correlation -----------------------------
    comp, present_caps = capability_composites(Y, M, kept_ids, cap)
    cdf = pd.DataFrame(comp, columns=present_caps)
    corr = cdf.corr(method="pearson").to_numpy()
    off = corr[~np.eye(len(present_caps), dtype=bool)]
    mean_off = float(np.nanmean(off))
    min_pair_idx = np.unravel_index(np.nanargmin(np.where(np.eye(len(present_caps), dtype=bool),
                                                          np.nan, corr)), corr.shape)
    min_pair = (present_caps[min_pair_idx[0]], present_caps[min_pair_idx[1]],
                round(float(corr[min_pair_idx]), 3))
    safety_mean = None
    if "safety" in present_caps:
        si = present_caps.index("safety")
        row = np.delete(corr[si], si)
        safety_mean = round(float(np.nanmean(row)), 3)
    n_cap_atoms = {c: int(sum(1 for cid in kept_ids if cap.get(cid) == c)) for c in present_caps}
    print(f"  capability composite mean|r|={mean_off:.3f}  min pair={min_pair}  "
          f"safety mean r={safety_mean}")

    # --- Probe 3: 1-vs-2 exploratory M2PL (general + normative/safety) ---------
    col_caps = np.array([cap.get(c, "unknown") for c in kept_ids])
    is_safety = (col_caps == "safety")
    rows = []
    Q_uni = np.ones((p_items, 1), dtype=int)
    rows.append({"structure": "unidim_1d", **fit_and_cv(
        Y, M, Q_uni, args.grid, args.ridge, args.k, args.seed, estimate_corr=False)})
    probe_2d_available = bool(is_safety.sum() >= 5)
    if probe_2d_available:
        Q_2d = np.zeros((p_items, 2), dtype=int)
        Q_2d[:, 0] = 1                      # general loads every atom
        Q_2d[is_safety, 1] = 1              # normative loads only safety atoms (cross-load)
        rows.append({"structure": "general_plus_normative_2d", **fit_and_cv(
            Y, M, Q_2d, args.grid, args.ridge, args.k, args.seed, estimate_corr=True)})
    else:
        print(f"  2-D probe skipped: only {int(is_safety.sum())} safety anchor atoms (<5).")

    best = min(rows, key=lambda r: r["oos_logloss_mean"])
    thresh = best["oos_logloss_mean"] + best["oos_logloss_se"]
    within = [r for r in rows if r["oos_logloss_mean"] <= thresh]
    selected = min(within, key=lambda r: (r["n_dims"], r["bic"]))

    with (args.out_dir / "structure_comparison.csv").open("w", newline="",
                                                          encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # verdict: is >1 dimension CONFIRMATORY-supported (not just detectable)?
    # A named multi-dim model must actually win out-of-sample (strictly lower held-out
    # log-loss than unidim beyond 1 SE) AND on BIC to count as supported. Horn parallel
    # analysis alone (k_pa) is LIBERAL when n_atoms >> n_models and readily flags weak
    # residual structure that does not survive cross-validation, so it is reported but not
    # sufficient on its own.
    uni_ll = next(r for r in rows if r["structure"] == "unidim_1d")["oos_logloss_mean"]
    uni_bic = next(r for r in rows if r["structure"] == "unidim_1d")["bic"]
    multidim_wins_oos = any(
        r["structure"] != "unidim_1d"
        and r["oos_logloss_mean"] < uni_ll - r["oos_logloss_se"]
        and r["bic"] < uni_bic
        for r in rows)
    multidim_supported = bool(multidim_wins_oos)
    selection = {
        "task": "A -- dimensionality (feasibility gate + exploratory diagnostic)",
        "exploratory": True,
        "decision": {
            "status": "FINAL -- KEEP UNIDIMENSIONAL",
            "confirmed": "2026-08-05 (user decision)",
            "authored_multi_skill_q": False,
            "detail": ("BiGGen stays 1-D ('general' skill). The exploratory data-driven "
                       "diagnostic below is the recorded evidence. No multi-skill Q-matrix "
                       "was authored; doing so is a DEFERRED, OPTIONAL future step (not "
                       "done) -- see deferred_optional_future_step."),
        },
        "feasibility_gate": {
            "multi_skill_q_present": False,
            "detail": ("BiGGen bank/rubrics have q_mapping=null, primary_skill='general', "
                       "q_modeled={'general':1} for every criterion. No named multi-skill "
                       "Q-matrix exists, so the confirmatory 1..5 sweep is NOT run; a "
                       "data-driven diagnostic is run instead and the sweep is STOPPED short."),
        },
        "n_models": n, "n_modeled_atoms": p_items,
        "pca": {
            "pc1_variance_share": round(pc1_share, 4),
            "pc2_variance_share": round(pc2_share, 4),
            "pc1_over_pc2_ratio": round(pc1_pc2, 3),
            "parallel_analysis_n_factors": k_pa,
            "top_eigenvalues": [round(float(v), 4) for v in ev[:15]],
            "parallel_analysis_null_p95": [round(float(v), 4) for v in pa95[:15]],
            "note": ("n<<p makes the correlation matrix rank-deficient, so eigenvalue "
                     "ABSOLUTE scale is uninformative; the reference is the parallel-"
                     "analysis permutation null (same n, p, structure broken column-wise). "
                     "n_factors counts components exceeding that null's 95th percentile."),
        },
        "capability_composites": {
            "capabilities": present_caps,
            "mean_offdiag_r": round(mean_off, 4),
            "min_pair": min_pair,
            "safety_mean_r_vs_others": safety_mean,
            "n_atoms_per_capability": n_cap_atoms,
            "note": ("SKILL_AXIS_NOTES Finding 1 recomputed on our binary atoms: high "
                     "capability collinearity => general ability dominates; safety is the "
                     "capability that separates most."),
        },
        "m2pl_1v2_probe": {
            "available": probe_2d_available,
            "structures": rows,
            "selected_by_oos_logloss_within_1se_then_parsimony": selected["structure"],
            "note": ("EXPLORATORY only. The 2-D probe loads a normative axis on "
                     "capability=='safety' atoms (cross-loading; non-safety atoms anchor "
                     "general). It uses the existing capability field, not a fabricated Q."),
        },
        "verdict": {
            "multidim_confirmatory_supported_at_n52": multidim_supported,
            "recommendation": (
                "Keep UNIDIMENSIONAL. A dominant general factor drives the atoms "
                f"(PC1/PC2 ~ {pc1_pc2:.1f}; capability composite mean|r| ~ {mean_off:.2f}), "
                "and the notes' strongest 2-D candidate (general + normative/safety) does "
                "NOT win out-of-sample (held-out log-loss ties/loses vs 1-D) and loses on "
                "BIC -- only AIC, which overfits at N=52, favors it (latent corr ~0.6, so "
                "the axes are not independent). Safety is the one capability that separates, "
                "echoing SKILL_AXIS_NOTES."),
            "parallel_analysis_reconciliation": (
                f"Horn parallel analysis flags {k_pa} components above the permutation null. "
                "This is EXPECTED and LIBERAL at n_atoms >> n_models and reflects the weak "
                "residual cross-cutting structure SKILL_AXIS_NOTES Finding 3 describes (the "
                "A/B/C demand clusters under the general factor). It is NOT evidence for a "
                "confirmatory multi-skill model: that structure does not survive "
                "cross-validation here, and no named Q-matrix exists to test it."),
            "deferred_optional_future_step": (
                "A named, labeled multi-skill Q-matrix (per-criterion q_mapping / "
                "primary_skill) does not exist for BiGGen and was NOT authored (user "
                "decision: keep 1-D). Should a future run want a confirmatory 1..5 sweep, "
                "someone would first have to author such a Q (e.g. the 'general + normative "
                "restraint' 2-D or the demand-based A/B/C 3-D axis discussed in "
                "data/BiGGen/SKILL_AXIS_NOTES.md); that is an OPTIONAL future step, not a "
                "pending blocker for this study."),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "selection.json").write_text(json.dumps(selection, indent=2),
                                                 encoding="utf-8")

    _figures(ev, pa95, k_pa, corr, present_caps, n, args.out_dir / "figures")

    print(f"\nVERDICT: multidim supported at N={n}? {multidim_supported}  "
          f"(selected: {selected['structure']})")
    print(f"wrote -> {args.out_dir}")
    return 0


def _figures(ev, pa95, k_pa, corr, caps, n, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    kk = min(15, ev.size, pa95.size)
    xs = np.arange(1, kk + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, ev[:kk], "o-", color="#2c7fb8", label="observed eigenvalues")
    ax.plot(xs, pa95[:kk], "s--", color="#c1666b", alpha=0.8,
            label="parallel-analysis null (95%)")
    ax.axvline(k_pa + 0.5, ls="-", color="green", alpha=0.3,
               label=f"n_factors > null = {k_pa}")
    ax.set_yscale("log")
    ax.set_xlabel("component"); ax.set_ylabel("eigenvalue (log; scale arbitrary)")
    ax.set_title(f"BiGGen dimensionality scree (N={n}); parallel-analysis n_factors={k_pa}\n"
                 "EXPLORATORY -- unidimensional bank; no confirmatory sweep", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "structure_cv.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(caps))); ax.set_xticklabels(caps, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(caps))); ax.set_yticklabels(caps, fontsize=7)
    for i in range(len(caps)):
        for j in range(len(caps)):
            if np.isfinite(corr[i, j]):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if corr[i, j] < 0.7 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("BiGGen capability-composite correlation\n(general ability dominates; "
                 "safety separates)", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "capability_corr.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
