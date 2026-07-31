"""Regenerate the TutorBench CAT recovery + pIRT figures with a CONTINUOUS ability axis.

Why this exists
---------------
``scripts/cat_eval_tutorbench.py`` estimates the full-bank ("truth") ability by EAP over
a fixed **Gauss-Hermite** grid with ``--grid 7`` nodes per dimension. For a peaked
posterior the EAP collapses onto a single quadrature node, so the reference axis snaps to
the 7 node coordinates -- exactly the ``{0, +/-1.15, +/-2.37, +/-3.75}`` columns visible in
the published scatters. The recovery scatter then looks like a handful of vertical
stripes rather than a cloud.

This script keeps the estimator identical (same M2PL link, same standard-normal prior,
same online MIRT Newton/Laplace CAT update) and only replaces the quadrature with a dense
grid, so the posterior mean can land anywhere and the horizontal axis becomes continuous.

* ``--grid-kind uniform`` (default): evenly spaced nodes on ``[-range, +range]`` per
  dimension, weighted by the standard-normal prior density. Even resolution everywhere,
  including the tails where Gauss-Hermite is sparsest.
* ``--grid-kind gh``: the original Gauss-Hermite rule, for an apples-to-apples check.

The node grid is evaluated in chunks so a dense 2-D or 3-D grid stays within memory: only
the ``(n_models, n_nodes)`` log-likelihood table is materialised.

Inputs (all already on disk -- no tutor-model inference is needed)
------------------------------------------------------------------
* ``--matrix``  staging/response_matrix.csv  -- 82 models x 6180 criteria, 0/1/NaN. Built
  by the node-side grading chain from the cached tutorbench responses + Qwen judge.
* ``--bank``    data/TutorBench/rubrics_qmatrix_calibrated_{2,3}skill.jsonl -- the frozen
  per-criterion ``discrimination`` (Q-masked 3-vector) and scalar ``difficulty``.

Only criteria whose ``irt_params.source`` marks them calibrated AND that appear as a
matrix column are used. Latent dimensions with no non-zero loading are dropped, so the
2-skill bank runs as 2-D (content collapsed to "correctness", + scaffolding) and the
3-skill bank as 3-D.

Outputs (``--out-dir``)
-----------------------
* ``figures/recovery_scatter_<dim>.png``  -- CAT vs full-bank ability, per dimension
* ``figures/pirt_calibration.png``        -- predicted vs observed pass-rate
* ``cat_per_model.csv``                   -- per-model abilities, CAT length, SEs
* ``metrics.json``                        -- recovery r, pIRT MAE, axis-continuity stats

Usage
-----
    python scripts/regen_cat_figures.py --bank data/TutorBench/rubrics_qmatrix_calibrated_2skill.jsonl \
        --out-dir regenerated_figures/2pl
    python scripts/regen_cat_figures.py --bank data/TutorBench/rubrics_qmatrix_calibrated_3skill.jsonl \
        --out-dir regenerated_figures/3pl --grid 61
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SKILLS = ("content", "diagnosis", "scaffolding")
# The 2-skill bank collapses content+diagnosis into a single axis that the published
# report labels "correctness"; it is stored in the `content` slot.
COLLAPSED_LABEL = {"content": "correctness"}

CALIBRATED_MARKER = "calibrated"


# ---------------------------------------------------------------------------
# bank + matrix loading
# ---------------------------------------------------------------------------


def bank_irt_params(path: Path) -> dict:
    """The ``irt_params`` block from the first calibrated record in the bank."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ip = rec.get("irt_params") or {}
            if CALIBRATED_MARKER in str(ip.get("source", "")):
                return ip
    return {}


def bank_provenance(path: Path) -> dict:
    """The matrix this bank records that it was calibrated against (``matrix_csv`` +
    ``matrix_sha256``). Empty dict if the bank carries no provenance."""
    return bank_irt_params(path).get("provenance") or {}


def modeled_skill_names(bank: Path) -> list[str] | None:
    """The SEMANTIC skill names this bank models, from its own provenance.

    The JSON `discrimination` keys are fixed slots (content/diagnosis/scaffolding), but a
    bank may repurpose them. The 3-skill bank records:

        "slot_repurposing": "a_content->correctness, a_diagnosis->scaffolding,
                             a_scaffolding->presentation"

    so its slot named "scaffolding" actually holds *presentation*. Trusting the slot names
    would silently mislabel every downstream axis, so read `modeled_skills` instead. It is
    ordered to match the non-zero slots in slot order, which is how align() pairs them.
    """
    ip = bank_irt_params(bank)
    names = ip.get("modeled_skills")
    if isinstance(names, list) and names and all(isinstance(s, str) for s in names):
        return [str(s) for s in names]
    return None


def verify_matrix(bank: Path, matrix: Path) -> dict:
    """Compare the matrix's sha256 against the bank's recorded one.

    Item parameters are only meaningful on the response matrix they were fit from, so a
    mismatch silently degrades every downstream number. Loud by design.
    """
    prov = bank_provenance(bank)
    want = prov.get("matrix_sha256")
    got = hashlib.sha256(matrix.read_bytes()).hexdigest()
    ok = bool(want) and want == got
    print(f"provenance: bank expects {prov.get('matrix_csv')}")
    print(f"            sha256 expected={str(want)[:24]}... actual={got[:24]}...")
    print("            " + ("MATCH - matrix and bank are aligned" if ok else
                            "*** MISMATCH - results will not reproduce the published fit ***"))
    return {"expected_matrix": prov.get("matrix_csv"), "expected_sha256": want,
            "actual_sha256": got, "aligned": ok}


def resolve_matrix(bank: Path, explicit: Path | None) -> Path:
    """Use ``--matrix`` when given, else the matrix the bank names in its provenance."""
    if explicit is not None:
        return explicit
    named = bank_provenance(bank).get("matrix_csv")
    if named:
        p = (ROOT / named).resolve()
        if p.is_file():
            return p
    return ROOT / "staging" / "response_matrix_full_nonopt.csv"


def load_bank(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read calibrated criteria -> (criterion_ids, A (n_items, 3), b (n_items,))."""
    items: list[str] = []
    rows: list[list[float]] = []
    diffs: list[float] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            source = str((rec.get("irt_params") or {}).get("source", ""))
            if CALIBRATED_MARKER not in source:
                continue
            disc = rec.get("discrimination")
            diff = rec.get("difficulty")
            if not isinstance(disc, dict) or diff is None:
                continue
            a = [float(disc.get(s, 0.0) or 0.0) for s in SKILLS]
            if not np.all(np.isfinite(a)) or not np.isfinite(float(diff)):
                continue
            items.append(str(rec["criterion_id"]))
            rows.append(a)
            diffs.append(float(diff))
    return items, np.asarray(rows, dtype=float), np.asarray(diffs, dtype=float)


def align(matrix: pd.DataFrame, items: list[str], A: np.ndarray, b: np.ndarray,
          modeled: list[str] | None = None):
    """Restrict to criteria present in BOTH the bank and the matrix; drop dead dims.

    ``modeled`` is the bank's ``modeled_skills`` list (see :func:`modeled_skill_names`).
    When supplied and its length matches the surviving dimension count, it names the axes;
    otherwise we fall back to the raw slot names, which is only correct for banks that do
    not repurpose slots.
    """
    keep = [i for i, c in enumerate(items) if c in matrix.columns]
    items = [items[i] for i in keep]
    A = A[keep]
    b = b[keep]

    active = [k for k in range(len(SKILLS)) if np.any(np.abs(A[:, k]) > 1e-9)]
    if modeled is not None and len(modeled) == len(active):
        dim_names = list(modeled)
    else:
        # Fallback: raw slot names, with the historical content->correctness relabel when
        # diagnosis was collapsed away.
        collapsed = (SKILLS.index("diagnosis") not in active
                     and SKILLS.index("content") in active)
        dim_names = [COLLAPSED_LABEL[SKILLS[k]] if (collapsed and SKILLS[k] in COLLAPSED_LABEL)
                     else SKILLS[k] for k in active]
    A = A[:, active]

    sub = matrix.reindex(columns=items)
    Yraw = sub.to_numpy(dtype=float)
    mask = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    return items, A, b, Y, mask, dim_names


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------


def build_grid(n_dims: int, nodes_per_dim: int, kind: str, half_range: float):
    """Return (grid (n_nodes, n_dims), log_prior (n_nodes,)) normalised to sum 1."""
    if kind == "gh":
        x, w = np.polynomial.hermite_e.hermegauss(nodes_per_dim)
        logw1d = np.log(w / np.sqrt(2.0 * np.pi))
    else:
        x = np.linspace(-half_range, half_range, nodes_per_dim)
        # Standard-normal density weights; the constant spacing factor cancels in the
        # normalisation below, so a plain density is sufficient.
        logw1d = -0.5 * x**2

    mesh = np.meshgrid(*([x] * n_dims), indexing="ij")
    grid = np.stack([m.reshape(-1) for m in mesh], axis=1).astype(float)

    logmesh = np.meshgrid(*([logw1d] * n_dims), indexing="ij")
    log_prior = np.sum(np.stack([m.reshape(-1) for m in logmesh], axis=1), axis=1)
    return grid, log_prior - logsumexp(log_prior)


def eap_all_models(Y, mask, A, b, grid, log_prior, chunk: int) -> np.ndarray:
    """Full-bank EAP ability for every model, chunked over quadrature nodes.

    Mathematically identical to ``cat_eval_tutorbench.eap_full`` (per-node log-likelihood
    from observed cells only -> posterior -> posterior mean), just vectorised over models
    and evaluated in node blocks so a dense grid fits in memory.
    """
    ym = np.where(mask, Y, 0.0)
    nm = np.where(mask, 1.0 - Y, 0.0)
    n_models, n_nodes = Y.shape[0], grid.shape[0]

    ll = np.empty((n_models, n_nodes), dtype=float)
    for start in range(0, n_nodes, chunk):
        stop = min(start + chunk, n_nodes)
        eta = A @ grid[start:stop].T - b[:, None]          # (n_items, blk)
        ll[:, start:stop] = ym @ log_expit(eta) + nm @ log_expit(-eta)

    joint = ll + log_prior[None, :]
    post = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    return post @ grid


# ---------------------------------------------------------------------------
# adaptive CAT (unchanged estimator: online MIRT Newton/Laplace, grid-free)
# ---------------------------------------------------------------------------


def mirt_update(theta, U, m, b_i, y):
    """PRD Eqs 2-3. `m` is the Q-masked discrimination vector for the administered item."""
    p = float(expit(float(m @ theta) - b_i))
    info = p * (1.0 - p) * np.outer(m, m)
    U_new = np.linalg.inv(np.linalg.inv(U) + info)
    U_new = (U_new + U_new.T) / 2.0
    return theta + U_new @ m * (float(y) - p), U_new


def eap_subset(y, idx, A, b, grid, log_prior):
    """Batch EAP over ONLY the administered items.

    Identical math to :func:`eap_all_models`, restricted to one person's administered
    subset. Because the likelihood is a product over items, this is exactly order
    invariant -- unlike the sequential update, which linearises at whatever theta was
    current when each item arrived and never revisits it.
    """
    Ai, bi, yi = A[idx], b[idx], y[idx]
    eta = Ai @ grid.T - bi[:, None]
    ll = yi @ log_expit(eta) + (1.0 - yi) @ log_expit(-eta)
    joint = ll + log_prior
    post = np.exp(joint - logsumexp(joint))
    return post @ grid


def _mwle_neg_obj_grad(theta, A, b, y, ridge):
    """Negative MWLE objective and gradient.

    Objective (maximised):  ln L(theta) + 1/2 * ln det I(theta)

    The penalty is the Jeffreys / Firth bias-reduction term. In ONE dimension Warm's
    weight is w(theta) = sqrt(I(theta)), so ln w = 1/2 ln I -- i.e. this reduces exactly
    to Warm's WLE, and the multidimensional case replaces I by det I.

    It also supplies the finiteness that plain MLE lacks: for an all-fail pattern ln L
    keeps rising as theta -> -inf, but I(theta) -> 0 so 1/2 ln det I -> -inf, which pins
    the maximum at an interior point.
    """
    n_dims = theta.shape[0]
    eta = A @ theta - b
    p = expit(eta)
    w = p * (1.0 - p)
    info = (A * w[:, None]).T @ A + ridge * np.eye(n_dims)

    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e10, np.zeros(n_dims)

    ll = float(y @ log_expit(eta) + (1.0 - y) @ log_expit(-eta))
    obj = ll + 0.5 * logdet

    # d/dtheta_k [1/2 ln det I] = 1/2 tr(I^-1 dI/dtheta_k), and
    # dI/dtheta_k = sum_i w_i(1-2p_i) a_ik a_i a_i^T, so the trace collapses to
    # sum_i w_i(1-2p_i) a_ik (a_i^T I^-1 a_i).
    iinv = np.linalg.inv(info)
    c = w * (1.0 - 2.0 * p)
    quad = np.einsum("ij,jk,ik->i", A, iinv, A)
    grad = A.T @ (y - p) + 0.5 * (A * (c * quad)[:, None]).sum(axis=0)
    return -obj, -grad


def mwle_subset(y, idx, A, b, theta0, ridge=1e-6, bound=12.0):
    """Multidimensional WLE over the administered items, started from ``theta0``.

    Returns (theta, converged). Falls back to ``theta0`` when the information matrix is
    too near-singular to penalise (e.g. the CAT administered almost nothing loading on
    one skill, so that dimension is unidentified from this subset).
    """
    from scipy.optimize import minimize

    Ai, bi, yi = A[idx], b[idx], y[idx].astype(float)
    if Ai.shape[0] < Ai.shape[1]:
        return np.asarray(theta0, dtype=float), False
    try:
        res = minimize(_mwle_neg_obj_grad, np.asarray(theta0, dtype=float),
                       args=(Ai, bi, yi, ridge), jac=True, method="L-BFGS-B",
                       bounds=[(-bound, bound)] * Ai.shape[1])
    except Exception:
        return np.asarray(theta0, dtype=float), False
    if not np.all(np.isfinite(res.x)):
        return np.asarray(theta0, dtype=float), False
    return res.x, bool(res.success)


def estimate_final(kind, y, order, A, b, grid, log_prior, theta_online, ridge):
    """Modular final-ability estimator. Selection is unchanged in every mode.

    * ``online`` -- report the accumulated sequential Gaussian theta (production).
    * ``batch``  -- discard it; re-estimate by EAP over the administered items.
    * ``mwle``   -- EAP as the starting point, then multidimensional WLE.

    Returns (theta, fell_back).
    """
    if kind == "online" or not len(order):
        return theta_online, False
    idx = np.asarray(order, dtype=int)
    eap = eap_subset(y, idx, A, b, grid, log_prior)
    if kind == "batch":
        return eap, False
    theta, ok = mwle_subset(y, idx, A, b, eap, ridge=ridge)
    return theta, (not ok)


def run_cat_person(y, mask, A, b, min_items, max_items, se_target):
    """Max-Fisher-info adaptive administration over the model's observed items."""
    n_dims = A.shape[1]
    theta = np.zeros(n_dims)
    U = np.eye(n_dims)
    remaining = set(int(i) for i in np.where(mask)[0])
    n_cross = [None] * n_dims
    n_admin = 0
    order: list[int] = []

    while remaining and n_admin < max_items:
        rem = np.fromiter(remaining, dtype=int)
        m = A[rem]
        p = expit(m @ theta - b[rem])
        info = p * (1.0 - p) * np.sum(m * m, axis=1)
        pick = int(rem[int(np.argmax(info))])
        theta, U = mirt_update(theta, U, A[pick], float(b[pick]), int(y[pick]))
        remaining.discard(pick)
        order.append(pick)
        n_admin += 1
        se = np.sqrt(np.diag(U))
        for d in range(n_dims):
            if n_cross[d] is None and float(se[d]) < se_target:
                n_cross[d] = n_admin
        if n_admin >= min_items and float(np.max(se)) < se_target:
            break

    return theta, n_admin, np.sqrt(np.diag(U)), n_cross, order


# ---------------------------------------------------------------------------
# metrics + figures
# ---------------------------------------------------------------------------


def pearson(x, y):
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 2 or xa.std() == 0 or ya.std() == 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def continuity(values: np.ndarray) -> dict:
    """How grid-quantised is the reference axis? More distinct values = more continuous."""
    v = np.asarray(values, float)
    return {
        "n_points": int(v.size),
        "n_distinct_rounded_3dp": int(np.unique(np.round(v, 3)).size),
        "distinct_fraction": float(np.unique(np.round(v, 3)).size / max(v.size, 1)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def make_figures(df, dim_names, agg, fig_dir: Path, se_target: float, est_label: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def _save(fig, name):
        p = fig_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths[name] = str(p)

    for dim in dim_names:
        full = df[f"theta_full_{dim}"].to_numpy()
        cat = df[f"theta_cat_{dim}"].to_numpy()
        r = agg["recovery_r"][dim]
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.scatter(full, cat, s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
        lo = min(full.min(), cat.min()) - 0.3
        hi = max(full.max(), cat.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ability ({dim})")
        ax.set_ylabel(f"CAT ability ({dim})")
        ax.set_title(f"TutorBench CAT recovery: {dim}{est_label}\n"
                     f"Pearson r = {r:.3f} (n = {len(df)} models)")
        ax.legend(loc="upper left", fontsize=9)
        _save(fig, f"recovery_scatter_{dim}.png")

    pred = df["pred_acc_cat"].to_numpy()
    act = df["obs_acc"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(act, pred, s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
    lo = min(pred.min(), act.min()) - 0.02
    hi = max(pred.max(), act.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("actual observed pass-rate")
    ax.set_ylabel("predicted accuracy (from CAT ability)")
    ax.set_title(f"TutorBench pIRT calibration{est_label}\n"
                 f"MAE = {agg['pirt_mae_cat']:.4f} (n = {len(df)} models)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "pirt_calibration.png")
    return paths


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path, default=None,
                   help="response matrix CSV; defaults to the one the bank's provenance names.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--grid", type=int, default=None,
                   help="nodes per latent dim (default: 201 for 2-D, 61 for 3-D).")
    p.add_argument("--grid-kind", choices=("uniform", "gh"), default="uniform")
    p.add_argument("--range", type=float, default=6.0,
                   help="half-width of the uniform grid (default 6.0).")
    p.add_argument("--chunk", type=int, default=4096, help="quadrature nodes per block.")
    p.add_argument("--se-target", type=float, default=0.3)
    p.add_argument("--min-items", type=int, default=1)
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--final-estimator", choices=("online", "batch", "mwle"), default="online",
                   help="'online' = report the accumulated sequential Gaussian theta "
                        "(production behaviour). 'batch' = keep the sequential update for "
                        "item SELECTION only, then re-estimate by EAP over the administered "
                        "items (order-invariant). 'mwle' = EAP init, then multidimensional "
                        "Warm WLE (also drops the prior's inward pull at the tails).")
    p.add_argument("--mwle-ridge", type=float, default=1e-6,
                   help="ridge added to I(theta) for numerical stability in MWLE.")
    args = p.parse_args()

    print("=" * 72)
    print(f"regenerating CAT figures :: bank={args.bank.name}")
    print("=" * 72)

    args.matrix = resolve_matrix(args.bank, args.matrix)
    prov = verify_matrix(args.bank, args.matrix)
    print()

    matrix = pd.read_csv(args.matrix, index_col=0)
    items, A_full, b_full = load_bank(args.bank)
    modeled = modeled_skill_names(args.bank)
    items, A, b, Y, mask, dim_names = align(matrix, items, A_full, b_full, modeled)
    models = list(matrix.index)
    n_dims = A.shape[1]

    slot_map = bank_provenance(args.bank).get("slot_map")
    print(f"skills    : modeled_skills={modeled} -> axes {dim_names}")
    if slot_map:
        print(f"            slot_map (JSON key -> what it holds): {slot_map}")

    if args.grid is None:
        args.grid = 201 if n_dims == 2 else 61

    print(f"bank      : {len(items)} calibrated items usable, {n_dims} active dims "
          f"{dim_names}")
    for k, name in enumerate(dim_names):
        col = A[:, k]
        print(f"            a_{name}: {int((np.abs(col) > 1e-9).sum())} loading, "
              f"median(nonzero)={np.median(col[np.abs(col) > 1e-9]):.3f}")
    print(f"matrix    : {len(models)} models x {len(items)} items "
          f"({int(mask.sum())} observed, {mask.mean()*100:.1f}% filled)")

    grid, log_prior = build_grid(n_dims, args.grid, args.grid_kind, args.range)
    print(f"quadrature: {args.grid_kind} grid, {args.grid} nodes/dim -> "
          f"{grid.shape[0]:,} nodes (was 7/dim in the original)")

    print("computing full-bank EAP ...")
    theta_full = eap_all_models(Y, mask, A, b, grid, log_prior, args.chunk)

    print("running adaptive CAT ...")
    rows = []
    n_fallback = 0
    for r, model in enumerate(models):
        th, n_admin, se_final, n_cross, order = run_cat_person(
            Y[r], mask[r], A, b, args.min_items, args.max_items, args.se_target
        )
        th, fell_back = estimate_final(args.final_estimator, Y[r], order, A, b,
                                       grid, log_prior, th, args.mwle_ridge)
        n_fallback += int(fell_back)
        obs = np.where(mask[r])[0]
        row = {"model": model, "n_obs_items": int(mask[r].sum()),
               "cat_n_items": n_admin,
               "obs_acc": float(Y[r][obs].mean()) if obs.size else np.nan,
               "pred_acc_cat": float(expit(A[obs] @ th - b[obs]).mean()) if obs.size else np.nan,
               "pred_acc_full": float(expit(A[obs] @ theta_full[r] - b[obs]).mean())
               if obs.size else np.nan}
        for k, name in enumerate(dim_names):
            row[f"theta_full_{name}"] = float(theta_full[r, k])
            row[f"theta_cat_{name}"] = float(th[k])
            row[f"final_se_{name}"] = float(se_final[k])
            row[f"n_cross_{name}"] = n_cross[k]
        rows.append(row)
    df = pd.DataFrame(rows)

    agg = {
        "recovery_r": {d: pearson(df[f"theta_full_{d}"], df[f"theta_cat_{d}"])
                       for d in dim_names},
        "pirt_mae_cat": float((df["pred_acc_cat"] - df["obs_acc"]).abs().mean()),
        "pirt_mae_full": float((df["pred_acc_full"] - df["obs_acc"]).abs().mean()),
        "axis_continuity": {d: continuity(df[f"theta_full_{d}"].to_numpy())
                            for d in dim_names},
    }

    est_label = {"online": "", "batch": " (batch EAP refit)",
                 "mwle": " (batch EAP + MWLE)"}[args.final_estimator]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "cat_per_model.csv", index=False)
    fig_paths = make_figures(df, dim_names, agg, args.out_dir / "figures", args.se_target,
                             est_label)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Recovery/pIRT figures with a dense-quadrature (continuous) ability axis.",
        "config": {"bank": str(args.bank), "matrix": str(args.matrix),
                   "grid_kind": args.grid_kind, "nodes_per_dim": args.grid,
                   "n_nodes": int(grid.shape[0]), "range": args.range,
                   "se_target": args.se_target, "min_items": args.min_items,
                   "max_items": args.max_items,
                   "final_estimator": args.final_estimator,
                   "mwle_ridge": args.mwle_ridge},
        "provenance": prov,
        "mwle_fallback_models": n_fallback,
        "bank": {"n_items_used": len(items), "n_dims": n_dims, "dims": dim_names,
                 "modeled_skills": modeled, "slot_map": slot_map,
                 **{f"a_{d}_n_loading": int((np.abs(A[:, k]) > 1e-9).sum())
                    for k, d in enumerate(dim_names)}},
        "n_models": int(len(df)),
        **agg,
        "figures": fig_paths,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    for d in dim_names:
        c = agg["axis_continuity"][d]
        print(f"{d:12s}: recovery r={agg['recovery_r'][d]:.3f}  "
              f"distinct x-values={c['n_distinct_rounded_3dp']}/{c['n_points']}  "
              f"range=[{c['min']:.2f}, {c['max']:.2f}]")
    print(f"pIRT MAE    : CAT={agg['pirt_mae_cat']:.4f}  "
          f"full-bank={agg['pirt_mae_full']:.4f}")
    print(f"estimator   : {args.final_estimator}"
          + (f"  (MWLE fell back to EAP for {n_fallback}/{len(df)} models)"
             if args.final_estimator == "mwle" else ""))
    print(f"\nwrote -> {args.out_dir}")
    for name in fig_paths:
        print(f"  figures/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
