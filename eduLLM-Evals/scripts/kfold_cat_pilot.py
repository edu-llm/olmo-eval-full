"""PILOT: CAT simulation on held-out persons using a frozen k-fold M2PL item bank.

This is the OPTIONAL stretch of the k-fold cross-validation study
(``scripts/kfold_cv_mirt.py``). It is a PILOT, on ONE fold only, and is clearly
labelled as such -- it is a sanity check that the frozen 2-skill item params support
adaptive testing of an unseen model, NOT a full CAT evaluation.

Design
------
* Use the frozen item params from fold ``--fold`` of a completed k-fold run
  (``staging/kfold/fold_<f>_item_params.csv`` + ``fold_assignments.json``). These
  were fit on the TRAIN models only, so the fold's TEST models are genuinely unseen.
* For each held-out (TEST) model, run adaptive item selection over that model's
  OBSERVED responses on the fitted items:
    - start theta=(0,0), U=diag(1,1);
    - next item = the observed, not-yet-administered item with MAX Fisher
      information ``p(1-p) * (m . m)`` at the current ability (max-info rule);
    - administer it (use the model's actual observed judge outcome) and update
      (theta, U) via the exact MIRT Newton/Laplace step reused from
      ``tutor_cat.mirt.update`` (PRD Eqs 1-3);
    - stop when both per-dim SEs < ``--se-stop`` (after >= ``--min-items``) or at
      ``--max-items``.
* Compare the CAT ability estimate to the FULL-response ability (EAP over all of the
  model's observed items under the same frozen params) and report ability-recovery
  error and predicted-score error.

Reuse
-----
The MIRT update, pass-probability, and Fisher-info are reused verbatim from
``tutor_cat.mirt`` (dimension-agnostic; here used with the 2-d collapsed params).
Nothing is written back to any bank; outputs land under the k-fold out dir.

Usage
-----
    python scripts/kfold_cat_pilot.py --fold 0
    python scripts/kfold_cat_pilot.py --kfold-dir staging/kfold --fold 0 --se-stop 0.4
"""

from __future__ import annotations

import argparse
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

import importlib.util  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


cm = _load_module("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cp = cm.cp
from tutor_cat import mirt as tc_mirt  # noqa: E402  (reused Fisher-info + update)

DEFAULT_MATRIX = ROOT / "staging" / "response_matrix.csv"
DEFAULT_KFOLD_DIR = ROOT / "staging" / "kfold"


def eap_full(y_obs: np.ndarray, mask: np.ndarray, A: np.ndarray, b: np.ndarray,
             grid: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Full-response EAP ability from ALL observed items (reference ability)."""
    ym = np.where(mask, y_obs, 0.0)
    nm = np.where(mask, 1.0 - y_obs, 0.0)
    eta = A @ grid.T - b[:, None]
    ll = ym @ log_expit(eta) + nm @ log_expit(-eta)   # (n_nodes,)
    joint = ll + log_prior
    post = np.exp(joint - logsumexp(joint))
    return post @ grid


def run_cat_person(y: np.ndarray, mask: np.ndarray, A: np.ndarray, b: np.ndarray,
                   min_items: int, max_items: int, se_stop: float) -> dict:
    """Max-Fisher-info adaptive administration for one person over observed items."""
    obs_idx = np.where(mask)[0]
    theta = np.zeros(A.shape[1])
    U = np.eye(A.shape[1])
    ones_q = np.ones(A.shape[1])
    administered: list[int] = []
    remaining = set(int(i) for i in obs_idx)

    n_admin = 0
    while remaining and n_admin < max_items:
        # Fisher info for each remaining item at current theta: p(1-p)*(m.m).
        rem = np.fromiter(remaining, dtype=int)
        m = A[rem]                              # (n_rem, dims); already Q-masked
        p = expit(m @ theta - b[rem])
        info = p * (1 - p) * np.sum(m * m, axis=1)
        pick = int(rem[int(np.argmax(info))])
        theta, U, _ = tc_mirt.update(theta, U, A[pick], ones_q, float(b[pick]),
                                     int(y[pick]))
        administered.append(pick)
        remaining.discard(pick)
        n_admin += 1
        se = tc_mirt.standard_errors(U)
        if n_admin >= min_items and float(np.max(se)) < se_stop:
            break

    return {"theta_cat": theta, "n_items": n_admin,
            "se": tc_mirt.standard_errors(U).tolist()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--kfold-dir", type=Path, default=DEFAULT_KFOLD_DIR)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--min-items", type=int, default=5)
    p.add_argument("--max-items", type=int, default=60)
    p.add_argument("--se-stop", type=float, default=0.40,
                   help="stop when both per-dim SEs < this (after --min-items).")
    args = p.parse_args()

    mat = cp.load_matrix(args.matrix)
    with (args.kfold_dir / "fold_assignments.json").open(encoding="utf-8") as fh:
        folds = json.load(fh)["folds"]
    test_models = folds[str(args.fold)]

    params = pd.read_csv(args.kfold_dir / f"fold_{args.fold}_item_params.csv")
    items = params["criterion_id"].tolist()
    A = params[["a_correctness", "a_scaffolding"]].to_numpy(dtype=float)
    b = params["b"].to_numpy(dtype=float)

    grid = cm.build_grid(2, args.grid)
    base_logw = cm.base_log_weights(2, args.grid)
    log_prior = cm.prior_log_weights(grid, base_logw, np.eye(2))

    sub = mat.loc[test_models].reindex(columns=items)
    Yraw = sub.to_numpy(dtype=float)
    Mobs = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)

    rows = []
    n_items_used, ability_err, score_err = [], [], []
    for r, model in enumerate(test_models):
        mask = Mobs[r]
        if mask.sum() < args.min_items:
            continue
        theta_full = eap_full(Y[r], mask, A, b, grid, log_prior)
        res = run_cat_person(Y[r], mask, A, b, args.min_items, args.max_items,
                             args.se_stop)
        theta_cat = res["theta_cat"]
        # predicted-score error: mean predicted P(pass) over the person's observed
        # items under CAT theta vs full-response theta.
        obs = np.where(mask)[0]
        p_cat = expit(A[obs] @ theta_cat - b[obs]).mean()
        p_full = expit(A[obs] @ theta_full - b[obs]).mean()
        aerr = float(np.linalg.norm(theta_cat - theta_full))
        serr = float(abs(p_cat - p_full))
        n_items_used.append(res["n_items"])
        ability_err.append(aerr)
        score_err.append(serr)
        rows.append({
            "model": model, "n_obs_items": int(mask.sum()),
            "cat_n_items": res["n_items"],
            "theta_cat_correctness": round(float(theta_cat[0]), 4),
            "theta_cat_scaffolding": round(float(theta_cat[1]), 4),
            "theta_full_correctness": round(float(theta_full[0]), 4),
            "theta_full_scaffolding": round(float(theta_full[1]), 4),
            "ability_l2_err": round(aerr, 4),
            "pred_score_cat": round(float(p_cat), 4),
            "pred_score_full": round(float(p_full), 4),
            "pred_score_abs_err": round(serr, 4),
            "final_se_correctness": round(res["se"][0], 4),
            "final_se_scaffolding": round(res["se"][1], 4),
        })

    detail = pd.DataFrame(rows)
    out_csv = args.kfold_dir / f"cat_pilot_fold_{args.fold}.csv"
    detail.to_csv(out_csv, index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "PILOT": True,
        "note": ("PILOT CAT on ONE fold's held-out (unseen) models using that fold's "
                 "frozen 2-skill M2PL params. Max-Fisher-info selection, MIRT update "
                 "reused from tutor_cat.mirt. Not a full CAT evaluation."),
        "fold": args.fold,
        "n_test_models": len(test_models),
        "n_scored": len(rows),
        "stopping_rule": {"min_items": args.min_items, "max_items": args.max_items,
                          "se_stop": args.se_stop},
        "median_items_to_converge": (float(np.median(n_items_used)) if n_items_used else None),
        "mean_items_to_converge": (float(np.mean(n_items_used)) if n_items_used else None),
        "median_ability_l2_err_vs_full": (float(np.median(ability_err)) if ability_err else None),
        "median_pred_score_abs_err_vs_full": (float(np.median(score_err)) if score_err else None),
        "max_pred_score_abs_err_vs_full": (float(np.max(score_err)) if score_err else None),
    }
    with (args.kfold_dir / f"cat_pilot_fold_{args.fold}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
