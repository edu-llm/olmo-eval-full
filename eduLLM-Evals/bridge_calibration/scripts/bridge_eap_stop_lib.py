"""EAP-posterior-SD stop primitives for the SCENARIO-LEVEL Bridge study (1-D).

Mirrors the WildBench (``prototype_eap_stop.py``) and BiGGen (``biggen_eap_stop_lib.py``)
1-D EAP-stop implementations. This changes ONLY the CAT *stop rule* -- it does not modify
the of-record engine, refit the bank, or re-run the SE_param bootstrap. The adaptive
SCENARIO SELECTION is the production engine's (unchanged): we run the engine to a long
forced length to obtain each model's adaptive administration ORDER, then apply the
EAP-posterior stop rule POST-HOC on that order:

  at each scenario boundary compute the EAP posterior SD over the administered items so far
  on a fine theta grid (posterior proportional to L(administered|theta)*N(0,1)); STOP at the
  first scenario with n_scenarios >= floor AND posterior SD <= SE target, else continue to
  the forced cap (= "hits cap").

Bridge is 1-D (``overall`` skill) like both references, so this is a single-integral
posterior SD (no multidim). The fine theta grid mirrors Bridge's of-record fine-EAP
reference (321 nodes over [-8, 8], standard-normal prior; see
``experiments/05_oos_recovery/recovery_metrics.json``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import log_expit, logsumexp

# Of-record locked defaults (Bridge): SE target 0.15, min_scenarios 12.
EAP_L_MAX = 60      # forced administration cap (scenarios); matches exp-06b max_scenarios
EAP_TARGET = 0.15   # of-record SE target (posterior SD)
EAP_FLOOR = 12      # of-record min_scenarios floor


def eap_grid(nodes: int = 321, half: float = 8.0):
    """Fine uniform theta grid + standard-normal log-prior (normalised).

    Defaults mirror Bridge's of-record fine-EAP reference (321 nodes over [-8, 8]).
    """
    gg = np.linspace(-half, half, nodes)
    lp = -0.5 * gg ** 2
    lp = lp - logsumexp(lp)
    return gg, lp


def eap_walk(order, yrow, col, scen_of, a, b, gg, lp):
    """Per-scenario cumulative EAP (mean, SD) along an administration order.

    ``order`` = criterion_ids in administration order; ``yrow`` = the model's responses
    (aligned to ``col``); ``a``/``b`` = 1-D discriminations/difficulties (aligned to ``col``).
    Returns a list of dicts {n_scen, n_crit, eap_mean, eap_sd, idx} recorded at each scenario
    boundary (idx = cumulative col indices administered so far).
    """
    ll_data = np.zeros(gg.size)
    out = []
    cur = None
    n_scen = 0
    n_crit = 0
    cum = []
    for cid in order:
        j = col.get(cid)
        if j is None:
            continue
        sid = scen_of.get(cid)
        if sid != cur:
            if cur is not None:
                post = np.exp(ll_data + lp - logsumexp(ll_data + lp))
                mean = float(post @ gg)
                sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
                out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                            "eap_sd": sd, "idx": np.array(cum, dtype=int)})
            cur = sid
            n_scen += 1
        eta = a[j] * gg - b[j]
        ll_data = ll_data + yrow[j] * log_expit(eta) + (1.0 - yrow[j]) * log_expit(-eta)
        n_crit += 1
        cum.append(j)
    if cur is not None:
        post = np.exp(ll_data + lp - logsumexp(ll_data + lp))
        mean = float(post @ gg)
        sd = float(np.sqrt(max(post @ (gg ** 2) - mean ** 2, 0.0)))
        out.append({"n_scen": n_scen, "n_crit": n_crit, "eap_mean": mean,
                    "eap_sd": sd, "idx": np.array(cum, dtype=int)})
    return out


def eap_stop_point(walk, floor=EAP_FLOOR, target=EAP_TARGET):
    """First scenario step meeting (n_scen>=floor AND eap_sd<=target); else last (capped)."""
    for step in walk:
        if step["n_scen"] >= floor and step["eap_sd"] <= target:
            return step, True
    return (walk[-1], False) if walk else (None, False)


def administer_eap(models, bank_path, matrix_path, scenarios_path, dims, a, b, col, scen_of,
                   Yfull, row_of, gg, lp, seed=20260729, floor=EAP_FLOOR, target=EAP_TARGET,
                   l_max=EAP_L_MAX, workers=6, runs_dir=None):
    """Administer the EAP-stop CAT for ``models``: one forced-long engine run + post-hoc stop.

    Returns {model: {order, scenarios_administered, criteria_administered, eap_sd, eap_mean,
    idx, hit_cap, precision_reached}}. ``a``/``b``/``col``/``scen_of``/``Yfull``/``row_of`` are
    aligned to the SAME bank as ``bank_path``. Selection is the production engine's; only the
    stop differs. Mirrors ``biggen_eap_stop_lib.administer_eap``.
    """
    import shutil

    import scripts.scenario_cat_lib as scat  # lazy (heavy engine import)

    rd = runs_dir or str(Path(bank_path).parent / "_eap_runs")
    spec = scat.RunSpec(seed=seed, top_n=5, max_se=0.0, min_evals_per_skill=0,
                        min_scenarios=l_max, max_scenarios=l_max, selection="trace",
                        mode="cat", runs_dir=rd)
    res = scat.run_models(models, bank_path, matrix_path, scenarios_path, "clamp", dims,
                          spec, workers=workers)
    out = {}
    for r in res:
        m = r["model"]
        yrow = Yfull[row_of[m]]
        walk = eap_walk(r["order"], yrow, col, scen_of, a, b, gg, lp)
        step, reached = eap_stop_point(walk, floor, target)
        if step is None:
            out[m] = {"order": [], "scenarios_administered": 0, "criteria_administered": 0,
                      "eap_sd": float("nan"), "eap_mean": float("nan"),
                      "idx": np.array([], dtype=int), "hit_cap": True,
                      "precision_reached": False}
            continue
        k = step["n_crit"]
        out[m] = {"order": [c for c in r["order"] if c in col][:k],
                  "scenarios_administered": step["n_scen"],
                  "criteria_administered": step["n_crit"],
                  "eap_sd": step["eap_sd"], "eap_mean": step["eap_mean"],
                  "idx": step["idx"], "hit_cap": bool(not reached),
                  "precision_reached": bool(reached)}
    shutil.rmtree(rd, ignore_errors=True)
    return out


def make_folds(models, k, seed):
    """Deterministic person (model) folds, matching the of-record k-fold convention."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def dist(s):
    """Compact distribution summary (median / mean / q1 / q3)."""
    s = np.asarray(s, float)
    return {"median": round(float(np.median(s)), 4), "mean": round(float(np.mean(s)), 4),
            "q1": round(float(np.quantile(s, .25)), 4), "q3": round(float(np.quantile(s, .75)), 4)}
