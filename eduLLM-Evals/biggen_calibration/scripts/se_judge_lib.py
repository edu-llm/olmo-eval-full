"""SE_judge measurement-error primitives for the BiGGen gemini-3 recalibration (Phase 3).

LOCAL scratch library. It CALLS the frozen production engine (scenario_cat_lib,
calibrate_mirt, biggen_eap_stop_lib, biggen_recovery_grid) -- it never edits it. This
module only holds the judge-measurement-error layer:

  * per-stratum Beta posteriors for the confusion rates (alpha=false-fail, beta=false-pass)
    fit Jeffreys-style from the 250-cell human-gold 2x2 counts (joined judge verdicts);
  * the single-de-bias per-criterion prior pass-rate pi_c;
  * the Bayes true-label posterior P(y | y*, alpha_s, beta_s, pi_c);
  * the CORRELATED-within-criterion resample (one latent uniform per criterion applied to
    all 52 models' cells in that column -- gemini-3 @ temp 0 mis-grades a whole criterion
    systematically, not independently per cell). The comonotone single-uniform coupling
    preserves each cell's Bayes-posterior marginal exactly while inducing full
    within-column correlation.

Judge stamp: gemini-3-flash-preview.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 8 BiGGen capability strata (the confusion.json / criterion_to_capability granularity).
STRATA = ("grounding", "instruction_following", "planning", "reasoning",
          "refinement", "safety", "theory_of_mind", "tool_usage")

JUDGE = "gemini-3-flash-preview"

# Overall confusion CI (scenario-clustered) from biggen_gemini3_confusion.json, used for the
# systematic bias band corners.
ALPHA_OVERALL = 0.22522522522522523
BETA_OVERALL = 0.02158273381294964
ALPHA_CI = (0.1471, 0.3131)
BETA_CI = (0.0, 0.0496)


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def stratum_counts(gold_labels_path: Path, verdicts_path: Path) -> dict:
    """Join the 250 human-gold labels to the gemini-3 judge verdicts; per-stratum 2x2.

    Returns {stratum: {n, gold_pass, gold_fail, false_fail, false_pass}} plus an "overall"
    aggregate. ``unscorable`` verdicts map to fail (production fail-closed), matching the
    confusion.json convention.
    """
    gold = {d["gold_case_id"]: d for d in read_jsonl(gold_labels_path)}
    ver = {d["gold_case_id"]: d for d in read_jsonl(verdicts_path)}
    out = {s: dict(n=0, gold_pass=0, gold_fail=0, false_fail=0, false_pass=0) for s in STRATA}
    overall = dict(n=0, gold_pass=0, gold_fail=0, false_fail=0, false_pass=0)
    n_missing = 0
    for cid, g in gold.items():
        v = ver.get(cid)
        if v is None:
            n_missing += 1
            continue
        s = g["stratum"]
        gl = g["gold_label"]
        jv = "fail" if v.get("unscorable") else v["verdict"]
        for bucket in (out.setdefault(s, dict(n=0, gold_pass=0, gold_fail=0,
                                              false_fail=0, false_pass=0)), overall):
            bucket["n"] += 1
            if gl == "pass":
                bucket["gold_pass"] += 1
                if jv == "fail":
                    bucket["false_fail"] += 1
            else:
                bucket["gold_fail"] += 1
                if jv == "pass":
                    bucket["false_pass"] += 1
    out["_overall"] = overall
    out["_n_missing_join"] = n_missing
    return out


def _beta_params(counts: dict) -> dict:
    """Jeffreys Beta posteriors for a stratum's alpha (false-fail) and beta (false-pass).

    alpha_s ~ Beta(false_fail + 0.5, gold_pass - false_fail + 0.5)
    beta_s  ~ Beta(false_pass + 0.5, gold_fail - false_pass + 0.5)
    """
    a1 = counts["false_fail"] + 0.5
    a2 = counts["gold_pass"] - counts["false_fail"] + 0.5
    b1 = counts["false_pass"] + 0.5
    b2 = counts["gold_fail"] - counts["false_pass"] + 0.5
    return {"alpha_a": a1, "alpha_b": a2, "beta_a": b1, "beta_b": b2}


def strata_posteriors(gold_labels_path: Path, verdicts_path: Path) -> dict:
    """Per-stratum Beta posteriors (params + mean + 95% CI) plus the raw counts."""
    from scipy.stats import beta as Bdist

    counts = stratum_counts(gold_labels_path, verdicts_path)
    post = {}
    for s in STRATA:
        c = counts[s]
        p = _beta_params(c)
        a1, a2, b1, b2 = p["alpha_a"], p["alpha_b"], p["beta_a"], p["beta_b"]
        post[s] = {
            "counts": c,
            "alpha_a": a1, "alpha_b": a2, "beta_a": b1, "beta_b": b2,
            "alpha_mean": a1 / (a1 + a2),
            "alpha_ci95": [float(Bdist.ppf(0.025, a1, a2)), float(Bdist.ppf(0.975, a1, a2))],
            "beta_mean": b1 / (b1 + b2),
            "beta_ci95": [float(Bdist.ppf(0.025, b1, b2)), float(Bdist.ppf(0.975, b1, b2))],
        }
    post["_overall_counts"] = counts["_overall"]
    post["_n_missing_join"] = counts["_n_missing_join"]
    return post


def stratum_index(criterion_ids, crit_to_cap_path: Path) -> np.ndarray:
    """Map each criterion_id to its stratum index in STRATA (fallback: reasoning if unknown)."""
    import csv

    cap_of = {}
    with open(crit_to_cap_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cap_of[row["criterion_id"]] = row["capability"]
    sidx = {s: i for i, s in enumerate(STRATA)}
    out = np.empty(len(criterion_ids), dtype=int)
    n_unknown = 0
    for j, c in enumerate(criterion_ids):
        cap = cap_of.get(c)
        if cap in sidx:
            out[j] = sidx[cap]
        else:
            out[j] = sidx["reasoning"]
            n_unknown += 1
    if n_unknown:
        print(f"  [se_judge_lib] WARNING {n_unknown} criteria without a stratum "
              f"-> defaulted to 'reasoning'")
    return out


def sample_rates(post: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw (alpha_s, beta_s) for the 8 strata from their Beta posteriors (one per stratum)."""
    alpha = np.empty(len(STRATA))
    beta = np.empty(len(STRATA))
    for i, s in enumerate(STRATA):
        p = post[s]
        alpha[i] = rng.beta(p["alpha_a"], p["alpha_b"])
        beta[i] = rng.beta(p["beta_a"], p["beta_b"])
    return alpha, beta


def debias_pi(p_obs_c: np.ndarray, alpha_s: np.ndarray, beta_s: np.ndarray) -> np.ndarray:
    """Per-criterion single-iteration de-biased prior pass-rate.

    pi_c = clamp((p_obs,c - beta_s) / (1 - alpha_s - beta_s), 0, 1).
    ``alpha_s``/``beta_s`` are already broadcast to per-criterion (via the stratum index).
    """
    denom = np.clip(1.0 - alpha_s - beta_s, 1e-6, None)
    return np.clip((p_obs_c - beta_s) / denom, 0.0, 1.0)


def bayes_pass_probs(pi_c: np.ndarray, alpha_s: np.ndarray, beta_s: np.ndarray):
    """Bayes P(y=pass | y*) per criterion, for observed-pass and observed-fail cells.

    r_pass = P(y=pass | y*=pass) = pi(1-alpha) / [pi(1-alpha) + (1-pi)beta]
    r_fail = P(y=pass | y*=fail) = pi*alpha   / [pi*alpha   + (1-pi)(1-beta)]
    """
    num_p = pi_c * (1.0 - alpha_s)
    den_p = num_p + (1.0 - pi_c) * beta_s
    r_pass = np.where(den_p > 0, num_p / np.clip(den_p, 1e-12, None), pi_c)
    num_f = pi_c * alpha_s
    den_f = num_f + (1.0 - pi_c) * (1.0 - beta_s)
    r_fail = np.where(den_f > 0, num_f / np.clip(den_f, 1e-12, None), pi_c)
    return r_pass, r_fail


def resample_true_labels(Yobs: np.ndarray, Mobs: np.ndarray, r_pass: np.ndarray,
                         r_fail: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Correlated-within-criterion true-label resample (one latent uniform per column).

    ``Yobs`` (n_models, n_items) observed 0/1 labels; ``Mobs`` the observed mask. ``r_pass``/
    ``r_fail`` are per-item (length n_items) Bayes pass-probs. Draw one u_c ~ U(0,1) per
    column; a cell's resampled TRUE label is pass iff u_c <= r (r_pass for observed-pass
    cells, r_fail for observed-fail cells). Comonotone coupling => full within-column
    correlation, exact per-cell marginals. Missing cells stay 0 (and are masked out
    downstream by ``Mobs``).
    """
    n_items = Yobs.shape[1]
    u = rng.random(n_items)                       # one latent per criterion
    thr = np.where(Yobs > 0.5, r_pass[None, :], r_fail[None, :])
    Ytrue = (u[None, :] <= thr).astype(float)
    return np.where(Mobs, Ytrue, 0.0)
