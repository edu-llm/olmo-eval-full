"""SE_judge measurement-error primitives for the TutorEval gemini-3 recalibration (Phase 3).

LOCAL scratch library. It holds ONLY the judge-measurement-error layer; it never touches the
frozen production engine (``scenario_cat_lib``, ``calibrate_mirt``, the tutoreval OOS-grid
fold machinery) -- the driver imports those read-only. This module mirrors the BiGGen
``se_judge_lib.py`` (the most recent corrected version) with the TutorEval-specific confusion
model:

  * Per-stratum Beta posteriors for the false-fail rate alpha, fit Jeffreys-style from the
    100-cell human-gold 2x2 counts. TutorEval strata = ``primary_skill``. Only
    ``conceptual_understanding`` carries enough gold-PASS cells (15 >= 8 threshold) for its OWN
    alpha; ``quantitative_procedural`` (3 gold-PASS) and missing/``unknown`` POOL to the overall
    alpha. beta (false-pass) = 0 pooled across ALL strata (zero false-passes in 100 gold cells).
  * The single-de-bias per-criterion prior pass-rate pi_c.
  * The Bayes true-label posterior P(y | y*, alpha_s, beta_s=0, pi_c). With beta=0 an observed
    PASS is certainly true (the judge never wrongly passes); only observed FAILs can flip up.
  * The CORRELATED-within-criterion resample (one latent uniform per criterion column applied to
    all 52 models' cells) -- gemini-3 @ temp 0 mis-grades a whole criterion systematically, not
    independently per cell. The comonotone single-uniform coupling preserves each cell's Bayes
    marginal exactly while inducing full within-column correlation.

beta=0 nuance (vs BiGGen where beta>0): de-bias is directionally UP only -- observed theta /
pass-rate is a FLOOR. The alpha CI is wide ([0.20, 0.70]; only 20 gold-PASS cells), so the
de-bias magnitude band 1/(1-alpha) in [1.25x, 3.33x] is wide even though its direction is firm.

Judge stamp: gemini-3-flash-preview.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JUDGE = "gemini-3-flash-preview"

# TutorEval primary_skill stratum that carries its OWN alpha (>=8 gold-PASS cells); every other
# primary_skill (quantitative_procedural, missing/unknown) POOLS to the overall alpha.
OWN_ALPHA_STRATUM = "conceptual_understanding"

# Overall confusion CI (scenario-clustered) from tutoreval_gemini3_confusion.json, used for the
# systematic bias-band corners. beta is a hard 0 (no false-pass observed anywhere).
ALPHA_OVERALL = 0.45
BETA_OVERALL = 0.0
ALPHA_CI = (0.20, 0.70)
BETA_CI = (0.0, 0.0)


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_confusion(confusion_path: Path) -> dict:
    """Load the measured confusion json (authoritative per-stratum gold 2x2 counts)."""
    return json.loads(Path(confusion_path).read_text(encoding="utf-8"))


def _jeffreys_alpha_beta(gold_pass, false_fail, gold_fail, false_pass):
    """Jeffreys Beta posterior params for a stratum's alpha (false-fail) and beta (false-pass).

    alpha ~ Beta(false_fail + 0.5, gold_pass - false_fail + 0.5)
    beta  ~ Beta(false_pass + 0.5, gold_fail - false_pass + 0.5)
    (beta is reported for completeness but the TutorEval sampler pins beta=0 -- see notes.)
    """
    return {
        "alpha_a": false_fail + 0.5, "alpha_b": gold_pass - false_fail + 0.5,
        "beta_a": false_pass + 0.5, "beta_b": gold_fail - false_pass + 0.5,
    }


def alpha_posteriors(confusion: dict, own_alpha_min: int = 8) -> dict:
    """Beta-from-gold-counts alpha posteriors: the OWN-alpha stratum and the pooled OVERALL.

    Returns {"own": {stratum, params, mean}, "overall": {...}, "beta": 0.0, "meta": {...}}.
    beta is pinned to 0 (zero false-pass in the 100 gold cells; confusion beta CI = [0, 0]).
    """
    from scipy.stats import beta as Bdist

    ov = confusion["overall"]
    overall = _jeffreys_alpha_beta(ov["gold_pass"], ov["false_fail"],
                                   ov["gold_fail"], ov["false_pass"])
    by = confusion.get("by_stratum", {})
    own_c = by[OWN_ALPHA_STRATUM]
    own = _jeffreys_alpha_beta(own_c["gold_pass"], own_c["false_fail"],
                               own_c["gold_fail"], own_c["false_pass"])

    def _mean_ci(pa, pb):
        m = pa / (pa + pb)
        return m, [float(Bdist.ppf(0.025, pa, pb)), float(Bdist.ppf(0.975, pa, pb))]

    ov_mean, ov_ci = _mean_ci(overall["alpha_a"], overall["alpha_b"])
    own_mean, own_ci = _mean_ci(own["alpha_a"], own["alpha_b"])
    return {
        "overall": {"alpha_a": overall["alpha_a"], "alpha_b": overall["alpha_b"],
                    "alpha_mean": ov_mean, "alpha_ci95": ov_ci, "counts": ov},
        "own": {"stratum": OWN_ALPHA_STRATUM,
                "alpha_a": own["alpha_a"], "alpha_b": own["alpha_b"],
                "alpha_mean": own_mean, "alpha_ci95": own_ci, "counts": own_c},
        "beta": 0.0,
        "own_alpha_min_gold_pass": own_alpha_min,
        "own_alpha_supported": own_c["gold_pass"] >= own_alpha_min,
    }


def stratum_own_mask(primary_skills) -> np.ndarray:
    """Boolean per-criterion mask: True where the criterion uses the OWN (conceptual) alpha.

    Everything else (quantitative_procedural, missing/None, unknown) pools to the overall alpha.
    """
    return np.array([str(s) == OWN_ALPHA_STRATUM for s in primary_skills], dtype=bool)


def sample_alpha_beta(post: dict, own_mask: np.ndarray,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw per-criterion (alpha_c, beta_c) for one replicate.

    Two alpha draws per replicate (one OWN, one OVERALL) from their Beta posteriors, broadcast
    to criteria by ``own_mask``. beta_c is pinned to 0 for all criteria.
    """
    a_own = rng.beta(post["own"]["alpha_a"], post["own"]["alpha_b"])
    a_ov = rng.beta(post["overall"]["alpha_a"], post["overall"]["alpha_b"])
    alpha_c = np.where(own_mask, a_own, a_ov)
    beta_c = np.zeros(own_mask.shape[0])
    return alpha_c, beta_c


def debias_pi(p_obs_c: np.ndarray, alpha_c: np.ndarray, beta_c: np.ndarray) -> np.ndarray:
    """Per-criterion single-iteration de-biased prior pass-rate.

    pi_c = clamp((p_obs,c - beta_c) / (1 - alpha_c - beta_c), 0, 1). With beta=0 this reduces to
    clamp(p_obs,c / (1 - alpha_c), 0, 1).
    """
    denom = np.clip(1.0 - alpha_c - beta_c, 1e-6, None)
    return np.clip((p_obs_c - beta_c) / denom, 0.0, 1.0)


def bayes_pass_probs(pi_c: np.ndarray, alpha_c: np.ndarray, beta_c: np.ndarray):
    """Bayes P(y=pass | y*) per criterion, for observed-pass and observed-fail cells.

    r_pass = P(y=pass | y*=pass) = pi(1-alpha) / [pi(1-alpha) + (1-pi)beta]
    r_fail = P(y=pass | y*=fail) = pi*alpha   / [pi*alpha   + (1-pi)(1-beta)]
    With beta=0: r_pass = 1 wherever pi>0 (an observed pass is certainly true), and r_fail =
    pi*alpha / [pi*alpha + (1-pi)] (only observed FAILs can flip UP to pass).
    """
    num_p = pi_c * (1.0 - alpha_c)
    den_p = num_p + (1.0 - pi_c) * beta_c
    r_pass = np.where(den_p > 0, num_p / np.clip(den_p, 1e-12, None), pi_c)
    num_f = pi_c * alpha_c
    den_f = num_f + (1.0 - pi_c) * (1.0 - beta_c)
    r_fail = np.where(den_f > 0, num_f / np.clip(den_f, 1e-12, None), pi_c)
    return r_pass, r_fail


def resample_true_labels(Yobs: np.ndarray, Mobs: np.ndarray, r_pass: np.ndarray,
                         r_fail: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Correlated-within-criterion true-label resample (one latent uniform per column).

    ``Yobs`` (n_models, n_items) observed 0/1 labels; ``Mobs`` the observed mask. ``r_pass``/
    ``r_fail`` are per-item (length n_items) Bayes pass-probs. Draw one u_c ~ U(0,1) per column;
    a cell's resampled TRUE label is pass iff u_c <= r (r_pass for observed-pass cells, r_fail
    for observed-fail cells). Comonotone coupling => full within-column correlation, exact
    per-cell marginals. Missing cells stay 0 (masked out downstream by ``Mobs``).
    """
    n_items = Yobs.shape[1]
    u = rng.random(n_items)                       # one latent per criterion
    thr = np.where(Yobs > 0.5, r_pass[None, :], r_fail[None, :])
    Ytrue = (u[None, :] <= thr).astype(float)
    return np.where(Mobs, Ytrue, 0.0)
