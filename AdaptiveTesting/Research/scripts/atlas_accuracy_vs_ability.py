#!/usr/bin/env python3
"""Locate & validate the ATLAS "same accuracy, different ability" phenomenon.

ATLAS's core claim is that raw accuracy throws away information that IRT ability
(theta) retains: two models can score the *same* fraction correct yet differ in
ability, because one solves the harder / more discriminating items while the
other coasts on the easy ones. This script surfaces concrete OpenLM examples of
that and then shows the ability gap is real signal (it predicts held-out
performance), not calibration noise.

It reuses the frozen ATLAS-style 3PL banks and the exact conventions from
``atlas_error_plots_openlm.py``:
  - ``load_bank``  applies the a>0 filter and converts (d, g) -> (a, b=-d/a, c=g)
  - ``load_matrix`` reads a model x item response matrix
  - ``eap_se``     computes the full-bank EAP theta (Gaussian prior, 81 nodes)

For each benchmark we build the full model x item matrix (train + test rows, the
same a>0 item columns) and, for every non-degenerate model, compute
  (1) full-bank accuracy  = mean correctness over the a>0 items
  (2) full-bank EAP theta = ability over those items.

Two passes:
  PASS 1 (examples, descriptive): find model pairs with |accuracy diff| <= tol
    and rank by |theta gap|; for each we confirm the mechanism (the higher-theta
    model got harder / more discriminating items right) and attach a held-out
    accuracy gap on a random half of the bank.
  PASS 2 (validation, anti-noise): split the a>0 bank into halves A/B (seed 7);
    among all pairs with near-equal *half-A* accuracy, test whether the higher
    half-A-theta model scores higher on the untouched half B. If ability
    predicts held-out performance at fixed accuracy, that is the evidence.

Usage:
  uv run python AdaptiveTesting/Research/scripts/atlas_accuracy_vs_ability.py \
      --benches ifeval,math,gpqa,musr,bbh --acc-tol 0.005 --seed 7

Outputs under
  AdaptiveTesting/Research/01_MCQ_ATLAS/data/accuracy_vs_ability/
    <bench>_examples.csv           (model_a, model_b, accuracy, theta_a, theta_b,
                                     theta_gap, heldoutB_gap, + mechanism cols)
    validation_summary.csv         (per-bench held-out predictiveness of theta)
    README.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the ATLAS OpenLM helpers verbatim (a>0 filter, (d,g)->(a,b,c), EAP theta).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_error_plots_openlm import (  # noqa: E402
    BENCH_PATHS,
    NODES,
    PRIORW,
    eap_se,
    load_bank,
    load_matrix,
)

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/accuracy_vs_ability"


def eap_theta_se(R: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Vectorized EAP theta + posterior SD for every row of R (models x items).

    Numerically identical to calling ``eap_se`` per row (same nodes, prior,
    clipping); batched via BLAS so the multi-thousand-item bbh bank stays fast.
    Returns ``(theta, se)`` arrays, each shape (M,).
    """
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)  # (K, n)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    logp = np.log(p)  # (K, n)
    log1mp = np.log(1 - p)  # (K, n)
    ll = R @ logp.T + (1.0 - R) @ log1mp.T  # (M, K)
    ll -= ll.max(axis=1, keepdims=True)
    w = np.exp(ll) * PRIORW[None, :]  # (M, K)
    w /= w.sum(axis=1, keepdims=True)
    theta = w @ NODES  # (M,)
    var = (w * (NODES[None, :] - theta[:, None]) ** 2).sum(axis=1)
    return theta, np.sqrt(var)


def load_full_matrix(bench: str) -> tuple[Path, pd.DataFrame]:
    """Combined train+test response matrix (all models, shared item columns)."""
    bank_csv, test_csv = BENCH_PATHS[bench]
    train_csv = test_csv.parent / test_csv.name.replace("test", "train")
    dtest = load_matrix(test_csv)
    dtrain = load_matrix(train_csv)
    if list(dtrain.columns) != list(dtest.columns):
        common = [c for c in dtrain.columns if c in set(dtest.columns)]
        dtrain, dtest = dtrain[common], dtest[common]
    full = pd.concat([dtrain, dtest], axis=0)
    full = full[~full.index.duplicated(keep="first")]
    return bank_csv, full


def near_equal_pairs(acc: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """All index pairs (i, j) with |acc[i]-acc[j]| <= tol via a sorted window."""
    order = np.argsort(acc, kind="mergesort")
    sa = acc[order]
    n = len(order)
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for k in range(i + 1, n):
            if sa[k] - sa[i] > tol:
                break
            pairs.append((int(order[i]), int(order[k])))
    return pairs


def mean_ab_correct(row: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    m = row > 0.5
    if not m.any():
        return float("nan"), float("nan")
    return float(a[m].mean()), float(b[m].mean())


def run_bench(bench: str, acc_tol: float, seed: int, n_examples: int) -> dict:
    bank_csv, full = load_full_matrix(bench)
    idxs, a, b, c = load_bank(bank_csv, list(full.columns))
    R_all = full[idxs].to_numpy(float)  # (M, n) over a>0 items
    models = np.array(full.index)
    n_items = len(idxs)

    acc = R_all.mean(axis=1)
    theta, se = eap_theta_se(R_all, a, b, c)
    # faithfulness: the batched EAP equals the reference per-row eap_se helper.
    for m in range(min(3, len(R_all))):
        ref_t, ref_s = eap_se(R_all[m], a, b, c)
        assert abs(ref_t - theta[m]) < 1e-9 and abs(ref_s - se[m]) < 1e-9

    # --- degeneracy filter: drop at/below-chance and near-constant response rows.
    guess_floor = float(c.mean())  # theta -> -inf gives mean(P)=mean(c)
    minority = np.minimum(R_all.sum(axis=1), (1 - R_all).sum(axis=1))
    keep = (acc > guess_floor) & (acc < 0.995) & (minority >= 5)
    n_drop = int((~keep).sum())

    R = R_all[keep]
    models = models[keep]
    acc = acc[keep]
    theta = theta[keep]
    se = se[keep]
    M = len(models)

    # --- seed-7 random split of the a>0 items into halves A / B (held out).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    half = n_items // 2
    A, B = perm[:half], perm[half:]
    aA, bA, cA = a[A], b[A], c[A]
    accA = R[:, A].mean(axis=1)
    accB = R[:, B].mean(axis=1)
    thetaA, _ = eap_theta_se(R[:, A], aA, bA, cA)

    # per-model mean discrimination / difficulty of correctly-answered items
    mean_a_corr = np.array([mean_ab_correct(R[m], a, b)[0] for m in range(M)])
    mean_b_corr = np.array([mean_ab_correct(R[m], a, b)[1] for m in range(M)])

    # ------------------------------------------------------------------ PASS 1
    # Illustrative examples: near-equal full-bank accuracy pairs, restricted to a
    # meaningful, well-identified region (accuracy in [p10, p90] of kept models)
    # with a theta gap that exceeds ~2 SD of its EAP measurement error (z >= 2)
    # so the gap is not calibration noise, and where the mechanism holds -- the
    # higher-theta model's correct answers sit on more discriminating and/or
    # harder items. Rank by |theta gap|; keep distinct models. Held-out gaps are
    # reported as-is (never selected on).
    lo_band, hi_band = np.percentile(acc, [10, 90])
    ranked = []
    for i, j in near_equal_pairs(acc, acc_tol):
        if not (lo_band <= acc[i] <= hi_band and lo_band <= acc[j] <= hi_band):
            continue
        gap = abs(theta[i] - theta[j])
        if gap / np.sqrt(se[i] ** 2 + se[j] ** 2 + 1e-12) < 2.0:
            continue
        hi, lo = (i, j) if theta[i] >= theta[j] else (j, i)
        if not (mean_a_corr[hi] > mean_a_corr[lo] or mean_b_corr[hi] > mean_b_corr[lo]):
            continue
        ranked.append((hi, lo, gap))
    ranked.sort(key=lambda t: -t[2])

    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for hi, lo, _ in ranked:
        if hi in used or lo in used:
            continue
        selected.append((hi, lo))
        used.update((hi, lo))
        if len(selected) >= n_examples:
            break

    rows = []
    for hi, lo in selected:  # hi = higher-theta model = "a"
        disc_gap = mean_a_corr[hi] - mean_a_corr[lo]
        diff_gap = mean_b_corr[hi] - mean_b_corr[lo]
        rows.append(
            {
                "model_a": models[hi],
                "model_b": models[lo],
                "accuracy": (acc[hi] + acc[lo]) / 2.0,
                "theta_a": theta[hi],
                "theta_b": theta[lo],
                "theta_gap": theta[hi] - theta[lo],
                "heldoutB_gap": accB[hi] - accB[lo],
                "se_a": se[hi],
                "se_b": se[lo],
                "z_gap": (theta[hi] - theta[lo]) / np.sqrt(se[hi] ** 2 + se[lo] ** 2 + 1e-12),
                "acc_a": acc[hi],
                "acc_b": acc[lo],
                "acc_diff": abs(acc[hi] - acc[lo]),
                "n_items": n_items,
                "n_correct_a": int(round(acc[hi] * n_items)),
                "n_correct_b": int(round(acc[lo] * n_items)),
                "mean_disc_correct_a": mean_a_corr[hi],
                "mean_disc_correct_b": mean_a_corr[lo],
                "mean_diff_correct_a": mean_b_corr[hi],
                "mean_diff_correct_b": mean_b_corr[lo],
                "disc_correct_gap": disc_gap,
                "diff_correct_gap": diff_gap,
                "mechanism_confirmed": bool(disc_gap > 0 or diff_gap > 0),
            }
        )
    ex_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ PASS 2
    # Held-out validation: among pairs with near-equal HALF-A accuracy, does the
    # higher half-A-theta model win on the untouched half B? Half A determines
    # both the accuracy match and the ability predictor; half B is never seen.
    band_lo, band_hi = np.percentile(accA, [10, 90])
    in_band = (accA >= band_lo) & (accA <= band_hi)

    def validate(tol: float, mask: np.ndarray | None = None) -> dict:
        wins = losses = ties = 0
        signed_gaps = []
        for i, j in near_equal_pairs(accA, tol):
            if mask is not None and not (mask[i] and mask[j]):
                continue
            dtheta = thetaA[i] - thetaA[j]
            if dtheta == 0:
                continue
            hi, lo = (i, j) if dtheta > 0 else (j, i)
            gap = accB[hi] - accB[lo]
            signed_gaps.append(gap)
            if gap > 0:
                wins += 1
            elif gap < 0:
                losses += 1
            else:
                ties += 1
        decided = wins + losses
        return {
            "n_pairs": wins + losses + ties,
            "n_decided": decided,
            "frac_higher_theta_wins": (wins / decided) if decided else float("nan"),
            "mean_heldoutB_gap": float(np.mean(signed_gaps)) if signed_gaps else float("nan"),
        }

    v_tol = validate(acc_tol)
    v_exact = validate(0.0)
    v_band = validate(acc_tol, in_band)

    summary = {
        "benchmark": bench,
        "n_models": M,
        "n_items": n_items,
        "guess_floor": guess_floor,
        "n_dropped_degenerate": n_drop,
        "acc_tol": acc_tol,
        "n_pairs_tol": v_tol["n_pairs"],
        "n_decided_tol": v_tol["n_decided"],
        "frac_higher_theta_wins_tol": v_tol["frac_higher_theta_wins"],
        "mean_heldoutB_gap_tol": v_tol["mean_heldoutB_gap"],
        "n_pairs_exact": v_exact["n_pairs"],
        "n_decided_exact": v_exact["n_decided"],
        "frac_higher_theta_wins_exact": v_exact["frac_higher_theta_wins"],
        "mean_heldoutB_gap_exact": v_exact["mean_heldoutB_gap"],
        "n_pairs_midband": v_band["n_pairs"],
        "frac_higher_theta_wins_midband": v_band["frac_higher_theta_wins"],
        "mean_heldoutB_gap_midband": v_band["mean_heldoutB_gap"],
    }

    print(
        f"[{bench}] M={M} items={n_items} floor={guess_floor:.3f} dropped={n_drop} "
        f"examples={len(ex_df)} | "
        f"|dAcc|<={acc_tol}: pairs={v_tol['n_pairs']} "
        f"win={v_tol['frac_higher_theta_wins']:.3f} "
        f"heldoutB_gap={v_tol['mean_heldoutB_gap']:+.4f} | "
        f"exact: pairs={v_exact['n_pairs']} win={v_exact['frac_higher_theta_wins']:.3f} "
        f"gap={v_exact['mean_heldoutB_gap']:+.4f} | "
        f"midband: win={v_band['frac_higher_theta_wins']:.3f} "
        f"gap={v_band['mean_heldoutB_gap']:+.4f}",
        flush=True,
    )
    return {"examples": ex_df, "summary": summary}


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional ``tabulate`` dependency)."""

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([head, sep, *body])


def write_readme(out_dir: Path, summary_df: pd.DataFrame, acc_tol: float, seed: int) -> None:
    lines = [
        "# ATLAS accuracy-vs-ability examples (OpenLM)",
        "",
        "Concrete cases where two models have **near-identical full-bank accuracy** but a",
        "**meaningful gap in IRT ability** (full-bank EAP theta), plus a held-out test that",
        "the ability gap is real signal rather than calibration noise.",
        "",
        "## Method",
        "",
        "- Banks: frozen ATLAS-style 3PL `irt_item_parameters_combined.csv` per benchmark;",
        "  items with `a1 <= 0` dropped (`load_bank`), `(d, g)` -> `(a, b=-d/a, c=g)`.",
        "- Response matrix: train + test rows stacked over the shared a>0 item columns.",
        "- Per model: full-bank accuracy = mean correctness; ability = full-bank EAP theta",
        "  (`eap_se`, Gaussian prior, 81 nodes) -- identical conventions to",
        "  `atlas_error_plots_openlm.py`.",
        "- Degenerate models removed: accuracy at/below the bank guessing floor `mean(c)`,",
        "  accuracy >= 0.995, or near-constant response vectors (minority count < 5).",
        "",
        "## Files",
        "",
        "- `<bench>_examples.csv` -- curated illustrative pairs. Candidates are pairs with",
        f"  |accuracy diff| <= {acc_tol}, both models in the central 10-90 percentile of",
        "  accuracy (well-identified region), a theta gap exceeding ~2x its combined EAP",
        "  posterior SD (`z_gap >= 2`, i.e. the gap is not measurement noise), and the",
        "  mechanism holding (the higher-theta model's correct answers sit on more",
        "  discriminating and/or harder items). Pairs are ranked by |theta gap| with",
        "  distinct models. `model_a` is the higher-theta model, so `theta_gap > 0`.",
        f"  `heldoutB_gap` = model_a - model_b accuracy on a random held-out half (seed {seed})",
        "  of the a>0 items -- reported as-is, never selected on; positive means the",
        "  higher-ability model also scores higher out of sample. `mean_disc_correct_*` /",
        "  `mean_diff_correct_*` are the mean discrimination (a) / difficulty (b) of each",
        "  model's correctly-answered items; `mechanism_confirmed` is True when the",
        "  higher-theta model's correct items are more discriminating and/or harder.",
        "- `validation_summary.csv` -- per-benchmark held-out predictiveness of theta.",
        "",
        "## Held-out validation (Pass 2)",
        "",
        f"Split the a>0 bank into halves A/B (random, seed {seed}). Among **all** pairs with",
        f"near-equal **half-A** accuracy (|diff| <= {acc_tol}; and the exact-tie subset),",
        "the higher half-A-theta model is predicted to score higher on the untouched half",
        "B. Half A sets both the accuracy match and the ability predictor; half B is never",
        "seen, so this is a genuine train/test split of items (no filtering on outcome).",
        "`frac_higher_theta_wins` is that fraction among decided (non-tied) pairs",
        "(chance = 0.50); `mean_heldoutB_gap` is the mean half-B accuracy advantage of the",
        "higher-theta model. Values > 0.50 and > 0 mean ability predicts held-out",
        "performance at fixed accuracy.",
        "",
        "### Summary",
        "",
        "`_tol` = pairs within |accuracy diff| <= acc_tol; `_exact` = exactly-equal",
        "accuracy; `_midband` = the |diff|<=tol pairs restricted to the central 10-90",
        "percentile of half-A accuracy (where theta is best identified).",
        "",
        df_to_markdown(
            summary_df[
                [
                    "benchmark",
                    "n_models",
                    "n_items",
                    "guess_floor",
                    "frac_higher_theta_wins_tol",
                    "mean_heldoutB_gap_tol",
                    "frac_higher_theta_wins_exact",
                    "mean_heldoutB_gap_exact",
                    "frac_higher_theta_wins_midband",
                    "mean_heldoutB_gap_midband",
                ]
            ]
        ),
        "",
        "### Findings",
        "",
    ]
    for _, r in summary_df.iterrows():
        f = r["frac_higher_theta_wins_tol"]
        g = r["mean_heldoutB_gap_tol"]
        tag = "STRONG" if f >= 0.58 else "MODEST" if f >= 0.53 else "WEAK/ABSENT"
        lines.append(
            f"- **{r['benchmark']}** ({tag}): higher-theta model wins held-out half B in "
            f"{f:.1%} of equal-accuracy pairs (mean held-out gap {g:+.4f})."
        )
    lines += [
        "",
        "Caveats: on `bbh` the bank is very large (~4k a>0 items) so the fixed 81-node EAP",
        "grid resolves theta only to ~0.1 steps; its near-equal-accuracy theta gaps are",
        "coarse and its held-out effect is modest. On `math` and `gpqa` most models sit near",
        "the accuracy/guessing floor, so theta gaps at fixed accuracy carry little held-out",
        "signal (gpqa is at chance; its largest in-sample gaps even reverse out of sample --",
        "the reason the held-out test matters).",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benches", default="ifeval,math,gpqa,musr,bbh")
    ap.add_argument("--acc-tol", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]

    summaries = []
    for bench in benches:
        if bench not in BENCH_PATHS:
            print(f"[{bench}] unknown, skipping", flush=True)
            continue
        res = run_bench(bench, args.acc_tol, args.seed, args.n_examples)
        res["examples"].to_csv(args.out_dir / f"{bench}_examples.csv", index=False)
        summaries.append(res["summary"])

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.out_dir / "validation_summary.csv", index=False)
    write_readme(args.out_dir, summary_df, args.acc_tol, args.seed)
    print(f"\nwrote examples + validation_summary.csv + README.md under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
