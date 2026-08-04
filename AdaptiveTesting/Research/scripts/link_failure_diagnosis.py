#!/usr/bin/env python3
"""Diagnose the ATLAS-replication link results on OpenLM benchmarks.

Corrected framing: the apparent gpqa/musr "link failure" was a BAD-ITEM artifact.
The 3PL banks (mirt fits) contain many non-positive-discrimination items (a<=0,
fit failures / polarity flips). ATLAS's canonical loader drops them; leaving them
in corrupts the latent-trait estimate and collapses theta-vs-score correlation.
After the drop, gpqa/musr link fine (r~0.74/0.75). bbh is the genuinely weak case.

This script:
  1. Quantifies the item-quality mechanism per benchmark: %a<=0, %0<a<0.2,
     discrimination distribution before/after filtering, and the KEY comparison --
     full-bank EAP theta vs actual score correlation WITH vs WITHOUT the a<=0
     filter (unfiltered should collapse, filtered should recover).
  2. Diagnoses bbh: score variance, error-vs-actual bias/shrinkage (from the
     replication's own CAT output), and item p-value distribution.

Bank/matrix loading + EAP conventions reuse the sibling replication script
``atlas_error_plots_openlm.py`` (which already implements the a<=0 filter).

Usage:
  export PYTHONPATH=/Users/arhant/Documents/EDLM/olmo-eval-full/eduLLM-Evals
  uv run python AdaptiveTesting/Research/scripts/link_failure_diagnosis.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "AdaptiveTesting/Experiments"
RESEARCH = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS"
REPLICATION = RESEARCH / "data/atlas_replication"
OUT_DIR = RESEARCH / "data/link_failure_diagnosis"
# pre-filter (a<=0 kept) CAT results, produced before the corrected loader
OLD_SUMMARY = RESEARCH / "data/openlm_atlas3pl_results_summary.csv"
NEW_SUMMARY = REPLICATION / "summary_pirt_mae_sd_se.csv"

os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplcache"))

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()

CHANCE = {"bbh": None, "gpqa": 0.25, "math": None, "musr": 0.5, "ifeval": None}

# bench -> (combined bank csv, held-out test matrix csv)  [same as replication]
BENCH_PATHS = {
    "math": (EXP / "openlm_atlas_3pl/math/calibration/irt_item_parameters_combined.csv",
             EXP / "openlm_atlas_3pl/math/data/response_matrix_test.csv"),
    "bbh": (EXP / "openlm_atlas_3pl/bbh/calibration/irt_item_parameters_combined.csv",
            EXP / "openlm_atlas_3pl/bbh/data/response_matrix_test.csv"),
    "musr": (EXP / "openlm_atlas_3pl/musr/calibration/irt_item_parameters_combined.csv",
             EXP / "openlm_atlas_3pl/musr/data/response_matrix_test.csv"),
    "gpqa": (EXP / "openlm_gpqa_atlas_3pl/calibration/irt_item_parameters_combined.csv",
             EXP / "openlm_gpqa_atlas_3pl/data/gpqa_response_matrix_test.csv"),
    "ifeval": (EXP / "openlm_atlas_3pl/ifeval/calibration/irt_item_parameters_combined.csv",
               EXP / "openlm_atlas_3pl/ifeval/data/response_matrix_test.csv"),
}

BENCHES = ["math", "bbh", "gpqa", "musr", "ifeval"]


def eap_theta(resp, a, b, c):
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    return float((NODES * w).sum())


def load_bank_raw(bank_csv: Path):
    """All fitted items (before any a<=0 drop): a, b, c keyed by item index."""
    pdf = pd.read_csv(bank_csv)
    name_col = pdf.columns[0]
    raw = {}
    for _, row in pdf.iterrows():
        key = str(row[name_col]).lstrip("X")
        try:
            idx = int(float(key))
        except ValueError:
            continue
        a = float(row["a1"])
        d = float(row["d"])
        g = float(row["g"]) if "g" in pdf.columns else 0.0
        b = -d / a if (np.isfinite(a) and a != 0) else np.nan
        raw[idx] = (a, b, float(np.clip(g, 0.0, 0.999)))
    return raw


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "model"}).set_index("model")
    df.columns = [int(c) for c in df.columns]
    return df.astype(float)


def theta_vec(resp_mat, a, b, c):
    return np.array([eap_theta(resp_mat[i], a, b, c) for i in range(resp_mat.shape[0])])


def load_cat_r(summary_csv: Path, se: float = 0.3) -> dict:
    """Read Pearson r per benchmark from an atlas-style CAT summary csv."""
    if not summary_csv.exists():
        return {}
    df = pd.read_csv(summary_csv)
    out = {}
    if "se_target" in df.columns:  # replication summary (long form)
        sub = df[np.isclose(df["se_target"], se)]
        for _, r in sub.iterrows():
            out[str(r["benchmark"])] = float(r["r"])
    else:  # old openlm summary (one row per bench, SE<=0.3)
        for _, r in df.iterrows():
            out[str(r["benchmark"])] = float(r["r"])
    return out


def diagnose(bench: str, cat_new: dict, cat_old: dict) -> dict:
    bank_csv, test_csv = BENCH_PATHS[bench]
    mat = load_matrix(test_csv)
    raw = load_bank_raw(bank_csv)
    models = list(mat.index)

    # items present in both bank and matrix with finite (a, b)
    in_bank = [i for i in mat.columns
               if i in raw and np.isfinite(raw[i][0]) and np.isfinite(raw[i][1])]
    a_all = np.array([raw[i][0] for i in in_bank], float)

    # ground-truth target: full-benchmark accuracy over ALL usable bank items
    # (filter-INDEPENDENT, so unfiltered vs filtered theta are judged on one target)
    resp_full = mat[in_bank].to_numpy(float)
    actual = np.nanmean(resp_full, axis=1)

    # UNFILTERED estimation set: all in-bank items (a of any sign)
    au = a_all
    bu = np.array([raw[i][1] for i in in_bank], float)
    cu = np.array([raw[i][2] for i in in_bank], float)
    theta_unf = theta_vec(resp_full, au, bu, cu)

    # FILTERED estimation set: a>0 (ATLAS canonical loader = replication behaviour)
    filt = [i for i in in_bank if raw[i][0] > 0]
    af = np.array([raw[i][0] for i in filt], float)
    bf = np.array([raw[i][1] for i in filt], float)
    cf = np.array([raw[i][2] for i in filt], float)
    resp_filt = mat[filt].to_numpy(float)
    theta_filt = theta_vec(resp_filt, af, bf, cf)

    r_unf = float(np.corrcoef(theta_unf, actual)[0, 1])
    r_filt = float(np.corrcoef(theta_filt, actual)[0, 1])

    n_fit = len(a_all)
    n_le0 = int(np.sum(a_all <= 0))
    n_0_02 = int(np.sum((a_all > 0) & (a_all < 0.2)))

    scored_scores = np.nanmean(resp_filt, axis=1)  # per-model acc on scored items
    pvals = np.nanmean(resp_filt, axis=0)           # per-item p-value (scored items)

    # CAT diagnostics from the (post-filter) replication per-bench output
    cat = {"mean_items": np.nan, "mae": np.nan, "mean_error": np.nan,
           "pred_sd": np.nan, "actual_sd": np.nan, "slope_pred_actual": np.nan,
           "theta_mae": np.nan}
    rep_csv = REPLICATION / bench / "pirt_vs_actual_se_0.3.csv"
    if rep_csv.exists():
        rep = pd.read_csv(rep_csv)
        act = rep["actual_accuracy"].to_numpy()
        pir = rep["pirt_accuracy"].to_numpy()
        err = rep["error"].to_numpy()
        cat["mean_items"] = float(rep["n_subset_items"].mean())
        cat["mae"] = float(np.abs(err).mean())
        cat["mean_error"] = float(err.mean())
        cat["pred_sd"] = float(pir.std())
        cat["actual_sd"] = float(act.std())
        if np.ptp(act) > 0:
            cat["slope_pred_actual"] = float(np.polyfit(act, pir, 1)[0])
    trec = REPLICATION / bench / "theta_recovery_se_0.3.csv"
    if trec.exists():
        cat["theta_mae"] = float(pd.read_csv(trec)["theta_abs_error"].mean())

    row = {
        "benchmark": bench,
        "n_models": len(models),
        "n_items_fit": n_fit,
        "n_items_a_le0": n_le0,
        "pct_a_le0": 100.0 * n_le0 / max(n_fit, 1),
        "n_items_0_a_0p2": n_0_02,
        "pct_0_a_0p2": 100.0 * n_0_02 / max(n_fit, 1),
        "n_items_scored_filtered": len(filt),
        "a_median_unfiltered": float(np.median(au)),
        "a_median_filtered": float(np.median(af)),
        "a_iqr_filtered": float(np.percentile(af, 75) - np.percentile(af, 25)),
        # KEY mechanism metric: same full-benchmark target, only bank filter differs
        "r_theta_score_unfiltered": r_unf,
        "r_theta_score_filtered": r_filt,
        # headline CAT correlations from the pipelines themselves
        "cat_r_prefilter_old": cat_old.get(bench, np.nan),
        "cat_r_postfilter_new": cat_new.get(bench, np.nan),
        "score_mean_full": float(np.mean(actual)),
        "score_sd_full": float(np.std(actual, ddof=1)),
        "score_sd_scored": float(np.std(scored_scores, ddof=1)),
        "score_range_scored": float(np.ptp(scored_scores)),
        "chance": CHANCE[bench],
        "pval_mean": float(np.mean(pvals)),
        "pval_sd": float(np.std(pvals, ddof=1)),
        "cat_mean_items": cat["mean_items"],
        "cat_mae": cat["mae"],
        "cat_mean_error": cat["mean_error"],
        "cat_pred_sd": cat["pred_sd"],
        "cat_actual_sd": cat["actual_sd"],
        "cat_slope_pred_actual": cat["slope_pred_actual"],
        "cat_theta_mae": cat["theta_mae"],
    }
    return {"row": row, "actual": actual, "theta_unf": theta_unf,
            "theta_filt": theta_filt, "a_all": a_all, "a_filt": af,
            "pvals": pvals, "rep_csv": rep_csv, "r_full_theta": r_filt}


def fig_mechanism(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benches = [b for b in ["math", "bbh", "gpqa", "musr", "ifeval"] if b in results]
    colors = {"math": "#2166ac", "bbh": "#4daf4a", "gpqa": "#d73027",
              "musr": "#984ea3", "ifeval": "#ff7f00"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (A) theta-score r: unfiltered vs filtered
    ax = axes[0, 0]
    x = np.arange(len(benches))
    r_unf = [results[b]["row"]["r_theta_score_unfiltered"] for b in benches]
    r_filt = [results[b]["row"]["r_theta_score_filtered"] for b in benches]
    ax.bar(x - 0.2, r_unf, 0.4, label="unfiltered (a<=0 kept)", color="#bbbbbb")
    ax.bar(x + 0.2, r_filt, 0.4, label="filtered (a>0 only)",
           color=[colors[b] for b in benches])
    for xi, ru, rf in zip(x, r_unf, r_filt):
        ax.text(xi - 0.2, ru + 0.02 * np.sign(ru or 1), f"{ru:.2f}",
                ha="center", va="bottom" if ru >= 0 else "top", fontsize=8)
        ax.text(xi + 0.2, rf + 0.02, f"{rf:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(benches)
    ax.set_ylabel("Pearson r(full-bank theta, actual score)")
    ax.set_title("(A) theta-accuracy r drops when a<=0 items are kept\n"
                 "cause is bad items, not the benchmark")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (B) %a<=0 per bench
    ax = axes[0, 1]
    pcts = [results[b]["row"]["pct_a_le0"] for b in benches]
    p02 = [results[b]["row"]["pct_0_a_0p2"] for b in benches]
    ax.bar(x, pcts, 0.6, color=[colors[b] for b in benches], label="a<=0")
    ax.bar(x, p02, 0.6, bottom=pcts, color="none", edgecolor="black", hatch="//",
           label="0<a<0.2")
    for xi, p in zip(x, pcts):
        ax.text(xi, p + 1, f"{p:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(benches)
    ax.set_ylabel("% of fitted bank items")
    ax.set_title("(B) Fraction of items with non-positive or near-zero a")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (C) discrimination distribution before vs after (gpqa + musr overlaid)
    ax = axes[1, 0]
    for b in ["gpqa", "musr"]:
        if b not in results:
            continue
        aall = results[b]["a_all"]
        ax.hist(np.clip(aall, -8, 8), bins=48, range=(-8, 8), histtype="step",
                lw=2, color=colors[b], density=True, label=f"{b} unfiltered")
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.set_xlabel("item discrimination a (clipped to [-8,8])")
    ax.set_ylabel("density")
    ax.set_title("(C) Unfiltered discrimination distribution (red line at a=0)\n"
                 "gpqa and musr have about half their items at a<=0")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (D) unfiltered vs filtered theta scatter vs actual for gpqa (illustrative)
    ax = axes[1, 1]
    b = "gpqa"
    act = results[b]["actual"]
    tu = results[b]["theta_unf"]
    tf = results[b]["theta_filt"]
    ax.scatter(tu, act, s=20, alpha=0.5, color="#bbbbbb",
               label=f"unfiltered theta (r={results[b]['row']['r_theta_score_unfiltered']:.2f})")
    ax.scatter(tf, act, s=20, alpha=0.6, color=colors[b],
               label=f"filtered theta (r={results[b]['row']['r_theta_score_filtered']:.2f})")
    ax.set_xlabel("full-bank EAP theta")
    ax.set_ylabel("actual gpqa score")
    ax.set_title("(D) gpqa: dropping a<=0 items raises theta-score r")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Why gpqa and musr showed low link r: a<=0 items in the bank",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "filter_mechanism.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def fig_bbh(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"math": "#2166ac", "bbh": "#4daf4a", "gpqa": "#d73027",
              "musr": "#984ea3"}
    benches = [b for b in ["math", "bbh", "gpqa", "musr"] if b in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (1) score variance across models (bbh has plenty of spread -> not the issue)
    ax = axes[0, 0]
    data = [results[b]["actual"] for b in benches]
    bp = ax.boxplot(data, tick_labels=benches, showmeans=True, patch_artist=True)
    for patch, b in zip(bp["boxes"], benches):
        patch.set_facecolor(colors[b])
        patch.set_alpha(0.5)
    for i, b in enumerate(benches, 1):
        ax.text(i, results[b]["actual"].max() + 0.01,
                f"SD={results[b]['row']['score_sd_full']:.3f}",
                ha="center", fontsize=8)
    ax.set_ylabel("per-model full-benchmark score")
    ax.set_title("(1) Score spread across models\n"
                 "bbh spread is wide, so low variance is not the cause")
    ax.grid(True, axis="y", alpha=0.3)

    # (2) early-stop under-sampling: full-bank theta-r vs CAT r, + mean #items
    ax = axes[0, 1]
    x = np.arange(len(benches))
    r_full = [results[b]["row"]["r_theta_score_filtered"] for b in benches]
    r_cat = [results[b]["row"]["cat_r_postfilter_new"] for b in benches]
    ax.bar(x - 0.2, r_full, 0.4, color="#377eb8", label="full-bank theta r")
    ax.bar(x + 0.2, r_cat, 0.4, color="#e41a1c", label="CAT r (SE<=0.3)")
    for xi, b in zip(x, benches):
        ni = results[b]["row"]["cat_mean_items"]
        ax.text(xi + 0.2, results[b]["row"]["cat_r_postfilter_new"] + 0.02,
                f"{ni:.0f} it", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(benches)
    ax.set_ylabel("Pearson r vs actual")
    ax.set_title("(2) bbh has the largest full-bank to CAT r drop\n"
                 "CAT stops near the 8-item minimum and under-samples")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (3) prediction over-dispersion: CAT pred vs actual (bbh predictions too spread)
    ax = axes[1, 0]
    for b in benches:
        rep_csv = results[b]["rep_csv"]
        if not rep_csv.exists():
            continue
        rep = pd.read_csv(rep_csv)
        act = rep["actual_accuracy"].to_numpy()
        pir = rep["pirt_accuracy"].to_numpy()
        ax.scatter(act, pir, s=16, alpha=0.35, color=colors[b],
                   label=f"{b}: pred SD={pir.std():.2f} vs act SD={act.std():.2f}")
    lo, hi = 0.05, 0.75
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.5, label="y=x")
    ax.set_xlabel("actual score")
    ax.set_ylabel("CAT p-IRT prediction (SE<=0.3)")
    ax.set_title("(3) bbh predictions are over-dispersed (MAE 0.11, highest)\n"
                 "few high-a items give noisy, extreme theta")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (4) item p-value distribution + discrimination note
    ax = axes[1, 1]
    for b in benches:
        pv = results[b]["pvals"]
        ax.hist(pv, bins=40, range=(0, 1), histtype="step", lw=2,
                color=colors[b], density=True,
                label=f"{b}: p-val mean={np.mean(pv):.2f}, a_med={results[b]['row']['a_median_filtered']:.1f}")
    ax.set_xlabel("item p-value (proportion correct)")
    ax.set_ylabel("density")
    ax.set_title("(4) Item difficulty (filtered items)\n"
                 "bbh a-median about 4 (highest), so SE is overconfident and stops early")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("bbh diagnosis: early-stop under-sampling of a heterogeneous suite "
                 "(not bad items, not low variance)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out = OUT_DIR / "bbh_diagnosis.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cat_new = load_cat_r(NEW_SUMMARY, se=0.3)
    cat_old = load_cat_r(OLD_SUMMARY)
    results = {}
    rows = []
    for b in BENCHES:
        res = diagnose(b, cat_new, cat_old)
        results[b] = res
        rows.append(res["row"])
        r = res["row"]
        print(f"[{b}] %a<=0={r['pct_a_le0']:.1f} "
              f"a_med(unf/filt)={r['a_median_unfiltered']:.2f}/{r['a_median_filtered']:.2f} "
              f"r_theta(unf->filt)={r['r_theta_score_unfiltered']:.3f}->"
              f"{r['r_theta_score_filtered']:.3f} | CAT r(old->new)="
              f"{r['cat_r_prefilter_old']:.3f}->{r['cat_r_postfilter_new']:.3f} "
              f"score_sd={r['score_sd_full']:.4f} cat_items={r['cat_mean_items']:.1f} "
              f"cat_mae={r['cat_mae']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    csv_out = OUT_DIR / "diagnosis_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"wrote {csv_out}", flush=True)

    fig_mechanism(results)
    fig_bbh(results)


if __name__ == "__main__":
    main()
