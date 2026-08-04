#!/usr/bin/env python3
"""Item-matched (fixed test length) 1PL vs 2PL vs 3PL correlation comparison on pedagogy.

The SE-stopped comparison in ``../run_pl_expanded78.py`` answers "how good is each bank at
its own SE operating point", where every bank spends a different number of items. This
script answers the fair, budget-matched question instead: if you give 2PL and 3PL the SAME
number of items that 1PL spends, what correlation do they reach, and where (if anywhere)
does 2PL or 3PL overtake 1PL and stay above it?

Because the Fisher-information CAT administers items in a fixed order (the SE target only
decides where to stop), the correlation at exactly N items is a read-off: take each held-out
model's running prediction after N administered items, then Pearson r vs actual full-bank
pedagogy accuracy across the held-out models. One CAT trace per model, every fixed length N.

Method (reused, not reinvented):

* Data: expanded 78-model pedagogy matrix via ``load_benchmark`` on the expanded
  ``_mcq_data``. Partition is the CONTROLLED design shared with
  ``run_expanded_calibration`` / ``run_pl_expanded78``: seed-7 split of the 52 OLD models
  gives 40 OLD-train and 12 OLD held-out; calib66 = those 40 plus the 26 new models. The
  eval set is the fixed 12 OLD held-out models. calib40 (the 40 OLD-train alone) is also
  computed for continuity with the original 52-model / 40-train baseline.
* Banks: 1PL girth ``rasch_mml``, 2PL girth ``twopl_mml``, 3PL
  ``se_sweep_small_pool.fit_3pl_mml``, all via ``se_sweep_small_pool.fit_bank`` with the
  uniform usability filter (finite, a>0, 0<=c<1). Item filter ``filter_items`` on the
  train slice.
* CAT: ``se_sweep.full_cat_traces`` + ``pred_meanprob``, ``MIN_ITEMS=8``, c-aware for all
  three banks so the comparison is fair (at c=0 it reduces to the 2PL/1PL CAT).

Validation gate: reading each bank's trace with the SE-stopping rule must reproduce the
controlled SE-stopped numbers already reported (calib66 1PL SE0.3 r=0.734, 2PL SE0.3
r=0.905, and the calib40 counterparts). 1PL/2PL are hard-gated (tol 5e-3); 3PL is reported
as a soft check because its thin-pool EM fit is only reproducible within one environment.

CPU-local, torch-free, ``uv run`` friendly. Writes only into this ``item_matched/`` folder
and the shared figures directory; nothing else is touched.

Usage:
  PYTHONPATH=eduLLM-Evals uv run python AdaptiveTesting/Research/01_MCQ_ATLAS/data/\
pedagogy_feasibility/pl_1_2_3_comparison/pedagogy_expanded78/item_matched/run_item_matched.py
"""

from __future__ import annotations

# Keep BLAS modest so the girth fits do not oversubscribe the machine.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[7]  # .../olmo-eval-full
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
SCRIPTS = REPO / "AdaptiveTesting/Research/scripts"
_PED = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility"
EXPANDED = _PED / "expanded_calibration"
FIG_DIR = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/figures"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EXPANDED))
sys.path.insert(0, str(REPO / "eduLLM-Evals"))

import se_sweep as S  # noqa: E402  full_cat_traces / pred_meanprob / MIN_ITEMS
import se_sweep_small_pool as SP  # noqa: E402  fit_bank / fit_3pl_mml
from run_expanded_calibration import N_TEST_CAP, NEW_STEMS, SEED  # noqa: E402
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402

BENCH = "pedagogy"
MCQ_DIR = EXPANDED / "_mcq_data"
MODEL_TYPES = ("1PL", "2PL", "3PL")
SE_TARGETS = (0.3, 0.15)
MIN_ITEMS = S.MIN_ITEMS  # 8
REF_NS = (12, 20, 43, 100)  # requested fixed-length reference points
COLORS = {"1PL": "#55A868", "2PL": "#DD8452", "3PL": "#4C72B0"}

# Controlled SE-stopped references from ../pl_1_2_3_expanded_controlled.csv.
# (pool, model, se) -> r. 1PL/2PL are hard-gated; 3PL is a soft check (thin-pool EM).
REF_SE_STOP_R = {
    (66, "1PL", 0.3): 0.7340,
    (66, "2PL", 0.3): 0.9049,
    (66, "3PL", 0.3): 0.5067,
    (66, "1PL", 0.15): 0.9255,
    (66, "2PL", 0.15): 0.9580,
    (66, "3PL", 0.15): 0.5751,
    (40, "1PL", 0.3): 0.8903,
    (40, "2PL", 0.3): 0.7163,
    (40, "3PL", 0.3): 0.8195,
    (40, "1PL", 0.15): 0.9407,
    (40, "2PL", 0.15): 0.8996,
    (40, "3PL", 0.15): 0.5422,
}
HARD_GATE = {"1PL", "2PL"}
VALID_TOL = 5e-3


# --------------------------------------------------------------------------- #
# Partition (identical to run_pl_expanded78.seed7_split)                       #
# --------------------------------------------------------------------------- #
def seed7_split(models: list[str]):
    """Reproduce the original seed-7 held-out split over a sorted model list."""
    models = sorted(models)
    rng = np.random.default_rng(SEED)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = min(N_TEST_CAP, len(models) // 3)
    test = sorted(order[:n_test])
    train = order[n_test:]
    return train, test


# --------------------------------------------------------------------------- #
# Fit one usable bank + eval traces                                           #
# --------------------------------------------------------------------------- #
def fit_usable_bank(mat, train: list[str], model_type: str):
    """Filter items on the train slice, fit the bank, apply the uniform usability
    filter, and return (items, a, b, c)."""
    kept_mat, _report = filter_items(mat.loc[sorted(train)], benchmark=BENCH)
    items, a, b, c = SP.fit_bank(kept_mat, model_type)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    usable = np.isfinite(a) & (a > 0) & np.isfinite(b) & np.isfinite(c) & (c >= 0) & (c < 1.0)
    items = [it for it, keep in zip(items, usable, strict=False) if keep]
    return items, a[usable], b[usable], c[usable]


def eval_traces(mat, items, a, b, c, eval_models):
    resp = {m: mat.loc[m, items].to_numpy(float) for m in eval_models}
    return {m: S.full_cat_traces(resp[m], a, b, c, S.pred_meanprob) for m in eval_models}


# --------------------------------------------------------------------------- #
# Read-offs                                                                    #
# --------------------------------------------------------------------------- #
def pred_at_n(trace, n: int) -> float:
    """Running prediction after exactly n administered items; hold the last
    prediction if the CAT trace ended before n."""
    if n <= len(trace):
        return float(trace[n - 1][2])
    return float(trace[-1][2])


def se_stop_readoff(trace, se: float):
    """(#items, pred, reached) at the first step with SE<=target and n>=MIN_ITEMS;
    fall back to the full CAT end if never reached (matches run_pl_expanded78)."""
    for ni, s, pr in trace:
        if ni >= MIN_ITEMS and s <= se:
            return int(ni), float(pr), True
    return int(trace[-1][0]), float(trace[-1][2]), False


def corr_mae(preds: np.ndarray, acts: np.ndarray):
    r = float(np.corrcoef(preds, acts)[0, 1]) if len(preds) >= 2 else float("nan")
    mae = float(np.mean(np.abs(preds - acts)))
    return r, mae


# --------------------------------------------------------------------------- #
# One pool: fit all banks, build traces, corr-vs-N, SE-stopped read-offs       #
# --------------------------------------------------------------------------- #
def run_pool(mat, actual, train, eval_models, pool_label: int):
    banks = {}
    for mt in MODEL_TYPES:
        t0 = time.time()
        items, a, b, c = fit_usable_bank(mat, train, mt)
        traces = eval_traces(mat, items, a, b, c, eval_models)
        n_bank = len(items)
        acts = np.array([actual[m] for m in eval_models], float)

        # corr-vs-N over the integer grid MIN_ITEMS..n_bank
        curve = []
        for n in range(MIN_ITEMS, n_bank + 1):
            preds = np.array([pred_at_n(traces[m], n) for m in eval_models], float)
            r, mae = corr_mae(preds, acts)
            curve.append({"N": n, "r": r, "mae": mae})

        # SE-stopped read-offs (validation + 1PL operating-point budgets)
        se_stopped = {}
        for se in SE_TARGETS:
            n_items, preds, reached = [], [], []
            for m in eval_models:
                ni, pr, rc = se_stop_readoff(traces[m], se)
                n_items.append(ni)
                preds.append(pr)
                reached.append(rc)
            r, mae = corr_mae(np.array(preds, float), acts)
            se_stopped[se] = {
                "r": r,
                "mae": mae,
                "mean_items": float(np.mean(n_items)),
                "frac_reached": float(np.mean(reached)),
            }

        banks[mt] = {
            "n_bank": n_bank,
            "curve": curve,
            "curve_by_n": {row["N"]: row for row in curve},
            "se_stopped": se_stopped,
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(
            f"[pool{pool_label}] {mt} bank={n_bank} "
            f"SE0.3 r={se_stopped[0.3]['r']:.4f} items={se_stopped[0.3]['mean_items']:.1f} | "
            f"N=43 r={banks[mt]['curve_by_n'].get(43, {'r': float('nan')})['r']:.4f} "
            f"({banks[mt]['fit_seconds']}s)",
            flush=True,
        )
    return banks


def r_at(banks, mt, n: int) -> float:
    """r at fixed length n for one bank, holding the last available N if n exceeds
    the bank size (the full-bank read-off)."""
    cbn = banks[mt]["curve_by_n"]
    if n in cbn:
        return cbn[n]["r"]
    n_bank = banks[mt]["n_bank"]
    if n > n_bank:
        return cbn[n_bank]["r"]
    return cbn[max(MIN_ITEMS, min(cbn))]["r"]


def mae_at(banks, mt, n: int) -> float:
    cbn = banks[mt]["curve_by_n"]
    if n in cbn:
        return cbn[n]["mae"]
    n_bank = banks[mt]["n_bank"]
    if n > n_bank:
        return cbn[n_bank]["mae"]
    return cbn[max(MIN_ITEMS, min(cbn))]["mae"]


def crossover(banks, top: str, base: str, nmax: int):
    """Smallest N* in [MIN_ITEMS, nmax] such that r_top(N) >= r_base(N) for all
    N>=N* (top overtakes base and stays above). Returns (kind, N*):
      'always'  top >= base across every budget (no crossover needed),
      'stays'   top < base below N* and >= base from N* up,
      'never'   top < base at the top of the range (never stays above)."""
    last_below = None
    for n in range(MIN_ITEMS, nmax + 1):
        if r_at(banks, top, n) < r_at(banks, base, n):
            last_below = n
    if last_below is None:
        return "always", MIN_ITEMS
    if last_below >= nmax:
        return "never", None
    return "stays", last_below + 1


# --------------------------------------------------------------------------- #
# Outputs                                                                      #
# --------------------------------------------------------------------------- #
def write_corr_vs_n(path: Path, banks, n_eval: int):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bank", "N", "r", "mae", "n_eval_models"])
        for mt in MODEL_TYPES:
            for row in banks[mt]["curve"]:
                w.writerow([mt, row["N"], round(row["r"], 4), round(row["mae"], 4), n_eval])


def reference_points(banks):
    """Ordered (label, N) reference budgets: the requested fixed grid plus the 1PL
    SE operating points (mean items rounded to an integer test length)."""
    pts = [(f"N={n}", n) for n in REF_NS]
    n_se03 = int(round(banks["1PL"]["se_stopped"][0.3]["mean_items"]))
    n_se015 = int(round(banks["1PL"]["se_stopped"][0.15]["mean_items"]))
    pts.append((f"1PL_SE0.3_budget(~{n_se03}i)", n_se03))
    pts.append((f"1PL_SE0.15_budget(~{n_se015}i)", n_se015))
    return pts


def write_matched_budget(path: Path, banks):
    pts = reference_points(banks)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bank", "reference_label", "reference_N", "r", "mae"])
        for label, n in pts:
            for mt in MODEL_TYPES:
                w.writerow(
                    [mt, label, n, round(r_at(banks, mt, n), 4), round(mae_at(banks, mt, n), 4)]
                )


def write_summary(path: Path, banks66, banks40, cross66, cross40):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "pool",
                "bank",
                "bank_items",
                "se03_mean_items",
                "se03_r_sestop",
                "se015_mean_items",
                "se015_r_sestop",
                "r_N12",
                "r_N20",
                "r_N43",
                "r_N100",
                "crossover_vs_1PL",
            ]
        )
        for pool, banks, cross in ((66, banks66, cross66), (40, banks40, cross40)):
            for mt in MODEL_TYPES:
                b = banks[mt]
                cx = "-" if mt == "1PL" else _cross_str(cross[mt])
                w.writerow(
                    [
                        pool,
                        mt,
                        b["n_bank"],
                        round(b["se_stopped"][0.3]["mean_items"], 2),
                        round(b["se_stopped"][0.3]["r"], 4),
                        round(b["se_stopped"][0.15]["mean_items"], 2),
                        round(b["se_stopped"][0.15]["r"], 4),
                        round(r_at(banks, mt, 12), 4),
                        round(r_at(banks, mt, 20), 4),
                        round(r_at(banks, mt, 43), 4),
                        round(r_at(banks, mt, 100), 4),
                        cx,
                    ]
                )


def _cross_str(cx) -> str:
    kind, n = cx
    if kind == "always":
        return "at_or_above_all_budgets"
    if kind == "never":
        return "never_stays_above"
    return f"stays_above_from_N={n}"


# --------------------------------------------------------------------------- #
# Figure                                                                       #
# --------------------------------------------------------------------------- #
def make_figure(banks, out_path: Path, n_eval: int, cross_2pl, cross_3pl):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1})
    fig, ax = plt.subplots(figsize=(9.6, 6.2))

    all_r = []
    for mt in MODEL_TYPES:
        ns = [row["N"] for row in banks[mt]["curve"]]
        rs = [row["r"] for row in banks[mt]["curve"]]
        all_r.extend(rs)
        ax.plot(ns, rs, "-", color=COLORS[mt], linewidth=2.6, label=f"{mt}", zorder=3)

    n_budget = int(round(banks["1PL"]["se_stopped"][0.3]["mean_items"]))
    ax.axvline(n_budget, color="#333333", linestyle="--", linewidth=1.6, zorder=2)
    ax.annotate(
        f"1PL SE0.3 budget\n~{n_budget} items",
        xy=(n_budget, 0.30),
        xytext=(n_budget * 1.15, 0.16),
        fontsize=11,
        fontweight="bold",
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.3),
    )

    # matched-budget dots at the 1PL SE0.3 budget for every bank
    for mt in MODEL_TYPES:
        rv = r_at(banks, mt, n_budget)
        ax.plot(
            [n_budget],
            [rv],
            "o",
            color=COLORS[mt],
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{rv:.3f}",
            xy=(n_budget, rv),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10.5,
            fontweight="bold",
            color=COLORS[mt],
        )

    # crossover annotations (only when a genuine crossover exists)
    y0 = min(all_r) - 0.06
    if cross_2pl[0] == "stays":
        nx = cross_2pl[1]
        ax.axvline(nx, color=COLORS["2PL"], linestyle=":", linewidth=1.4, alpha=0.8, zorder=2)
        ax.annotate(
            f"2PL overtakes 1PL\nat N={nx}",
            xy=(nx, r_at(banks, "2PL", nx)),
            xytext=(nx * 0.42, y0 + 0.10),
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["2PL"],
            arrowprops=dict(arrowstyle="->", color=COLORS["2PL"], linewidth=1.2),
        )
    if cross_3pl[0] == "stays":
        nx = cross_3pl[1]
        ax.axvline(nx, color=COLORS["3PL"], linestyle=":", linewidth=1.4, alpha=0.8, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("fixed test length (items administered, log scale)", fontweight="bold")
    ax.set_ylabel("Pearson r (CAT-predicted vs actual pedagogy accuracy)", fontweight="bold")
    ax.set_ylim(max(0.0, y0), 1.0)
    ax.set_title(
        "Pedagogy item-matched: correlation vs fixed test length\n"
        f"66-model calibration pool, {n_eval} held-out models (c-aware CAT for all banks)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="IRT model", loc="lower right", fontsize=12, title_fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(banks66, banks40) -> bool:
    print("\n=== validation: SE-stopped read-offs reproduce controlled numbers ===", flush=True)
    ok = True
    for pool, banks in ((66, banks66), (40, banks40)):
        for mt in MODEL_TYPES:
            for se in SE_TARGETS:
                got = banks[mt]["se_stopped"][se]["r"]
                ref = REF_SE_STOP_R[(pool, mt, se)]
                gate = mt in HARD_GATE
                match = abs(got - ref) <= VALID_TOL
                if gate:
                    ok &= match
                flag = ("OK" if match else "MISMATCH") if gate else ("ok" if match else "soft-diff")
                print(
                    f"  pool{pool} {mt} SE<={se:<4} got={got:.4f} ref={ref:.4f} "
                    f"{'[gate]' if gate else '[soft]'} {flag}",
                    flush=True,
                )
    verdict = "PASSED" if ok else "FAILED"
    print(f"=== validation {verdict} (hard gate 1PL/2PL, tol={VALID_TOL}) ===\n", flush=True)
    return ok


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mat = load_benchmark(MCQ_DIR, BENCH).dropna(axis=0, how="any")
    all_models = list(mat.index)
    new_models = [m for m in all_models if m.replace("/", "__") in NEW_STEMS]
    old_models = [m for m in all_models if m.replace("/", "__") not in NEW_STEMS]
    assert len(new_models) == 26, f"expected 26 new, got {len(new_models)}"
    assert len(old_models) == 52, f"expected 52 old, got {len(old_models)}"

    old_train, old_test = seed7_split(old_models)  # 40 train, 12 held-out
    calib40 = sorted(old_train)
    calib66 = sorted(list(old_train) + list(new_models))
    eval_models = old_test  # fixed 12-model eval set for both pools
    actual = {m: float(v) for m, v in mat.mean(axis=1).to_dict().items()}
    print(
        f"loaded {len(all_models)} models, {mat.shape[1]} items; "
        f"calib40={len(calib40)} calib66={len(calib66)} eval={len(eval_models)}",
        flush=True,
    )

    banks66 = run_pool(mat, actual, calib66, eval_models, 66)
    banks40 = run_pool(mat, actual, calib40, eval_models, 40)

    # crossovers over the common (identical) bank range
    nmax66 = min(banks66[mt]["n_bank"] for mt in MODEL_TYPES)
    nmax40 = min(banks40[mt]["n_bank"] for mt in MODEL_TYPES)
    cross66 = {
        "2PL": crossover(banks66, "2PL", "1PL", nmax66),
        "3PL": crossover(banks66, "3PL", "1PL", nmax66),
    }
    cross40 = {
        "2PL": crossover(banks40, "2PL", "1PL", nmax40),
        "3PL": crossover(banks40, "3PL", "1PL", nmax40),
    }

    ok = validate(banks66, banks40)

    # deliverables (primary = 66-pool controlled)
    n_eval = len(eval_models)
    p_corr = args.out_dir / "pedagogy_item_matched_corr_vs_N.csv"
    write_corr_vs_n(p_corr, banks66, n_eval)
    print("wrote", p_corr)

    p_tab = args.out_dir / "pedagogy_matched_budget_table.csv"
    write_matched_budget(p_tab, banks66)
    print("wrote", p_tab)

    # secondary = 40-pool controlled (continuity with the original 52-model baseline)
    p_corr40 = args.out_dir / "pedagogy_item_matched_corr_vs_N_pool40.csv"
    write_corr_vs_n(p_corr40, banks40, n_eval)
    print("wrote", p_corr40)

    p_sum = args.out_dir / "pedagogy_item_matched_summary.csv"
    write_summary(p_sum, banks66, banks40, cross66, cross40)
    print("wrote", p_sum)

    fig_path = FIG_DIR / "pedagogy_item_matched_corr_vs_items.png"
    make_figure(banks66, fig_path, n_eval, cross66["2PL"], cross66["3PL"])
    print("wrote", fig_path)

    # headline to stdout
    pts = reference_points(banks66)
    print("\nMATCHED-BUDGET r (66-model pool, 12 held-out)")
    print(f"  {'reference':<26} {'N':>4}  {'1PL':>7} {'2PL':>7} {'3PL':>7}")
    for label, n in pts:
        print(
            f"  {label:<26} {n:>4}  "
            f"{r_at(banks66, '1PL', n):>7.3f} {r_at(banks66, '2PL', n):>7.3f} "
            f"{r_at(banks66, '3PL', n):>7.3f}"
        )
    print("\nCROSSOVERS (66-model pool, top overtakes 1PL and stays above)")
    print(f"  2PL vs 1PL: {_cross_str(cross66['2PL'])}")
    print(f"  3PL vs 1PL: {_cross_str(cross66['3PL'])}")
    print("\nCROSSOVERS (40-model pool)")
    print(f"  2PL vs 1PL: {_cross_str(cross40['2PL'])}")
    print(f"  3PL vs 1PL: {_cross_str(cross40['3PL'])}")

    return {
        "ok": ok,
        "banks66": banks66,
        "banks40": banks40,
        "cross66": cross66,
        "cross40": cross40,
        "n_eval": n_eval,
    }


if __name__ == "__main__":
    main()
