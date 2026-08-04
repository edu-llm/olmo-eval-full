#!/usr/bin/env python3
"""Item-matched (fixed test length) correlation comparison on ARC.

Question answered here: if you give 2PL and 3PL the SAME NUMBER of items that
1PL uses, what correlation do they reach, and where does 1PL overtake them?

Because the Fisher-information CAT fixes item ORDER (the SE target only decides
where to STOP), "correlation at exactly N items" is well defined: read each
held-out model's running prediction after N administered items, then compute the
Pearson r (and MAE) of that prediction against the model's actual full-benchmark
accuracy, across the held-out models. We reuse the exact CAT machinery from
`se_sweep.py` (Fisher-information item selection, EAP theta/SE, the mean-prob
IRT predictor) and read each bank's per-model trace at every fixed N.

Two readouts of the same traces:
  * hold_at_floor (PRIMARY, matches the task's `full_cat_traces` behaviour): run
    the CAT to the SE=0.10 bank floor; if a model's CAT already ended before N,
    hold its last (floor) prediction. This mirrors se_sweep.full_cat_traces.
  * forced_N (robustness): administer exactly N items to every model regardless
    of SE, i.e. never stop early. Identical to hold_at_floor for every N up to
    the SE=0.10 stop of the slowest bank; differs only where a bank would have
    stopped (mainly 2PL past ~100 items). Reported so the high-N comparison is
    transparent, not an artefact of the stopping rule.

Local banks (PRIMARY, clean apples-to-apples): same in-house ARC matrix, same
held-out set for all three banks (seed 7, n_test 13, full train pool).
  1PL: se_sweep.load_local method="rasch"    2PL: method="girth"
  3PL: se_sweep_small_pool.fit_bank(kept,"3PL") on the SAME kept train matrix,
       with its usability filter, then the c-aware CAT.
Predictor: pred_meanprob for all three (fair, identical predictor).

ATLAS 3PL (SECONDARY, clearly caveated): the published ATLAS ARC 3PL bank on
ATLAS's OWN 60 held-out models and OWN response matrix (pred_pirt). This is a
DIFFERENT test set, so it is not strictly item-matched to the local banks; it is
plotted only as a reference line.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402  (imported after the BLAS thread caps above)

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
DATA_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS"
sys.path.insert(0, str(REPO / "eduLLM-Evals"))
sys.path.insert(0, str(SCRIPTS))

import se_sweep as S  # noqa: E402  reuse prob/eap_se/predictors/load_local/load_atlas/NODES
import se_sweep_small_pool as SP  # noqa: E402  reuse fit_bank (1PL/2PL/3PL) + 3PL MML-EM

MIN_ITEMS = S.MIN_ITEMS  # 8
SE_FLOOR = 0.10  # the "bank floor" full_cat_traces stops at

# 1PL reference operating points (mean CAT items at each SE target, full pool),
# from the published SE sweep (se_sweep_local_arc_1pl / small-pool N=50 full).
ONEPL_SE_ITEMS = {
    "SE0.30": 43.38,
    "SE0.25": 63.69,
    "SE0.20": 100.46,
    "SE0.15": 182.00,
    "SE0.12": 303.54,
}
HEADLINE_NS = [20, 43, 71, 100]

COLORS = {"1PL": "#55A868", "2PL": "#DD8452", "3PL": "#4C72B0", "ATLAS_3PL": "#8172B3"}


# --------------------------------------------------------------------------- #
# Traces: one CAT per model, run to the FULL bank (no early stop). The SE=0.10  #
# floor readout is just a truncation of this same trace, so hold_at_floor and   #
# forced_N are guaranteed consistent up to each bank's floor.                   #
# --------------------------------------------------------------------------- #
def cat_trace_full(resp_all, a, b, c, predictor, max_items=None):
    """Reuse se_sweep's Fisher-information selection + EAP + predictor, but run
    to the full bank (or `max_items`) with no SE early stop. Returns a list of
    (n_items, se, pred) with one entry per administered item, so index i holds
    the running prediction after (i + 1) items."""
    n = len(a)
    cap = n if max_items is None else min(n, max_items)
    used = np.zeros(n, bool)
    theta, order, steps = 0.0, [], []
    for _ in range(cap):
        p = S.prob(theta, a, b, c)
        info = (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.asarray(order)
        theta, se = S.eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        steps.append((len(order), se, predictor(resp_all, order, theta, a, b, c)))
    return steps


def floor_len(steps):
    """Number of items at the SE=0.10 stop (>= MIN_ITEMS); full length if never
    reached. Matches se_sweep.full_cat_traces' break condition exactly."""
    for ni, se, _pr in steps:
        if ni >= MIN_ITEMS and se <= SE_FLOOR:
            return ni
    return steps[-1][0]


def pred_at(steps, N, stop_len):
    """Running prediction after N items. If the CAT ended before N (N > stop_len)
    hold the last prediction at stop_len; otherwise read step N directly."""
    k = min(N, stop_len)
    return steps[k - 1][2]


# --------------------------------------------------------------------------- #
# Banks                                                                        #
# --------------------------------------------------------------------------- #
def load_local_3pl(mcq_dir, bench, n_test, seed):
    """Same split/actual as se_sweep.load_local, but fit a 3PL bank on the kept
    train matrix via se_sweep_small_pool.fit_bank + its usability filter."""
    from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark

    mat = load_benchmark(mcq_dir, bench).dropna(axis=0, how="any")
    models = list(mat.index)
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    test = sorted(order[:n_test])
    train = order[n_test:]
    actual = mat.mean(axis=1).to_dict()
    kept, _ = filter_items(mat.loc[train], benchmark=bench)
    items, a, b, c = SP.fit_bank(kept, "3PL")
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    usable = np.isfinite(a) & (a > 0) & np.isfinite(b) & np.isfinite(c) & (c >= 0) & (c < 1.0)
    items = [it for it, keep in zip(items, usable, strict=False) if keep]
    a, b, c = a[usable], b[usable], c[usable]
    resp = {m: mat.loc[m, items].to_numpy(float) for m in test}
    return items, a, b, c, resp, {m: float(actual[m]) for m in test}, S.pred_meanprob


def build_banks(mcq_dir, bench, n_test, seed, atlas_n_test, atlas_seed, atlas_cap):
    """Return {bank_name: dict(a,b,c,resp,actual,predictor,traces,floor,bank_size,max_N)}.
    Traces are full-bank (local) or capped (ATLAS) and shared by both readouts."""
    banks = {}

    def add_local(name, method):
        items, a, b, c, resp, actual, predictor = S.load_local(mcq_dir, bench, method, n_test, seed)
        banks[name] = dict(
            a=np.asarray(a, float),
            b=np.asarray(b, float),
            c=np.asarray(c, float),
            resp=resp,
            actual=actual,
            predictor=predictor,
            bank_size=len(items),
        )

    add_local("1PL", "rasch")
    add_local("2PL", "girth")
    items3, a3, b3, c3, resp3, actual3, pred3 = load_local_3pl(mcq_dir, bench, n_test, seed)
    banks["3PL"] = dict(
        a=a3, b=b3, c=c3, resp=resp3, actual=actual3, predictor=pred3, bank_size=len(items3)
    )

    # sanity: all three local banks must share the identical held-out set/actual
    ref_test = sorted(banks["1PL"]["actual"])
    for nm in ("2PL", "3PL"):
        assert sorted(banks[nm]["actual"]) == ref_test, f"{nm} test set differs"
        for m in ref_test:
            assert abs(banks[nm]["actual"][m] - banks["1PL"]["actual"][m]) < 1e-12, (
                f"{nm} actual differs for {m}"
            )

    # ATLAS 3PL (own test set) - secondary, caveated
    try:
        items_a, aa, ba, ca, respa, actuala, preda = S.load_atlas(atlas_n_test, atlas_seed)
        banks["ATLAS_3PL"] = dict(
            a=np.asarray(aa, float),
            b=np.asarray(ba, float),
            c=np.asarray(ca, float),
            resp=respa,
            actual=actuala,
            predictor=preda,
            bank_size=len(items_a),
            cap=atlas_cap,
        )
    except Exception as e:  # secondary; never block the primary comparison
        print(f"[warn] ATLAS bank unavailable, skipping secondary line: {e}")

    # compute the shared full-bank traces once per bank
    for name, B in banks.items():
        cap = B.get("cap")
        max_N = min(B["bank_size"], cap) if cap else B["bank_size"]
        traces = {
            m: cat_trace_full(B["resp"][m], B["a"], B["b"], B["c"], B["predictor"], cap)
            for m in B["resp"]
        }
        B["traces"] = traces
        B["floor"] = {m: floor_len(traces[m]) for m in traces}
        B["max_N"] = max_N
        print(
            f"[{name}] bank={B['bank_size']} items, held-out={len(traces)} models, "
            f"trace_cap={max_N}, mean floor items={np.mean(list(B['floor'].values())):.1f}"
        )
    return banks


# --------------------------------------------------------------------------- #
# Correlation-vs-N and reference readouts                                      #
# --------------------------------------------------------------------------- #
def corr_at_N(B, N, mode):
    """Pearson r and MAE of the running prediction after N items vs actual."""
    preds, acts = [], []
    for m, steps in B["traces"].items():
        stop = B["floor"][m] if mode == "hold_at_floor" else steps[-1][0]
        preds.append(pred_at(steps, N, stop))
        acts.append(B["actual"][m])
    preds = np.asarray(preds)
    acts = np.asarray(acts)
    if preds.std() < 1e-12 or acts.std() < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(preds, acts)[0, 1])
    mae = float(np.mean(np.abs(preds - acts)))
    return r, mae, len(preds)


def curve(B, mode):
    """(N, r, mae, n) for every integer N from MIN_ITEMS to the bank's max_N."""
    out = []
    for N in range(MIN_ITEMS, B["max_N"] + 1):
        r, mae, n = corr_at_N(B, N, mode)
        out.append((N, r, mae, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq-dir", default=str(REPO / "AdaptiveTesting/Inputs/Open/LLM-Judge/mcq"))
    ap.add_argument("--bench", default="arc_challenge")
    ap.add_argument("--n-test", type=int, default=13)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--atlas-n-test", type=int, default=60)
    ap.add_argument("--atlas-seed", type=int, default=7)
    ap.add_argument(
        "--atlas-cap",
        type=int,
        default=350,
        help="cap ATLAS trace length (secondary line; covers all reference Ns)",
    )
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "data/se_sweep/item_matched")
    ap.add_argument("--fig-dir", type=Path, default=DATA_ROOT / "figures")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="skip trace computation; rebuild figures and README from _cache.json",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplcache"))

    if args.report_only:
        order, curves, crossovers, reference_Ns, ref_notes, meta = load_cache(args.out_dir)
        emit_reports(args, order, curves, crossovers, reference_Ns, ref_notes, meta)
        return

    banks = build_banks(
        args.mcq_dir,
        args.bench,
        args.n_test,
        args.seed,
        args.atlas_n_test,
        args.atlas_seed,
        args.atlas_cap,
    )
    local = [nm for nm in ("1PL", "2PL", "3PL") if nm in banks]
    order = local + (["ATLAS_3PL"] if "ATLAS_3PL" in banks else [])

    # ---- correlation-vs-N curves (both readouts) ----
    curves = {
        mode: {nm: curve(banks[nm], mode) for nm in order} for mode in ("hold_at_floor", "forced_N")
    }

    def write_curve_csv(path, mode):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["bank", "N", "r", "mae", "n_eval_models"])
            for nm in order:
                for N, r, mae, n in curves[mode][nm]:
                    w.writerow([nm, N, "" if np.isnan(r) else round(r, 4), round(mae, 4), n])

    write_curve_csv(args.out_dir / "item_matched_corr_vs_N.csv", "hold_at_floor")
    write_curve_csv(args.out_dir / "item_matched_corr_vs_N_forcedN.csv", "forced_N")

    # ---- reference / matched-budget table ----
    ref_notes = {}
    for n in HEADLINE_NS:
        ref_notes.setdefault(n, []).append("headline")
    for se, mi in ONEPL_SE_ITEMS.items():
        ref_notes.setdefault(int(round(mi)), []).append(f"1PL_{se}")
    reference_Ns = sorted(ref_notes)

    mb_fields = ["bank", "reference_N", "ref_note", "r", "mae", "n_eval_models"]

    def matched_rows(mode):
        rows = []
        for nm in order:
            B = banks[nm]
            for N in [n for n in reference_Ns if n <= B["max_N"]]:
                r, mae, n = corr_at_N(B, N, mode)
                rows.append(
                    {
                        "bank": nm,
                        "reference_N": N,
                        "ref_note": "; ".join(ref_notes[N]),
                        "r": "" if np.isnan(r) else round(r, 4),
                        "mae": round(mae, 4),
                        "n_eval_models": n,
                    }
                )
        return rows

    def write_matched_csv(path, mode):
        rows = matched_rows(mode)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=mb_fields)
            w.writeheader()
            w.writerows(rows)
        return rows

    write_matched_csv(args.out_dir / "matched_budget_table.csv", "hold_at_floor")
    write_matched_csv(args.out_dir / "matched_budget_table_forcedN.csv", "forced_N")

    # ---- crossover: smallest N above which 1PL r >= competitor r for all larger N ----
    def crossover(mode, competitor):
        c1 = {N: r for (N, r, _m, _n) in curves[mode]["1PL"]}
        cc = {N: r for (N, r, _m, _n) in curves[mode][competitor]}
        common = sorted(set(c1) & set(cc))
        # scan from the top: the crossover is one past the last N where 1PL < competitor
        last_below = None
        for N in common:
            r1, rc = c1[N], cc[N]
            if np.isnan(r1) or np.isnan(rc):
                continue
            if r1 < rc - 1e-9:
                last_below = N
        if last_below is None:
            return common[0]  # 1PL >= competitor everywhere on the grid
        idx = common.index(last_below)
        return common[idx + 1] if idx + 1 < len(common) else None  # None = never overtakes

    crossovers = []
    for mode in ("hold_at_floor", "forced_N"):
        for comp in ("2PL", "3PL"):
            if comp in banks:
                xo = crossover(mode, comp)
                crossovers.append(
                    {
                        "mode": mode,
                        "pair": f"1PL_over_{comp}",
                        "crossover_N": "never" if xo is None else xo,
                    }
                )
    with open(args.out_dir / "crossover.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mode", "pair", "crossover_N"])
        w.writeheader()
        w.writerows(crossovers)

    # ---- validation against the SE sweep ----
    print("\n=== VALIDATION (hold_at_floor readout) ===")
    r_2pl_8, _, _ = corr_at_N(banks["2PL"], 8, "hold_at_floor")
    r_1pl_43, _, _ = corr_at_N(banks["1PL"], 43, "hold_at_floor")
    r_2pl_71, _, _ = corr_at_N(banks["2PL"], 71, "hold_at_floor")
    print(f"2PL @ N=8   r={r_2pl_8:.4f}  (SE sweep SE0.30 -> 8 items, r=0.8615)")
    print(f"1PL @ N=43  r={r_1pl_43:.4f}  (SE sweep SE0.30 -> 43.4 items, r=0.976)")
    print(f"2PL @ N=71  r={r_2pl_71:.4f}  (SE sweep SE0.12 -> 71 items, r=0.965)")
    ok = True
    if abs(r_2pl_8 - 0.8615) > 0.005:
        print(f"FAIL: 2PL@8 r={r_2pl_8:.4f} deviates from 0.8615 by >0.005")
        ok = False
    if abs(r_1pl_43 - 0.976) > 0.03:
        print(
            f"WARN: 1PL@43 r={r_1pl_43:.4f} deviates from 0.976 by >0.03 "
            f"(fixed-N vs fixed-SE slicing differ; investigate if large)"
        )
    if not ok:
        raise SystemExit("Validation FAILED - stopping before reporting results.")
    print("validation OK")

    # ---- cache (so figures/README can be rebuilt fast via --report-only) ----
    meta = {
        nm: {
            "bank_size": int(B["bank_size"]),
            "n_models": int(len(B["traces"])),
            "mean_floor": float(np.mean(list(B["floor"].values()))),
        }
        for nm, B in banks.items()
    }
    dump_cache(args.out_dir, order, curves, crossovers, reference_Ns, ref_notes, meta)

    # ---- figures + console summary + README ----
    emit_reports(args, order, curves, crossovers, reference_Ns, ref_notes, meta)
    print(f"\nwrote outputs to {args.out_dir}")


# --------------------------------------------------------------------------- #
# Cache + reporting orchestration                                              #
# --------------------------------------------------------------------------- #
def dump_cache(out_dir, order, curves, crossovers, reference_Ns, ref_notes, meta):
    import json

    payload = {
        "order": order,
        "curves": {
            mode: {nm: [[N, r, mae, n] for (N, r, mae, n) in rows] for nm, rows in d.items()}
            for mode, d in curves.items()
        },
        "crossovers": crossovers,
        "reference_Ns": reference_Ns,
        "ref_notes": {str(k): v for k, v in ref_notes.items()},
        "meta": meta,
    }
    (out_dir / "_cache.json").write_text(json.dumps(payload))


def load_cache(out_dir):
    import json

    d = json.loads((out_dir / "_cache.json").read_text())
    curves = {
        mode: {nm: [tuple(x) for x in rows] for nm, rows in banks_.items()}
        for mode, banks_ in d["curves"].items()
    }
    ref_notes = {int(k): v for k, v in d["ref_notes"].items()}
    return d["order"], curves, d["crossovers"], d["reference_Ns"], ref_notes, d["meta"]


def emit_reports(args, order, curves, crossovers, reference_Ns, ref_notes, meta):
    make_figures(args.fig_dir, order, curves, crossovers)
    print_summary(order, curves, crossovers, reference_Ns, ref_notes)
    write_readme(args.out_dir, order, meta, curves, crossovers, reference_Ns, ref_notes, args)


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def _xo_val(crossovers, mode, pair):
    for c in crossovers:
        if c["mode"] == mode and c["pair"] == pair:
            return c["crossover_N"]
    return None


def make_figures(fig_dir, order, curves, crossovers):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "1PL": "Local 1PL",
        "2PL": "Local 2PL",
        "3PL": "Local 3PL",
        "ATLAS_3PL": "ATLAS 3PL (different test set)",
    }

    # ---- main figure: correlation vs fixed test length (hold_at_floor) ----
    fig, ax = plt.subplots(figsize=(11.0, 6.6), constrained_layout=True)
    for nm in order:
        pts = curves["hold_at_floor"][nm]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        style = "--" if nm == "ATLAS_3PL" else "-"
        lw = 1.9 if nm == "ATLAS_3PL" else 2.4
        ax.plot(xs, ys, style, color=COLORS[nm], lw=lw, label=labels[nm])

    ax.axvline(43, color="#333333", ls=":", lw=1.8)
    ax.text(
        43,
        0.515,
        "  1PL SE0.30 budget (43 items)",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

    c1 = dict((p[0], p[1]) for p in curves["hold_at_floor"]["1PL"])
    xo2 = _xo_val(crossovers, "hold_at_floor", "1PL_over_2PL")
    xo3 = _xo_val(crossovers, "hold_at_floor", "1PL_over_3PL")
    ann = []
    if isinstance(xo2, int) and xo2 == xo3:
        ax.plot([xo2], [c1[xo2]], "D", color="white", mec="black", mew=1.8, ms=13, zorder=6)
        ann.append(f"1PL passes both 2PL and 3PL for good at N={xo2}")
    else:
        if isinstance(xo2, int):
            ax.plot([xo2], [c1[xo2]], "o", color="white", mec="black", mew=1.6, ms=12, zorder=6)
            ann.append(f"1PL passes 2PL for good at N={xo2}")
        if isinstance(xo3, int):
            ax.plot([xo3], [c1[xo3]], "s", color="white", mec="black", mew=1.6, ms=12, zorder=6)
            ann.append(f"1PL passes 3PL for good at N={xo3}")
    if ann:
        ax.text(
            0.015,
            0.985,
            "Crossover (item-matched):\n" + "\n".join(ann),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#888888", alpha=0.95),
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Fixed test length (number of items administered, log scale)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Pearson r  (predicted vs actual full-ARC accuracy)", fontsize=14, fontweight="bold"
    )
    ax.set_title(
        "ARC item-matched comparison: correlation vs fixed test length\n"
        "same held-out models and same predictor for local 1PL, 2PL, 3PL",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", alpha=0.28)
    from matplotlib.ticker import ScalarFormatter

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([8, 20, 43, 71, 100, 200, 400, 759])
    ax.legend(loc="lower right", fontsize=12, framealpha=0.93)
    out_main = fig_dir / "arc_item_matched_corr_vs_items.png"
    fig.savefig(out_main, dpi=150)
    plt.close(fig)
    print("wrote", out_main)

    # ---- supplementary: hold-at-floor vs forced-N (robustness) ----
    fig, ax = plt.subplots(figsize=(11.0, 6.6), constrained_layout=True)
    for nm in [n for n in order if n != "ATLAS_3PL"]:
        for mode, ls, alpha, tag in (
            ("hold_at_floor", "-", 1.0, "hold at SE floor"),
            ("forced_N", ":", 0.9, "forced N items"),
        ):
            pts = curves[mode][nm]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, ls, color=COLORS[nm], lw=2.2, alpha=alpha, label=f"{nm} ({tag})")
    ax.axvline(43, color="#333333", ls=":", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(
        "Fixed test length (number of items administered, log scale)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Pearson r", fontsize=14, fontweight="bold")
    ax.set_title(
        "Robustness: hold-at-SE-floor vs forced-N readout (local banks)\n"
        "identical up to each bank's SE=0.10 stop; they diverge only past it",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", alpha=0.28)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([8, 20, 43, 71, 100, 200, 400, 759])
    ax.legend(loc="lower right", fontsize=10, ncol=3, framealpha=0.93)
    out_supp = fig_dir / "arc_item_matched_floor_vs_forced.png"
    fig.savefig(out_supp, dpi=150)
    plt.close(fig)
    print("wrote", out_supp)


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _r_at(curves, mode, nm, N):
    for n, r, _m, _c in curves[mode][nm]:
        if n == N:
            return r
    return None


def print_summary(order, curves, crossovers, reference_Ns, ref_notes):
    print("\n=== MATCHED-BUDGET TABLE (hold_at_floor; Pearson r) ===")
    hdr = "N".rjust(5) + "  " + "".join(nm.rjust(10) for nm in order) + "   note"
    print(hdr)
    for N in reference_Ns:
        line = str(N).rjust(5) + "  "
        for nm in order:
            r = _r_at(curves, "hold_at_floor", nm, N)
            line += ("--" if r is None else f"{r:.4f}").rjust(10)
        line += "   " + "; ".join(ref_notes[N])
        print(line)
    print("\n=== CROSSOVER (smallest N above which 1PL r >= competitor for all larger N) ===")
    for c in crossovers:
        print(f"  {c['mode']:<14} {c['pair']:<16} -> N = {c['crossover_N']}")


def write_readme(out_dir, order, meta, curves, crossovers, reference_Ns, ref_notes, args):
    def rr(mode, nm, N):
        r = _r_at(curves, mode, nm, N)
        return "n/a" if r is None or (isinstance(r, float) and np.isnan(r)) else f"{r:.4f}"

    xo2 = _xo_val(crossovers, "hold_at_floor", "1PL_over_2PL")
    xo3 = _xo_val(crossovers, "hold_at_floor", "1PL_over_3PL")
    xo2f = _xo_val(crossovers, "forced_N", "1PL_over_2PL")
    xo3f = _xo_val(crossovers, "forced_N", "1PL_over_3PL")

    lines = []
    lines.append("# ARC item-matched (fixed test length) correlation comparison\n")
    lines.append(
        "Question: if you give 2PL and 3PL the same NUMBER of items that 1PL uses, "
        "what correlation do they reach, and where does 1PL overtake them?\n"
    )

    lines.append("## Method\n")
    lines.append(
        "The Fisher-information CAT fixes item ORDER; the SE target only decides where to "
        "stop. So the correlation at exactly N items is well defined: run one CAT per "
        "held-out model, then read that model's running prediction after N administered "
        "items and correlate against its actual full-ARC accuracy across models. We reuse "
        "the exact `se_sweep.py` machinery (Fisher-information selection, EAP theta and SE, "
        "and the `pred_meanprob` predictor for all three local banks, so the predictor is "
        "identical and the comparison is fair).\n"
    )
    lines.append(
        "- Data: in-house ARC matrix, `arc_challenge`, seed 7, n_test 13, full train pool. "
        f"Same {meta['1PL']['n_models']} held-out models and same actual accuracies for "
        "1PL, 2PL, and 3PL.\n"
    )
    lines.append(
        f"- Banks (all {meta['1PL']['bank_size']} items for the local set): 1PL via "
        "`load_local` method=rasch, 2PL via method=girth, 3PL via "
        '`se_sweep_small_pool.fit_bank(kept, "3PL")` on the SAME kept train matrix with its '
        "usability filter, using the c-aware CAT.\n"
    )
    lines.append(
        "- Two readouts of the same traces: `hold_at_floor` (PRIMARY) runs each CAT to the "
        "SE=0.10 bank floor and holds the last prediction if the CAT ended before N (this "
        "mirrors `se_sweep.full_cat_traces`); `forced_N` (robustness) administers exactly N "
        "items with no early stop. They are identical for every N up to a bank's SE=0.10 "
        "stop and differ only past it (mostly 2PL beyond about 100 items).\n"
    )

    lines.append("## Validation against the SE sweep\n")
    lines.append(
        f"- 2PL at N=8: r = {rr('hold_at_floor', '2PL', 8)} (SE sweep SE0.30 uses exactly 8 "
        "items, r = 0.8615). Exact match expected and observed.\n"
    )
    lines.append(
        f"- 1PL at N=43: r = {rr('hold_at_floor', '1PL', 43)} (SE sweep SE0.30 uses 43.4 items, "
        "r = 0.976). Close; small gap is the fixed-N vs fixed-SE slicing.\n"
    )
    lines.append(
        f"- 2PL at N=71: r = {rr('hold_at_floor', '2PL', 71)} (SE sweep SE0.12 uses 71 items, "
        "r = 0.965).\n"
    )

    lines.append("## Matched-budget table (PRIMARY, hold_at_floor; Pearson r)\n")
    head = "| N | " + " | ".join(order) + " | reference |"
    sep = "|" + "---|" * (len(order) + 2)
    lines.append(head)
    lines.append(sep)
    for N in reference_Ns:
        cells = " | ".join(rr("hold_at_floor", nm, N) for nm in order)
        lines.append(f"| {N} | {cells} | {'; '.join(ref_notes[N])} |")
    lines.append("")
    lines.append(
        "N=43 is 1PL's SE0.30 operating point (its headline budget). Reference notes tag the "
        "headline set {20, 43, 71, 100} and the 1PL SE operating points (SE0.30=43, SE0.25=64, "
        "SE0.20=100, SE0.15=182, SE0.12=304 items).\n"
    )

    lines.append("## Forced-N readout (robustness; Pearson r at the same budgets)\n")
    lines.append(head)
    lines.append(sep)
    for N in reference_Ns:
        cells = " | ".join(rr("forced_N", nm, N) for nm in order)
        lines.append(f"| {N} | {cells} | {'; '.join(ref_notes[N])} |")
    lines.append("")
    lines.append(
        "Forcing 2PL to actually take N items (instead of holding at its SE=0.10 floor) is the "
        "only place the two readouts differ; it shows whether extra items past 2PL's floor help "
        "or not.\n"
    )

    lines.append("## Crossover (smallest N above which 1PL wins for the rest of the range)\n")
    lines.append(
        f"- 1PL overtakes 2PL for good at N = {xo2} items (hold_at_floor); N = {xo2f} (forced_N)."
    )
    lines.append(
        f"- 1PL overtakes 3PL for good at N = {xo3} items (hold_at_floor); N = {xo3f} (forced_N).\n"
    )

    def mm(mode, nm, N):
        for n, _r, mae, _c in curves[mode][nm]:
            if n == N:
                return f"{mae:.3f}"
        return "n/a"

    r1_43 = rr("hold_at_floor", "1PL", 43)
    r2_43 = rr("hold_at_floor", "2PL", 43)
    r3_43 = rr("hold_at_floor", "3PL", 43)
    r1_20 = rr("hold_at_floor", "1PL", 20)
    r2_20 = rr("hold_at_floor", "2PL", 20)
    r3_20 = rr("hold_at_floor", "3PL", 20)
    r3_304 = rr("hold_at_floor", "3PL", 304)
    lines.append("## Plain-language verdict\n")
    lines.append(
        f"At 1PL's own SE0.30 budget of 43 items, 1PL reaches r = {r1_43} while 2PL reaches "
        f"r = {r2_43} and 3PL reaches r = {r3_43}, so at a matched 43-item budget the richer "
        "models do not beat 1PL. The order flips only at very short tests: at N=20 items the "
        f"c-aware and discriminating banks lead, with 3PL highest among the local banks "
        f"(r = {r3_20}), 2PL next (r = {r2_20}), and 1PL last (r = {r1_20}), because those "
        "banks spread models apart with very few items while 1PL still needs items to "
        "accumulate the same information. Once you can afford a few dozen items 1PL pulls ahead "
        f"and stays ahead: it passes 2PL for good at N = {xo2} items and passes 3PL for good at "
        f"N = {xo3} items, and it keeps improving with test length while the others plateau. "
        "3PL never leads again at any larger matched budget here, and it carries two extra "
        f"caveats: its absolute calibration is poor (MAE about {mm('hold_at_floor', '3PL', 43)} "
        f"at N=43 versus about {mm('hold_at_floor', '1PL', 43)} for 1PL, from the c-inflated "
        "mean-prob predictor), and its rank correlation actually drifts DOWN as you add items "
        f"(r = {r3_304} at N=304) because many thin-sample items are pinned at the a and c "
        "bounds and inject noise. Bottom line: at 1PL's item budget the extra 2PL and 3PL "
        "parameters do not buy accuracy, and 1PL is the better bank at every budget beyond a "
        "very short screening test of about two dozen items.\n"
    )

    lines.append("## Files\n")
    lines.append("- `item_matched_corr_vs_N.csv` primary curves (bank, N, r, mae, n_eval_models).")
    lines.append("- `item_matched_corr_vs_N_forcedN.csv` robustness curves (forced-N readout).")
    lines.append("- `matched_budget_table.csv` r at the reference budgets (primary readout).")
    lines.append("- `matched_budget_table_forcedN.csv` same budgets, forced-N readout.")
    lines.append("- `crossover.csv` crossover N for each pair and readout.")
    lines.append("- `../../../figures/arc_item_matched_corr_vs_items.png` main figure.")
    lines.append("- `../../../figures/arc_item_matched_floor_vs_forced.png` robustness figure.")
    lines.append("")
    lines.append("## Secondary reference: ATLAS 3PL\n")
    if "ATLAS_3PL" in meta:
        lines.append(
            "The ATLAS 3PL line uses the published ATLAS ARC 3PL bank on ATLAS's OWN 60 "
            "held-out models and OWN response matrix with the `pred_pirt` predictor. It is a "
            "DIFFERENT test set (different models, different response matrix, different "
            "predictor), so it is NOT strictly item-matched to the local banks and is shown only "
            "as a reference line. Its numbers should not be read as a like-for-like comparison "
            "against local 1PL/2PL/3PL.\n"
        )
    else:
        lines.append("ATLAS bank was unavailable at run time; secondary line omitted.\n")

    (out_dir / "README.md").write_text("\n".join(lines))
    print("wrote", out_dir / "README.md")


if __name__ == "__main__":
    main()
