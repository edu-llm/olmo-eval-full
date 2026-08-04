#!/usr/bin/env python3
"""Experiment 3 (expanded): 1PL vs 2PL vs 3PL on the 78-model pedagogy MCQ set.

The original Experiment 3 (``pl_1_2_3_comparison/run_pl_comparison.py``) compared
1PL/2PL/3PL on a 52-model / 40-train pedagogy pool. We have since scored 26 more
models on the identical 920 pedagogy items, taking the pool to 78 models. This
script re-runs the 1PL/2PL/3PL comparison on the expanded set and reports it under
three sampling designs, so the calibration-set effect is read fairly:

1. Controlled (FAIR): fix the eval set to the original seed-7 held-out 12 models,
   then calibrate a bank on the original 40 train models (calib40) versus those
   same 40 plus the 26 new models (calib66). Identical eval set and item axis, so
   the only thing that changes is how many models the bank saw. This reproduces the
   1PL/2PL columns of ``expanded_calibration/controlled_comparison.csv`` and adds
   3PL.

2. K-fold CV (FAIR): K=6 model-level cross-validation on the full set; every model
   is held out once and predicted from a bank fit on the rest, then all
   out-of-sample predictions are pooled for one lower-variance recovery r. Run on
   OLD (52) and NEW (78). Reproduces the 1PL/2PL rows of
   ``expanded_calibration/kfold_cv_summary.csv`` and adds 3PL (OLD and NEW).

3. Naive single split (CAVEATED): the seed-7 draw of 12 held-out from 78 and 66
   train. The seed-7 draw of 12 from 78 happens to span a very narrow accuracy
   band, which deflates Pearson r. Included for completeness and flagged; the fair
   designs lead the conclusions.

Machinery is reused, not reimplemented:

* Fitters via ``se_sweep_small_pool.fit_bank``: 1PL girth ``rasch_mml``, 2PL girth
  ``twopl_mml``, 3PL the self-contained MML-EM ``fit_3pl_mml`` (girth's
  ``threepl_mml`` is broken under scipy>=1.15).
* CAT via ``se_sweep.full_cat_traces`` + ``pred_meanprob``, ``MIN_ITEMS=8``, stop at
  ``SE<=target``. At c=0 this reduces exactly to the 2PL/1PL CAT, so 1PL/2PL
  reproduce the prior numbers.
* Item filter ``filter_items`` on the train slice (ATLAS point-biserial rules).
* Uniform usability filter (finite, a>0, 0<=c<1) on every fitted bank.

The old/new model partition (``NEW_STEMS``) and the seed-7 split are imported from
``expanded_calibration/run_expanded_calibration.py`` so the partitions are shared.

Usage:
  uv run python AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility/\
pl_1_2_3_comparison/pedagogy_expanded78/run_pl_expanded78.py [--workers 6]
"""

from __future__ import annotations

# Keep BLAS modest so parallel fits do not oversubscribe the machine.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]  # .../olmo-eval-full
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
SCRIPTS = REPO / "AdaptiveTesting/Research/scripts"
_PED = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility"
EXPANDED = _PED / "expanded_calibration"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EXPANDED))
sys.path.insert(0, str(REPO / "eduLLM-Evals"))

import se_sweep as S  # noqa: E402  CAT: full_cat_traces / pred_meanprob / MIN_ITEMS
import se_sweep_small_pool as SP  # noqa: E402  fit_bank / fit_3pl_mml
from run_expanded_calibration import N_TEST_CAP, NEW_STEMS, SEED  # noqa: E402
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402

BENCH = "pedagogy"
MCQ_DIR = EXPANDED / "_mcq_data"
MODEL_TYPES = ("1PL", "2PL", "3PL")
SE_TARGETS = (0.3, 0.15)
K = 6
COLORS = {"1PL": "#55A868", "2PL": "#DD8452", "3PL": "#4C72B0"}

# References for validation (1PL/2PL only; the fair designs the expanded scripts ran).
# controlled_comparison.csv: (r_calib40, r_calib66)
REF_CONTROLLED = {
    ("2PL", 0.3): (0.7163, 0.9049),
    ("2PL", 0.15): (0.8996, 0.958),
    ("1PL", 0.3): (0.8903, 0.734),
    ("1PL", 0.15): (0.9407, 0.9255),
}
# kfold_cv_summary.csv pooled r
REF_KFOLD_OLD = {
    ("2PL", 0.3): 0.6846,
    ("2PL", 0.15): 0.8338,
    ("1PL", 0.3): 0.7835,
    ("1PL", 0.15): 0.8819,
}
REF_KFOLD_NEW = {
    ("2PL", 0.3): 0.7654,
    ("2PL", 0.15): 0.8782,
    ("1PL", 0.3): 0.7577,
    ("1PL", 0.15): 0.8931,
}
# expanded_diagnostic_summary.csv naive single split on 78 (cross-check, not a hard gate)
REF_NAIVE = {
    ("2PL", 0.3): 0.8238,
    ("2PL", 0.15): 0.7272,
    ("1PL", 0.3): 0.4571,
    ("1PL", 0.15): 0.7238,
}
VALID_TOL = 5e-3


# --------------------------------------------------------------------------- #
# Splits (identical to run_expanded_calibration.split_models / run_kfold_cv)   #
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


def kfold_folds(models: list[str], k: int = K):
    """Reproduce run_kfold_cv fold assignment: seed-7 permutation, stride f::K."""
    model_list = sorted(models)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(model_list))
    return [[model_list[i] for i in perm[f::k]] for f in range(k)]


# --------------------------------------------------------------------------- #
# Core fit + CAT for one (train, test, model_type) cell                        #
# --------------------------------------------------------------------------- #
_STATE: dict = {}


def _init_worker(mcq_dir: str, bench: str):
    mat = load_benchmark(mcq_dir, bench).dropna(axis=0, how="any")
    _STATE["mat"] = mat
    _STATE["actual"] = {m: float(v) for m, v in mat.mean(axis=1).to_dict().items()}


def _fit_cat(mat, actual, train, test, model_type, ses):
    t0 = time.time()
    kept_mat, report = filter_items(mat.loc[sorted(train)], benchmark=BENCH)
    n_filtered = kept_mat.shape[1]

    items, a, b, c = SP.fit_bank(kept_mat, model_type)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)

    usable = np.isfinite(a) & (a > 0) & np.isfinite(b) & np.isfinite(c) & (c >= 0) & (c < 1.0)
    n_dropped = int(np.sum(~usable))
    items = [it for it, keep in zip(items, usable, strict=False) if keep]
    a, b, c = a[usable], b[usable], c[usable]
    n_bank = len(items)
    fit_secs = time.time() - t0

    a_lo = int(np.sum(a <= 0.05)) if n_bank else 0
    a_hi = int(np.sum(a >= 5.9)) if n_bank else 0
    c_hi = int(np.sum(c >= 0.45)) if (n_bank and model_type == "3PL") else 0

    resp = {m: mat.loc[m, items].to_numpy(float) for m in test}
    traces = {m: S.full_cat_traces(resp[m], a, b, c, S.pred_meanprob) for m in test}

    per_se = {}
    for se in ses:
        recs = []
        for m in test:
            chosen, reached = None, False
            for ni, s, pr in traces[m]:
                if ni >= S.MIN_ITEMS and s <= se:
                    chosen, reached = (ni, pr), True
                    break
            if chosen is None:
                chosen = (traces[m][-1][0], traces[m][-1][2])
            recs.append(
                {
                    "model": m,
                    "n_items": int(chosen[0]),
                    "pred": float(chosen[1]),
                    "act": float(actual[m]),
                    "reached": bool(reached),
                }
            )
        per_se[se] = recs

    diag = {
        "n_train": len(train),
        "n_items_raw": report.n_items_raw,
        "kept_filtered": n_filtered,
        "bank_items": n_bank,
        "n_dropped_unusable": n_dropped,
        "a_min": round(float(a.min()), 3) if n_bank else float("nan"),
        "a_max": round(float(a.max()), 3) if n_bank else float("nan"),
        "a_pinned_lo": a_lo,
        "a_pinned_hi": a_hi,
        "c_min": round(float(c.min()), 3) if n_bank else float("nan"),
        "c_max": round(float(c.max()), 3) if n_bank else float("nan"),
        "c_pinned_hi": c_hi,
        "fit_seconds": round(fit_secs, 1),
    }
    return per_se, diag


def run_job(job: dict) -> dict:
    mat = _STATE["mat"]
    actual = _STATE["actual"]
    per_se, diag = _fit_cat(mat, actual, job["train"], job["test"], job["model_type"], SE_TARGETS)
    return {"key": job["key"], "per_se": {se: recs for se, recs in per_se.items()}, "diag": diag}


# --------------------------------------------------------------------------- #
# Aggregation helpers                                                          #
# --------------------------------------------------------------------------- #
def _agg(recs: list[dict]) -> dict:
    pred = np.array([r["pred"] for r in recs], float)
    act = np.array([r["act"] for r in recs], float)
    nit = np.array([r["n_items"] for r in recs], float)
    reached = np.array([r["reached"] for r in recs], bool)
    r = float(np.corrcoef(pred, act)[0, 1]) if len(pred) >= 2 else float("nan")
    return {
        "r": r,
        "mae": float(np.mean(np.abs(pred - act))),
        "mean_items": float(np.mean(nit)),
        "frac_reached": float(np.mean(reached)),
        "n_eval": int(len(pred)),
    }


# --------------------------------------------------------------------------- #
# Figure                                                                       #
# --------------------------------------------------------------------------- #
def make_figure(controlled: dict, out_path: Path, n_test: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    x = np.arange(len(MODEL_TYPES))
    w = 0.38
    pool_styles = [("calib40", "40-model pool", "#BBBBBB"), ("calib66", "66-model pool", "#4C72B0")]
    for ax, se in zip(axes, (0.3, 0.15), strict=False):
        for i, (label, legend, color) in enumerate(pool_styles):
            vals = [controlled[(label, mt, se)]["r"] for mt in MODEL_TYPES]
            xs = x + (i - 0.5) * w
            ax.bar(xs, vals, w, label=legend, color=color, edgecolor="black", linewidth=0.6)
            for xi, v in zip(xs, vals, strict=False):
                ax.annotate(
                    f"{v:.3f}",
                    (xi, v),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    xytext=(0, 1),
                    textcoords="offset points",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_TYPES)
        ax.set_title(f"Held-out Pearson r at SE<={se:g}")
        ax.set_xlabel("IRT model")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Pearson r (CAT-predicted vs actual full accuracy)")
    # legend sits in the right panel where the 3PL bars leave clear white space
    axes[1].legend(title="calibration pool", loc="upper right", fontsize=9)
    fig.suptitle(
        "1PL vs 2PL vs 3PL on pedagogy, controlled 40 vs 66 calibration pool\n"
        f"66-model calibration pool (78-model pedagogy set), {n_test} held-out",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mat = load_benchmark(MCQ_DIR, BENCH).dropna(axis=0, how="any")
    all_models = list(mat.index)
    new_models = [m for m in all_models if m.replace("/", "__") in NEW_STEMS]
    old_models = [m for m in all_models if m.replace("/", "__") not in NEW_STEMS]
    assert len(new_models) == 26, f"expected 26 new, got {len(new_models)}"
    assert len(old_models) == 52, f"expected 52 old, got {len(old_models)}"
    old_set = set(old_models)
    print(
        f"loaded {len(all_models)} models, {mat.shape[1]} items; "
        f"OLD={len(old_models)} NEW_added={len(new_models)}",
        flush=True,
    )

    # --- design partitions -------------------------------------------------- #
    old_train, old_test = seed7_split(old_models)  # 40 train, 12 test (faithful OLD split)
    calib40 = sorted(old_train)
    calib66 = sorted(list(old_train) + list(new_models))
    naive_train, naive_test = seed7_split(all_models)  # 66 train, 12 test on the full 78
    old_folds = kfold_folds(old_models)
    new_folds = kfold_folds(all_models)

    # --- build the job list ------------------------------------------------- #
    jobs = []
    for mt in MODEL_TYPES:
        jobs.append(
            {
                "key": ("controlled", "calib40", mt),
                "train": calib40,
                "test": old_test,
                "model_type": mt,
            }
        )
        jobs.append(
            {
                "key": ("controlled", "calib66", mt),
                "train": calib66,
                "test": old_test,
                "model_type": mt,
            }
        )
        jobs.append(
            {
                "key": ("naive", "single66", mt),
                "train": naive_train,
                "test": naive_test,
                "model_type": mt,
            }
        )
        for f, fold in enumerate(old_folds):
            train = [m for m in sorted(old_models) if m not in set(fold)]
            jobs.append(
                {"key": ("kfold", "old52", mt, f), "train": train, "test": fold, "model_type": mt}
            )
        for f, fold in enumerate(new_folds):
            train = [m for m in sorted(all_models) if m not in set(fold)]
            jobs.append(
                {"key": ("kfold", "new78", mt, f), "train": train, "test": fold, "model_type": mt}
            )

    print(f"running {len(jobs)} fit+CAT jobs (workers={args.workers})", flush=True)

    def _log(res):
        k = res["key"]
        d = res["diag"]
        print(
            f"done {'.'.join(str(x) for x in k):<28} bank={d['bank_items']:<4} "
            f"a=[{d['a_min']},{d['a_max']}] cmax={d['c_max']} ({d['fit_seconds']}s)",
            flush=True,
        )

    def run_sequential():
        _init_worker(str(MCQ_DIR), BENCH)
        out = []
        for job in jobs:
            res = run_job(job)
            out.append(res)
            _log(res)
        return out

    results = []
    if args.workers > 1:
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker, initargs=(str(MCQ_DIR), BENCH)
            ) as ex:
                for res in ex.map(run_job, jobs):
                    results.append(res)
                    _log(res)
        except (PermissionError, OSError) as e:
            print(f"process pool unavailable ({e}); running sequentially", flush=True)
            results = run_sequential()
    else:
        results = run_sequential()

    by_key = {res["key"]: res for res in results}

    # --- controlled --------------------------------------------------------- #
    controlled = {}
    controlled_diag = {}
    for label in ("calib40", "calib66"):
        for mt in MODEL_TYPES:
            res = by_key[("controlled", label, mt)]
            controlled_diag[(label, mt)] = res["diag"]
            for se in SE_TARGETS:
                controlled[(label, mt, se)] = _agg(res["per_se"][se])

    # --- kfold (pool folds) ------------------------------------------------- #
    kfold = {}
    for set_label in ("old52", "new78"):
        for mt in MODEL_TYPES:
            for se in SE_TARGETS:
                pooled = []
                for f in range(K):
                    pooled.extend(by_key[("kfold", set_label, mt, f)]["per_se"][se])
                kfold[(set_label, mt, se)] = _agg(pooled)
                if set_label == "new78":
                    sub = [r for r in pooled if r["model"] in old_set]
                    kfold[("new_on_old52", mt, se)] = _agg(sub)

    # --- naive -------------------------------------------------------------- #
    naive = {}
    naive_diag = {}
    for mt in MODEL_TYPES:
        res = by_key[("naive", "single66", mt)]
        naive_diag[mt] = res["diag"]
        for se in SE_TARGETS:
            naive[(mt, se)] = _agg(res["per_se"][se])

    # ------------------------------------------------------------------ #
    # write CSVs                                                          #
    # ------------------------------------------------------------------ #
    # 1) controlled
    p1 = args.out_dir / "pl_1_2_3_expanded_controlled.csv"
    with open(p1, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model_type",
                "se_stop",
                "r_40",
                "r_66",
                "delta_r",
                "items_40",
                "items_66",
                "mae_40",
                "mae_66",
                "frac_reached_40",
                "frac_reached_66",
                "kept_items_40",
                "kept_items_66",
            ]
        )
        for se in SE_TARGETS:
            for mt in MODEL_TYPES:
                a40 = controlled[("calib40", mt, se)]
                a66 = controlled[("calib66", mt, se)]
                w.writerow(
                    [
                        mt,
                        se,
                        round(a40["r"], 4),
                        round(a66["r"], 4),
                        round(a66["r"] - a40["r"], 4),
                        round(a40["mean_items"], 2),
                        round(a66["mean_items"], 2),
                        round(a40["mae"], 4),
                        round(a66["mae"], 4),
                        round(a40["frac_reached"], 2),
                        round(a66["frac_reached"], 2),
                        controlled_diag[("calib40", mt)]["bank_items"],
                        controlled_diag[("calib66", mt)]["bank_items"],
                    ]
                )
    print("wrote", p1)

    # 2) kfold
    p2 = args.out_dir / "pl_1_2_3_expanded_kfold.csv"
    with open(p2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model_type",
                "se_stop",
                "K",
                "r_old52",
                "r_new78",
                "r_new_on_old52",
                "items_old52",
                "items_new78",
                "mae_old52",
                "mae_new78",
                "n_eval_old52",
                "n_eval_new78",
            ]
        )
        for se in SE_TARGETS:
            for mt in MODEL_TYPES:
                o = kfold[("old52", mt, se)]
                n = kfold[("new78", mt, se)]
                s = kfold[("new_on_old52", mt, se)]
                w.writerow(
                    [
                        mt,
                        se,
                        K,
                        round(o["r"], 4),
                        round(n["r"], 4),
                        round(s["r"], 4),
                        round(o["mean_items"], 2),
                        round(n["mean_items"], 2),
                        round(o["mae"], 4),
                        round(n["mae"], 4),
                        o["n_eval"],
                        n["n_eval"],
                    ]
                )
    print("wrote", p2)

    # 3) naive single split
    p3 = args.out_dir / "pl_1_2_3_expanded_single_split.csv"
    caveat = (
        "CAVEAT: naive seed-7 draw of 12 held-out from 78 spans a narrow accuracy band, "
        "which deflates Pearson r; use the controlled and k-fold designs for conclusions"
    )
    with open(p3, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["model_type", "se_stop", "r", "items", "mae", "frac_reached", "kept_items", "caveat"]
        )
        for se in SE_TARGETS:
            for mt in MODEL_TYPES:
                d = naive[(mt, se)]
                w.writerow(
                    [
                        mt,
                        se,
                        round(d["r"], 4),
                        round(d["mean_items"], 2),
                        round(d["mae"], 4),
                        round(d["frac_reached"], 2),
                        naive_diag[mt]["bank_items"],
                        caveat,
                    ]
                )
    print("wrote", p3)

    # 4) fit diagnostics at the 66-model pool (40 shown for contrast)
    p4 = args.out_dir / "fit_diagnostics.csv"
    with open(p4, "w", newline="") as fh:
        fields = [
            "pool",
            "model_type",
            "n_train",
            "n_items_raw",
            "kept_filtered",
            "bank_items",
            "n_dropped_unusable",
            "a_min",
            "a_max",
            "a_pinned_lo",
            "a_pinned_hi",
            "c_min",
            "c_max",
            "c_pinned_hi",
            "fit_seconds",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for label in ("calib40", "calib66"):
            for mt in MODEL_TYPES:
                d = controlled_diag[(label, mt)]
                w.writerow({"pool": label, "model_type": mt, **d})
    print("wrote", p4)

    # 5) per-model predictions (audit trail across all designs)
    p5 = args.out_dir / "per_model_predictions.csv"
    with open(p5, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "design",
                "label",
                "model_type",
                "se_stop",
                "model",
                "actual_full",
                "pred",
                "n_items",
                "reached",
            ]
        )
        for (design, label, mt, *rest), res in sorted(by_key.items(), key=lambda kv: str(kv[0])):
            for se in SE_TARGETS:
                for r in res["per_se"][se]:
                    fold = rest[0] if rest else ""
                    lab = f"{label}" + (f"_fold{fold}" if fold != "" else "")
                    w.writerow(
                        [
                            design,
                            lab,
                            mt,
                            se,
                            r["model"],
                            round(r["act"], 6),
                            round(r["pred"], 6),
                            r["n_items"],
                            int(r["reached"]),
                        ]
                    )
    print("wrote", p5)

    # ------------------------------------------------------------------ #
    # figure                                                             #
    # ------------------------------------------------------------------ #
    fig_path = args.out_dir / "pl_1_2_3_expanded78.png"
    make_figure(controlled, fig_path, len(old_test))
    print("wrote", fig_path)

    # ------------------------------------------------------------------ #
    # validation                                                         #
    # ------------------------------------------------------------------ #
    print("\n=== validation: 1PL/2PL reproduce prior fair-design numbers ===", flush=True)
    ok = True

    def _check(tag, got, ref):
        nonlocal ok
        match = got is not None and abs(got - ref) <= VALID_TOL
        ok &= match
        flag = "OK" if match else "MISMATCH"
        print(f"  {tag:<34} got={got:.4f} ref={ref:.4f} {flag}", flush=True)

    for mt in ("1PL", "2PL"):
        for se in SE_TARGETS:
            r40_ref, r66_ref = REF_CONTROLLED[(mt, se)]
            _check(
                f"controlled {mt} SE<={se:g} r_40", controlled[("calib40", mt, se)]["r"], r40_ref
            )
            _check(
                f"controlled {mt} SE<={se:g} r_66", controlled[("calib66", mt, se)]["r"], r66_ref
            )
    for mt in ("1PL", "2PL"):
        for se in SE_TARGETS:
            _check(
                f"kfold OLD52 {mt} SE<={se:g} r",
                kfold[("old52", mt, se)]["r"],
                REF_KFOLD_OLD[(mt, se)],
            )
            _check(
                f"kfold NEW78 {mt} SE<={se:g} r",
                kfold[("new78", mt, se)]["r"],
                REF_KFOLD_NEW[(mt, se)],
            )
    print("  --- naive single-split cross-check (not a gate) ---", flush=True)
    for mt in ("1PL", "2PL"):
        for se in SE_TARGETS:
            got = naive[(mt, se)]["r"]
            ref = REF_NAIVE[(mt, se)]
            print(
                f"  naive {mt} SE<={se:g} r  got={got:.4f} ref={ref:.4f} "
                f"{'OK' if abs(got - ref) <= VALID_TOL else 'DIFF'}",
                flush=True,
            )
    print(f"=== validation {'PASSED' if ok else 'FAILED'} (tol={VALID_TOL}) ===\n", flush=True)

    # ------------------------------------------------------------------ #
    # headline table to stdout                                           #
    # ------------------------------------------------------------------ #
    print("HEADLINE  controlled (r_40 -> r_66)   |  k-fold (old52 / new78)")
    for se in SE_TARGETS:
        print(f"  SE<={se:g}")
        for mt in MODEL_TYPES:
            c40 = controlled[("calib40", mt, se)]["r"]
            c66 = controlled[("calib66", mt, se)]["r"]
            ko = kfold[("old52", mt, se)]["r"]
            kn = kfold[("new78", mt, se)]["r"]
            it40 = controlled[("calib40", mt, se)]["mean_items"]
            it66 = controlled[("calib66", mt, se)]["mean_items"]
            print(
                f"    {mt}:  r {c40:.4f} -> {c66:.4f}  (items {it40:.1f} -> {it66:.1f})   |  "
                f"kfold {ko:.4f} / {kn:.4f}"
            )

    print("\n3PL bounds-pinning (controlled banks):")
    for label in ("calib40", "calib66"):
        d = controlled_diag[(label, "3PL")]
        print(
            f"  {label}: bank={d['bank_items']} a_pinned_lo={d['a_pinned_lo']} "
            f"a_pinned_hi={d['a_pinned_hi']} c_pinned_hi={d['c_pinned_hi']} "
            f"a=[{d['a_min']},{d['a_max']}] c=[{d['c_min']},{d['c_max']}]"
        )

    return {
        "controlled": controlled,
        "controlled_diag": controlled_diag,
        "kfold": kfold,
        "naive": naive,
        "naive_diag": naive_diag,
        "ok": ok,
        "n_test": len(old_test),
    }


if __name__ == "__main__":
    main()
