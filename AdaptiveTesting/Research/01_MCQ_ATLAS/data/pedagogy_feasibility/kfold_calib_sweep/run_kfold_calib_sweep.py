#!/usr/bin/env python3
"""K-fold CV learning curve for the pedagogy CAT diagnostic: pooled recovery
correlation as the calibration pool grows from 10 to 70 models, 1PL vs 2PL.

Design (k-fold at each calibration pool size N):
  * K-fold model-level CV (K=10) over the 78 pedagogy models, so every model is
    held out exactly once and predicted out-of-sample. Each fold holds out ~7-8
    models and leaves ~70-71 for training.
  * At each pool size N in {10..70}, for each fold, sample exactly N calibration
    models from that fold's training portion, calibrate the bank on them, filter
    items on that calibration slice, run the SE-stopped CAT on the fold's held-out
    models, and read off the predicted accuracy at each SE target. Pool the
    out-of-sample predictions across all folds, then compute ONE pooled Pearson r
    vs the actual full-bank (920-item) pedagogy accuracy. That pooled r is the
    correlation "at that N".
  * Repeat the calibration subsampling with a few seeds and report the mean and SD
    of pooled r across seeds as an uncertainty band. At N=70 the subsample is
    essentially the whole training portion, so the band is tiny by construction.

Machinery is REUSED, not reinvented:
  * Fitters: ``se_sweep_small_pool.fit_bank(kept_df, model_type)`` -> girth
    ``rasch_mml`` (1PL) / ``twopl_mml`` (2PL), with the same usability filter
    (finite, a>0) applied downstream as in ``se_sweep_small_pool.run_cell``. NO 3PL.
  * Item filter: ``tutor_cat.mcq_irt.matrix.filter_items`` (ATLAS point-biserial).
  * CAT: ``se_sweep.full_cat_traces`` + ``se_sweep.pred_meanprob``, MIN_ITEMS=8,
    read off (items, prediction) at each SE target from each model's trace.

The girth 2PL fits dominate runtime, so unique (model_type, fold, calibration
subset) cells are fitted once (identical subsets across seeds are deduplicated)
and dispatched over a process pool.

CPU-only, torch-free, ``uv run`` friendly, local ``.mplcache``. Writes only into
``kfold_calib_sweep/`` plus the single requested figure in ``figures/``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Keep BLAS single-threaded so many parallel girth fits do not oversubscribe. On
# macOS numpy links Apple Accelerate, which honours VECLIB_MAXIMUM_THREADS and
# ignores the OpenMP/OpenBLAS/MKL knobs, so set all of them before numpy imports.
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402  (must follow the BLAS thread caps above)

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
REPO = HERE.parents[5]  # .../olmo-eval-full
sys.path.insert(0, str(REPO / "eduLLM-Evals"))
sys.path.insert(0, str(REPO / "AdaptiveTesting" / "Research" / "scripts"))

import se_sweep as S  # noqa: E402  (full_cat_traces / pred_meanprob / MIN_ITEMS)
from se_sweep_small_pool import fit_bank  # noqa: E402  (girth rasch_mml / twopl_mml)
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402

MCQ_DIR = (REPO / "AdaptiveTesting" / "Research" / "01_MCQ_ATLAS" / "data"
           / "pedagogy_feasibility" / "expanded_calibration" / "_mcq_data")
FIG_PATH = (REPO / "AdaptiveTesting" / "Research" / "01_MCQ_ATLAS" / "figures"
            / "pedagogy_kfold_corr_vs_calibN.png")
BENCH = "pedagogy"
SE_TARGETS = [0.3, 0.15]
PRIMARY_SE = 0.3
MODEL_TYPES = ("1PL", "2PL")

FOLD_SEED = 7          # partition seed (matches run_kfold_cv.py convention)
SUBSAMPLE_SEED_BASE = 1000  # per-(seed, fold, N) calibration subsample seeds

# Prior 6-fold NEW/78 pooled r at SE<=0.3 (expanded_calibration/kfold_cv_summary.csv).
BASELINE_R = {"2PL": 0.7654, "1PL": 0.7577}

# --------------------------------------------------------------------------- #
# Worker (one unique calibration cell)                                         #
# --------------------------------------------------------------------------- #
_MAT = None  # per-worker global, set by the pool initializer
_USE_USABILITY = True  # drop non-finite / a<=0 items (matches se_sweep_small_pool)


def _init_worker(mat, use_usability=True):
    global _MAT, _USE_USABILITY
    _MAT = mat
    _USE_USABILITY = use_usability


def subsample(train, n_pool, seed_key):
    """Sample exactly n_pool models from train; the whole train if n_pool>=len."""
    if n_pool >= len(train):
        return list(train)
    rng = np.random.default_rng(seed_key)
    idx = rng.permutation(len(train))[:n_pool]
    return [train[i] for i in idx]


def run_cell(job):
    """Fit one bank on a calibration subset and read off held-out CAT predictions.

    Returns (key, {se: {model: (pred, n_items)}}, n_bank_items).
    """
    key, model_type, test, sub = job
    mat = _MAT

    kept, _rep = filter_items(mat.loc[sub], benchmark=BENCH)
    items, a, b, c = fit_bank(kept, model_type)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)

    # Same usability filter as se_sweep_small_pool.run_cell (drop non-finite / a<=0).
    # Disabled only for the validation control that mirrors the 6-fold baseline,
    # which kept every calibrated item.
    if _USE_USABILITY:
        usable = (
            np.isfinite(a) & (a > 0) & np.isfinite(b) & np.isfinite(c)
            & (c >= 0) & (c < 1.0)
        )
        items = [it for it, keep in zip(items, usable, strict=False) if keep]
        a, b, c = a[usable], b[usable], c[usable]
    n_bank = len(items)

    out = {se: {} for se in SE_TARGETS}
    for m in test:
        resp = mat.loc[m, items].to_numpy(float)
        steps = S.full_cat_traces(resp, a, b, c, S.pred_meanprob)
        for se in SE_TARGETS:
            chosen = None
            for (ni, s, pr) in steps:
                if ni >= S.MIN_ITEMS and s <= se:
                    chosen = (pr, ni)
                    break
            if chosen is None:  # SE target never reached: use full-CAT endpoint
                chosen = (steps[-1][2], steps[-1][0])
            out[se][m] = chosen
    return key, out, n_bank


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def build_folds(models, k):
    rng = np.random.default_rng(FOLD_SEED)
    perm = rng.permutation(len(models))
    return [[models[i] for i in perm[f::k]] for f in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="10,20,30,40,50,60,70")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3, help="number of subsample seeds")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--se-list", default="0.3,0.15")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--no-usability", action="store_true",
                    help="validation control: keep every calibrated item (mirrors "
                         "the 6-fold baseline); default applies the usability filter")
    ap.add_argument("--figure-only", action="store_true",
                    help="regenerate the figure and README from the existing CSV "
                         "without recomputing the sweep")
    args = ap.parse_args()
    use_usability = not args.no_usability

    if args.figure_only:
        regenerate_from_csv(args.out_dir)
        return

    global SE_TARGETS
    SE_TARGETS = [float(x) for x in args.se_list.split(",")]
    Ns = [int(x) for x in args.pools.split(",")]
    K = args.k
    n_seeds = args.seeds
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mat = load_benchmark(MCQ_DIR, BENCH).dropna(axis=0, how="any")
    models = list(mat.index)
    actual = {m: float(v) for m, v in mat.mean(axis=1).items()}
    print(f"loaded {len(models)} models x {mat.shape[1]} items; K={K} seeds={n_seeds} "
          f"pools={Ns} usability_filter={use_usability}", flush=True)

    folds = build_folds(models, K)
    fold_train = [[m for m in models if m not in set(fold)] for fold in folds]
    print("fold test sizes:", [len(f) for f in folds],
          "| train sizes:", [len(t) for t in fold_train], flush=True)

    # Build unique fit cells (dedup identical subsets across seeds) and a request
    # map from (N, model_type, seed) -> [(fold_idx, key), ...].
    jobs = {}
    req = {}
    for N in Ns:
        for mt in MODEL_TYPES:
            for s in range(n_seeds):
                lst = []
                for f in range(K):
                    sub = subsample(fold_train[f], N, [SUBSAMPLE_SEED_BASE + s, f, N])
                    subt = tuple(sorted(sub))
                    key = (mt, f, subt)
                    if key not in jobs:
                        jobs[key] = (key, mt, folds[f], list(sub))
                    lst.append((f, key))
                req[(N, mt, s)] = lst

    job_list = list(jobs.values())
    n_2pl = sum(1 for j in job_list if j[1] == "2PL")
    print(f"unique calibration cells: {len(job_list)} ({n_2pl} are 2PL); "
          f"dispatching over {args.workers} workers", flush=True)

    # Run all unique cells.
    results = {}
    t0 = time.time()
    done = [0]

    def _log(res):
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(job_list):
            print(f"  cells {done[0]}/{len(job_list)}  ({time.time() - t0:.0f}s)", flush=True)

    def run_sequential():
        for job in job_list:
            key, out, nb = run_cell(job)
            results[key] = (out, nb)
            _log((key, out, nb))

    if args.workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                     initargs=(mat, use_usability)) as ex:
                for key, out, nb in ex.map(run_cell, job_list):
                    results[key] = (out, nb)
                    _log((key, out, nb))
        except (PermissionError, OSError) as e:
            print(f"process pool unavailable ({e}); running sequentially", flush=True)
            _init_worker(mat, use_usability)
            results.clear()
            run_sequential()
    else:
        _init_worker(mat, use_usability)
        run_sequential()

    print(f"all cells done in {time.time() - t0:.0f}s", flush=True)

    # Pool across folds for each (N, model_type, seed), then aggregate over seeds.
    def pooled(N, mt, s, se):
        preds, acts, nits = [], [], []
        for _f, key in req[(N, mt, s)]:
            out, _nb = results[key]
            for m, (pred, ni) in out[se].items():
                preds.append(pred)
                acts.append(actual[m])
                nits.append(ni)
        preds = np.asarray(preds)
        acts = np.asarray(acts)
        r = float(np.corrcoef(preds, acts)[0, 1])
        mae = float(np.mean(np.abs(preds - acts)))
        return r, mae, float(np.mean(nits)), len(preds)

    rows = []
    for N in Ns:
        for mt in MODEL_TYPES:
            for se in SE_TARGETS:
                rs, maes, its, npts = [], [], [], []
                for s in range(n_seeds):
                    r, mae, mit, n = pooled(N, mt, s, se)
                    rs.append(r)
                    maes.append(mae)
                    its.append(mit)
                    npts.append(n)
                rows.append({
                    "N": N, "model": mt, "se_target": se,
                    "pooled_r_mean": round(float(np.mean(rs)), 4),
                    "pooled_r_sd": round(float(np.std(rs, ddof=0)), 4),
                    "mae": round(float(np.mean(maes)), 4),
                    "mean_items": round(float(np.mean(its)), 2),
                    "n_seeds": n_seeds, "K": K,
                    "n_pooled": int(np.mean(npts)),
                })

    # ------------------------------------------------------------------ #
    # CSV                                                                 #
    # ------------------------------------------------------------------ #
    csv_path = args.out_dir / "kfold_poolsize_sweep.csv"
    fields = ["N", "model", "se_target", "pooled_r_mean", "pooled_r_sd", "mae",
              "mean_items", "n_seeds", "K"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in sorted(rows, key=lambda x: (x["model"], x["se_target"], x["N"])):
            w.writerow({k: row[k] for k in fields})
    print("wrote", csv_path, flush=True)

    # ------------------------------------------------------------------ #
    # Console summary + validation                                        #
    # ------------------------------------------------------------------ #
    def cell(N, mt, se):
        for row in rows:
            if row["N"] == N and row["model"] == mt and row["se_target"] == se:
                return row
        return None

    print("\n== pooled Pearson r at SE<=0.3 (mean +/- SD over seeds) ==", flush=True)
    header = "  N   " + "".join(f"{n:>10}" for n in Ns)
    print(header, flush=True)
    for mt in MODEL_TYPES:
        line = f"  {mt}: " + "".join(
            f"{cell(n, mt, PRIMARY_SE)['pooled_r_mean']:>6.3f}"
            f"+-{cell(n, mt, PRIMARY_SE)['pooled_r_sd']:<3.2f}" for n in Ns)
        print(line, flush=True)

    cross = crossover_N(rows, Ns)
    if cross is None:
        print("\n2PL never overtakes 1PL at SE<=0.3 over the swept range", flush=True)
    else:
        print(f"\n2PL overtakes 1PL at SE<=0.3 near N={cross:.1f}", flush=True)

    top_N = max(Ns)
    print(f"\n== validation at N={top_N} vs prior 6-fold NEW/78 ==", flush=True)
    for mt in MODEL_TYPES:
        got = cell(top_N, mt, PRIMARY_SE)["pooled_r_mean"]
        base = BASELINE_R[mt]
        print(f"  {mt}: N={top_N} pooled r={got:.3f}  (prior 6-fold={base:.3f}, "
              f"delta={got - base:+.3f})", flush=True)

    # ------------------------------------------------------------------ #
    # Figure + README (skipped for the --no-usability validation control  #
    # so it never clobbers the primary figure/README)                     #
    # ------------------------------------------------------------------ #
    if not args.no_usability:
        make_figure(rows, Ns, n_seeds, K, cross)
        write_readme(args.out_dir, rows, Ns, n_seeds, K, cross)


def regenerate_from_csv(out_dir):
    """Rebuild the figure and README from an existing kfold_poolsize_sweep.csv."""
    csv_path = out_dir / "kfold_poolsize_sweep.csv"
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "N": int(r["N"]), "model": r["model"],
                "se_target": float(r["se_target"]),
                "pooled_r_mean": float(r["pooled_r_mean"]),
                "pooled_r_sd": float(r["pooled_r_sd"]),
                "mae": float(r["mae"]), "mean_items": float(r["mean_items"]),
                "n_seeds": int(r["n_seeds"]), "K": int(r["K"]),
            })
    Ns = sorted({r["N"] for r in rows})
    n_seeds = rows[0]["n_seeds"]
    K = rows[0]["K"]
    cross = crossover_N(rows, Ns)
    make_figure(rows, Ns, n_seeds, K, cross)
    write_readme(out_dir, rows, Ns, n_seeds, K, cross)


def crossover_N(rows, Ns):
    """First N (linearly interpolated) where 2PL pooled r exceeds 1PL at SE<=0.3."""
    def r_at(mt, n):
        for row in rows:
            if row["N"] == n and row["model"] == mt and row["se_target"] == PRIMARY_SE:
                return row["pooled_r_mean"]
        return np.nan

    diff = [r_at("2PL", n) - r_at("1PL", n) for n in Ns]
    for i in range(1, len(Ns)):
        if diff[i - 1] < 0 <= diff[i]:
            x0, x1 = Ns[i - 1], Ns[i]
            d0, d1 = diff[i - 1], diff[i]
            return x0 + (x1 - x0) * (0 - d0) / (d1 - d0)
    if diff[0] >= 0:  # 2PL already ahead at the smallest N
        return float(Ns[0])
    return None


def make_figure(rows, Ns, n_seeds, K, cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(mt):
        m = [next(r["pooled_r_mean"] for r in rows
                  if r["N"] == n and r["model"] == mt and r["se_target"] == PRIMARY_SE)
             for n in Ns]
        sd = [next(r["pooled_r_sd"] for r in rows
                   if r["N"] == n and r["model"] == mt and r["se_target"] == PRIMARY_SE)
              for n in Ns]
        return np.asarray(m), np.asarray(sd)

    colors = {"1PL": "#55A868", "2PL": "#C44E52"}
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    for mt in ("1PL", "2PL"):
        m, sd = series(mt)
        ax.plot(Ns, m, "o-", color=colors[mt], lw=2.8, ms=9, label=mt, zorder=3)
        ax.fill_between(Ns, m - sd, m + sd, color=colors[mt], alpha=0.20, zorder=1)

    lo = min(min(series(mt)[0] - series(mt)[1]) for mt in ("1PL", "2PL"))
    hi = max(max(series(mt)[0] + series(mt)[1]) for mt in ("1PL", "2PL"))
    ax.set_ylim(lo - 0.03, hi + 0.04)

    # Crossover marker, placed in the empty lower-middle band so it clears both
    # the curves (top) and the legend (lower right).
    if cross is not None and Ns[0] < cross < Ns[-1]:
        ax.axvline(cross, color="#444444", ls="--", lw=1.8, zorder=2)
        ymid = lo + 0.10
        ax.annotate(f"2PL overtakes 1PL\nnear N={cross:.0f}",
                    xy=(cross, ymid), xytext=(cross - 1.5, ymid),
                    fontsize=12.5, fontweight="bold", color="#333333",
                    ha="right", va="center")

    ax.set_xlabel("Number of calibration models (N)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Pooled Pearson r (predicted vs actual, SE<=0.3)",
                  fontsize=13.5, fontweight="bold")
    ax.set_title(f"Pedagogy K-fold CV learning curve: recovery r vs calibration pool size\n"
                 f"K={K} folds, {n_seeds} seeds; shaded band = SD across seeds",
                 fontsize=12.5, fontweight="bold")
    ax.set_xticks(Ns)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="IRT model", fontsize=13, title_fontsize=13, loc="lower right")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print("wrote", FIG_PATH, flush=True)


def write_readme(out_dir, rows, Ns, n_seeds, K, cross):
    def get(mt, n, se, field="pooled_r_mean"):
        for r in rows:
            if r["N"] == n and r["model"] == mt and r["se_target"] == se:
                return r[field]
        return None

    top_N = max(Ns)
    lines = []
    lines.append("# Pedagogy K-fold CV learning curve: recovery correlation vs "
                 "calibration pool size\n")
    lines.append("## Design\n")
    lines.append(
        f"- Data: expanded pedagogy set (78 models x 920 items), "
        f"`expanded_calibration/_mcq_data` via `load_benchmark(.., \"pedagogy\")`.\n"
        f"- Model-level {K}-fold CV over the 78 models (partition seed {FOLD_SEED}); "
        f"each fold holds out ~7-8 models, leaving ~70-71 for training, so every "
        f"model is predicted out-of-sample exactly once.\n"
        f"- At each calibration pool size N in {{{', '.join(str(n) for n in Ns)}}}, for "
        f"each fold we sample exactly N calibration models from that fold's training "
        f"portion, filter items (ATLAS point-biserial `filter_items`) on that slice, "
        f"calibrate the bank (girth `rasch_mml` for 1PL, `twopl_mml` for 2PL, with the "
        f"finite/a>0 usability filter), run the SE-stopped Fisher-information CAT "
        f"(MIN_ITEMS=8, mean-probability predictor) on the fold's held-out models, and "
        f"read off the prediction at each SE target from each model's trace.\n"
        f"- Predictions are pooled across all folds; the pooled Pearson r vs the actual "
        f"full-bank (920-item) pedagogy accuracy is the correlation at that N.\n"
        f"- The calibration subsampling is repeated with {n_seeds} seeds; we report the "
        f"mean and SD of pooled r across seeds (the band). At N={top_N} the subsample is "
        f"essentially the whole training portion, so the band is tiny by construction.\n"
        f"- 1PL and 2PL only; NO 3PL.\n"
    )

    lines.append("\n## Pooled Pearson r at each N (SE<=0.3)\n")
    lines.append("| N | 1PL r (SD) | 2PL r (SD) | 2PL - 1PL |")
    lines.append("|---:|:---:|:---:|:---:|")
    for n in Ns:
        r1 = get("1PL", n, PRIMARY_SE)
        s1 = get("1PL", n, PRIMARY_SE, "pooled_r_sd")
        r2 = get("2PL", n, PRIMARY_SE)
        s2 = get("2PL", n, PRIMARY_SE, "pooled_r_sd")
        lines.append(f"| {n} | {r1:.3f} ({s1:.3f}) | {r2:.3f} ({s2:.3f}) | {r2 - r1:+.3f} |")

    lines.append("\n## Pooled Pearson r at each N (SE<=0.15)\n")
    lines.append("| N | 1PL r (SD) | 2PL r (SD) | 2PL - 1PL |")
    lines.append("|---:|:---:|:---:|:---:|")
    for n in Ns:
        r1 = get("1PL", n, 0.15)
        s1 = get("1PL", n, 0.15, "pooled_r_sd")
        r2 = get("2PL", n, 0.15)
        s2 = get("2PL", n, 0.15, "pooled_r_sd")
        if r1 is None or r2 is None:
            continue
        lines.append(f"| {n} | {r1:.3f} ({s1:.3f}) | {r2:.3f} ({s2:.3f}) | {r2 - r1:+.3f} |")

    lines.append("\n## Crossover\n")
    if cross is None:
        lines.append("Over the swept range, 2PL does not overtake 1PL at SE<=0.3.\n")
    elif cross <= Ns[0]:
        lines.append(f"2PL is already at or above 1PL at the smallest pool (N={Ns[0]}).\n")
    else:
        lines.append(f"2PL overtakes 1PL at SE<=0.3 near N={cross:.0f} calibration models. "
                     f"Below that pool size 1PL's regularized single-parameter bank is the "
                     f"safer recovery model; above it the extra 2PL discrimination parameter "
                     f"pays off.\n")

    lines.append("\n## Validation vs prior 6-fold numbers\n")
    lines.append(f"At N={top_N} (essentially the full training portion) the {K}-fold pooled r "
                 f"should sit near the previously reported 6-fold NEW/78 numbers "
                 f"(2PL SE0.3 ~= {BASELINE_R['2PL']:.3f}, "
                 f"1PL SE0.3 ~= {BASELINE_R['1PL']:.3f}).\n\n")
    lines.append(f"| model | N={top_N} pooled r (SE0.3) | prior 6-fold | delta |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for mt in MODEL_TYPES:
        got = get(mt, top_N, PRIMARY_SE)
        base = BASELINE_R[mt]
        lines.append(f"| {mt} | {got:.3f} | {base:.3f} | {got - base:+.3f} |")

    lines.append("\n## Plain-language summary\n")
    r1_lo = get("1PL", Ns[0], PRIMARY_SE)
    r1_hi = get("1PL", top_N, PRIMARY_SE)
    r2_lo = get("2PL", Ns[0], PRIMARY_SE)
    r2_hi = get("2PL", top_N, PRIMARY_SE)
    if cross is None:
        cross_txt = ("closes almost the entire gap but does not fully overtake 1PL "
                     "within the swept range")
    elif cross <= Ns[0]:
        cross_txt = "is at or above 1PL from the smallest pool onward"
    else:
        cross_txt = f"overtakes 1PL around N={cross:.0f} calibration models"
    if r2_hi > r1_hi:
        top_txt = (f"the 2PL bank gives the best recovery (2PL r={r2_hi:.3f} vs "
                   f"1PL r={r1_hi:.3f}), matching the earlier full-set result")
    else:
        top_txt = (f"1PL and 2PL are essentially tied (1PL r={r1_hi:.3f} vs "
                   f"2PL r={r2_hi:.3f}), matching the earlier full-set result")
    lines.append(
        f"This learning curve shows how well a short adaptive test recovers a model's "
        f"full pedagogy score as you add more calibration models. With very few "
        f"calibration models the simpler 1PL bank is the more reliable choice "
        f"(1PL r={r1_lo:.3f} vs 2PL r={r2_lo:.3f} at N={Ns[0]}), because a single "
        f"difficulty parameter per item is easy to estimate from thin data while 2PL "
        f"discriminations are noisy. As the pool grows, 2PL improves faster and "
        f"{cross_txt}. By N={top_N} {top_txt}. In short, use 1PL when calibration data "
        f"is scarce and move to 2PL once you have enough models to estimate "
        f"discriminations reliably.\n"
    )

    (out_dir / "README.md").write_text("\n".join(lines))
    print("wrote", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
