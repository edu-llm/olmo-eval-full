#!/usr/bin/env python3
"""Full K-fold cross-validation of the OpenLM ATLAS 3PL p-IRT CAT for one benchmark.

Every model receives exactly ONE out-of-sample prediction. We shuffle all models
(seed 7), split into K folds, and for each fold recalibrate the 3PL item bank on the
other K-1 folds using the validated pipeline in ``openlm_trainsize_sweep.py`` (chunked
R ``mirt`` 3PL fit, mean-sigma chunk linking with polarity flip, a>0 item filter). We
then run the SE<=target adaptive Fisher-information CAT defined in
``atlas_error_plots_openlm.py`` (``adaptive_trace``/``read_off``, MIN_ITEMS=8, max 400
items, EAP theta) on the held-out fold. Concatenating the folds yields one OOS row per
model, directly comparable to the single-split ``atlas_replication`` result.

Per held-out model we record: predicted accuracy (p-IRT), actual full-benchmark
accuracy (mean correctness over all usable a>0 bank items), and #CAT items.

Outputs go to
  AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv/<bench>/
    pirt_vs_actual_cv_se<SE>.csv   (model, pirt, actual, items, fold, n_bank)
    results.json                   (Pearson r, Slope, Accuracy MAE, 95% CI MAE,
                                    Avg CAT items, Bank size, n models, wall-clock)

CPU-local, uv run. No AWS, no git.

Usage:
  uv run python AdaptiveTesting/Research/scripts/openlm_atlas_cv.py \
      --bench ifeval --folds 10 --seed 7 --se 0.3 --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import atlas_error_plots_openlm as atlas  # noqa: E402  (same-dir helper module)
import openlm_trainsize_sweep as sweep  # noqa: E402  (validated calibration pipeline)

REPO = SCRIPTS.parents[2]
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv"


def make_folds(models: list[str], seed: int, k: int) -> list[list[str]]:
    """Shuffle models with a fixed seed and partition into k folds (every model once)."""
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    return [[order[i] for i in grp] for grp in np.array_split(np.arange(len(order)), k)]


def diagnose_fold(step_dir: Path, se_target: float, floor_se: float, max_items: int):
    """Run the ATLAS SE<=se_target adaptive CAT on the held-out fold's test matrix.

    Uses the re-calibrated per-fold bank (a>0 items only). Returns per-model rows
    (model, pirt, actual, items) plus the usable bank size for this fold.
    """
    test = sweep.load_matrix(step_dir / "data" / "response_matrix_test.csv")
    idxs, a, b, c = sweep.load_bank(step_dir / "calibration", list(test.columns))
    rows = []
    for m in test.index:
        resp = test.loc[m, idxs].to_numpy(float)
        actual = float(resp.mean())  # actual accuracy over all usable a>0 items
        steps = atlas.adaptive_trace(resp, a, b, c, floor_se, max_items)
        ni, _se_reached, _theta, pred = atlas.read_off(steps, se_target)
        rows.append({"model": m, "pirt": float(pred), "actual": actual, "items": int(ni)})
    return rows, len(idxs)


def compute_metrics(df: pd.DataFrame, bank_sizes: list[int], args) -> dict:
    p = df["pirt"].to_numpy(float)
    a = df["actual"].to_numpy(float)
    ae = np.abs(p - a)
    n = len(df)
    r = float(np.corrcoef(p, a)[0, 1])
    slope = float(np.polyfit(a, p, 1)[0])  # OLS slope of predicted on actual (ideal 1.0)
    mae = float(ae.mean())
    half = 1.96 * float(ae.std()) / np.sqrt(n)  # np.std -> matches atlas_replication SE
    bank_mean = float(np.mean(bank_sizes)) if bank_sizes else float("nan")
    return {
        "Benchmark": args.bench,
        "Pearson r": round(r, 4),
        "Slope": round(slope, 4),
        "Accuracy MAE": round(mae, 4),
        "95% CI MAE": [round(mae - half, 4), round(mae + half, 4)],
        "Avg CAT items": round(float(df["items"].mean()), 2),
        "Bank size": round(bank_mean, 1),
        "n models": int(n),
        "bank_size_per_fold": [int(x) for x in bank_sizes],
        "se_target": args.se,
        "folds": args.folds,
        "seed": args.seed,
    }


def _write_outputs(
    out_dir: Path, df: pd.DataFrame, bank_sizes: list[int], args, wall_s: float, final: bool
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["model", "pirt", "actual", "items", "fold", "n_bank"]
    df[cols].to_csv(out_dir / f"pirt_vs_actual_cv_se{args.se:g}.csv", index=False)
    metrics = compute_metrics(df, bank_sizes, args)
    metrics["wall_clock_s"] = round(wall_s, 1)
    metrics["complete"] = bool(final)
    (out_dir / "results.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run_cv(args: argparse.Namespace) -> None:
    bench = args.bench
    out_dir = OUT_ROOT / bench
    work = out_dir / "_work"
    fit_r, link_r = sweep.write_helper_scripts(work)

    print(f"[{bench}] building master matrix...", flush=True)
    mat = sweep.build_master(bench)
    folds = make_folds(list(mat.index), args.seed, args.folds)
    # sanity: folds partition the model set exactly (each model out-of-sample once)
    flat = [m for f in folds for m in f]
    assert len(flat) == len(set(flat)) == len(mat), "folds must partition all models"
    print(
        f"[{bench}] models={len(mat)} items={mat.shape[1]} folds={args.folds} "
        f"seed={args.seed} SE<={args.se} fold_sizes={[len(f) for f in folds]}",
        flush=True,
    )

    all_rows: list[dict] = []
    bank_sizes: list[int] = []
    t0 = time.time()
    n_run = args.folds if args.limit_folds <= 0 else min(args.limit_folds, args.folds)
    for k, test_models in enumerate(folds):
        fold_dir = work / f"fold{k}"
        cache = fold_dir / "fold_results.csv"
        if cache.exists() and not args.force:
            fdf = pd.read_csv(cache)
            all_rows.extend(fdf.to_dict("records"))
            bank_sizes.append(int(fdf["n_bank"].iloc[0]))
            print(
                f"[{bench}] fold {k}: cached ({len(fdf)} models, "
                f"bank={int(fdf['n_bank'].iloc[0])})",
                flush=True,
            )
            continue
        if k >= n_run:
            print(f"[{bench}] fold {k}: skipped (limit-folds={n_run})", flush=True)
            continue

        train_models = [m for j, f2 in enumerate(folds) if j != k for m in f2]
        tf = time.time()
        ends, n_kept = sweep.prepare_step(mat, train_models, test_models, fold_dir, args.chunk_size)
        sweep.fit_and_link(fold_dir, ends, args.ncycles, args.workers, fit_r, link_r)
        rows, nb = diagnose_fold(fold_dir, args.se, args.floor_se, args.max_items)
        for r in rows:
            r["fold"] = k
            r["n_bank"] = nb
        pd.DataFrame(rows).to_csv(cache, index=False)
        all_rows.extend(rows)
        bank_sizes.append(nb)

        fp = np.array([r["pirt"] for r in rows])
        fa = np.array([r["actual"] for r in rows])
        fr = float(np.corrcoef(fp, fa)[0, 1]) if fp.std() > 0 else float("nan")
        print(
            f"[{bench}] fold {k}: n_train={len(train_models)} n_test={len(test_models)} "
            f"kept_items={n_kept} bank={nb} fold_r={fr:.4f} "
            f"items={np.mean([r['items'] for r in rows]):.1f} ({round(time.time() - tf, 1)}s)",
            flush=True,
        )
        # incremental save so partial progress survives interruptions
        _write_outputs(
            out_dir, pd.DataFrame(all_rows), bank_sizes, args, time.time() - t0, final=False
        )

    df = pd.DataFrame(all_rows)
    complete = len(df) == len(mat)
    metrics = _write_outputs(out_dir, df, bank_sizes, args, time.time() - t0, final=complete)
    _print_row(metrics, complete)


def _print_row(m: dict, complete: bool) -> None:
    print("\n" + "=" * 78, flush=True)
    print(
        f"OpenLM ATLAS {m['folds']}-fold CV  |  SE<={m['se_target']}  |  seed {m['seed']}"
        + ("" if complete else "   (PARTIAL)"),
        flush=True,
    )
    print("=" * 78, flush=True)
    print(f"  Benchmark      : {m['Benchmark']}", flush=True)
    print(f"  Pearson r      : {m['Pearson r']}", flush=True)
    print(f"  Slope          : {m['Slope']}", flush=True)
    print(f"  Accuracy MAE   : {m['Accuracy MAE']}", flush=True)
    print(f"  95% CI MAE     : [{m['95% CI MAE'][0]}, {m['95% CI MAE'][1]}]", flush=True)
    print(f"  Avg CAT items  : {m['Avg CAT items']}", flush=True)
    print(f"  Bank size      : {m['Bank size']}", flush=True)
    print(f"  n models       : {m['n models']}", flush=True)
    print(f"  wall-clock (s) : {m['wall_clock_s']}", flush=True)
    print("=" * 78, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default="ifeval")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--se", type=float, default=0.3, help="SE stopping target")
    p.add_argument(
        "--floor-se",
        type=float,
        default=0.095,
        help="tight SE floor for the single full CAT trace per model",
    )
    p.add_argument("--max-items", type=int, default=400)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--workers", type=int, default=2, help="R chunk-fit workers (keep modest)")
    p.add_argument(
        "--limit-folds",
        type=int,
        default=0,
        help="only compute the first N fold indices this run (0 = all)",
    )
    p.add_argument("--force", action="store_true", help="recompute folds even if cached")
    args = p.parse_args()
    run_cv(args)


if __name__ == "__main__":
    main()
