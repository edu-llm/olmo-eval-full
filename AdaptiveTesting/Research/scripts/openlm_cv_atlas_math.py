#!/usr/bin/env python3
"""Full K-fold cross-validation ATLAS replication for one OpenLM benchmark (MATH).

Every model receives exactly ONE out-of-sample p-IRT prediction. We split the
cleaned model set into K folds (shuffled, fixed seed), and for each fold we
recalibrate the 3PL item bank on the other K-1 folds using the SAME validated
pipeline as ``openlm_trainsize_sweep.py`` (chunked R ``mirt`` 3PL fit, mean-sigma
chunk linking with polarity flip, a>0 item filter) and then run the SE-target
adaptive CAT from ``atlas_error_plots_openlm.py`` (``adaptive_trace``:
MIN_ITEMS=8, max 400 items, EAP theta, Fisher-information selection) on the
held-out fold. Concatenating all folds yields one out-of-sample row per model,
from which we compute the ATLAS accuracy-recovery metrics.

Per-model ``actual`` accuracy is the mean correctness over the usable (a>0) bank
items of that fold, matching the ``actual_accuracy`` field in
``atlas_error_plots_openlm.py`` (apples-to-apples with the p-IRT prediction,
which is defined over the same bank).

This module deliberately uses a benchmark-specific filename to avoid colliding
with concurrent sibling CV runs; it only imports the stable shared pipeline
modules, never another CV driver.

Outputs (per benchmark) under
  AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv/<bench>/
    per_model.csv   (model, pirt, actual, items, fold)
    results.json    (the reporting row + provenance)

Usage:
  uv run python AdaptiveTesting/Research/scripts/openlm_cv_atlas_math.py \
      --bench math --folds 10 --seed 7 --se 0.3 --workers 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import atlas_error_plots_openlm as atlas
import numpy as np
import openlm_trainsize_sweep as sweep
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv"


def make_folds(models: list[str], k: int, seed: int) -> list[list[str]]:
    """Shuffle models with a fixed seed, then split into k near-equal folds.

    Matches sklearn ``KFold(shuffle=True)`` semantics: permute the model list
    once, then take contiguous chunks. Every model lands in exactly one fold.
    """
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    return [list(chunk) for chunk in np.array_split(np.array(order, dtype=object), k)]


def run_fold(mat, train_models, test_models, step_dir, args, fit_r, link_r, ncycles):
    """Recalibrate on train_models, run the SE-target CAT on test_models.

    Returns (rows, n_bank). ``rows`` are per-held-out-model dicts with the raw
    fields (model, pirt, actual, items).
    """
    ends, _ = sweep.prepare_step(mat, train_models, test_models, step_dir, args.chunk_size)
    sweep.fit_and_link(step_dir, ends, ncycles, args.workers, fit_r, link_r)
    test = sweep.load_matrix(step_dir / "data" / "response_matrix_test.csv")
    idxs, a, b, c = sweep.load_bank(step_dir / "calibration", list(test.columns))
    n_bank = len(idxs)
    if n_bank == 0:
        raise RuntimeError("empty a>0 bank after linking")
    rows = []
    for mid in test.index:
        resp = test.loc[mid, idxs].to_numpy(float)
        actual = float(resp.mean())
        # floor_se == the SE target: adaptive_trace stops at the first step with
        # SE<=target and >=MIN_ITEMS, exactly what read_off(steps, target) returns.
        # Equivalent to the tight-floor multi-target trace in
        # atlas_error_plots_openlm.py for a single target, but much cheaper.
        steps = atlas.adaptive_trace(resp, a, b, c, floor_se=args.se, max_items=args.max_items)
        ni, _se_reached, _theta, pred = atlas.read_off(steps, args.se)
        rows.append({"model": str(mid), "pirt": float(pred),
                     "actual": actual, "items": int(ni)})
    return rows, n_bank


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _load_fold_cache(step_dir: Path):
    cache = step_dir / "fold_result.json"
    if cache.exists():
        try:
            obj = json.loads(cache.read_text())
            return obj["rows"], int(obj["n_bank"])
        except Exception:
            return None
    return None


def _save_fold_cache(step_dir: Path, rows, n_bank: int) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "fold_result.json").write_text(
        json.dumps({"n_bank": int(n_bank), "rows": rows})
    )


def _write_partial(out_dir: Path, all_rows: list[dict]) -> None:
    if not all_rows:
        return
    pd.DataFrame(all_rows, columns=["model", "pirt", "actual", "items", "fold"]).to_csv(
        out_dir / "per_model.csv", index=False
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="math")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--se", type=float, default=0.3, help="CAT SE stopping target")
    ap.add_argument("--max-items", type=int, default=400, dest="max_items")
    ap.add_argument("--chunk-size", type=int, default=100, dest="chunk_size")
    ap.add_argument("--ncycles", type=int, default=500, help="mirt EM NCYCLES (standard)")
    ap.add_argument("--workers", type=int, default=2, help="R chunk-fit workers per fold (<=2)")
    ap.add_argument("--retries", type=int, default=1,
                    help="extra attempts per fold on failure (standard settings, clean dir)")
    ap.add_argument("--work-root", default="")
    ap.add_argument("--matrix-cache", default="", dest="matrix_cache",
                    help="pickle path for the built master matrix (default: work/master_<bench>.pkl)")
    args = ap.parse_args()

    bench = args.bench
    out_dir = OUT_ROOT / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    work = (Path(args.work_root) / bench) if args.work_root else (OUT_ROOT / "_work" / bench)
    fit_r, link_r = sweep.write_helper_scripts(work)

    t_start = time.time()
    # Cache the (memory-heavy) build_master parse to disk so it runs exactly once.
    # This lets the run resume cheaply if it is OOM-killed under concurrent load,
    # and keeps the fold loop itself light on memory.
    cache_path = Path(args.matrix_cache) if args.matrix_cache else work / f"master_{bench}.pkl"
    if cache_path.exists():
        print(f"[{bench}] loading cached master matrix {cache_path}...", flush=True)
        mat = pd.read_pickle(cache_path)
    else:
        print(f"[{bench}] building master matrix (all models) via sweep.build_master...", flush=True)
        mat = sweep.build_master(bench)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".pkl.tmp")
        mat.to_pickle(tmp)
        tmp.replace(cache_path)  # atomic: a kill mid-write never leaves a partial cache
        print(f"[{bench}] cached master matrix -> {cache_path}", flush=True)
    models = list(mat.index)
    folds = make_folds(models, args.folds, args.seed)
    print(
        f"[{bench}] models={len(models)} items={mat.shape[1]} folds={args.folds} "
        f"seed={args.seed} se={args.se} fold_sizes={[len(f) for f in folds]}",
        flush=True,
    )

    all_rows: list[dict] = []
    bank_sizes: list[int] = []
    dropped: list[int] = []

    for i, test_models in enumerate(folds):
        step_dir = work / f"fold{i}"
        cached = _load_fold_cache(step_dir)
        if cached is not None:
            rows, n_bank = cached
            for r in rows:
                r["fold"] = i
            all_rows.extend(rows)
            bank_sizes.append(n_bank)
            print(f"[{bench}] fold {i}: CACHED held-out={len(rows)} bank={n_bank} "
                  f"cumulative_models={len(all_rows)}", flush=True)
            _write_partial(out_dir, all_rows)
            continue

        test_set = set(test_models)
        train_models = [m for m in models if m not in test_set]
        t0 = time.time()
        rows = None
        n_bank = 0
        for attempt in range(1, args.retries + 2):
            try:
                if attempt > 1:
                    shutil.rmtree(step_dir, ignore_errors=True)
                rows, n_bank = run_fold(
                    mat, train_models, test_models, step_dir, args, fit_r, link_r, args.ncycles,
                )
                break
            except Exception as exc:  # keep CV going; drop a truly unrecoverable fold
                print(f"[{bench}] fold {i} attempt {attempt}/{args.retries + 1} "
                      f"FAILED: {exc}", flush=True)
                rows = None
        if rows is None:
            dropped.append(i)
            print(f"[{bench}] fold {i} DROPPED after {args.retries + 1} attempts", flush=True)
            continue

        _save_fold_cache(step_dir, rows, n_bank)
        for r in rows:
            r["fold"] = i
        all_rows.extend(rows)
        bank_sizes.append(n_bank)
        print(f"[{bench}] fold {i}: held-out={len(rows)} bank={n_bank} "
              f"({round(time.time() - t0, 1)}s) cumulative_models={len(all_rows)}", flush=True)
        _write_partial(out_dir, all_rows)

    if not all_rows:
        raise SystemExit(f"[{bench}] no folds succeeded; nothing to report")

    df = pd.DataFrame(all_rows, columns=["model", "pirt", "actual", "items", "fold"])
    df.to_csv(out_dir / "per_model.csv", index=False)

    pirt = df["pirt"].to_numpy(float)
    actual = df["actual"].to_numpy(float)
    items = df["items"].to_numpy(float)
    n = len(df)
    abs_err = np.abs(pirt - actual)
    mae = float(abs_err.mean())
    std_abs = float(abs_err.std())  # ddof=0, matches atlas_error_plots_openlm.py
    se_mae = std_abs / np.sqrt(n)
    ci_low, ci_high = mae - 1.96 * se_mae, mae + 1.96 * se_mae
    r = float(np.corrcoef(pirt, actual)[0, 1])
    slope = float(np.polyfit(actual, pirt, 1)[0])
    avg_items = float(items.mean())
    bank_mean = float(np.mean(bank_sizes))

    row = {
        "Benchmark": bench,
        "Pearson r": round(r, 4),
        "Slope": round(slope, 4),
        "Accuracy MAE": round(mae, 4),
        "95% CI MAE": [round(ci_low, 4), round(ci_high, 4)],
        "Avg CAT items": round(avg_items, 2),
        "Bank size": round(bank_mean, 1),
        "n_models": n,
    }
    results = {
        **row,
        "se_target": args.se,
        "folds": args.folds,
        "seed": args.seed,
        "bank_size_mean": round(bank_mean, 2),
        "bank_size_min": int(min(bank_sizes)),
        "bank_size_max": int(max(bank_sizes)),
        "bank_size_per_fold": [int(x) for x in bank_sizes],
        "fold_sizes": [len(f) for f in folds],
        "n_folds_used": args.folds - len(dropped),
        "dropped_folds": dropped,
        "mae_std_abs_err": round(std_abs, 6),
        "mae_se": round(se_mae, 6),
        "actual_accuracy_definition": (
            "mean correctness over usable (a>0) bank items per fold (ATLAS convention)"
        ),
        "pipeline": (
            "openlm_trainsize_sweep: chunked R mirt 3PL + mean-sigma polarity link + a>0 filter; "
            "atlas_error_plots_openlm.adaptive_trace CAT (MIN_ITEMS=8, max 400, Fisher/EAP)"
        ),
        "wall_clock_s": round(time.time() - t_start, 1),
        "wall_clock": _hms(time.time() - t_start),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== 10-fold CV ATLAS replication (out-of-sample, SE<={args.se:g}) ===", flush=True)
    print(f"Benchmark      : {bench}", flush=True)
    print(f"Pearson r      : {row['Pearson r']}", flush=True)
    print(f"Slope          : {row['Slope']}", flush=True)
    print(f"Accuracy MAE   : {row['Accuracy MAE']}", flush=True)
    print(f"95% CI MAE     : [{row['95% CI MAE'][0]}, {row['95% CI MAE'][1]}]", flush=True)
    print(f"Avg CAT items  : {row['Avg CAT items']}", flush=True)
    print(f"Bank size      : {row['Bank size']} (min {results['bank_size_min']}, "
          f"max {results['bank_size_max']})", flush=True)
    print(f"n models       : {n}", flush=True)
    if dropped:
        print(f"DROPPED FOLDS  : {dropped}", flush=True)
    print(f"wall-clock     : {results['wall_clock']} ({results['wall_clock_s']}s)", flush=True)
    print(f"\nwrote {out_dir / 'per_model.csv'} and {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
