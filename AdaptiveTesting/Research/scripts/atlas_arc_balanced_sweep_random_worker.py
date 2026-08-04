#!/usr/bin/env python3
"""Parallel worker: RANDOM strategy + full-pool reference for the balanced ATLAS ARC sweep.

One of three concurrent CPU-local workers of the `atlas_arc_balanced_sweep` experiment.
This worker computes the full-pool reference and the random calibration train-size sweep,
writing its rows to `results_random.csv` so they merge cleanly with the sibling workers'
per-strategy CSVs (same shared setup -> identical pool, test set, and metric conventions).

Everything shared is reused unchanged (imported, never edited):
  * atlas_arc_balanced_sweep.load_data    -> combined ARC model x item matrix, the ~3.7k-model
    train pool, and the fixed 417-model held-out ARC test set (shared column intersection)
  * oracle_subset_selection.draw_pool      -> capped 500-model "full pool" (numpy rng SEED)
  * oracle_subset_selection.select_random  -> N-model random draw from pool (numpy rng 1000+seed)
  * openlm_trainsize_sweep.prepare_step / fit_and_link -> chunked R mirt 3PL fit +
    mean-sigma linking with polarity flip + a>0 item filter
  * openlm_trainsize_sweep.load_matrix / load_bank      -> held-out matrix + fitted item bank
  * atlas_error_plots_openlm.adaptive_trace / read_off  -> SE<=0.3 Fisher-info EAP CAT
    (MIN_ITEMS=8, max 400 items)

Random @ N=500 draws the entire capped pool (a permutation of all 500 models), so its bank
and held-out CAT are identical to the full-pool reference; those rows reuse the full-pool
metrics instead of recomputing the same heavy calibration.

Usage (CPU-local, uv):
  uv run python AdaptiveTesting/Research/scripts/atlas_arc_balanced_sweep_random_worker.py
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import atlas_arc_balanced_sweep as arc  # noqa: E402  shared harness: ARC data loading + paths
import atlas_error_plots_openlm as errplots  # noqa: E402  SE<=0.3 Fisher CAT (adaptive_trace/read_off)
import openlm_trainsize_sweep as sweep  # noqa: E402  chunked R mirt 3PL calibration + bank/matrix IO
import oracle_subset_selection as oss  # noqa: E402  draw_pool + select_random (shared draws)

# --- shared setup (must match sibling workers EXACTLY so rows merge) ---
POOL_SEED = 12345
POOL_CAP = 500
N_GRID = [100, 200, 300, 400, 500]
SEEDS = [0, 1, 2]
SE = 0.3
FLOOR_SE = 0.3  # CAT stop target; read_off at SE<=0.3 is invariant to any tighter floor
MAX_ITEMS = 400
CHUNK_SIZE = 100
NCYCLES = 500
TEST_SEED = 7  # only used if the test set is capped (it is not here: all 417 kept)

# --- local execution knobs (do not affect the recorded numbers) ---
JOBS = 4  # parallel calibration tasks; <=4 as three workers share the box
WORKERS = 1  # R chunk-fit threads per task

OUT_ROOT = arc.OUT_ROOT
RESULTS = OUT_ROOT / "results_random.csv"
WORK = OUT_ROOT / "_work_random" / arc.BENCH  # isolated from sibling work dirs
FIELDS = [
    "strategy",
    "N",
    "seed",
    "se_target",
    "r",
    "mae",
    "items",
    "n_bank_items",
    "n_pool",
    "n_test",
]


def run_cat(step_dir: Path) -> dict:
    """SE<=0.3 Fisher-info EAP CAT on the held-out matrix using the freshly fitted bank."""
    test = sweep.load_matrix(step_dir / "data" / "response_matrix_test.csv")
    idxs, a, b, c = sweep.load_bank(step_dir / "calibration", list(test.columns))
    preds, acts, items = [], [], []
    for mid in test.index:
        resp = test.loc[mid, idxs].to_numpy(float)
        acts.append(float(resp.mean()))
        steps = errplots.adaptive_trace(resp, a, b, c, FLOOR_SE, MAX_ITEMS)
        ni, _se, _theta, pred = errplots.read_off(steps, SE)
        preds.append(pred)
        items.append(ni)
    preds = np.asarray(preds)
    acts = np.asarray(acts)
    r = float(np.corrcoef(preds, acts)[0, 1]) if preds.std() > 0 else float("nan")
    return {
        "corr": round(r, 4),
        "mae": round(float(np.mean(np.abs(preds - acts))), 4),
        "mean_items": round(float(np.mean(items)), 2),
        "n_bank_items": len(idxs),
    }


def calibrate_and_cat(subset, step_dir, fit_r, link_r, mat, test_models) -> dict:
    """Chunked R mirt 3PL calibration on `subset`, then the SE<=0.3 held-out CAT."""
    ends, _ = sweep.prepare_step(mat, subset, test_models, step_dir, CHUNK_SIZE)
    sweep.fit_and_link(step_dir, ends, NCYCLES, WORKERS, fit_r, link_r)
    return run_cat(step_dir)


def _row(strategy, n, seed, m, n_pool, n_test) -> dict:
    return {
        "strategy": strategy,
        "N": n,
        "seed": seed,
        "se_target": SE,
        "r": m["corr"],
        "mae": m["mae"],
        "items": m["mean_items"],
        "n_bank_items": m["n_bank_items"],
        "n_pool": n_pool,
        "n_test": n_test,
    }


def _key(r: dict) -> tuple:
    return (str(r["strategy"]), int(r["N"]), int(r["seed"]), float(r["se_target"]))


def write_results(rows: list[dict]) -> None:
    """Merge/incremental write so partial progress survives and re-runs are idempotent."""
    import pandas as pd

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(RESULTS).to_dict("records") if RESULTS.exists() else []
    merged = {_key(r): r for r in existing}
    for r in rows:
        merged[_key(r)] = r
    out = sorted(
        merged.values(),
        key=lambda r: (str(r["strategy"]), float(r["se_target"]), int(r["N"]), int(r["seed"])),
    )
    with RESULTS.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k) for k in FIELDS})


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fit_r, link_r = sweep.write_helper_scripts(WORK)

    mat, train_pool, test_models = arc.load_data(SimpleNamespace(n_test=0, test_seed=TEST_SEED))
    pool = oss.draw_pool(train_pool, POOL_CAP, POOL_SEED)
    n_pool, n_test = len(pool), len(test_models)
    grid = [n for n in N_GRID if n <= n_pool]
    print(
        f"[arc-random] pool={n_pool} test={n_test} grid={grid} "
        f"jobs={JOBS} workers={WORKERS} pool_seed={POOL_SEED}",
        flush=True,
    )

    # Heavy calibrations: the full pool + random draws for N < pool_cap. Random @ N=pool_cap
    # selects the whole pool, so it reuses the full-pool metrics (see module docstring).
    tasks: list[tuple[str, int, int]] = [("full_pool", n_pool, 0)]
    for n in grid:
        if n >= n_pool:
            continue
        for seed in SEEDS:
            tasks.append(("random", n, seed))

    def run_task(strategy: str, n: int, seed: int):
        if strategy == "full_pool":
            subset, sd = list(pool), WORK / "full_pool"
        else:
            subset, sd = oss.select_random(pool, n, seed), WORK / f"random_s{seed}_n{n}"
        t0 = time.time()
        m = calibrate_and_cat(subset, sd, fit_r, link_r, mat, test_models)
        return _row(strategy, n, seed, m, n_pool, n_test), round(time.time() - t0, 1)

    rows: list[dict] = []
    full_metrics: dict | None = None
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(run_task, s, n, sd): (s, n, sd) for (s, n, sd) in tasks}
        for fut in as_completed(futs):
            s, n, sd = futs[fut]
            try:
                row, secs = fut.result()
                rows.append(row)
                if s == "full_pool":
                    full_metrics = {
                        "corr": row["r"],
                        "mae": row["mae"],
                        "mean_items": row["items"],
                        "n_bank_items": row["n_bank_items"],
                    }
                print(
                    f"[arc-random] {s:<10} N={n:<4} seed={sd} bank={row['n_bank_items']:<4} "
                    f"SE{SE} r={row['r']} mae={row['mae']} items={row['items']} ({secs}s)",
                    flush=True,
                )
            except Exception as exc:  # keep sweeping past a bad fit
                print(f"[arc-random] {s} N={n} seed={sd} FAILED: {exc}", flush=True)
                rows.append(
                    _row(
                        s,
                        n,
                        sd,
                        {"corr": float("nan"), "mae": float("nan"),
                         "mean_items": float("nan"), "n_bank_items": 0},
                        n_pool,
                        n_test,
                    )
                )
            write_results(rows)

    # random @ N == pool_cap is the whole pool -> identical bank/CAT to the full-pool reference.
    if full_metrics is not None and n_pool in grid:
        for seed in SEEDS:
            rows.append(_row("random", n_pool, seed, full_metrics, n_pool, n_test))

    write_results(rows)
    print(f"[arc-random] DONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
