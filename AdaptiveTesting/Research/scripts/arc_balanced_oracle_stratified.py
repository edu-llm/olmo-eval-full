#!/usr/bin/env python3
"""ATLAS ARC balanced-pool sweep -- ORACLE_STRATIFIED worker.

Reverse-engineers the theoretical-best latent-ability-spanning subset from the
final full-pool calibration: calibrate the whole capped pool, read each pool
model's fitted EAP theta (its oracle latent ability) from the linked bank,
split the pool into latent-ability quantile groups, then for each N spread the
selection across those groups (and across theta within each group), recalibrate
on that subset, and run the ATLAS Fisher/EAP CAT on the held-out test set.

Only the choice of the N calibration models changes across N; the capped pool
(500 train models drawn with default_rng(12345), sorted) and the held-out set
(all 417 test models) are the fixed shared setup used by every sibling worker.

All heavy lifting reuses the validated shared library functions (not edited):
  openlm_trainsize_sweep : prepare_step, fit_and_link, write_helper_scripts,
                           load_matrix, load_bank, eap_se
  atlas_error_plots_openlm: adaptive_trace, read_off (SE<=0.3, MIN_ITEMS=8,
                           max 400 items, Fisher-information / EAP CAT)
  oracle_subset_selection : full_pool_calibration, latent_groups,
                           select_oracle_stratified
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import openlm_trainsize_sweep as sweep  # noqa: E402
import oracle_subset_selection as oss  # noqa: E402
from atlas_error_plots_openlm import adaptive_trace, read_off  # noqa: E402

REPO = SCRIPTS.parents[2]
DATA = REPO / "AdaptiveTesting/Inputs/ATLAS/data"
TRAIN_CSV = DATA / "gaussian_sampled_arc_response_matrix_train.csv"
TEST_CSV = DATA / "gaussian_sampled_arc_response_matrix_test.csv"
OUT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_arc_balanced_sweep"

# Shared setup (must match sibling workers EXACTLY)
POOL_SEED = 12345
POOL_CAP = 500
GROUPS = 5
SE_TARGET = 0.3
N_GRID = [100, 200, 300, 400, 500]
FLOOR_SE = 0.3          # trace floor; read_off at SE<=0.3 is invariant to any floor<=0.3
MAX_ITEMS = 400
CHUNK_SIZE = 100
NCYCLES = 500
WORKERS = 2             # R chunk-fit workers per calibration
BENCH = "arc"

FIELDS = [
    "strategy", "N", "seed", "se_target",
    "r", "mae", "items", "n_bank_items", "n_pool", "n_test",
]


def load_arc_matrix(path: Path) -> pd.DataFrame:
    """Load a wide ATLAS response matrix, dropping any trailing non-item
    (e.g. score) column and coercing item columns to int labels / float 0-1."""
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "model"}).set_index("model")
    item_cols = [c for c in df.columns if str(c).strip().lstrip("-").isdigit()]
    df = df[item_cols]
    df.columns = [int(c) for c in df.columns]
    return df.astype(float)


def cat_metrics(step_dir: Path) -> dict:
    """Run the ATLAS Fisher/EAP adaptive CAT (adaptive_trace + read_off at
    SE<=0.3) on the held-out test matrix of a calibrated step and return the
    held-out accuracy-recovery metrics."""
    test = sweep.load_matrix(step_dir / "data" / "response_matrix_test.csv")
    idxs, a, b, c = sweep.load_bank(step_dir / "calibration", list(test.columns))
    preds, acts, nitems = [], [], []
    for mid in test.index:
        resp = test.loc[mid, idxs].to_numpy(float)
        acts.append(float(resp.mean()))
        steps = adaptive_trace(resp, a, b, c, FLOOR_SE, MAX_ITEMS)
        ni, _se, _theta, pred = read_off(steps, SE_TARGET)
        preds.append(pred)
        nitems.append(ni)
    preds = np.asarray(preds)
    acts = np.asarray(acts)
    r = float(np.corrcoef(preds, acts)[0, 1]) if preds.std() > 0 else float("nan")
    return {
        "r": round(r, 4),
        "mae": round(float(np.mean(np.abs(preds - acts))), 4),
        "items": round(float(np.mean(nitems)), 2),
        "n_bank_items": len(idxs),
    }


def write_latent_groups(bounds: list[dict]) -> None:
    path = OUT / "latent_groups_arc.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "theta_lo", "theta_hi", "size", "mean_theta"])
        w.writeheader()
        w.writerows(bounds)
    print(f"[arc] wrote {path}", flush=True)


def write_results(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: int(r["N"]))
    path = OUT / "results_stratified.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    print(f"[arc] wrote {path}", flush=True)


def cross_check(pool: list[str], theta_by_model: dict[str, float]) -> None:
    """Sanity-check the reproduced pool + oracle thetas against any existing
    sibling artifacts (selected_subsets.csv) without modifying them."""
    ss = OUT / "selected_subsets.csv"
    if not ss.exists():
        return
    df = pd.read_csv(ss)
    ref500 = df[(df["strategy"] == "oracle_stratified") & (df["N"] == 500)]
    if not len(ref500):
        return
    pool_set, ref_set = set(pool), set(ref500["model"])
    inter = len(pool_set & ref_set)
    print(f"[xcheck] pool vs existing N=500 set: {inter}/{len(ref_set)} overlap "
          f"(pool={len(pool_set)})", flush=True)
    ref_theta = dict(zip(ref500["model"], ref500["theta"], strict=False))
    diffs = [abs(theta_by_model[m] - ref_theta[m]) for m in pool_set & ref_set
             if m in theta_by_model]
    if diffs:
        print(f"[xcheck] oracle theta max|Δ|={max(diffs):.4f} "
              f"mean|Δ|={np.mean(diffs):.4f} vs existing", flush=True)


def main() -> None:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ncpu = os.cpu_count() or 4
    jobs = min(4, max(1, ncpu // WORKERS)) or 1
    print(f"[arc] cpu={ncpu} jobs={jobs} workers={WORKERS}", flush=True)

    train_df = load_arc_matrix(TRAIN_CSV)
    test_df = load_arc_matrix(TEST_CSV)
    train_models = list(train_df.index)
    test_models = list(test_df.index)
    print(f"[arc] train={train_df.shape} test={test_df.shape}", flush=True)

    # Capped pool: EXACTLY 500 of the 3,747 train models, default_rng(12345), sorted.
    pool = oss.draw_pool(train_models, POOL_CAP, POOL_SEED)
    assert len(pool) == POOL_CAP, len(pool)

    mat = pd.concat([train_df.loc[pool], test_df])
    assert mat.index.is_unique, "pool and test models overlap"

    work = OUT / "_work" / BENCH
    fit_r, link_r = sweep.write_helper_scripts(work)
    args = SimpleNamespace(
        chunk_size=CHUNK_SIZE, ncycles=NCYCLES, workers=WORKERS, work_root=str(OUT / "_work"),
    )
    ses = [SE_TARGET]

    # Step 1: full-pool calibration -> per-model oracle EAP theta (+ reference r).
    _full_metrics, theta_by_model, _s = oss.full_pool_calibration(
        args, BENCH, mat, test_models, pool, fit_r, link_r, ses
    )

    # Step 2: latent-ability quantile groups.
    group_of, bounds = oss.latent_groups(theta_by_model, GROUPS)
    write_latent_groups(bounds)
    for bnd in bounds:
        print(f"[arc] group {bnd['group']}: theta[{bnd['theta_lo']},{bnd['theta_hi']}] "
              f"size={bnd['size']} mean={bnd['mean_theta']}", flush=True)
    cross_check(pool, theta_by_model)

    full_dir = work / "full_pool"

    # Step 3: oracle_stratified selection + calibration + CAT for each N.
    def run_n(n: int) -> dict:
        subset = oss.select_oracle_stratified(theta_by_model, group_of, pool, GROUPS, n)
        if n >= POOL_CAP:
            step_dir = full_dir  # N=500 == full pool: reuse its calibrated bank
        else:
            step_dir = work / f"oracle_stratified_s0_n{n}"
            ends, _ = sweep.prepare_step(mat, subset, test_models, step_dir, CHUNK_SIZE)
            sweep.fit_and_link(step_dir, ends, NCYCLES, WORKERS, fit_r, link_r)
        m = cat_metrics(step_dir)
        return {
            "strategy": "oracle_stratified", "N": n, "seed": 0, "se_target": SE_TARGET,
            "r": m["r"], "mae": m["mae"], "items": m["items"],
            "n_bank_items": m["n_bank_items"], "n_pool": POOL_CAP, "n_test": len(test_models),
        }

    rows: list[dict] = []
    subset_ns = [n for n in N_GRID if n < POOL_CAP]
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_n, n): n for n in subset_ns}
        for fut in as_completed(futs):
            n = futs[fut]
            row = fut.result()
            rows.append(row)
            print(f"[arc] oracle_stratified N={n:<4} bank={row['n_bank_items']:<4} "
                  f"SE{SE_TARGET} r={row['r']} mae={row['mae']} items={row['items']}", flush=True)
    if POOL_CAP in N_GRID:  # N=500 reuses the full-pool bank (no extra calibration)
        row = run_n(POOL_CAP)
        rows.append(row)
        print(f"[arc] oracle_stratified N={POOL_CAP:<4} bank={row['n_bank_items']:<4} "
              f"SE{SE_TARGET} r={row['r']} mae={row['mae']} items={row['items']}", flush=True)

    write_results(rows)

    print("\n=== oracle_stratified r-by-N (SE<=0.3, ARC, pool=500, test=417) ===", flush=True)
    for r in sorted(rows, key=lambda r: r["N"]):
        print(f"  N={r['N']:<4} r={r['r']:<7} mae={r['mae']:<7} items={r['items']:<6} "
              f"bank={r['n_bank_items']}", flush=True)
    print(f"[arc] done in {round(time.time() - t_start, 1)}s", flush=True)


if __name__ == "__main__":
    main()
