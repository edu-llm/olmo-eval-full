#!/usr/bin/env python3
"""Calibration parameter-range vs 7B-model recovery experiment.

Fix the TEST set to 7B models (params_b >= 6.5). Compare two calibration pools on how
well the resulting 3PL bank recovers those 7B models' benchmark accuracy:
  Config A (upward extrapolation): calibrate on params_b in [0, 5).
  Config B (in-range / interpolation): calibrate on params_b in [3, 7], with the
     >= 6.5 test models removed so the calibration band is effectively [3, 6.5).

Both configs use the SAME held-out 7B test set. Calibration N is matched between A and B
by subsampling both pools to the smaller pool's size (fixed per-seed RNG), so the only
difference is the parameter range, not pool size. We run several seeds and report
mean +/- sd of the Pearson r (p-IRT predicted vs actual full-benchmark accuracy) and MAE.

Everything downstream of model selection reuses the validated OpenLM 3PL pipeline in
`openlm_trainsize_sweep.py` (chunked R mirt 3PL, mean-sigma linking, a>0 item filter,
EAP/Fisher CAT, p-IRT accuracy recovery).

Outputs under AdaptiveTesting/Research/01_MCQ_ATLAS/data/calib_range_to_7b/.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

import openlm_trainsize_sweep as sw

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
PARAMS_CSV = REPO / "AdaptiveTesting/Research/05_Data_Availability/data/openlm_download_status.csv"
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/calib_range_to_7b"

TEST_MIN = 6.5  # "7B test models" definition

FIELDS = [
    "benchmark", "config", "band", "se_target", "seed", "corr", "mae",
    "mean_items", "n_calib", "n_test", "n_bank_items",
]


def load_params() -> dict[str, float]:
    df = pd.read_csv(PARAMS_CSV)
    df = df[df["status"] == "ok"]
    return dict(zip(df["model"], df["params_b"].astype(float)))


def select_pools(mat_models: list[str], params: dict[str, float]):
    """Return (test_models, poolA, poolB) restricted to models present in the matrix."""
    have = [m for m in mat_models if m in params]
    test = [m for m in have if params[m] >= TEST_MIN]
    pool_a = [m for m in have if 0.0 <= params[m] < 5.0]  # A: [0,5), excludes test
    pool_b = [m for m in have if 3.0 <= params[m] <= 7.0 and params[m] < TEST_MIN]  # B: [3,6.5)
    return test, pool_a, pool_b


def run_bench(bench: str, seeds: list[int], ses: list[float], chunk_size: int,
              ncycles: int, workers: int, rows: list[dict]) -> dict:
    work = OUT_ROOT / "_work" / bench
    fit_r, link_r = sw.write_helper_scripts(work)
    params = load_params()

    print(f"[{bench}] building master matrix...", flush=True)
    mat = sw.build_master(bench)
    test, pool_a, pool_b = select_pools(list(mat.index), params)
    n_match = min(len(pool_a), len(pool_b))
    pa = np.array([params[m] for m in pool_a]) if pool_a else np.array([])
    pb = np.array([params[m] for m in pool_b]) if pool_b else np.array([])
    print(
        f"[{bench}] matrix models={len(mat)} items={mat.shape[1]} | "
        f"test(>= {TEST_MIN})={len(test)} poolA[0,5)={len(pool_a)} "
        f"poolB[3,6.5)={len(pool_b)} matched_N={n_match}",
        flush=True,
    )

    meta = {
        "bench": bench, "n_test": len(test), "n_pool_a": len(pool_a),
        "n_pool_b": len(pool_b), "n_match": n_match,
        "a_med": float(np.median(pa)) if pa.size else float("nan"),
        "a_min": float(pa.min()) if pa.size else float("nan"),
        "a_max": float(pa.max()) if pa.size else float("nan"),
        "b_med": float(np.median(pb)) if pb.size else float("nan"),
        "b_min": float(pb.min()) if pb.size else float("nan"),
        "b_max": float(pb.max()) if pb.size else float("nan"),
        "test_med": float(np.median([params[m] for m in test])) if test else float("nan"),
    }
    if n_match < 8 or len(test) < 3:
        print(f"[{bench}] band too thin (matched_N={n_match}, n_test={len(test)}); skipping.",
              flush=True)
        meta["skipped"] = True
        return meta
    meta["skipped"] = False

    configs = {"A": ("[0,5)", pool_a), "B": ("[3,6.5)", pool_b)}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for cfg, (band, pool) in configs.items():
            sub = [pool[i] for i in rng.permutation(len(pool))[:n_match]]
            step_dir = work / f"{cfg}_n{n_match}_seed{seed}"
            t0 = time.time()
            try:
                ends, _ = sw.prepare_step(mat, sub, test, step_dir, chunk_size)
                sw.fit_and_link(step_dir, ends, ncycles, workers, fit_r, link_r)
                metrics = sw.diagnose(step_dir, ses)
                for se in ses:
                    m = metrics[se]
                    rows.append({
                        "benchmark": bench, "config": cfg, "band": band,
                        "se_target": se, "seed": seed, "corr": m["corr"],
                        "mae": m["mae"], "mean_items": m["mean_items"],
                        "n_calib": n_match, "n_test": len(test),
                        "n_bank_items": m["n_bank_items"],
                    })
                main = metrics[ses[0]]
                print(
                    f"[{bench}] cfg={cfg} band={band} seed={seed} "
                    f"SE{ses[0]}: r={main['corr']} MAE={main['mae']} "
                    f"bank={main['n_bank_items']} ({round(time.time()-t0,1)}s)",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{bench}] cfg={cfg} seed={seed} FAILED: {exc}", flush=True)
                for se in ses:
                    rows.append({
                        "benchmark": bench, "config": cfg, "band": band,
                        "se_target": se, "seed": seed, "corr": float("nan"),
                        "mae": float("nan"), "mean_items": float("nan"),
                        "n_calib": n_match, "n_test": len(test), "n_bank_items": 0,
                    })
            _write_csv(rows)
    return meta


def _write_csv(rows: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benches", default="ifeval,math")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--se-list", default="0.3,0.2")
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    ses = [float(s) for s in args.se_list.split(",")]

    rows: list[dict] = []
    metas: list[dict] = []
    for bench in benches:
        metas.append(run_bench(bench, seeds, ses, args.chunk_size, args.ncycles,
                               args.workers, rows))
        pd.DataFrame(metas).to_csv(OUT_ROOT / "pools_meta.csv", index=False)
    _write_csv(rows)
    print(f"DONE -> {OUT_ROOT / 'results.csv'} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
