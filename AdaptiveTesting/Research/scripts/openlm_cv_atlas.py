#!/usr/bin/env python3
"""K-fold cross-validated ATLAS p-IRT accuracy recovery on an OpenLM benchmark.

This gives every model exactly ONE out-of-sample prediction: we partition the
models into K folds and, for each fold, recalibrate the 3PL item bank on the
other K-1 folds (the SAME validated pipeline as ``openlm_trainsize_sweep.py`` --
chunked R ``mirt`` 3PL fit, mean-sigma chunk linking with polarity flip, a>0 item
filter) and then run the SE-target adaptive CAT on the held-out fold. The CAT is
defined exactly as in ``atlas_error_plots_openlm.py`` (``adaptive_trace`` with
Fisher-information selection, EAP theta, MIN_ITEMS=8, up to --max-items items,
p-IRT predicted accuracy and actual accuracy over all usable a>0 bank items),
so the concatenated out-of-sample rows are directly comparable to the single
90/10-split ATLAS replication.

For GPQA the raw ``Inputs/OpenLM/gpqa/*.csv`` are not present locally, so
``sweep.build_master("gpqa")`` resolves no data; we fall back to reconstructing
the full model x item matrix by concatenating the committed ATLAS-format train +
test matrices under ``Experiments/openlm_gpqa_atlas_3pl/data`` (the exact cleaned
matrix ``prepare_matrix.py`` produced).

Usage:
  uv run python AdaptiveTesting/Research/scripts/openlm_cv_atlas.py \
      --bench gpqa --folds 10 --seed 7 --se 0.3 --workers 2

Outputs (under AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv/<bench>/):
  per_model_se<SE>.csv   (model, pirt, actual, items, fold)
  results.json           (one comparable results row + provenance)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import atlas_error_plots_openlm as atlas
import openlm_trainsize_sweep as sweep

csv.field_size_limit(sys.maxsize)

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
EXP = REPO / "AdaptiveTesting/Experiments"
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv"

# Plain mean-sigma linker (no polarity flip), identical to the script that built the
# frozen single-split banks (Experiments/openlm_gpqa_atlas_3pl/calibration). On GPQA,
# sweep's default polarity-flip linker rescues ~half the chunks with mis-oriented a1,
# which are miscalibrated and bias p-IRT high; the no-flip linker + a>0 filter drops
# them, reproducing the single-split methodology so the CV is comparable.
NOFLIP_LINK_R = REPO / "AdaptiveTesting/Inputs/ATLAS/scripts/02_link_chunks_custom.r"

# Where the committed ATLAS-format matrices live when build_master finds no raw data.
FALLBACK_MATRICES = {
    "gpqa": (
        EXP / "openlm_gpqa_atlas_3pl/data/gpqa_response_matrix_train.csv",
        EXP / "openlm_gpqa_atlas_3pl/data/gpqa_response_matrix_test.csv",
    ),
    "ifeval": (
        EXP / "openlm_atlas_3pl/ifeval/data/response_matrix_train.csv",
        EXP / "openlm_atlas_3pl/ifeval/data/response_matrix_test.csv",
    ),
    "math": (
        EXP / "openlm_atlas_3pl/math/data/response_matrix_train.csv",
        EXP / "openlm_atlas_3pl/math/data/response_matrix_test.csv",
    ),
    "bbh": (
        EXP / "openlm_atlas_3pl/bbh/data/response_matrix_train.csv",
        EXP / "openlm_atlas_3pl/bbh/data/response_matrix_test.csv",
    ),
    "musr": (
        EXP / "openlm_atlas_3pl/musr/data/response_matrix_train.csv",
        EXP / "openlm_atlas_3pl/musr/data/response_matrix_test.csv",
    ),
}


def build_full_matrix(bench: str) -> tuple[pd.DataFrame, str]:
    """Full model x item 0/1 matrix (all models). Prefer sweep.build_master; if the
    raw OpenLM CSVs are absent, reconstruct from committed ATLAS train+test matrices."""
    try:
        mat = sweep.build_master(bench)
    except Exception:
        mat = pd.DataFrame()
    if len(mat) > 0:
        return mat.astype(int), "build_master"

    if bench not in FALLBACK_MATRICES:
        raise SystemExit(f"no raw data for {bench} and no fallback matrices registered")
    train_csv, test_csv = FALLBACK_MATRICES[bench]
    if not train_csv.is_file() or not test_csv.is_file():
        raise SystemExit(f"fallback matrices missing: {train_csv} / {test_csv}")
    train = sweep.load_matrix(train_csv)
    test = sweep.load_matrix(test_csv)
    if list(train.columns) != list(test.columns):
        common = [c for c in train.columns if c in set(test.columns)]
        train, test = train[common], test[common]
    full = pd.concat([train, test], axis=0)
    full = full[~full.index.duplicated(keep="first")].astype(int)
    return full, "reconstructed_atlas_train_test"


def make_folds(models: list[str], k: int, seed: int) -> list[list[str]]:
    """Shuffle models with the given seed, then split into k near-equal folds."""
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    return [[order[i] for i in idx] for idx in np.array_split(np.arange(len(order)), k)]


def run_fold(
    mat: pd.DataFrame,
    train_models: list[str],
    test_models: list[str],
    step_dir: Path,
    args,
    fit_r: Path,
    link_r: Path,
) -> tuple[list[dict], int]:
    """Calibrate on train_models, run the held-out adaptive CAT on test_models.

    Returns (per-model rows, usable a>0 bank size). Matches atlas_error_plots_openlm:
    adaptive_trace (Fisher info, EAP theta), read off at SE<=args.se, actual accuracy
    over all a>0 bank items, p-IRT predicted accuracy."""
    ends, _ = sweep.prepare_step(mat, train_models, test_models, step_dir, args.chunk_size)
    sweep.fit_and_link(step_dir, ends, args.ncycles, args.workers, fit_r, link_r)

    test = sweep.load_matrix(step_dir / "data" / "response_matrix_test.csv")
    idxs, a, b, c = sweep.load_bank(step_dir / "calibration", list(test.columns))
    n_bank = len(idxs)
    if n_bank == 0:
        raise RuntimeError(f"empty a>0 bank for fold at {step_dir}")

    rows: list[dict] = []
    for mid in test.index:
        resp = test.loc[mid, idxs].to_numpy(float)
        actual = float(resp.mean())
        steps = atlas.adaptive_trace(resp, a, b, c, args.floor_se, args.max_items)
        ni, _se, _theta, pred = atlas.read_off(steps, args.se)
        rows.append({"model": mid, "pirt": pred, "actual": actual, "items": ni})
    return rows, n_bank


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default="gpqa")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--se", type=float, default=0.3, help="SE stopping target")
    p.add_argument("--floor-se", type=float, default=0.095,
                   help="tight SE floor for the single full CAT trace per model")
    p.add_argument("--max-items", type=int, default=400)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--workers", type=int, default=2, help="R chunk-fit workers per fold")
    p.add_argument("--link", choices=["noflip", "flip"], default="noflip",
                   help="chunk linker: 'noflip' matches the frozen single-split bank "
                        "(comparable); 'flip' is sweep's default polarity-flip linker")
    p.add_argument("--work-root", default="")
    args = p.parse_args()

    bench = args.bench
    out_dir = OUT_ROOT / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_root) / bench if args.work_root else OUT_ROOT / "_work" / bench
    fit_r, flip_link_r = sweep.write_helper_scripts(work)
    link_r = NOFLIP_LINK_R if args.link == "noflip" else flip_link_r
    suffix = "" if args.link == "noflip" else "_sweepflip"

    mat, source = build_full_matrix(bench)
    print(f"[{bench}] matrix source={source} models={mat.shape[0]} items={mat.shape[1]}",
          flush=True)

    folds = make_folds(list(mat.index), args.folds, args.seed)
    sizes = [len(f) for f in folds]
    assert sum(sizes) == mat.shape[0]
    assert len({m for f in folds for m in f}) == mat.shape[0]  # disjoint + cover
    print(f"[{bench}] {args.folds} folds seed={args.seed} sizes={sizes} SE<={args.se}",
          flush=True)

    t0 = time.time()
    all_rows: list[dict] = []
    bank_sizes: list[int] = []
    for k, test_models in enumerate(folds):
        train_models = [m for m in mat.index if m not in set(test_models)]
        step_dir = work / f"fold{k}"
        fold_csv = step_dir / f"fold_results_{args.link}.csv"
        ft0 = time.time()
        if fold_csv.is_file():
            fdf = pd.read_csv(fold_csv)
            if len(fdf) == len(test_models):
                rows = fdf.to_dict("records")
                n_bank = int(fdf["n_bank"].iloc[0]) if "n_bank" in fdf else -1
                print(f"[{bench}] fold {k} CACHED n={len(rows)} bank={n_bank}", flush=True)
                for r in rows:
                    all_rows.append({"model": r["model"], "pirt": float(r["pirt"]),
                                     "actual": float(r["actual"]), "items": int(r["items"]),
                                     "fold": k})
                if n_bank > 0:
                    bank_sizes.append(n_bank)
                continue

        rows, n_bank = run_fold(mat, train_models, test_models, step_dir, args, fit_r, link_r)
        for r in rows:
            r["fold"] = k
            r["n_bank"] = n_bank
        step_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(fold_csv, index=False)
        all_rows.extend(rows)
        bank_sizes.append(n_bank)
        pr = np.array([r["pirt"] for r in rows])
        ac = np.array([r["actual"] for r in rows])
        rr = float(np.corrcoef(pr, ac)[0, 1]) if pr.std() > 0 else float("nan")
        print(f"[{bench}] fold {k} n={len(rows)} bank={n_bank} "
              f"r={rr:.4f} items={np.mean([r['items'] for r in rows]):.1f} "
              f"({round(time.time() - ft0, 1)}s)", flush=True)
        _write_outputs(bench, out_dir, all_rows, bank_sizes, args, time.time() - t0,
                       partial=True)

    wall = time.time() - t0
    summary = _write_outputs(bench, out_dir, all_rows, bank_sizes, args, wall, partial=False)
    _print_report(summary)


def _write_outputs(bench, out_dir: Path, rows: list[dict], bank_sizes: list[int],
                   args, wall_s: float, partial: bool) -> dict:
    suffix = "" if args.link == "noflip" else "_sweepflip"
    df = pd.DataFrame(rows)[["model", "pirt", "actual", "items", "fold"]]
    df.to_csv(out_dir / f"per_model_se{args.se:g}{suffix}.csv", index=False)

    pirt = df["pirt"].to_numpy(float)
    actual = df["actual"].to_numpy(float)
    items = df["items"].to_numpy(float)
    abs_err = np.abs(pirt - actual)
    n = len(df)
    r = float(np.corrcoef(pirt, actual)[0, 1]) if n > 1 and pirt.std() > 0 else float("nan")
    slope = float(np.polyfit(actual, pirt, 1)[0]) if n > 1 else float("nan")
    mae = float(abs_err.mean())
    half = 1.96 * float(np.std(abs_err)) / np.sqrt(n) if n > 0 else float("nan")
    bank_mean = float(np.mean(bank_sizes)) if bank_sizes else float("nan")

    summary = {
        "benchmark": bench,
        "se_target": args.se,
        "n_folds": args.folds,
        "seed": args.seed,
        "n_models": int(n),
        "pearson_r": round(r, 4),
        "slope": round(slope, 4),
        "accuracy_mae": round(mae, 4),
        "mae_95ci": [round(mae - half, 4), round(mae + half, 4)],
        "avg_cat_items": round(float(items.mean()), 2),
        "bank_size": round(bank_mean, 1),
        "bank_size_per_fold": [int(x) for x in bank_sizes],
        "wall_clock_s": round(wall_s, 1),
        "partial": partial,
        "linking": args.link,
        "cat": {"definition": "atlas_error_plots_openlm.adaptive_trace",
                "min_items": atlas.MIN_ITEMS, "max_items": args.max_items,
                "floor_se": args.floor_se, "a_filter": "a>0"},
    }
    with (out_dir / f"results{suffix}.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _print_report(s: dict) -> None:
    lo, hi = s["mae_95ci"]
    print("\n=== CV out-of-sample results (SE<=%(se)s) ===" % {"se": s["se_target"]}, flush=True)
    print("Benchmark, Pearson r, Slope, Accuracy MAE, 95% CI MAE, Avg CAT items, Bank size",
          flush=True)
    print(f"{s['benchmark']}, {s['pearson_r']}, {s['slope']}, {s['accuracy_mae']}, "
          f"[{lo}, {hi}], {s['avg_cat_items']}, {s['bank_size']}", flush=True)
    print(f"n models = {s['n_models']}   wall-clock = {s['wall_clock_s']}s", flush=True)


if __name__ == "__main__":
    main()
