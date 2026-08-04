#!/usr/bin/env python3
"""2PL 10-fold cross-validation of the OpenLM ATLAS p-IRT CAT for MuSR.

This is the 2PL twin of the 3PL MuSR CV that produced
``atlas_replication_cv/musr/{pirt_vs_actual_cv_se0.3.csv,results.json}`` via
``openlm_atlas_cv.py``. It is identical in EVERY respect except the item-response
model used at calibration: the R ``mirt`` fit uses ``itemtype = "2PL"`` (guessing
fixed at 0 / no c parameter) instead of 3PL. Everything else is unchanged:

  * the same ``sweep.build_master("musr")`` model x item matrix (all models),
  * 10 folds over models shuffled with seed 7 (same ``make_folds`` as the 3PL run),
  * per-fold recalibration on the other 9 folds with the SAME chunked ``mirt`` EM
    (chunk_size 100, ncycles 500, GenRandomPars retry) + mean-sigma polarity-flip
    chunk linking (``sweep.LINK_R_TEXT``, the MuSR 3PL default) + a>0 item filter,
  * the SAME atlas adaptive CAT (``atlas_error_plots_openlm.adaptive_trace`` /
    ``read_off``: MIN_ITEMS=8, max 400 items, Fisher-information selection, EAP
    theta, read off at SE<=0.3),
  * the SAME per-model ``actual`` = mean correctness over that fold's usable (a>0)
    bank items.

So the concatenated out-of-sample rows are directly comparable to the 3PL MuSR CV.
We additionally record, per fold, the calibrated bank size and the a<=0 drop rate
under 2PL (which may differ from the 3PL drop rate).

Outputs (never overwriting the 3PL results) go to
  AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv/musr/2pl/
    pirt_vs_actual_cv_se<SE>.csv   (model, pirt, actual, items, fold, n_bank)
    results.json                   (reporting row + drop rate + per-fold diagnostics)

CPU-local, uv run. No AWS, no git.

Usage:
  uv run python AdaptiveTesting/Research/scripts/openlm_cv_atlas_musr_2pl.py \
      --bench musr --folds 10 --seed 7 --se 0.3 --workers 2
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

# The ONLY change from the 3PL pipeline: itemtype 3PL -> 2PL (c fixed at 0). We
# derive the 2PL fit script from the validated 3PL one by string substitution so
# that the EM settings, GenRandomPars retry ladder, EAP scoring, coef export and
# output filenames stay byte-for-byte identical -- guaranteeing the comparison
# isolates the response model. The linker is the same polarity-flip mean-sigma
# linker the 3PL MuSR run used (sweep.LINK_R_TEXT).
FIT_R_TEXT_2PL = (
    sweep.FIT_R_TEXT.replace('itemtype = "3PL"', 'itemtype = "2PL"')
    .replace("Fitting 3PL chunk", "Fitting 2PL chunk")
    .replace("Saved 3PL chunk", "Saved 2PL chunk")
)
assert 'itemtype = "2PL"' in FIT_R_TEXT_2PL and "3PL" not in FIT_R_TEXT_2PL.split("mirt(")[1].split(")")[0]


def write_helper_scripts_2pl(work: Path) -> tuple[Path, Path]:
    """Write the 2PL chunk-fit script and the (unchanged) polarity-flip linker."""
    work.mkdir(parents=True, exist_ok=True)
    fit_r = work / "fit_chunk_2pl.r"
    link_r = work / "link_chunks_polarity.r"
    fit_r.write_text(FIT_R_TEXT_2PL)
    link_r.write_text(sweep.LINK_R_TEXT)
    return fit_r, link_r


def make_folds(models: list[str], seed: int, k: int) -> list[list[str]]:
    """Shuffle models with a fixed seed and partition into k folds (every model once).

    Byte-for-byte the same partition rule as ``openlm_atlas_cv.make_folds`` (the
    3PL MuSR driver), so 2PL folds are identical to the 3PL folds.
    """
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    return [[order[i] for i in grp] for grp in np.array_split(np.arange(len(order)), k)]


def bank_counts(calib_dir: Path, mat_cols: list[int]) -> tuple[int, int]:
    """(calibrated items mapping to a test column, usable a>0 items) for one fold."""
    pdf = pd.read_csv(calib_dir / "irt_item_parameters_combined.csv")
    name_col = pdf.columns[0]
    cols = set(mat_cols)
    n_total = n_usable = 0
    for _, row in pdf.iterrows():
        key = str(row[name_col]).lstrip("X")
        try:
            idx = int(float(key))
        except ValueError:
            continue
        if idx not in cols:
            continue
        n_total += 1
        a = float(row["a1"])
        if np.isfinite(a) and a > 0:
            n_usable += 1
    return n_total, n_usable


def diagnose_fold(step_dir: Path, se_target: float, floor_se: float, max_items: int):
    """Run the ATLAS SE<=se_target adaptive CAT on the held-out fold's test matrix.

    Uses the re-calibrated per-fold 2PL bank (a>0 items only). Returns per-model
    rows (model, pirt, actual, items), usable bank size, calibrated bank size.
    """
    test = sweep.load_matrix(step_dir / "data" / "response_matrix_test.csv")
    idxs, a, b, c = sweep.load_bank(step_dir / "calibration", list(test.columns))
    n_total, n_usable = bank_counts(step_dir / "calibration", list(test.columns))
    if len(idxs) == 0:
        raise RuntimeError("empty a>0 bank after linking")
    rows = []
    for m in test.index:
        resp = test.loc[m, idxs].to_numpy(float)  # actual over all usable a>0 items
        actual = float(resp.mean())
        steps = atlas.adaptive_trace(resp, a, b, c, floor_se, max_items)
        ni, _se_reached, _theta, pred = atlas.read_off(steps, se_target)
        rows.append({"model": m, "pirt": float(pred), "actual": actual, "items": int(ni)})
    return rows, len(idxs), n_total


def _fold_pearson(rows: list[dict]) -> float:
    p = np.array([r["pirt"] for r in rows], float)
    a = np.array([r["actual"] for r in rows], float)
    return float(np.corrcoef(p, a)[0, 1]) if p.std() > 0 and a.std() > 0 else float("nan")


def verify_fold_identity(folds: list[list[str]], se: float) -> str:
    """If the 3PL per-model CSV exists, assert our folds match it model-for-model."""
    ref = OUT_ROOT / "musr" / f"pirt_vs_actual_cv_se{se:g}.csv"
    if not ref.is_file():
        return "no 3PL reference CSV found -> fold-identity check skipped"
    rdf = pd.read_csv(ref)
    ref_fold = dict(zip(rdf["model"].astype(str), rdf["fold"].astype(int)))
    ours = {str(m): k for k, f in enumerate(folds) for m in f}
    if set(ours) != set(ref_fold):
        raise SystemExit(
            "fold-identity check FAILED: model set differs from the 3PL run "
            f"(ours={len(ours)} vs 3PL={len(ref_fold)})"
        )
    mism = [m for m in ours if ours[m] != ref_fold[m]]
    if mism:
        raise SystemExit(
            f"fold-identity check FAILED: {len(mism)} models land in a different "
            f"fold than the 3PL run (e.g. {mism[:3]})"
        )
    return f"fold-identity check PASSED: all {len(ours)} models match 3PL fold assignment"


def compute_metrics(df: pd.DataFrame, bank_sizes: list[int], calib_sizes: list[int],
                    fold_rs: list[float], args) -> dict:
    p = df["pirt"].to_numpy(float)
    a = df["actual"].to_numpy(float)
    ae = np.abs(p - a)
    n = len(df)
    r = float(np.corrcoef(p, a)[0, 1])
    slope = float(np.polyfit(a, p, 1)[0])  # OLS slope of predicted on actual (ideal 1.0)
    mae = float(ae.mean())
    half = 1.96 * float(ae.std()) / np.sqrt(n)  # np.std (ddof=0) -> matches 3PL CV
    bank_mean = float(np.mean(bank_sizes)) if bank_sizes else float("nan")
    # % items dropped: a<=0 fraction of the calibrated bank, averaged across folds.
    drop_fracs = [1 - u / t for u, t in zip(bank_sizes, calib_sizes) if t > 0]
    pct_dropped = round(100 * float(np.mean(drop_fracs)), 2) if drop_fracs else float("nan")
    # Stability probe: pooled r with the single worst fold removed (mirrors the
    # 3PL "excluding fold 9 -> r=0.809" note).
    worst = int(np.nanargmin(fold_rs)) if fold_rs else -1
    if worst >= 0 and "fold" in df:
        keep = df[df["fold"] != worst]
        r_excl = (float(np.corrcoef(keep["pirt"], keep["actual"])[0, 1])
                  if len(keep) > 1 else float("nan"))
    else:
        r_excl = float("nan")
    return {
        "Benchmark": args.bench,
        "Pearson r": round(r, 4),
        "Slope": round(slope, 4),
        "Accuracy MAE": round(mae, 4),
        "95% CI MAE": [round(mae - half, 4), round(mae + half, 4)],
        "Avg CAT items": round(float(df["items"].mean()), 2),
        "Bank size": round(bank_mean, 1),
        "% items dropped": pct_dropped,
        "n models": int(n),
        "itemtype": "2PL",
        "bank_size_per_fold": [int(x) for x in bank_sizes],
        "calibrated_items_per_fold": [int(x) for x in calib_sizes],
        "pct_dropped_per_fold": [round(100 * (1 - u / t), 2) if t > 0 else None
                                 for u, t in zip(bank_sizes, calib_sizes)],
        "fold_pearson_r": [round(x, 4) for x in fold_rs],
        "worst_fold": worst,
        "pearson_r_excl_worst_fold": round(r_excl, 4),
        "se_target": args.se,
        "folds": args.folds,
        "seed": args.seed,
        "linking": "polarity_flip_mean_sigma (sweep.LINK_R_TEXT; MuSR 3PL default)",
    }


def _write_outputs(out_dir: Path, df: pd.DataFrame, bank_sizes, calib_sizes, fold_rs,
                   args, wall_s: float, final: bool, note: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["model", "pirt", "actual", "items", "fold", "n_bank"]
    df[cols].to_csv(out_dir / f"pirt_vs_actual_cv_se{args.se:g}.csv", index=False)
    metrics = compute_metrics(df, bank_sizes, calib_sizes, fold_rs, args)
    metrics["wall_clock_s"] = round(wall_s, 1)
    metrics["fold_identity"] = note
    metrics["complete"] = bool(final)
    (out_dir / "results.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run_cv(args: argparse.Namespace) -> None:
    bench = args.bench
    out_dir = OUT_ROOT / bench / "2pl"
    work = out_dir / "_work"
    fit_r, link_r = write_helper_scripts_2pl(work)

    print(f"[{bench}/2PL] building master matrix...", flush=True)
    mat = sweep.build_master(bench)
    folds = make_folds(list(mat.index), args.seed, args.folds)
    flat = [m for f in folds for m in f]
    assert len(flat) == len(set(flat)) == len(mat), "folds must partition all models"
    note = verify_fold_identity(folds, args.se)
    print(f"[{bench}/2PL] {note}", flush=True)
    print(
        f"[{bench}/2PL] models={len(mat)} items={mat.shape[1]} folds={args.folds} "
        f"seed={args.seed} SE<={args.se} fold_sizes={[len(f) for f in folds]}",
        flush=True,
    )

    all_rows: list[dict] = []
    bank_sizes: list[int] = []
    calib_sizes: list[int] = []
    fold_rs: list[float] = []
    t0 = time.time()
    for k, test_models in enumerate(folds):
        fold_dir = work / f"fold{k}"
        cache = fold_dir / "fold_results.csv"
        meta = fold_dir / "fold_meta.json"
        if cache.exists() and meta.exists() and not args.force:
            fdf = pd.read_csv(cache)
            mj = json.loads(meta.read_text())
            all_rows.extend(fdf.to_dict("records"))
            bank_sizes.append(int(mj["n_bank"]))
            calib_sizes.append(int(mj["n_calibrated"]))
            fold_rs.append(_fold_pearson(fdf.to_dict("records")))
            print(f"[{bench}/2PL] fold {k}: cached ({len(fdf)} models, "
                  f"bank={mj['n_bank']}/{mj['n_calibrated']})", flush=True)
            continue

        train_models = [m for j, f2 in enumerate(folds) if j != k for m in f2]
        tf = time.time()
        ends, n_kept = sweep.prepare_step(mat, train_models, test_models, fold_dir, args.chunk_size)
        sweep.fit_and_link(fold_dir, ends, args.ncycles, args.workers, fit_r, link_r)
        rows, nb, n_total = diagnose_fold(fold_dir, args.se, args.floor_se, args.max_items)
        for r in rows:
            r["fold"] = k
            r["n_bank"] = nb
        pd.DataFrame(rows).to_csv(cache, index=False)
        meta.write_text(json.dumps({"n_bank": int(nb), "n_calibrated": int(n_total)}))
        all_rows.extend(rows)
        bank_sizes.append(nb)
        calib_sizes.append(n_total)
        fr = _fold_pearson(rows)
        fold_rs.append(fr)
        print(
            f"[{bench}/2PL] fold {k}: n_train={len(train_models)} n_test={len(test_models)} "
            f"kept={n_kept} bank={nb}/{n_total} (drop {100 * (1 - nb / n_total):.1f}%) "
            f"fold_r={fr:.4f} items={np.mean([r['items'] for r in rows]):.1f} "
            f"({round(time.time() - tf, 1)}s)",
            flush=True,
        )
        _write_outputs(out_dir, pd.DataFrame(all_rows), bank_sizes, calib_sizes, fold_rs,
                       args, time.time() - t0, final=False, note=note)

    df = pd.DataFrame(all_rows)
    complete = len(df) == len(mat)
    metrics = _write_outputs(out_dir, df, bank_sizes, calib_sizes, fold_rs, args,
                             time.time() - t0, final=complete, note=note)
    _print_row(metrics, complete)


def _print_row(m: dict, complete: bool) -> None:
    print("\n" + "=" * 78, flush=True)
    print(
        f"OpenLM ATLAS {m['folds']}-fold CV (2PL)  |  SE<={m['se_target']}  |  seed {m['seed']}"
        + ("" if complete else "   (PARTIAL)"),
        flush=True,
    )
    print("=" * 78, flush=True)
    print(f"  Benchmark        : {m['Benchmark']}", flush=True)
    print(f"  Pearson r        : {m['Pearson r']}", flush=True)
    print(f"  Slope            : {m['Slope']}", flush=True)
    print(f"  Accuracy MAE     : {m['Accuracy MAE']}", flush=True)
    print(f"  95% CI MAE       : [{m['95% CI MAE'][0]}, {m['95% CI MAE'][1]}]", flush=True)
    print(f"  Avg CAT items    : {m['Avg CAT items']}", flush=True)
    print(f"  Bank size        : {m['Bank size']}", flush=True)
    print(f"  % items dropped  : {m['% items dropped']}", flush=True)
    print(f"  n models         : {m['n models']}", flush=True)
    print(f"  per-fold r       : {m['fold_pearson_r']}", flush=True)
    print(f"  r excl worst fold: {m['pearson_r_excl_worst_fold']} (fold {m['worst_fold']})",
          flush=True)
    print(f"  wall-clock (s)   : {m['wall_clock_s']}", flush=True)
    print("=" * 78, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default="musr")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--se", type=float, default=0.3, help="SE stopping target")
    p.add_argument("--floor-se", type=float, default=0.095,
                   help="tight SE floor for the single full CAT trace per model")
    p.add_argument("--max-items", type=int, default=400)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--workers", type=int, default=2, help="R chunk-fit workers (keep modest)")
    p.add_argument("--force", action="store_true", help="recompute folds even if cached")
    args = p.parse_args()
    run_cv(args)


if __name__ == "__main__":
    main()
