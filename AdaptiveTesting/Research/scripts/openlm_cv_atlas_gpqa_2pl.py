#!/usr/bin/env python3
"""2PL 10-fold cross-validation of the OpenLM ATLAS p-IRT CAT for GPQA.

This is the 2PL twin of the 3PL GPQA CV that produced
``atlas_replication_cv/gpqa/{per_model_se0.3.csv,results.json}`` via
``openlm_cv_atlas.py --bench gpqa --link noflip`` (the headline GPQA row:
r=0.6518, slope=1.2553, MAE=0.0738, avg items 13.71, bank 575.6, n=1102, and an
a<=0 drop rate of 51.4% under 3PL). It is identical in EVERY respect except the
item-response model used at calibration: the R ``mirt`` fit uses
``itemtype = "2PL"`` (guessing fixed at 0 / no c parameter) instead of 3PL.

To guarantee "identical in every other respect", everything else is REUSED from
the exact 3PL GPQA harness by importing its building blocks from
``openlm_cv_atlas`` (the module that built the 3PL headline):

  * the SAME full model x item matrix (``build_full_matrix("gpqa")`` -> the
    reconstructed committed ATLAS train+test matrix, 1102 models),
  * the SAME 10 folds over models shuffled with seed 7 (``make_folds``),
  * per-fold recalibration on the other 9 folds via ``run_fold`` -> the SAME
    chunked ``mirt`` EM (chunk_size 100, ncycles 500, GenRandomPars retry ladder),
  * the NO-FLIP mean-sigma chunk linker (``02_link_chunks_custom.r`` =
    ``openlm_cv_atlas.NOFLIP_LINK_R``) that the GPQA 3PL headline used -- GPQA
    needs it because ~half its chunks fit reversed, and the default polarity-flip
    linker rescues+miscalibrates those, biasing p-IRT high,
  * the a>0 item filter and the SAME atlas adaptive CAT
    (``atlas_error_plots_openlm.adaptive_trace`` / ``read_off``: MIN_ITEMS=8,
    max 400 items, Fisher-information selection, EAP theta, read off at SE<=0.3),
  * the SAME per-model ``actual`` = mean correctness over that fold's usable
    (a>0) bank items.

The ONLY changed input is the mirt itemtype (3PL -> 2PL, via string substitution
on ``sweep.FIT_R_TEXT`` -- the same technique as ``openlm_cv_atlas_musr_2pl.py``),
so the concatenated out-of-sample rows are directly comparable to the 3PL GPQA CV.
We additionally record, per fold, the calibrated bank size and the a<=0 drop rate
under 2PL (which may differ from the 3PL 51.4%), and a short 3PL-vs-2PL comparison.

Outputs (never overwriting the 3PL results) go to
  AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv/gpqa/2pl/
    per_model_se<SE>.csv   (model, pirt, actual, items, fold)  [same columns as 3PL]
    results.json           (reporting row + drop rate + comparison to 3PL)

CPU-local, uv run. No AWS, no git.

Usage:
  uv run python AdaptiveTesting/Research/scripts/openlm_cv_atlas_gpqa_2pl.py \
      --bench gpqa --folds 10 --seed 7 --se 0.3 --workers 2
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

import atlas_error_plots_openlm as atlas  # noqa: E402  (CAT definition, MIN_ITEMS)
import openlm_cv_atlas as cv3pl  # noqa: E402  (the 3PL GPQA harness we mirror)
import openlm_trainsize_sweep as sweep  # noqa: E402  (validated calibration pipeline)

OUT_ROOT = cv3pl.OUT_ROOT

# The ONLY change from the 3PL pipeline: itemtype 3PL -> 2PL (c fixed at 0). We
# derive the 2PL fit script from the validated 3PL one by string substitution so
# the EM settings, GenRandomPars retry ladder, EAP scoring, coef export and output
# filenames stay byte-for-byte identical -- guaranteeing the comparison isolates
# the response model. (Same technique as openlm_cv_atlas_musr_2pl.py.)
FIT_R_TEXT_2PL = (
    sweep.FIT_R_TEXT.replace('itemtype = "3PL"', 'itemtype = "2PL"')
    .replace("Fitting 3PL chunk", "Fitting 2PL chunk")
    .replace("Saved 3PL chunk", "Saved 2PL chunk")
)
assert 'itemtype = "2PL"' in FIT_R_TEXT_2PL
assert "3PL" not in FIT_R_TEXT_2PL.split("mirt(")[1].split(")")[0]


def write_fit_2pl(work: Path) -> Path:
    """Write the 2PL chunk-fit R script (only the itemtype differs from 3PL)."""
    work.mkdir(parents=True, exist_ok=True)
    fit_r = work / "fit_chunk_2pl.r"
    fit_r.write_text(FIT_R_TEXT_2PL)
    return fit_r


def count_calibrated_items(calib_dir: Path, test_cols) -> tuple[int, int]:
    """(#calibrated items mapping to a test column, #of those with usable a>0).

    Mirrors ``sweep.load_bank``'s index parsing so the usable count equals the CAT
    bank size; the total is the denominator for the a<=0 drop rate.
    """
    pdf = pd.read_csv(calib_dir / "irt_item_parameters_combined.csv")
    name_col = pdf.columns[0]
    cols = set(test_cols)
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


def verify_fold_identity(folds: list[list[str]], se: float) -> str:
    """If the 3PL per-model CSV exists, assert our folds match it model-for-model."""
    ref = OUT_ROOT / "gpqa" / f"per_model_se{se:g}.csv"
    if not ref.is_file():
        return "no 3PL reference CSV found -> fold-identity check skipped"
    rdf = pd.read_csv(ref)
    ref_fold = dict(zip(rdf["model"].astype(str), rdf["fold"].astype(int), strict=True))
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


def _fold_r(rows: list[dict]) -> float:
    p = np.array([r["pirt"] for r in rows], float)
    a = np.array([r["actual"] for r in rows], float)
    return float(np.corrcoef(p, a)[0, 1]) if p.std() > 0 and a.std() > 0 else float("nan")


def compute_summary(df: pd.DataFrame, bank_sizes: list[int], calib_sizes: list[int],
                    fold_rs: list[float], args, wall_s: float, note: str,
                    partial: bool) -> dict:
    """Reporting row using the SAME formulas as openlm_cv_atlas._write_outputs
    (the 3PL harness) plus the 2PL drop-rate and per-fold diagnostics."""
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

    drop_fracs = [1 - u / t for u, t in zip(bank_sizes, calib_sizes, strict=True) if t > 0]
    pct_dropped = round(100 * float(np.mean(drop_fracs)), 2) if drop_fracs else float("nan")

    finite_rs = [x for x in fold_rs if np.isfinite(x)]
    worst = int(np.nanargmin(fold_rs)) if finite_rs else -1
    if worst >= 0 and "fold" in df:
        keep = df[df["fold"] != worst]
        r_excl = (float(np.corrcoef(keep["pirt"], keep["actual"])[0, 1])
                  if len(keep) > 1 and keep["pirt"].std() > 0 else float("nan"))
    else:
        r_excl = float("nan")

    return {
        "benchmark": args.bench,
        "itemtype": "2PL",
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
        "pct_items_dropped": pct_dropped,
        "calibrated_items_per_fold": [int(x) for x in calib_sizes],
        "pct_dropped_per_fold": [round(100 * (1 - u / t), 2) if t > 0 else None
                                 for u, t in zip(bank_sizes, calib_sizes, strict=True)],
        "fold_pearson_r": [round(x, 4) for x in fold_rs],
        "worst_fold": worst,
        "pearson_r_excl_worst_fold": round(r_excl, 4),
        "wall_clock_s": round(wall_s, 1),
        "partial": partial,
        "linking": "noflip",
        "cat": {"definition": "atlas_error_plots_openlm.adaptive_trace",
                "min_items": atlas.MIN_ITEMS, "max_items": args.max_items,
                "floor_se": args.floor_se, "a_filter": "a>0"},
        "fold_identity": note,
    }


def add_comparison(summary: dict) -> None:
    """Attach a 3PL-vs-2PL comparison block loaded from the frozen 3PL results.json.

    The 3PL results.json carries no drop-rate field, so we compare against the
    known GPQA 3PL a<=0 drop rate of 51.4% (613/1192 negative-a items)."""
    ref_path = OUT_ROOT / "gpqa" / "results.json"
    if not ref_path.is_file():
        summary["comparison_to_3pl"] = {"note": f"3PL results.json not found at {ref_path}"}
        return
    t = json.loads(ref_path.read_text())

    def num_cmp(key):
        a, b = t.get(key), summary.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return {"3pl": a, "2pl": b, "delta": round(b - a, 4)}
        return {"3pl": a, "2pl": b, "delta": None}

    s3, s2 = t.get("slope"), summary.get("slope")
    slope_block = {"3pl": s3, "2pl": s2}
    if isinstance(s3, (int, float)) and isinstance(s2, (int, float)):
        slope_block.update({
            "abs_dist_from_1_3pl": round(abs(s3 - 1.0), 4),
            "abs_dist_from_1_2pl": round(abs(s2 - 1.0), 4),
            "moved_toward_1": bool(abs(s2 - 1.0) < abs(s3 - 1.0)),
        })

    summary["comparison_to_3pl"] = {
        "source": str(ref_path),
        "pearson_r": num_cmp("pearson_r"),
        "slope": slope_block,
        "accuracy_mae": num_cmp("accuracy_mae"),
        "avg_cat_items": num_cmp("avg_cat_items"),
        "bank_size": num_cmp("bank_size"),
        "pct_items_dropped": {"3pl_reported": 51.4, "2pl": summary.get("pct_items_dropped")},
    }


def _write_outputs(out_dir: Path, df: pd.DataFrame, summary: dict, args) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["model", "pirt", "actual", "items", "fold"]
    df[cols].to_csv(out_dir / f"per_model_se{args.se:g}.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))


def run_cv(args: argparse.Namespace) -> None:
    bench = args.bench
    out_dir = OUT_ROOT / bench / "2pl"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    fit_r = write_fit_2pl(work)
    link_r = cv3pl.NOFLIP_LINK_R  # the GPQA 3PL headline's no-flip mean-sigma linker
    if not Path(link_r).is_file():
        raise SystemExit(f"no-flip linker not found at {link_r}")

    mat, source = cv3pl.build_full_matrix(bench)
    print(f"[{bench}/2PL] matrix source={source} models={mat.shape[0]} "
          f"items={mat.shape[1]}", flush=True)

    folds = cv3pl.make_folds(list(mat.index), args.folds, args.seed)
    sizes = [len(f) for f in folds]
    assert sum(sizes) == mat.shape[0]
    assert len({m for f in folds for m in f}) == mat.shape[0]  # disjoint + cover
    note = verify_fold_identity(folds, args.se)
    print(f"[{bench}/2PL] {note}", flush=True)
    print(f"[{bench}/2PL] {args.folds} folds seed={args.seed} sizes={sizes} "
          f"SE<={args.se} link=noflip fit=2PL", flush=True)

    all_rows: list[dict] = []
    bank_sizes: list[int] = []
    calib_sizes: list[int] = []
    fold_rs: list[float] = []
    t0 = time.time()
    for k, test_models in enumerate(folds):
        train_models = [m for m in mat.index if m not in set(test_models)]
        step_dir = work / f"fold{k}"
        cache = step_dir / "fold_results.csv"
        meta = step_dir / "fold_meta.json"
        ft0 = time.time()

        if cache.is_file() and meta.is_file() and not args.force:
            fdf = pd.read_csv(cache)
            mj = json.loads(meta.read_text())
            if len(fdf) == len(test_models):
                rows = fdf.to_dict("records")
                all_rows.extend(rows)
                bank_sizes.append(int(mj["n_bank"]))
                calib_sizes.append(int(mj["n_total"]))
                fold_rs.append(_fold_r(rows))
                print(f"[{bench}/2PL] fold {k} CACHED n={len(rows)} "
                      f"bank={mj['n_bank']}/{mj['n_total']}", flush=True)
                continue

        # Identical code path to the 3PL headline (openlm_cv_atlas.run_fold): only
        # the fit_r (2PL) differs; link_r is the SAME no-flip linker the 3PL used.
        rows, n_bank = cv3pl.run_fold(mat, train_models, test_models, step_dir,
                                      args, fit_r, link_r)
        test_cols = list(sweep.load_matrix(
            step_dir / "data" / "response_matrix_test.csv").columns)
        n_total, n_usable = count_calibrated_items(step_dir / "calibration", test_cols)
        if n_usable != n_bank:
            print(f"[{bench}/2PL] fold {k} NOTE n_usable({n_usable}) != n_bank("
                  f"{n_bank}); using CAT bank={n_bank} for size", flush=True)
        for r in rows:
            r["fold"] = k
        step_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows)[["model", "pirt", "actual", "items", "fold"]].to_csv(
            cache, index=False)
        meta.write_text(json.dumps({"n_bank": int(n_bank), "n_total": int(n_total)}))

        all_rows.extend(rows)
        bank_sizes.append(n_bank)
        calib_sizes.append(n_total)
        fr = _fold_r(rows)
        fold_rs.append(fr)
        drop = 100 * (1 - n_bank / n_total) if n_total else float("nan")
        print(f"[{bench}/2PL] fold {k} n={len(rows)} bank={n_bank}/{n_total} "
              f"(drop {drop:.1f}%) r={fr:.4f} "
              f"items={np.mean([r['items'] for r in rows]):.1f} "
              f"({round(time.time() - ft0, 1)}s)", flush=True)

        summ = compute_summary(pd.DataFrame(all_rows), bank_sizes, calib_sizes,
                               fold_rs, args, time.time() - t0, note, partial=True)
        add_comparison(summ)
        _write_outputs(out_dir, pd.DataFrame(all_rows), summ, args)

    wall = time.time() - t0
    df = pd.DataFrame(all_rows)
    summ = compute_summary(df, bank_sizes, calib_sizes, fold_rs, args, wall, note,
                           partial=(len(df) != mat.shape[0]))
    add_comparison(summ)
    _write_outputs(out_dir, df, summ, args)
    _print_report(summ)


def _print_report(s: dict) -> None:
    lo, hi = s["mae_95ci"]
    c = s.get("comparison_to_3pl", {})
    print("\n" + "=" * 82, flush=True)
    print(f"OpenLM ATLAS {s['n_folds']}-fold CV (2PL) | {s['benchmark']} | "
          f"SE<={s['se_target']} | seed {s['seed']}"
          + ("" if not s["partial"] else "   (PARTIAL)"), flush=True)
    print("=" * 82, flush=True)
    print("Benchmark, Pearson r, Slope, Accuracy MAE, 95% CI MAE, Avg CAT items, "
          "Bank size, % items dropped, n models", flush=True)
    print(f"{s['benchmark']}, {s['pearson_r']}, {s['slope']}, {s['accuracy_mae']}, "
          f"[{lo}, {hi}], {s['avg_cat_items']}, {s['bank_size']}, "
          f"{s['pct_items_dropped']}, {s['n_models']}", flush=True)
    print(f"per-fold r        : {s['fold_pearson_r']}", flush=True)
    print(f"r excl worst fold : {s['pearson_r_excl_worst_fold']} (fold {s['worst_fold']})",
          flush=True)
    print(f"drop per fold (%) : {s['pct_dropped_per_fold']}", flush=True)
    print(f"wall-clock (s)    : {s['wall_clock_s']}", flush=True)
    if c and "pearson_r" in c:
        print("-" * 82, flush=True)
        print("3PL -> 2PL (GPQA):", flush=True)
        print(f"  Pearson r      : {c['pearson_r']['3pl']} -> {c['pearson_r']['2pl']} "
              f"(delta {c['pearson_r']['delta']})", flush=True)
        sb = c["slope"]
        print(f"  Slope          : {sb['3pl']} -> {sb['2pl']} "
              f"(|dist from 1|: {sb.get('abs_dist_from_1_3pl')} -> "
              f"{sb.get('abs_dist_from_1_2pl')}, toward 1: {sb.get('moved_toward_1')})",
              flush=True)
        print(f"  Accuracy MAE   : {c['accuracy_mae']['3pl']} -> {c['accuracy_mae']['2pl']} "
              f"(delta {c['accuracy_mae']['delta']})", flush=True)
        print(f"  Avg CAT items  : {c['avg_cat_items']['3pl']} -> {c['avg_cat_items']['2pl']} "
              f"(delta {c['avg_cat_items']['delta']})", flush=True)
        print(f"  Bank size      : {c['bank_size']['3pl']} -> {c['bank_size']['2pl']} "
              f"(delta {c['bank_size']['delta']})", flush=True)
        print(f"  % items dropped: {c['pct_items_dropped']['3pl_reported']} (3PL) -> "
              f"{c['pct_items_dropped']['2pl']} (2PL)", flush=True)
    print("=" * 82, flush=True)


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
    p.add_argument("--workers", type=int, default=2,
                   help="R chunk-fit workers per fold (keep modest)")
    p.add_argument("--force", action="store_true", help="recompute folds even if cached")
    args = p.parse_args()
    if args.bench != "gpqa":
        print(f"[warn] this driver is specialized for GPQA (no-flip linker + "
              f"reconstructed matrix); got --bench {args.bench}", flush=True)
    run_cv(args)


if __name__ == "__main__":
    main()
