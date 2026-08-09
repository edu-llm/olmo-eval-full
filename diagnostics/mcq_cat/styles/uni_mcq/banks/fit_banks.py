"""Fit per-item IRT parameters for the pedagogy / piqa / socialiqa banks.

No prior study persisted these: each fit in memory, ran a CAT diagnostic, recorded a
correlation, and discarded the parameters. This produces the artifact vendoring needs.

Faithfulness to the calibration source matters more than elegance, so `load_benchmark`,
`filter_items` and `fit_bank` are transcribed verbatim from `tutor_cat/mcq_irt/matrix.py` and
`Research/scripts/se_sweep_small_pool.py`. `reproduce_check.py` is the gate that the
transcription is right; run it first.

Fit family per bank is a measured decision — see MCQ_BANK_VENDORING_PLAN.md §6.2.

The shipped fit uses EVERY model available for a bank, while the published correlations were
measured on a 40-model train slice with 12 held out. More calibration models makes a better
bank; the held-out split existed to validate the approach, and validating an approach is not
the same as shipping it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Keep BLAS modest, as the calibration scripts do, so a wide 2PL fit does not oversubscribe.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reproduce_check import (  # noqa: E402
    EXPECTED_RAW,
    filter_items,
    load_benchmark,
)

FIT_FAMILY = {"pedagogy": "1PL", "piqa": "2PL", "socialiqa": "1PL"}


def fit_bank(kept_df, model_type):
    """Verbatim from se_sweep_small_pool.fit_bank. girth wants items x persons."""
    items = list(kept_df.columns)
    if model_type == "1PL":
        from girth import rasch_mml

        est = rasch_mml(kept_df.to_numpy(dtype=int).T)
        b = np.asarray(est["Difficulty"], float)
        a = np.ones_like(b)
        c = np.zeros_like(b)
    elif model_type == "2PL":
        from girth import twopl_mml

        est = twopl_mml(kept_df.to_numpy(dtype=int).T)
        a = np.asarray(est["Discrimination"], float)
        b = np.asarray(est["Difficulty"], float)
        c = np.zeros_like(a)
    else:
        raise ValueError(model_type)
    return items, a, b, c


def usable(a, b, c):
    """The study's uniform usability filter: finite, a > 0, 0 <= c < 1."""
    return np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (a > 0) & (c >= 0) & (c < 1)


def position_of(item_id):
    """1-based position in the bank's raw enumeration.

    Ids are dense and their numeric suffix is the 0-based row index the 2026-08-01 loader
    enumerated, verified item by item in Phase 0. Positions must stay on this ORIGINAL
    numbering, with gaps where filter_items dropped an item: renumbering survivors 1..n would
    silently repoint every parameter at the wrong question.
    """
    return int(item_id.rsplit("_", 1)[1]) + 1


def item_stats(mat):
    X = mat.to_numpy(dtype=float)
    from reproduce_check import _point_biserial

    return pd.DataFrame(
        {"p_value": X.mean(axis=0), "point_biserial": _point_biserial(X)},
        index=pd.Index(list(mat.columns), name="item"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrices", required=True, type=Path, help="dir of per-bank response CSVs")
    ap.add_argument("--out", required=True, type=Path, help="where to write the fitted CSVs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for bank, family in FIT_FAMILY.items():
        mat = load_benchmark(args.matrices, bank)
        if mat.shape[1] != EXPECTED_RAW[bank]:
            raise SystemExit(f"{bank}: {mat.shape[1]} raw items, expected {EXPECTED_RAW[bank]}")

        stats = item_stats(mat)
        kept, _counts = filter_items(mat)
        print(f"\n{bank}  ({mat.shape[0]} models x {mat.shape[1]} items) -> {family}", flush=True)

        t0 = time.time()
        items, a, b, c = fit_bank(kept, family)
        secs = time.time() - t0

        ok = usable(a, b, c)
        df = pd.DataFrame(
            {
                "item_id": items,
                "position": [position_of(i) for i in items],
                "a": a,
                "b": b,
                "c": c,
                "p_value": stats.loc[items, "p_value"].to_numpy(),
                "point_biserial": stats.loc[items, "point_biserial"].to_numpy(),
            }
        )[ok].sort_values("position").reset_index(drop=True)
        df.to_csv(args.out / f"{bank}_item_params.csv", index=False)

        # Under Rasch, difficulty is a strictly monotone function of pass rate, so a Spearman
        # of anything but -1 means the fit is wrong. 2PL legitimately breaks that, because
        # discrimination also moves the predicted ranking.
        spearman = df["b"].corr(df["p_value"], method="spearman")
        meta = {
            "bank": bank,
            "fit_family": family,
            "fitter": {"1PL": "girth.rasch_mml", "2PL": "girth.twopl_mml"}[family],
            "n_models": int(mat.shape[0]),
            "n_items_raw": int(mat.shape[1]),
            "n_items_kept_by_filter": int(kept.shape[1]),
            "n_items_final": int(len(df)),
            "max_position": int(df["position"].max()),
            "a_min": float(df["a"].min()),
            "a_max": float(df["a"].max()),
            "spearman_b_vs_p_value": round(float(spearman), 4),
            "fit_seconds": round(secs, 1),
        }
        all_meta.append(meta)
        print(
            f"  {meta['n_items_final']} items in {secs:.1f}s  "
            f"a=[{meta['a_min']:.3f},{meta['a_max']:.3f}]  "
            f"spearman(b, p)={spearman:+.4f}",
            flush=True,
        )

    (args.out / "fit_metadata.json").write_text(json.dumps(all_meta, indent=2))
    print(f"\nwrote {args.out / 'fit_metadata.json'}")


if __name__ == "__main__":
    main()
