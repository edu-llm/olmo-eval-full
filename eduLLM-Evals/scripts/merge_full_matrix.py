"""Merge the big-run nonoptional response matrix with the supplement full-bank
matrix into ONE combined ``staging/response_matrix_full.csv`` -- read-only w.r.t.
both inputs (the big-run ``staging/response_matrix.csv`` is never modified).

Rebuild semantics (frozen for Calibration Run 5)
------------------------------------------------
The combined matrix has the FULL curated bank as columns (6,845 = 6,180
nonoptional + 665 optional). Its rows are the big-run model order.

* NONOPTIONAL columns (already present in the big run): the big-run cell wins
  wherever it is observed. The supplement only FILLS HOLES -- a cell that is NaN
  in the big run and observed in the supplement is taken from the supplement.
  Where BOTH runs graded the same model x criterion and DISAGREE, the big-run
  value is kept (prefer-big-run) and the disagreement is logged to the audit.
* OPTIONAL columns (new; never in the big run): taken entirely from the
  supplement (these are the 662 presentation ``style_surface`` criteria + 3
  ``rescope_optional`` conditionals).

Outputs (all under ``staging/``)
--------------------------------
* ``response_matrix_full.csv`` / ``.npy``     -- combined 82 x 6,845 matrix.
* ``response_matrix_full_manifest.json``      -- provenance (per-cell source
  tallies), dims, per-category fill, and the full disagreement audit.
* ``response_matrix_full_provenance.csv``     -- per-column: optional flag,
  big/supp/combined observed counts, hole-fills, disagreements.
* ``response_matrix_full_nonoptional.csv``    -- the 6,180 nonoptional columns
  of the combined (hole-filled) matrix, in big-run column order, for the
  core 2-skill [correctness, scaffolding] refresh.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIG = ROOT / "staging" / "response_matrix.csv"
DEFAULT_SUPP = ROOT / "staging" / "response_matrix_supp.csv"
DEFAULT_CURATED = ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl"
DEFAULT_STAGING = ROOT / "staging"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def load_bank_columns(path: Path) -> tuple[list[str], dict[str, dict]]:
    """Full curated column order (file order) + per-criterion optional/dimension."""
    columns: list[str] = []
    meta: dict[str, dict] = {}
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get("criterion_id") or "").strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)
            columns.append(cid)
            meta[cid] = {
                "optional": rec.get("optional") is True,
                "dimension": rec.get("dimension"),
            }
    return columns, meta


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="model")
    return df.apply(pd.to_numeric, errors="coerce")


def write_matrix_csv(matrix: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *matrix.columns])
        for model, series in matrix.iterrows():
            values: list[object] = []
            for value in series.to_numpy():
                values.append("" if pd.isna(value) else int(round(value)))
            writer.writerow([model, *values])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--big", type=Path, default=DEFAULT_BIG)
    p.add_argument("--supp", type=Path, default=DEFAULT_SUPP)
    p.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_STAGING)
    p.add_argument("--basename", type=str, default="response_matrix_full")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    full_cols, meta = load_bank_columns(args.curated)
    big = load_matrix(args.big)
    supp = load_matrix(args.supp)

    big_models = list(big.index)
    supp_models = set(supp.index)
    if set(big_models) != supp_models:
        only_big = sorted(set(big_models) - supp_models)
        only_supp = sorted(supp_models - set(big_models))
        raise SystemExit(
            f"model sets differ: only_big={only_big[:5]} only_supp={only_supp[:5]}"
        )

    # Canonical row order = big-run order; align supplement to it.
    supp = supp.reindex(index=big_models)

    big_cols = set(big.columns)
    # Ensure supplement has every curated column (all-NaN if it never graded it).
    supp = supp.reindex(columns=full_cols)

    combined = pd.DataFrame(
        np.nan, index=pd.Index(big_models, name="model"), columns=full_cols, dtype=float
    )
    # provenance code per cell: 0 = empty/NaN, 1 = big-run, 2 = supplement(hole-fill/optional)
    prov = np.zeros(combined.shape, dtype=np.int8)

    big_arr_full = np.array(big.reindex(columns=full_cols).to_numpy(), dtype=float)  # NaN for optional cols
    supp_arr = np.array(supp.to_numpy(), dtype=float)
    comb_arr = np.full(combined.shape, np.nan, dtype=float)

    big_obs_mask = ~np.isnan(big_arr_full)
    supp_obs_mask = ~np.isnan(supp_arr)

    # Big-run wins wherever observed.
    comb_arr[big_obs_mask] = big_arr_full[big_obs_mask]
    prov[big_obs_mask] = 1
    # Hole-fill / new optional columns: supplement observed AND big-run empty.
    fill_mask = supp_obs_mask & ~big_obs_mask
    comb_arr[fill_mask] = supp_arr[fill_mask]
    prov[fill_mask] = 2

    # Disagreement audit: both observed AND differ (prefer big-run, keep as is).
    overlap_mask = big_obs_mask & supp_obs_mask
    disagree_mask = overlap_mask & (big_arr_full != supp_arr)
    agree_mask = overlap_mask & (big_arr_full == supp_arr)

    col_index = {c: i for i, c in enumerate(full_cols)}
    optional_flags = np.array([bool(meta.get(c, {}).get("optional")) for c in full_cols])

    disagreements: list[dict] = []
    dis_rows, dis_cols = np.where(disagree_mask)
    for r, c in zip(dis_rows.tolist(), dis_cols.tolist()):
        disagreements.append({
            "model": big_models[r],
            "criterion_id": full_cols[c],
            "big_run_value": int(big_arr_full[r, c]),
            "supplement_value": int(supp_arr[r, c]),
            "resolved_to": "big_run",
        })

    combined = pd.DataFrame(comb_arr, index=combined.index, columns=full_cols)

    # ---- per-column provenance table ----
    nonopt_cols = [c for c in full_cols if not optional_flags[col_index[c]]]
    opt_cols = [c for c in full_cols if optional_flags[col_index[c]]]

    prov_rows = []
    for c in full_cols:
        j = col_index[c]
        prov_rows.append({
            "criterion_id": c,
            "optional": bool(optional_flags[j]),
            "dimension": meta.get(c, {}).get("dimension"),
            "big_observed": int(big_obs_mask[:, j].sum()),
            "supp_observed": int(supp_obs_mask[:, j].sum()),
            "combined_observed": int((prov[:, j] > 0).sum()),
            "hole_fills_from_supp": int((prov[:, j] == 2).sum()),
            "disagreements": int(disagree_mask[:, j].sum()),
        })
    prov_df = pd.DataFrame(prov_rows)
    prov_csv = args.out_dir / f"{args.basename}_provenance.csv"
    prov_df.to_csv(prov_csv, index=False)

    # ---- write matrices ----
    matrix_csv = args.out_dir / f"{args.basename}.csv"
    matrix_npy = args.out_dir / f"{args.basename}.npy"
    write_matrix_csv(combined, matrix_csv)
    np.save(matrix_npy, combined.to_numpy(dtype=np.float32))

    nonopt_df = combined[nonopt_cols]
    nonopt_csv = args.out_dir / f"{args.basename}_nonoptional.csv"
    write_matrix_csv(nonopt_df, nonopt_csv)

    # ---- tallies ----
    def _pf(arr: np.ndarray) -> dict:
        return {
            "pass": int(np.nansum(arr == 1.0)),
            "fail": int(np.nansum(arr == 0.0)),
            "nan": int(np.isnan(arr).sum()),
        }

    presentation_cols = [c for c in opt_cols if meta.get(c, {}).get("dimension") == "style_surface"]
    optional_other = [c for c in opt_cols if c not in set(presentation_cols)]

    def _cat_fill(cols: list[str]) -> dict:
        js = [col_index[c] for c in cols]
        obs = (prov[:, js] > 0).any(axis=0) if js else np.array([], dtype=bool)
        populated = int(obs.sum())
        return {"total_columns": len(cols), "populated_columns": populated,
                "all_nan_columns": len(cols) - populated}

    manifest = {
        "generated_at": _utcnow(),
        "note": (
            "COMBINED response matrix = big-run nonoptional (prefer-big-run) hole-filled "
            "by the full Qwen supplement, plus the optional/presentation columns from the "
            "supplement. Big-run cells win on overlap; supplement fills NaN holes only."
        ),
        "inputs": {
            "big_run_matrix": {"path": str(args.big), "shape": list(big.shape)},
            "supplement_matrix": {"path": str(args.supp), "shape": list(supp.shape)},
            "curated_bank": {"path": str(args.curated), "n_criteria": len(full_cols)},
        },
        "n_models": int(combined.shape[0]),
        "n_criteria": int(combined.shape[1]),
        "cell_count": int(combined.size),
        "column_order": "curated bank file order, ALL criteria (nonoptional + optional)",
        "row_order": "big-run model order",
        "cells": _pf(comb_arr),
        "provenance_cell_counts": {
            "from_big_run": int((prov == 1).sum()),
            "from_supplement_holefill_or_optional": int((prov == 2).sum()),
            "empty_nan": int((prov == 0).sum()),
        },
        "overlap_audit": {
            "cells_graded_by_both": int(overlap_mask.sum()),
            "agreements": int(agree_mask.sum()),
            "disagreements": int(disagree_mask.sum()),
            "resolution": "prefer big-run original value on every disagreement",
            "disagreements_detail": disagreements,
        },
        "hole_fill": {
            "nonoptional_holes_filled_by_supplement": int(
                (prov[:, [col_index[c] for c in nonopt_cols]] == 2).sum()
            ),
            "optional_new_column_cells": int(
                (prov[:, [col_index[c] for c in opt_cols]] == 2).sum()
            ),
        },
        "category_fill": {
            "nonoptional": _cat_fill(nonopt_cols),
            "optional_presentation_style_surface": _cat_fill(presentation_cols),
            "optional_other": _cat_fill(optional_other),
        },
        "outputs": {
            "matrix_csv": str(matrix_csv),
            "matrix_npy": str(matrix_npy),
            "provenance_csv": str(prov_csv),
            "nonoptional_csv": str(nonopt_csv),
        },
        "provenance": {"script": "scripts/merge_full_matrix.py"},
    }
    manifest_path = args.out_dir / f"{args.basename}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- console ----
    print(f"combined matrix    : {combined.shape[0]} models x {combined.shape[1]} criteria")
    print(f"cells pass/fail/NaN: {manifest['cells']['pass']} / {manifest['cells']['fail']} / "
          f"{manifest['cells']['nan']}")
    print(f"provenance         : big-run {manifest['provenance_cell_counts']['from_big_run']} | "
          f"supplement {manifest['provenance_cell_counts']['from_supplement_holefill_or_optional']} | "
          f"empty {manifest['provenance_cell_counts']['empty_nan']}")
    print(f"hole-fill          : nonoptional holes filled {manifest['hole_fill']['nonoptional_holes_filled_by_supplement']} | "
          f"optional new cells {manifest['hole_fill']['optional_new_column_cells']}")
    print(f"overlap audit      : both-graded {manifest['overlap_audit']['cells_graded_by_both']} "
          f"(agree {manifest['overlap_audit']['agreements']}, "
          f"disagree {manifest['overlap_audit']['disagreements']} -> kept big-run)")
    cf = manifest["category_fill"]
    print(f"category fill      : nonopt {cf['nonoptional']['populated_columns']}/"
          f"{cf['nonoptional']['total_columns']} | presentation "
          f"{cf['optional_presentation_style_surface']['populated_columns']}/"
          f"{cf['optional_presentation_style_surface']['total_columns']} | optional_other "
          f"{cf['optional_other']['populated_columns']}/{cf['optional_other']['total_columns']}")
    for out in (matrix_csv, matrix_npy, prov_csv, nonopt_csv, manifest_path):
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
