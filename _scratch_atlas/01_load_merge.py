"""Load + merge every ATLAS / OpenLM candidate pool into one dedup'd table.

Writes _scratch_atlas/candidates.csv
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "_scratch_atlas")


def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


# name -> (path, id_column)
POOLS: dict[str, tuple[str, str]] = {
    "atlas_cal_0p5_7b": (
        p("AdaptiveTesting", "Inputs", "ATLAS", "arc_0p5_7b", "cal_models_0p5_7b.csv"),
        "model",
    ),
    "atlas_heuristic": (
        p(
            "AdaptiveTesting",
            "Experiments",
            "atlas_recalibrate_0p5_7b",
            "results",
            "atlas_model_sizes_heuristic.csv",
        ),
        "model",
    ),
    "openlm_selected": (
        p("AdaptiveTesting", "Inputs", "OpenLM", "models_selected.csv"),
        "fullname",
    ),
    "openlm_download": (
        p("AdaptiveTesting", "Inputs", "OpenLM", "download_status.csv"),
        "model",
    ),
}

MATRICES = {
    "atlas_mat_arc_train": "gaussian_sampled_arc_response_matrix_train.csv",
    "atlas_mat_arc_test": "gaussian_sampled_arc_response_matrix_test.csv",
    "atlas_mat_arc_train_scores": "gaussian_sampled_arc_response_matrix_train_with_scores.csv",
    "atlas_mat_arc_0p5_7b": "gaussian_sampled_arc_response_matrix_train_with_scores_0p5_7b.csv",
}

records: dict[str, dict] = {}          # lowercase id -> record
pool_rowcounts: dict[str, int] = {}
pool_uniq: dict[str, int] = {}


def touch(model_id: str, pool: str) -> dict:
    key = model_id.strip().lower()
    rec = records.get(key)
    if rec is None:
        rec = {
            "key": key,
            "model": model_id.strip(),
            "aliases": set(),
            "pools": set(),
            "params_all": [],
            "average": None,
            "details_repo": None,
            "dl_status": None,
            "bench_rows": {},
            "splits": {},
        }
        records[key] = rec
    rec["aliases"].add(model_id.strip())
    rec["pools"].add(pool)
    return rec


# ---------------------------------------------------------------- tabular pools
for pool, (path, idcol) in POOLS.items():
    df = pd.read_csv(path)
    pool_rowcounts[pool] = len(df)
    seen = set()
    for _, row in df.iterrows():
        mid = str(row[idcol])
        if not mid or mid == "nan":
            continue
        seen.add(mid.lower())
        rec = touch(mid, pool)
        pb = row.get("params_b")
        if pd.notna(pb):
            try:
                rec["params_all"].append((pool, float(pb)))
            except (TypeError, ValueError):
                pass
        if pool == "openlm_selected":
            avg = row.get("average")
            if pd.notna(avg):
                rec["average"] = float(avg)
            if pd.notna(row.get("details_repo")):
                rec["details_repo"] = row["details_repo"]
        if pool == "openlm_download":
            rec["dl_status"] = row.get("status")
            br = row.get("benchmark_rows")
            if pd.notna(br) and str(br).strip() not in ("", "{}"):
                try:
                    rec["bench_rows"] = json.loads(br)
                except json.JSONDecodeError:
                    pass
    pool_uniq[pool] = len(seen)

# ------------------------------------------------------------- response matrices
mat_dir = p("AdaptiveTesting", "Inputs", "ATLAS", "data")
for pool, fname in MATRICES.items():
    path = os.path.join(mat_dir, fname)
    if not os.path.exists(path):
        continue
    ids = pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str)
    pool_rowcounts[pool] = len(ids)
    seen = set()
    for mid in ids:
        if not mid or mid == "nan":
            continue
        seen.add(mid.lower())
        touch(mid, pool)
    pool_uniq[pool] = len(seen)

# ------------------------------------------------------------ openlm split files
for path in sorted(glob.glob(p("AdaptiveTesting", "Experiments", "**", "data", "split_models.csv"), recursive=True)):
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    bench = rel.split("/")[2] if "openlm_atlas_3pl" in rel else "gpqa"
    if "openlm_atlas_3pl" in rel:
        bench = rel.split("openlm_atlas_3pl/")[1].split("/")[0]
    pool = f"openlm_split_{bench}"
    df = pd.read_csv(path)
    pool_rowcounts[pool] = len(df)
    seen = set()
    for _, row in df.iterrows():
        mid = str(row["model"])
        if not mid or mid == "nan":
            continue
        seen.add(mid.lower())
        rec = touch(mid, pool)
        rec["splits"][bench] = row.get("split")
    pool_uniq[pool] = len(seen)

# --------------------------------------------------------------------- resolve
rows = []
conflicts = 0
for key, rec in records.items():
    vals = sorted({round(v, 4) for _, v in rec["params_all"]})
    if len(vals) > 1:
        conflicts += 1
    # prefer the ATLAS cal / OpenLM values, else first available
    prio = ["openlm_selected", "openlm_download", "atlas_cal_0p5_7b", "atlas_heuristic"]
    params = None
    by_pool = dict(rec["params_all"])
    for pl in prio:
        if pl in by_pool:
            params = by_pool[pl]
            break
    if params is None and rec["params_all"]:
        params = rec["params_all"][0][1]
    org = rec["model"].split("/")[0] if "/" in rec["model"] else "(no-org)"
    name = rec["model"].split("/", 1)[1] if "/" in rec["model"] else rec["model"]
    rows.append(
        {
            "key": key,
            "model": rec["model"],
            "org": org,
            "name": name,
            "params_b": params,
            "params_variants": ";".join(str(v) for v in vals),
            "n_pools": len(rec["pools"]),
            "pools": ";".join(sorted(rec["pools"])),
            "in_atlas": int(any(pl.startswith("atlas") for pl in rec["pools"])),
            "in_openlm": int(any(pl.startswith("openlm") for pl in rec["pools"])),
            "average": rec["average"],
            "dl_status": rec["dl_status"],
            "n_bench": len(rec["bench_rows"]),
            "details_repo": rec["details_repo"],
            "n_aliases": len(rec["aliases"]),
            "aliases": ";".join(sorted(rec["aliases"])) if len(rec["aliases"]) > 1 else "",
        }
    )

cand = pd.DataFrame(rows).sort_values("model", key=lambda s: s.str.lower()).reset_index(drop=True)
cand.to_csv(os.path.join(OUT, "candidates.csv"), index=False)

# ----------------------------------------------------------------------- report
print("=" * 78)
print("POOL INVENTORY")
print("=" * 78)
inv = pd.DataFrame(
    [{"pool": k, "rows": pool_rowcounts[k], "unique_ids_ci": pool_uniq[k]} for k in pool_rowcounts]
).sort_values("pool")
print(inv.to_string(index=False))

print()
print("=" * 78)
print("MERGED CANDIDATE TABLE")
print("=" * 78)
print(f"total unique candidates (case-insensitive)   : {len(cand)}")
print(f"  with case-variant aliases                  : {(cand['n_aliases'] > 1).sum()}")
print(f"  params_b known                             : {cand['params_b'].notna().sum()}")
print(f"  params_b missing                           : {cand['params_b'].isna().sum()}")
print(f"  params_b conflicting across pools          : {conflicts}")
print(f"  with OpenLM 'average' score                : {cand['average'].notna().sum()}")
print()
inw = cand["params_b"].notna() & (cand["params_b"] > 0) & (cand["params_b"] <= 7.0)
print(f"IN 0-7B WINDOW (0 < params_b <= 7.0)         : {int(inw.sum())}")
print(f"  of which >= 0.15B                          : {int((inw & (cand['params_b'] >= 0.15)).sum())}")
print(f"  of which have OpenLM average               : {int((inw & cand['average'].notna()).sum())}")
print(f"  ATLAS-only                                 : {int((inw & (cand.in_atlas == 1) & (cand.in_openlm == 0)).sum())}")
print(f"  OpenLM-only                                : {int((inw & (cand.in_atlas == 0) & (cand.in_openlm == 1)).sum())}")
print(f"  in BOTH ATLAS and OpenLM                   : {int((inw & (cand.in_atlas == 1) & (cand.in_openlm == 1)).sum())}")
print(f"OUT of window (>7B)                          : {int((cand['params_b'] > 7.0).sum())}")
print()
print("pool-membership signature counts (top 20):")
print(cand["pools"].value_counts().head(20).to_string())
print()
print("params_b distribution (whole table):")
print(cand["params_b"].describe().to_string())
print()
print(f"wrote {os.path.join(OUT, 'candidates.csv')}")
