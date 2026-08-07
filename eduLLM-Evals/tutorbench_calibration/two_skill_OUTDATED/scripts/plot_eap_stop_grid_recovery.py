"""STUDY-ONLY companion analysis for the EAP-posterior op-point grid (TutorBench, 2-skill).

Reads the existing grid artifacts under ``reports/eap_stop_grid_tutorbench_2skill/`` and:

  1. Deliverable 1 -- SE_total mean/SD table keyed by SE_ability target (floor collapsed:
     median cell stats are floor-invariant since the median model stops well past the max
     floor of 15; the floor only clamps the fast-converging tail). Two versions: ALL 33
     models and EXCLUDING the two weakly-calibrated models
     (``Qwen/Qwen1.5-1.8B`` and ``BSC-LT/salamandra-7b-instruct``). Also checks the
     floor effect numerically (min stop length across all cells; per-model invariance).

  2. Deliverable 2 -- one combined 3-panel heatmap (floor x SE_ability target):
       A: median SE_total (correctness)   [from per_cell_grid.csv]
       B: median test length (scenarios)  [from per_cell_grid.csv]
       C: IN-SAMPLE recovery r (correctness) = Pearson r across models between MWLE theta
          at stop (per cell, from per_model_per_cell.csv) and the FULL-BANK reference theta
          (MWLE over ALL administered criteria per model), excluding the 2 weak models.

The FULL-BANK reference theta is recomputed here (not stored in the CSVs): it is the
deployed multidimensional MWLE (``scenario_cat_lib.mwle_subset``) over ALL bank&matrix
criteria, started from the dense-grid EAP mean -- identical estimator to the study script's
theta-at-stop, just evaluated on the whole administered set (order-invariant at exhaustion).

Nothing here is committed and the production engine is untouched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import scenario_cat_lib as scl  # noqa: E402

WEAK_MODELS = ["Qwen/Qwen1.5-1.8B", "BSC-LT/salamandra-7b-instruct"]
MAX_FLOOR = 15


def full_bank_reference_theta(bank_path, matrix_path, negative_policy, stop_nodes):
    """FULL-BANK reference theta per model = MWLE over ALL bank&matrix criteria, started
    from the dense-grid EAP mean. Order-invariant (whole administered set). Returns a
    DataFrame indexed by model with columns theta_ref_<dim>."""
    records, dims, _ = scl.load_fitted_bank(Path(bank_path), negative_policy)
    ids_all, A_all, b_all = scl.assemble_arrays(records, dims)
    matrix = pd.read_csv(matrix_path, index_col=0)
    keep = [i for i, c in enumerate(ids_all) if c in matrix.columns]
    ids = [ids_all[i] for i in keep]
    A, b = A_all[keep], b_all[keep]
    grid, log_prior = scl.build_grid(len(dims), stop_nodes)
    idx = np.arange(len(ids), dtype=int)
    rows = {}
    for model in matrix.index:
        Yrow = np.nan_to_num(matrix.loc[model].reindex(ids).to_numpy(dtype=float))
        mean, _ = scl.eap_subset_mean_var(Yrow, idx, A, b, grid, log_prior)
        theta_ref, ok = scl.mwle_subset(Yrow, idx, A, b, mean)
        rows[model] = {**{f"theta_ref_{d}": float(theta_ref[k]) for k, d in enumerate(dims)},
                       "mwle_ok": bool(ok)}
    return pd.DataFrame.from_dict(rows, orient="index"), dims, len(ids)


def collapse_over_floor(pmpc: pd.DataFrame, dims: list[str]):
    """One row per (se_target, model): floor-collapsed. Verifies the per-model stop is
    floor-invariant (it must be, since every stop is past the max floor). Returns the
    collapsed frame plus a bool that every (se_target, model) was floor-invariant."""
    keycols = ["length", "stop_reason"] + [f"se_total_{d}" for d in dims] \
        + [f"theta_mwle_{d}" for d in dims]
    invariant = True
    for (_se, _m), g in pmpc.groupby(["se_target", "model"]):
        if g["length"].nunique() != 1 or g["stop_reason"].nunique() != 1:
            invariant = False
    # take one representative row per (se_target, model): the smallest floor
    collapsed = (pmpc.sort_values("floor")
                 .groupby(["se_target", "model"], as_index=False).first())
    return collapsed[["se_target", "model", "floor"] + keycols], invariant


def se_total_table(collapsed: pd.DataFrame, dims: list[str], exclude=None):
    """MEAN +/- SD of SE_total across models per SE_ability target, + median length,
    %reach, %plateau. exclude: list of model ids to drop first."""
    df = collapsed
    if exclude:
        df = df[~df["model"].isin(exclude)]
    out = []
    for se_t, g in df.groupby("se_target"):
        row = {"se_target": se_t, "n_models": len(g)}
        for d in dims:
            v = g[f"se_total_{d}"].to_numpy(float)
            row[f"mean_se_total_{d}"] = float(v.mean())
            row[f"sd_se_total_{d}"] = float(v.std(ddof=1))
        row["median_len"] = float(g["length"].median())
        row["pct_reach"] = float((g["stop_reason"] == "precision_reached").mean())
        row["pct_plateau"] = float((g["stop_reason"] == "info_plateau").mean())
        out.append(row)
    return pd.DataFrame(out).sort_values("se_target").reset_index(drop=True)


def fmt_table_md(tbl: pd.DataFrame, dims: list[str], title: str) -> str:
    hdr = ("| SE_target | mean±SD SE_total (correctness) | mean±SD SE_total (scaffolding) "
           "| median len | %reach | %plateau |")
    sep = "|---|---|---|---|---|---|"
    lines = [f"**{title}**", "", hdr, sep]
    for _, r in tbl.iterrows():
        c = f"{r['mean_se_total_correctness']:.3f} ± {r['sd_se_total_correctness']:.3f}"
        s = f"{r['mean_se_total_scaffolding']:.3f} ± {r['sd_se_total_scaffolding']:.3f}"
        lines.append(f"| {r['se_target']:.2f} | {c} | {s} | {r['median_len']:.0f} "
                     f"| {r['pct_reach']*100:.0f}% | {r['pct_plateau']*100:.0f}% |")
    return "\n".join(lines)


def recovery_grid(pmpc, ref, floors, targets, skill, exclude):
    """Per-cell IN-SAMPLE Pearson r between theta_mwle_<skill> at stop and the full-bank
    reference theta, across models (excluding `exclude`)."""
    ref_col = f"theta_ref_{skill}"
    stop_col = f"theta_mwle_{skill}"
    grid = np.full((len(floors), len(targets)), np.nan)
    for i, fl in enumerate(floors):
        for j, se in enumerate(targets):
            sub = pmpc[(pmpc["floor"] == fl) & (np.isclose(pmpc["se_target"], se))].copy()
            if exclude:
                sub = sub[~sub["model"].isin(exclude)]
            sub = sub.merge(ref[[ref_col]], left_on="model", right_index=True, how="inner")
            x = sub[stop_col].to_numpy(float)
            y = sub[ref_col].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3:
                grid[i, j] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return grid


def _panel(ax, grid, floors, targets, title, fmt, cmap, vmin=None, vmax=None):
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(targets)), [f"{t:.2f}" for t in targets])
    ax.set_yticks(range(len(floors)), [str(f) for f in floors])
    ax.set_xlabel("SE_ability target")
    ax.set_ylabel("min_scenarios floor")
    ax.set_title(title, fontsize=9)
    for i in range(len(floors)):
        for j in range(len(targets)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, format(grid[i, j], fmt), ha="center", va="center",
                        color="black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-dir", type=Path,
                    default=ROOT / "reports/eap_stop_grid_tutorbench_2skill")
    ap.add_argument("--bank", type=Path,
                    default=ROOT / "data/TutorBench/rubrics_qmatrix_calibrated_2skill_fitted.jsonl")
    ap.add_argument("--matrix", type=Path,
                    default=ROOT / "tutorbench_tb33_grading/response_matrix/response_matrix.csv")
    ap.add_argument("--stop-nodes", type=int, default=161)
    ap.add_argument("--negative-policy", default="clamp")
    args = ap.parse_args()

    rd = args.report_dir
    pmpc = pd.read_csv(rd / "per_model_per_cell.csv")
    per_cell = pd.read_csv(rd / "per_cell_grid.csv")
    floors = sorted(per_cell["floor"].unique())
    targets = sorted(per_cell["se_target"].unique())

    print("[ref] recomputing FULL-BANK reference theta (MWLE over all criteria) ...")
    ref, dims, n_crit = full_bank_reference_theta(args.bank, args.matrix,
                                                  args.negative_policy, args.stop_nodes)
    print(f"[ref] dims={dims} n_criteria={n_crit} models={len(ref)} "
          f"mwle_converged={int(ref['mwle_ok'].sum())}/{len(ref)}")

    # ---- floor-inertness ----
    min_len_overall = int(pmpc["length"].min())
    per_cell_min = pmpc.groupby(["floor", "se_target"])["length"].min()
    min_len_per_cell = int(per_cell_min.min())
    collapsed, floor_invariant = collapse_over_floor(pmpc, dims)

    # ---- Deliverable 1 tables ----
    tbl_all = se_total_table(collapsed, dims, exclude=None)
    tbl_excl = se_total_table(collapsed, dims, exclude=WEAK_MODELS)
    tbl_all.to_csv(rd / "se_total_mean_sd_by_target_all33.csv", index=False)
    tbl_excl.to_csv(rd / "se_total_mean_sd_by_target_excl2.csv", index=False)

    md_all = fmt_table_md(tbl_all, dims, "Deliverable 1a — ALL 33 models")
    md_excl = fmt_table_md(tbl_excl, dims,
                           "Deliverable 1b — EXCLUDING 2 weak models (headline)")

    # ---- Deliverable 2: recovery grids + combined figure ----
    rec_corr = recovery_grid(pmpc, ref, floors, targets, "correctness", WEAK_MODELS)
    rec_scaf = recovery_grid(pmpc, ref, floors, targets, "scaffolding", WEAK_MODELS)

    se_grid = np.full((len(floors), len(targets)), np.nan)
    len_grid = np.full((len(floors), len(targets)), np.nan)
    for i, fl in enumerate(floors):
        for j, se in enumerate(targets):
            sub = per_cell[(per_cell["floor"] == fl) & (np.isclose(per_cell["se_target"], se))]
            if len(sub):
                se_grid[i, j] = float(sub["median_se_total_correctness"].iloc[0])
                len_grid[i, j] = float(sub["median_len"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    _panel(axes[0], se_grid, floors, targets,
           "A. Median SE_total (correctness)", ".3f", "viridis_r")
    _panel(axes[1], len_grid, floors, targets,
           "B. Median test length (scenarios)", ".0f", "magma_r")
    _panel(axes[2], rec_corr, floors, targets,
           "C. In-sample recovery r (correctness)\nlandscape only; headline recovery is "
           "OOS k-fold at the finalist", ".2f", "cividis", vmin=0.0, vmax=1.0)
    fig.suptitle(
        "EAP stop-rule op-point grid (TutorBench, 2-skill). Floor axis kept to show the "
        "striping: median cell stats are floor-invariant (median stop 20-33 scen > max "
        "floor 15 at every SE target). The floor still clamps the fast-converging tail "
        "(min stop 7 scen at floor 6 -> 15 at floor 15), but that minority does not move "
        "the medians.\nPanel C recovery excludes 2 weakly-calibrated models "
        "(Qwen/Qwen1.5-1.8B, BSC-LT/salamandra-7b-instruct); it is IN-SAMPLE landscape "
        "diagnostics only, not the headline OOS recovery.", fontsize=8, y=1.04)
    fig.tight_layout()
    out_png = rd / "heatmap_combined_se_len_recovery.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # recovery grids to CSV for the record
    pd.DataFrame(rec_corr, index=[f"floor_{f}" for f in floors],
                 columns=[f"se_{t}" for t in targets]).to_csv(
        rd / "recovery_r_correctness_grid.csv")
    pd.DataFrame(rec_scaf, index=[f"floor_{f}" for f in floors],
                 columns=[f"se_{t}" for t in targets]).to_csv(
        rd / "recovery_r_scaffolding_grid.csv")

    # ---- console report ----
    print("\n" + md_all + "\n\n" + md_excl + "\n")
    print(f"[floor-inertness] min stop length overall = {min_len_overall} scenarios; "
          f"min across every (floor,se) cell = {min_len_per_cell}; max floor = {MAX_FLOOR}; "
          f">= max floor: {min_len_overall >= MAX_FLOOR}; "
          f"per-model stop floor-invariant: {floor_invariant}")
    j_by_se = {t: k for k, t in enumerate(targets)}
    for se in (0.22, 0.25, 0.30):
        if se in j_by_se:
            j = j_by_se[se]
            # floor is inert; report the floor=6 row value (identical across floors)
            print(f"[recovery in-sample @ SE={se:.2f}] correctness r={rec_corr[0, j]:.3f} "
                  f"scaffolding r={rec_scaf[0, j]:.3f}")
    print(f"[figure] {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
