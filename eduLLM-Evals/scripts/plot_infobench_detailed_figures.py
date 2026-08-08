#!/usr/bin/env python3
"""Generate detailed figures for the 2026-08-04 InFoBench calibration snapshot.

The script is deliberately read-only with respect to the run artifacts. It
creates a separate figure directory and labels same-cohort or coarse-grid
diagnostics explicitly so they are not mistaken for final held-out evidence.
It validates the snapshot's selected structure and settings before rendering,
so a later run cannot silently inherit outdated labels or highlighted values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2f8fbd"
ORANGE = "#d95f02"
PURPLE = "#6a51a3"
RED = "#c44e52"
GREEN = "#3a923a"
GRAY = "#7f7f7f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/calibration/InFoBench_playbook_full"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def linear_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    return r, float(slope), float(intercept)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def validate_snapshot(run_dir: Path) -> None:
    config = read_json(run_dir / "study_config.json")
    structure = read_json(run_dir / "selection.json")
    cat = read_json(run_dir / "cat_configuration_selection.json")
    sensitivity = read_json(run_dir / "sensitivity_selection.json")
    observed = {
        "models": (config.get("input_expectations") or {}).get("models"),
        "structure": structure.get("selected_structure"),
        "se_target": cat.get("selected_se_target"),
        "grid": sensitivity.get("production_grid"),
        "ridge": sensitivity.get("selected_ridge"),
    }
    expected = {
        "models": 52,
        "structure": "overall_1d",
        "se_target": 0.25,
        "grid": 5,
        "ridge": 0.1,
    }
    if observed != expected:
        raise ValueError(
            "this renderer is frozen to the 2026-08-04 InFoBench snapshot; "
            f"expected {expected}, observed {observed}"
        )


def plot_item_parameters(run_dir: Path, out_dir: Path) -> None:
    rows = []
    bank = run_dir / "selected_bank" / "rubrics_fitted_only.jsonl"
    with bank.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "a": row["discrimination"]["instruction_following"],
                    "b": row["difficulty"],
                }
            )
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(frame["a"], bins=35, color=BLUE, edgecolor="white")
    axes[0].axvline(frame["a"].median(), color=ORANGE, linestyle="--")
    axes[0].set(xlabel="discrimination (a)", ylabel="# criteria", title="Discrimination")
    axes[0].text(
        0.98,
        0.95,
        f"median={frame['a'].median():.2f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )

    axes[1].hist(frame["b"], bins=35, color=ORANGE, edgecolor="white")
    axes[1].axvline(frame["b"].median(), color=BLUE, linestyle="--")
    axes[1].set(xlabel="difficulty (b)", ylabel="# criteria", title="Difficulty")
    axes[1].text(
        0.98,
        0.95,
        f"median={frame['b'].median():.2f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
    )

    axes[2].scatter(frame["b"], frame["a"], s=11, alpha=0.35, color=PURPLE)
    axes[2].axhline(0, color=GRAY, linewidth=1)
    axes[2].set(xlabel="difficulty (b)", ylabel="discrimination (a)", title="Item-parameter map")
    fig.suptitle(
        f"InFoBench fitted 1D item bank ({len(frame):,} criteria; parameters provisional at N=52)",
        fontsize=14,
    )
    fig.tight_layout()
    save(fig, out_dir, "01_item_parameter_distributions.png")


def plot_grid_ridge(run_dir: Path, out_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "sensitivity_summary.csv")
    selection = read_json(run_dir / "sensitivity_selection.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    colors = {3: ORANGE, 5: BLUE, 7: PURPLE}
    for grid, group in frame.groupby("grid"):
        group = group.sort_values("ridge")
        axes[0].errorbar(
            group["ridge"],
            group["cv_mean"],
            yerr=group["cv_se"],
            marker="o",
            capsize=3,
            label=f"grid {int(grid)}",
            color=colors.get(int(grid), GRAY),
        )
        axes[1].plot(
            group["ridge"],
            group["median_item_parameter_stability"],
            marker="o",
            label=f"grid {int(grid)}",
            color=colors.get(int(grid), GRAY),
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("ridge penalty")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("held-out log loss (lower is better)")
    axes[0].set_title("Held-out predictive fit")
    axes[1].set_ylabel("median cross-fold item-parameter stability")
    axes[1].set_title("Discrimination/difficulty stability")
    selected = frame.loc[
        frame["grid"].eq(selection["production_grid"])
        & frame["ridge"].eq(selection["selected_ridge"])
    ]
    if len(selected) != 1:
        raise ValueError("selected grid/ridge row is missing or ambiguous")
    axes[0].scatter(
        [selected.iloc[0]["ridge"]],
        [selected.iloc[0]["cv_mean"]],
        marker="*",
        s=180,
        color=RED,
        zorder=5,
    )
    fig.suptitle("InFoBench 1D calibration: grid and ridge sensitivity", fontsize=14)
    fig.tight_layout()
    save(fig, out_dir, "02_grid_ridge_validation.png")


def plot_efficiency(run_dir: Path, out_dir: Path) -> None:
    metrics = read_json(run_dir / "cat_vs_random" / "metrics.json")["modes"]
    cat = metrics["cat"]
    random = metrics["baseline"]
    labels = ["adaptive CAT", "random order"]
    colors = [BLUE, ORANGE]
    panels = [
        ("Mean scenarios", [cat["scenarios"]["mean"], random["scenarios"]["mean"]], 1),
        ("Mean criteria", [cat["criteria_mean"], random["criteria_mean"]], 1),
        (
            "MWLE recovery r",
            [
                cat["recovery_vs_full_bank_eap"]["mwle"]["instruction_following"]["r"],
                random["recovery_vs_full_bank_eap"]["mwle"]["instruction_following"]["r"],
            ],
            3,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (title, values, decimals) in zip(axes, panels):
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.bar_label(bars, fmt=f"%.{decimals}f", padding=3)
        ax.grid(axis="y", alpha=0.2)
        if "recovery" in title.lower():
            ax.set_ylim(0, 1.08)
    fig.suptitle(
        "InFoBench CAT efficiency vs random order (same-cohort replay, SE=0.25)\n"
        "Recovery panel is provisional because the full-bank reference used five EAP nodes",
        fontsize=13,
    )
    fig.tight_layout()
    save(fig, out_dir, "03_cat_vs_random_efficiency.png")


def plot_pirt_oos(run_dir: Path, out_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "estimator_oos_kfold" / "oos_per_model.csv")
    frame = frame.loc[frame["status"].eq("ok")].copy()
    x = frame["pirt_predicted_pass_rate_mwle"].to_numpy(float)
    y = frame["observed_fitted_pass_rate"].to_numpy(float)
    r, slope, intercept = linear_stats(x, y)
    mae = float(np.mean(np.abs(x - y)))
    lo = float(np.nanmin(np.r_[x, y]))
    hi = float(np.nanmax(np.r_[x, y]))
    pad = (hi - lo) * 0.06
    grid = np.linspace(lo - pad, hi + pad, 200)
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    ax.scatter(x, y, s=45, alpha=0.8, color=BLUE, edgecolor="#444444", linewidth=0.4)
    ax.plot(grid, grid, linestyle="--", color=GRAY, label="perfect agreement (y=x)")
    ax.plot(grid, intercept + slope * grid, color=ORANGE, linewidth=2, label=f"OLS slope={slope:.3f}")
    ax.set(
        xlim=(lo - pad, hi + pad),
        ylim=(lo - pad, hi + pad),
        xlabel="predicted pass rate (train-fold parameters + held-out MWLE theta)",
        ylabel="observed held-out pass rate",
    )
    ax.set_title(
        "InFoBench p-IRT predicted vs observed (5-fold held-out)\n"
        f"r={r:.3f}, MAE={mae:.3f}, slope={slope:.3f}, n={len(frame)}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, out_dir, "04_oos_pirt_predicted_vs_actual.png")


def plot_uncertainty(run_dir: Path, out_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "parameter_uncertainty" / "ability_se_components.csv")
    selected = frame.loc[(frame["se_target"].eq(0.25)) & frame["estimator"].eq("mwle")].copy()
    cols = ["se_ability", "se_param", "se_total"]
    means = selected[cols].mean()
    sds = selected[cols].std(ddof=1)
    labels = ["ability SE", "item-parameter SE", "total SE"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(labels, means, yerr=sds, capsize=4, color=[BLUE, ORANGE, PURPLE])
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=3)
    ax.axhline(0.25, color=RED, linestyle="--", label="nominal stopping SE=0.25")
    ax.set_ylabel("mean SE across 52 models")
    ax.set_title(
        "InFoBench MWLE uncertainty at selected CAT setting\n"
        "Total SE includes item uncertainty; order/path uncertainty is not included"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, out_dir, "05_mwle_uncertainty_selected_setting.png")

    aggregate = pd.read_csv(run_dir / "parameter_uncertainty" / "total_se_vs_target.csv")
    aggregate = aggregate.loc[aggregate["estimator"].eq("mwle")].sort_values("se_target")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(aggregate["se_target"], aggregate["median_se_ability"], marker="o", label="ability SE", color=BLUE)
    ax.plot(aggregate["se_target"], aggregate["median_se_param"], marker="s", label="item-parameter SE", color=ORANGE)
    ax.plot(aggregate["se_target"], aggregate["median_se_total"], marker="^", label="total SE", color=PURPLE)
    ax.plot([0.20, 0.35], [0.20, 0.35], linestyle="--", color=GRAY, label="reported target")
    ax.set(xlabel="CAT stopping-SE target", ylabel="median SE across models")
    ax2 = ax.twinx()
    ax2.plot(
        aggregate["se_target"],
        aggregate["mean_scenarios"],
        marker="D",
        linestyle=":",
        color=GREEN,
        label="mean scenarios",
    )
    ax2.set_ylabel("mean scenarios administered")
    handles, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels1 + labels2, loc="upper left")
    ax.grid(alpha=0.2)
    ax.set_title("InFoBench MWLE precision-length tradeoff including item uncertainty")
    fig.tight_layout()
    save(fig, out_dir, "06_mwle_total_se_vs_target.png")


def plot_order_stability(run_dir: Path, out_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "order_stability" / "per_model_spread.csv")
    values = frame["mwle_instruction_following_sd"].dropna().to_numpy(float)
    mean = float(np.mean(values))
    median = float(np.median(values))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.hist(values, bins=12, color=BLUE, alpha=0.8, edgecolor="black")
    ax.axvline(mean, color=ORANGE, linestyle="--", linewidth=2, label=f"mean SD={mean:.3f}")
    ax.axvline(0.25, color=RED, linestyle=":", linewidth=2, label="nominal SE target=0.25")
    ax.set(xlabel="across-order MWLE theta SD per model", ylabel="# models")
    ax.set_title(
        "InFoBench CAT order/seed dependence (8 orders)\n"
        f"median SD={median:.3f}, mean SD={mean:.3f}, max SD={values.max():.3f}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, out_dir, "07_order_seed_stability.png")


def plot_diagnostic_sweeps(run_dir: Path, out_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "cat_configuration_candidates.csv")
    floor = frame.loc[frame["kind"].eq("min_scenarios")].copy()
    floor["value"] = pd.to_numeric(floor["value"])
    floor = floor.sort_values("value")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(floor["value"], floor["mwle_recovery_r"], marker="o", color=BLUE, label="MWLE recovery r")
    ax.plot(
        floor["value"],
        floor["mwle_recovery_slope"],
        marker="s",
        linestyle="--",
        color=PURPLE,
        label="MWLE slope",
    )
    ax.set(xlabel="minimum-scenario floor", ylabel="recovery r / slope")
    ax2 = ax.twinx()
    ax2.plot(floor["value"], floor["mean_scenarios"], marker="^", color=ORANGE, label="mean scenarios")
    ax2.set_ylabel("mean scenarios administered")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")
    ax.grid(alpha=0.2)
    ax.set_title(
        "InFoBench recovery vs minimum-scenario floor\n"
        "Diagnostic same-cohort replay; recovery uses five-node full-bank reference"
    )
    fig.tight_layout()
    save(fig, out_dir, "08_minimum_scenario_sweep_DIAGNOSTIC.png")

    se = frame.loc[frame["kind"].eq("se_target")].copy()
    se["value"] = pd.to_numeric(se["value"])
    se = se.sort_values("value")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axes[0].plot(se["value"], se["mean_scenarios"], marker="o", color=BLUE)
    axes[0].set(xlabel="stopping-SE target", ylabel="mean scenarios", title="Test length")
    axes[1].plot(se["value"], se["mwle_recovery_r"], marker="o", color=BLUE, label="r")
    axes[1].plot(
        se["value"],
        se["mwle_recovery_slope"],
        marker="s",
        linestyle="--",
        color=PURPLE,
        label="slope",
    )
    axes[1].set(xlabel="stopping-SE target", ylabel="recovery r / slope", title="Recovery")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "InFoBench stopping-SE sweep (100% nominal convergence in every arm)\n"
        "Diagnostic same-cohort replay; recovery uses five-node full-bank reference",
        fontsize=13,
    )
    fig.tight_layout()
    save(fig, out_dir, "09_se_target_sweep_DIAGNOSTIC.png")


def plot_recovery_scatter(run_dir: Path, out_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "estimator_oos_kfold" / "oos_per_model.csv")
    frame = frame.loc[frame["status"].eq("ok")].copy()
    x = frame["theta_ref_instruction_following"].to_numpy(float)
    estimators = [
        ("online", "theta_online_instruction_following"),
        ("EAP", "theta_eap_instruction_following"),
        ("MWLE", "theta_mwle_instruction_following"),
    ]
    all_y = np.concatenate([frame[column].to_numpy(float) for _, column in estimators])
    lo = float(np.nanmin(np.r_[x, all_y]))
    hi = float(np.nanmax(np.r_[x, all_y]))
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 100)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (label, column) in zip(axes, estimators):
        y = frame[column].to_numpy(float)
        r, slope, _ = linear_stats(x, y)
        ax.scatter(x, y, s=35, alpha=0.72, color=BLUE, edgecolor="#555555", linewidth=0.3)
        ax.plot(grid, grid, linestyle="--", color=GRAY)
        ax.set_title(f"{label}: r={r:.3f}, slope={slope:.3f}")
        ax.set_xlabel("five-node reference EAP theta")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("held-out CAT theta")
    fig.suptitle(
        "PROVISIONAL InFoBench held-out estimator recovery\n"
        "The reference theta collapses onto five quadrature nodes; rerun with dense 1D scoring before publication",
        fontsize=13,
        color=RED,
    )
    fig.tight_layout()
    save(fig, out_dir, "10_oos_estimator_recovery_PROVISIONAL.png")


def plot_leaderboard(run_dir: Path, out_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "parameter_uncertainty" / "ability_se_components.csv")
    frame = frame.loc[(frame["se_target"].eq(0.25)) & frame["estimator"].eq("mwle")].copy()
    frame = frame.sort_values("theta")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.errorbar(
        frame["theta"],
        y,
        xerr=1.96 * frame["se_total"],
        fmt="o",
        markersize=3.5,
        color="black",
        ecolor="#d8a0a5",
        elinewidth=3,
        capsize=0,
        label="±1.96 total SE",
    )
    ax.errorbar(
        frame["theta"],
        y,
        xerr=1.96 * frame["se_ability"],
        fmt="none",
        ecolor="#4c66a4",
        elinewidth=1.2,
        label="±1.96 ability-only SE",
    )
    ax.set_yticks(y, frame["model"], fontsize=7)
    ax.set_xlabel("instruction-following theta (MWLE, selected CAT path)")
    ax.set_title(
        "Exploratory InFoBench CAT leaderboard (52 calibration models)\n"
        "Same cohort; total-SE bars exclude order/path variability"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save(fig, out_dir, "11_model_leaderboard_EXPLORATORY.png")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / "figures" / "detailed_figures").resolve()
    validate_snapshot(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_item_parameters(run_dir, out_dir)
    plot_grid_ridge(run_dir, out_dir)
    plot_efficiency(run_dir, out_dir)
    plot_pirt_oos(run_dir, out_dir)
    plot_uncertainty(run_dir, out_dir)
    plot_order_stability(run_dir, out_dir)
    plot_diagnostic_sweeps(run_dir, out_dir)
    plot_recovery_scatter(run_dir, out_dir)
    plot_leaderboard(run_dir, out_dir)
    print(f"Wrote detailed figures to {out_dir}")


if __name__ == "__main__":
    main()
