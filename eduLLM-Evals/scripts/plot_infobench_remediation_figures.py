#!/usr/bin/env python3
"""Render remediation-only InFoBench calibration and CAT figures.

The previous InFoBench figure renderer is frozen to the obsolete five-node
snapshot.  This renderer consumes only the completed remediation artifacts.
It intentionally refuses to make Phase-4 uncertainty, order-stability, pooled
CAT, final-policy, or deployment-bank figures because those results do not
exist in the remediation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT / "runs" / "calibration" / "InFoBench_remediation_v1_quadrature_resolution"
)
DEFAULT_CONFIG = REPO_ROOT / "configs" / "infobench_calibration_remediation_v1.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "infobench_calibration_remediation" / "figures"

COLORS = {
    "blue": "#2F80B7",
    "orange": "#D95F02",
    "green": "#2A9D6F",
    "red": "#C84630",
    "purple": "#7251A6",
    "gray": "#72777D",
    "light_gray": "#D9DDE1",
    "dark": "#28323C",
}

FIGURE_SPECS = [
    (
        "01_item_parameter_distributions.png",
        "Provisional item-parameter distributions",
        "The locked grid-61/ridge-0.1 fit contains 2,105 criteria: 2,097 "
        "currently exportable and 8 with nonpositive discrimination. This is "
        "a Phase-2 diagnostic, not a frozen deployment bank.",
    ),
    (
        "02_fit_grid_common_cell_equivalence.png",
        "Fit-grid common-cell equivalence",
        "Held-out log-loss and Brier-score changes are shown with family-cluster "
        "bootstrap 95% intervals. Both adjacent grid comparisons stay inside "
        "the frozen equivalence margin on the same 21,329 evaluation cells.",
    ),
    (
        "03_eap_numerical_stability.png",
        "EAP numerical stability gates",
        "Theta shifts are far below the locked tolerances across denser EAP "
        "grids, a wider integration bound, and an independent quadrature family. "
        "The 401-node normal-trapezoid grid therefore passed Phase 2.",
    ),
    (
        "04_nested_ridge_selection.png",
        "Nested-CV ridge selection",
        "Ridge was selected inside each outer training fold using disjoint inner "
        "predictions and the one-standard-error rule. All five folds selected 0.1.",
    ),
    (
        "05_cat_gate_outcomes.png",
        "CAT gate outcomes by fold",
        "Only outer folds 0, 1, and 4 produced any Phase-3 finalist. Failure "
        "counts overlap because a candidate can fail more than one gate.",
    ),
    (
        "06_inner_cv_cat_tradeoff_landscape.png",
        "Inner-CV CAT tradeoff landscape",
        "All 240 candidate-fold rows are development evidence. Points outlined "
        "in black pass every absolute gate; this is not outer-test performance.",
    ),
    (
        "07_minimum_scenario_sweep_DIAGNOSTIC.png",
        "Minimum-scenario sweep diagnostic",
        "A frozen trace/SE=0.20 slice shows how the minimum-scenario floor changes "
        "recovery slope and pass-rate MAE inside the outer training folds.",
    ),
    (
        "08_se_target_sweep_DIAGNOSTIC.png",
        "Conditional-SE target sweep diagnostic",
        "A frozen trace/floor=12 slice shows the inner-CV test-length and recovery "
        "tradeoff. It does not select a final stopping target.",
    ),
    (
        "09_partial_outer_recovery_INCOMPLETE.png",
        "Partial outer-fold recovery",
        "Only the 32 models in folds 0, 1, and 4 were scored because folds 2 and "
        "3 had no finalist. No pooled recovery estimate or final policy exists.",
    ),
    (
        "10_partial_outer_cat_vs_random_INCOMPLETE.png",
        "Partial outer CAT-versus-random diagnostics",
        "Per-fold CAT and random-baseline results are shown separately for the "
        "three scored folds. They must not be interpreted as an all-52-model result.",
    ),
]


class FigureInputError(RuntimeError):
    """Raised when remediation artifacts cannot support an honest figure set."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FigureInputError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FigureInputError(f"expected object at {path}")
    return value


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise FigureInputError(f"cannot read CSV input {path}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FigureInputError(f"{source} is missing columns: {missing}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureInputError(message)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _require(path.exists() and path.stat().st_size > 10_000, f"empty figure: {path}")


def _title(fig: plt.Figure, main: str, subtitle: str, *, warning: bool = False) -> None:
    fig.suptitle(main, fontsize=16, y=1.035)
    fig.text(
        0.5,
        0.985,
        subtitle,
        ha="center",
        va="top",
        fontsize=10,
        color=COLORS["red"] if warning else COLORS["gray"],
        weight="bold" if warning else "normal",
    )


def _format_fold(value: Any) -> str:
    return f"Fold {int(value)}"


def plot_item_parameters(frame: pd.DataFrame, output: Path) -> None:
    a_col = "a_instruction_following"
    _require_columns(frame, ["criterion_id", a_col, "b", "exportable"], output)
    frame = frame.copy()
    frame["exportable"] = frame["exportable"].astype(bool)
    exportable = frame[frame["exportable"]]
    rejected = frame[~frame["exportable"]]
    _require(len(frame) == 2105, f"expected 2,105 fitted items, found {len(frame)}")
    _require(len(exportable) == 2097, f"expected 2,097 exportable items, found {len(exportable)}")
    _require(len(rejected) == 8, f"expected 8 nonexportable items, found {len(rejected)}")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    _title(
        fig,
        "InFoBench provisional 1D 2PL item bank",
        "Locked Phase-2 fit · N=52 tutor models · 2,105 fitted / 2,097 exportable "
        "· not a final deployment bank",
    )

    axes[0].hist(exportable[a_col], bins=32, color=COLORS["blue"], edgecolor="white")
    med_a = float(exportable[a_col].median())
    axes[0].axvline(med_a, color=COLORS["orange"], linestyle="--", linewidth=2)
    axes[0].axvline(0, color=COLORS["dark"], linewidth=1)
    axes[0].set(title="Discrimination", xlabel="discrimination (a)", ylabel="# criteria")
    axes[0].text(0.97, 0.93, f"median = {med_a:.2f}", transform=axes[0].transAxes, ha="right")

    axes[1].hist(exportable["b"], bins=32, color=COLORS["orange"], edgecolor="white")
    med_b = float(exportable["b"].median())
    axes[1].axvline(med_b, color=COLORS["blue"], linestyle="--", linewidth=2)
    axes[1].axvline(0, color=COLORS["dark"], linewidth=1)
    axes[1].set(title="Difficulty", xlabel="difficulty (b)", ylabel="# criteria")
    axes[1].text(0.97, 0.93, f"median = {med_b:.2f}", transform=axes[1].transAxes, ha="right")

    axes[2].scatter(
        exportable["b"],
        exportable[a_col],
        s=13,
        alpha=0.35,
        color=COLORS["purple"],
        linewidth=0,
        label="exportable",
    )
    axes[2].scatter(
        rejected["b"],
        rejected[a_col],
        s=42,
        marker="x",
        color=COLORS["red"],
        linewidth=1.6,
        label="nonexportable",
    )
    axes[2].axhline(0, color=COLORS["dark"], linewidth=1)
    axes[2].set(title="Item-parameter map", xlabel="difficulty (b)", ylabel="discrimination (a)")
    axes[2].legend(loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.6)
    _save(fig, output)


def plot_common_cell_equivalence(gate: dict[str, Any], output: Path) -> None:
    comparisons = gate.get("comparisons", {})
    ordered = ["41_vs_61", "61_vs_81"]
    _require(gate.get("passed") is True, "common-cell gate did not pass")
    _require(all(key in comparisons for key in ordered), "common-cell comparisons are incomplete")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    _title(
        fig,
        "Fit-grid changes are equivalent on identical held-out cells",
        "21,329 common evaluation cells · family-cluster bootstrap (5,000 replicates) "
        "· shaded region is the frozen ±0.005 margin",
    )
    labels = ["41 → 61", "61 → 81"]
    y = np.arange(len(ordered))
    for ax, metric, label in zip(
        axes, ["log_loss", "brier"], ["Log loss", "Brier score"], strict=True
    ):
        centers, lows, highs = [], [], []
        for key in ordered:
            block = comparisons[key]["metrics"][metric]
            centers.append(float(block["pooled_shift_upper_minus_lower"]))
            lows.append(float(block["family_cluster_bootstrap_ci_95"][0]))
            highs.append(float(block["family_cluster_bootstrap_ci_95"][1]))
        margin = float(comparisons[ordered[0]]["metrics"][metric]["equivalence_margin"])
        ax.axvspan(-margin, margin, color=COLORS["green"], alpha=0.12, label="equivalence region")
        ax.axvline(0, color=COLORS["dark"], linewidth=1)
        for idx, (center, low, high) in enumerate(zip(centers, lows, highs, strict=True)):
            ax.errorbar(
                center,
                idx,
                xerr=np.array([[center - low], [high - center]]),
                fmt="o",
                markersize=7,
                capsize=4,
                color=COLORS["blue"],
                ecolor=COLORS["blue"],
            )
            ax.annotate(
                f"{center:+.5f}",
                xy=(center, idx),
                xytext=(8 if center >= 0 else -8, 0),
                textcoords="offset points",
                ha="left" if center >= 0 else "right",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y, labels)
        ax.set_xlim(-0.0055, 0.0055)
        ax.set_title(label)
        ax.set_xlabel("change: denser fit grid − smaller fit grid")
        ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.7)
        ax.invert_yaxis()
    axes[0].set_ylabel("fit-grid comparison")
    axes[1].legend(loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=3)
    _save(fig, output)


def plot_numerical_stability(
    primary: dict[str, Any],
    summary: pd.DataFrame,
    bound: dict[str, Any],
    cross_family: dict[str, Any],
    output: Path,
) -> None:
    ordered = ["401_vs_801", "801_vs_1601", "401_vs_1601"]
    comparisons = primary.get("comparisons", {})
    _require(
        primary.get("passed") is True and primary.get("locked_eap_grid") == 401,
        "401-node EAP grid was not locked",
    )
    _require(bound.get("passed") is True, "integration-bound gate failed")
    _require(cross_family.get("passed") is True, "cross-family gate failed")
    _require(all(key in comparisons for key in ordered), "primary grid comparisons are incomplete")
    _require_columns(summary, ["nodes", "n_models", "unique_theta_rounded_6"], output)
    _require(
        (summary["unique_theta_rounded_6"] == 52).all(), "theta estimates still show grid pile-up"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    _title(
        fig,
        "Phase 2 numerical stability gates all passed",
        "Fit grid 61 · ridge 0.1 diagnostic · normal-trapezoid EAP grid 401 "
        "locked on [−8, 8] · 52/52 unique model thetas",
    )

    x = np.arange(len(ordered))
    metrics = [
        ("median_absolute_theta_shift", "median", "o", COLORS["green"]),
        ("p95_absolute_theta_shift", "95th percentile", "s", COLORS["blue"]),
        ("maximum_absolute_theta_shift", "maximum", "^", COLORS["orange"]),
    ]
    for field, label, marker, color in metrics:
        values = [max(float(comparisons[key][field]), 1e-14) for key in ordered]
        axes[0].plot(x, values, marker=marker, linewidth=1.8, color=color, label=label)
    threshold = float(comparisons[ordered[0]]["thresholds"]["maximum_absolute_theta_shift"])
    axes[0].axhline(
        threshold, color=COLORS["red"], linestyle="--", label=f"max allowed = {threshold:g}"
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, ["401→801", "801→1601", "401→1601"])
    axes[0].set(title="Denser EAP grids", ylabel="absolute theta shift (log scale)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", which="both", color=COLORS["light_gray"], linewidth=0.6)

    check_labels = ["bound\nfull", "bound\nheld-out", "method\nfull", "method\nheld-out"]
    values = [
        float(bound["maximum_full_theta_shift"]),
        float(bound["maximum_heldout_theta_shift"]),
        float(cross_family["maximum_full_theta_shift"]),
        float(cross_family["maximum_heldout_theta_shift"]),
    ]
    colors = [COLORS["blue"], COLORS["blue"], COLORS["purple"], COLORS["purple"]]
    axes[1].bar(np.arange(4), [max(value, 1e-14) for value in values], color=colors, width=0.68)
    axes[1].axhline(0.005, color=COLORS["red"], linestyle="--", label="max allowed = 0.005")
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(4), check_labels)
    axes[1].set(
        title="Wider bound and independent method", ylabel="maximum theta shift (log scale)"
    )
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", which="both", color=COLORS["light_gray"], linewidth=0.6)

    observed_tail = float(bound["maximum_posterior_tail_mass_observed"])
    allowed_tail = float(bound["maximum_posterior_tail_mass_allowed"])
    axes[2].bar([0], [observed_tail], color=COLORS["green"], width=0.55)
    axes[2].axhline(
        allowed_tail, color=COLORS["red"], linestyle="--", label=f"max allowed = {allowed_tail:.0e}"
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks([0], ["maximum observed"])
    axes[2].set(title="Posterior tail mass beyond |θ| ≥ 7.5", ylabel="tail probability (log scale)")
    axes[2].text(0, observed_tail * 1.5, f"{observed_tail:.2e}", ha="center")
    axes[2].legend(fontsize=8.5)
    axes[2].grid(axis="y", which="both", color=COLORS["light_gray"], linewidth=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.5)
    _save(fig, output)


def plot_nested_ridge(frame: pd.DataFrame, output: Path) -> None:
    needed = [
        "outer_fold",
        "ridge",
        "mean_model_disjoint_log_loss",
        "se_model_disjoint_log_loss",
        "one_se_threshold",
        "within_one_se",
        "selected",
    ]
    _require_columns(frame, needed, output)
    _require(len(frame) == 15, f"expected 15 ridge rows, found {len(frame)}")
    selected = frame[frame["selected"].astype(bool)]
    _require(len(selected) == 5, f"expected one selected ridge per fold, found {len(selected)}")
    _require(np.allclose(selected["ridge"], 0.1), "not every fold selected ridge 0.1")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), sharex=True, sharey=True)
    _title(
        fig,
        "Nested ridge selection uses inner-fold prediction only",
        "Mean per-model disjoint log loss ± SE · strongest ridge within one SE "
        "of the fold minimum · all folds selected ridge 0.1",
    )
    for fold, ax in zip(range(5), axes.flat, strict=False):
        subset = frame[frame["outer_fold"] == fold].sort_values("ridge")
        ax.errorbar(
            subset["ridge"],
            subset["mean_model_disjoint_log_loss"],
            yerr=subset["se_model_disjoint_log_loss"],
            marker="o",
            capsize=3,
            color=COLORS["blue"],
            linewidth=1.6,
        )
        within = subset[subset["within_one_se"].astype(bool)]
        ax.scatter(
            within["ridge"],
            within["mean_model_disjoint_log_loss"],
            s=70,
            facecolors="none",
            edgecolors=COLORS["green"],
            linewidths=1.8,
            label="within one SE",
        )
        chosen = subset[subset["selected"].astype(bool)]
        ax.scatter(
            chosen["ridge"],
            chosen["mean_model_disjoint_log_loss"],
            s=60,
            color=COLORS["orange"],
            marker="D",
            zorder=4,
            label="selected",
        )
        threshold = float(subset["one_se_threshold"].iloc[0])
        ax.axhline(
            threshold, color=COLORS["gray"], linestyle="--", linewidth=1, label="one-SE threshold"
        )
        ax.set_xscale("log")
        ax.set_xticks([0.001, 0.01, 0.1], [".001", ".01", ".1"])
        ax.set_title(_format_fold(fold))
        ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    axes.flat[5].axis("off")
    axes[1, 0].set_xlabel("ridge")
    axes[1, 1].set_xlabel("ridge")
    axes[0, 0].set_ylabel("mean disjoint log loss")
    axes[1, 0].set_ylabel("mean disjoint log loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.91, 0.12))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2, h_pad=2)
    _save(fig, output)


def _failed_gate_counts(frame: pd.DataFrame) -> dict[str, int]:
    keys = [
        "recovery_slope",
        "disjoint_pass_rate_mae",
        "scenario_reduction_vs_random",
        "recovery_correlation_lower_95_ci",
        "absolute_disjoint_pass_rate_bias",
        "paired_ci_favors_cat",
    ]
    counts = {key: 0 for key in keys}
    for raw in frame["failed_gates"].fillna(""):
        failed = {part.strip() for part in str(raw).split("|") if part.strip()}
        for key in keys:
            counts[key] += int(key in failed)
    return counts


def plot_cat_gate_outcomes(
    frame: pd.DataFrame,
    finalists: dict[str, Any],
    gates: dict[str, Any],
    output: Path,
) -> None:
    _require_columns(frame, ["outer_fold", "all_gates_pass", "failed_gates"], output)
    _require(len(frame) == 240, f"expected 240 candidate-fold rows, found {len(frame)}")
    folds = finalists.get("folds", [])
    _require(len(folds) == 5, "expected five outer-fold summaries")
    pass_counts = [int(block["n_candidates_passing_all_absolute_gates"]) for block in folds]
    _require(pass_counts == [3, 5, 0, 0, 1], f"unexpected Phase-3 pass counts: {pass_counts}")
    _require(
        finalists.get("final_policy_freeze_authorized") is False,
        "a final policy was unexpectedly authorized",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [0.95, 1.4]})
    _title(
        fig,
        "Phase 3 CAT policy selection is incomplete",
        "Only 3/5 folds produced a finalist · 48 candidates tested per fold · "
        "no fallback, pooled winner, or final policy",
        warning=True,
    )
    fold_x = np.arange(5)
    colors = [COLORS["green"] if value > 0 else COLORS["red"] for value in pass_counts]
    axes[0].bar(fold_x, pass_counts, color=colors, width=0.65)
    axes[0].set_xticks(fold_x, [f"Fold {fold}" for fold in range(5)])
    axes[0].set_ylim(0, 6.2)
    axes[0].set(title="Candidates passing every absolute gate", ylabel="# of 48 candidates")
    for idx, value in enumerate(pass_counts):
        axes[0].text(idx, value + 0.18, str(value), ha="center", weight="bold")
        if value == 0:
            axes[0].text(
                idx,
                0.12,
                "no finalist",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=COLORS["red"],
                rotation=90,
            )
    axes[0].grid(axis="y", color=COLORS["light_gray"], linewidth=0.7)

    counts = _failed_gate_counts(frame)
    display = {
        "recovery_slope": (
            f"Recovery slope outside {gates['minimum_recovery_slope']:.1f}–"
            f"{gates['maximum_recovery_slope']:.1f}"
        ),
        "disjoint_pass_rate_mae": f"Pass-rate MAE > {gates['maximum_disjoint_pass_rate_mae']:.2f}",
        "scenario_reduction_vs_random": (
            f"Scenario reduction < {gates['minimum_scenario_reduction_vs_random']:.0%}"
        ),
        "recovery_correlation_lower_95_ci": (
            f"Recovery r lower CI < {gates['minimum_recovery_correlation_lower_95_ci']:.2f}"
        ),
        "absolute_disjoint_pass_rate_bias": (
            f"Absolute pass-rate bias > {gates['maximum_absolute_disjoint_pass_rate_bias']:.2f}"
        ),
        "paired_ci_favors_cat": "Paired CI does not favor CAT",
    }
    ordered = sorted(counts, key=counts.get)
    y = np.arange(len(ordered))
    axes[1].barh(y, [counts[key] for key in ordered], color=COLORS["orange"])
    axes[1].set_yticks(y, [display[key] for key in ordered])
    axes[1].set(
        title="Overlapping reasons candidates failed", xlabel="# of 240 candidate-fold rows"
    )
    axes[1].set_xlim(0, max(counts.values()) * 1.18)
    for idx, key in enumerate(ordered):
        axes[1].text(counts[key] + 2, idx, str(counts[key]), va="center")
    axes[1].grid(axis="x", color=COLORS["light_gray"], linewidth=0.7)
    axes[1].text(
        0.99,
        0.02,
        "Counts overlap: one row may fail multiple gates.",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=4)
    _save(fig, output)


def plot_tradeoff_landscape(frame: pd.DataFrame, gates: dict[str, Any], output: Path) -> None:
    needed = [
        "outer_fold",
        "selector",
        "mean_scenario_count",
        "recovery_slope",
        "disjoint_pass_rate_mae",
        "all_gates_pass",
    ]
    _require_columns(frame, needed, output)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharex=True, sharey=True)
    _title(
        fig,
        "Inner-CV CAT tradeoff landscape — development evidence only",
        "All 240 candidate-fold rows · color = disjoint pass-rate MAE · black outline "
        "= passes every absolute gate · not outer-test performance",
        warning=True,
    )
    vmin = float(frame["disjoint_pass_rate_mae"].min())
    vmax = float(frame["disjoint_pass_rate_mae"].max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    markers = {"trace": "o", "dopt": "^"}
    last = None
    for fold, ax in enumerate(axes):
        subset = frame[frame["outer_fold"] == fold]
        ax.axhspan(
            float(gates["minimum_recovery_slope"]),
            float(gates["maximum_recovery_slope"]),
            color=COLORS["green"],
            alpha=0.09,
        )
        for selector, marker in markers.items():
            values = subset[subset["selector"] == selector]
            failing = values[~values["all_gates_pass"].astype(bool)]
            passing = values[values["all_gates_pass"].astype(bool)]
            last = ax.scatter(
                failing["mean_scenario_count"],
                failing["recovery_slope"],
                c=failing["disjoint_pass_rate_mae"],
                cmap="viridis_r",
                norm=norm,
                marker=marker,
                s=35,
                alpha=0.72,
                linewidth=0,
            )
            ax.scatter(
                passing["mean_scenario_count"],
                passing["recovery_slope"],
                c=passing["disjoint_pass_rate_mae"],
                cmap="viridis_r",
                norm=norm,
                marker=marker,
                s=68,
                edgecolor="black",
                linewidth=1.2,
                zorder=3,
            )
        ax.set_title(f"Fold {fold} · {int(subset['all_gates_pass'].astype(bool).sum())} pass")
        ax.set_xlabel("mean scenarios")
        ax.grid(color=COLORS["light_gray"], linewidth=0.55)
    axes[0].set_ylabel("recovery slope")
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["gray"],
            label="trace",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=COLORS["gray"],
            label="D-opt",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="white",
            label="all gates pass",
            markersize=8,
        ),
    ]
    axes[0].legend(handles=legend, loc="lower right", fontsize=8)
    _require(last is not None, "tradeoff plot has no points")
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.14, top=0.80, wspace=0.12)
    color_axis = fig.add_axes((0.92, 0.18, 0.012, 0.58))
    cbar = fig.colorbar(last, cax=color_axis)
    cbar.set_label("disjoint pass-rate MAE (lower is better)")
    cbar.ax.axhline(
        float(gates["maximum_disjoint_pass_rate_mae"]), color=COLORS["red"], linewidth=2
    )
    _save(fig, output)


def _plot_fold_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    colors: list[str],
    median_label: str = "median across folds",
) -> None:
    piv = frame.pivot(index=x_col, columns="outer_fold", values=y_col).sort_index()
    for idx, fold in enumerate(piv.columns):
        ax.plot(
            piv.index,
            piv[fold],
            marker="o",
            linewidth=1,
            alpha=0.55,
            color=colors[idx],
            label=f"Fold {int(fold)}",
        )
    ax.plot(
        piv.index,
        piv.median(axis=1),
        marker="D",
        linewidth=2.8,
        color=COLORS["dark"],
        label=median_label,
        zorder=5,
    )


def plot_floor_sweep(frame: pd.DataFrame, gates: dict[str, Any], output: Path) -> None:
    subset = frame[
        (frame["selector"] == "trace")
        & np.isclose(frame["conditional_se_target"].astype(float), 0.20)
    ].copy()
    _require(len(subset) == 30, f"expected 30 floor-sweep rows, found {len(subset)}")
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["red"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
    _title(
        fig,
        "Minimum-scenario floor sweep — inner-CV diagnostic",
        "Frozen slice: trace selector, conditional SE target 0.20 · each thin line "
        "is an outer training fold · does not select a final policy",
        warning=True,
    )
    _plot_fold_lines(axes[0], subset, "minimum_scenarios", "recovery_slope", colors=colors)
    axes[0].axhspan(
        float(gates["minimum_recovery_slope"]),
        float(gates["maximum_recovery_slope"]),
        color=COLORS["green"],
        alpha=0.10,
    )
    axes[0].set(
        title="Recovery calibration", xlabel="required minimum scenarios", ylabel="recovery slope"
    )
    axes[0].grid(color=COLORS["light_gray"], linewidth=0.6)

    _plot_fold_lines(axes[1], subset, "minimum_scenarios", "disjoint_pass_rate_mae", colors=colors)
    limit = float(gates["maximum_disjoint_pass_rate_mae"])
    axes[1].axhspan(0, limit, color=COLORS["green"], alpha=0.10)
    axes[1].axhline(limit, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axes[1].set(
        title="Held-out pass-rate error",
        xlabel="required minimum scenarios",
        ylabel="disjoint pass-rate MAE",
    )
    axes[1].grid(color=COLORS["light_gray"], linewidth=0.6)
    axes[0].legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=3)
    _save(fig, output)


def plot_se_sweep(frame: pd.DataFrame, gates: dict[str, Any], output: Path) -> None:
    subset = frame[
        (frame["selector"] == "trace") & (frame["minimum_scenarios"].astype(int) == 12)
    ].copy()
    _require(len(subset) == 20, f"expected 20 SE-sweep rows, found {len(subset)}")
    floor_binds = all(
        subset.groupby("outer_fold")[column].nunique().max() == 1
        for column in ["mean_scenario_count", "recovery_slope", "disjoint_pass_rate_mae"]
    )
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["red"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    _title(
        fig,
        "Conditional-SE target sweep — inner-CV diagnostic",
        "Frozen slice: trace selector, minimum 12 scenarios · the floor binds, "
        "so changing SE target does not change this slice"
        if floor_binds
        else (
            "Frozen slice: trace selector, minimum 12 scenarios · lower SE target "
            "is stricter · each thin line is an outer training fold"
        ),
        warning=True,
    )
    _plot_fold_lines(axes[0], subset, "conditional_se_target", "mean_scenario_count", colors=colors)
    axes[0].set(title="Test length", xlabel="conditional SE target", ylabel="mean scenarios")
    axes[0].grid(color=COLORS["light_gray"], linewidth=0.6)

    _plot_fold_lines(axes[1], subset, "conditional_se_target", "recovery_slope", colors=colors)
    axes[1].axhspan(
        float(gates["minimum_recovery_slope"]),
        float(gates["maximum_recovery_slope"]),
        color=COLORS["green"],
        alpha=0.10,
    )
    axes[1].set(
        title="Recovery calibration", xlabel="conditional SE target", ylabel="recovery slope"
    )
    axes[1].grid(color=COLORS["light_gray"], linewidth=0.6)

    _plot_fold_lines(
        axes[2], subset, "conditional_se_target", "disjoint_pass_rate_mae", colors=colors
    )
    limit = float(gates["maximum_disjoint_pass_rate_mae"])
    axes[2].axhspan(0, limit, color=COLORS["green"], alpha=0.10)
    axes[2].axhline(limit, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axes[2].set(
        title="Held-out pass-rate error",
        xlabel="conditional SE target",
        ylabel="disjoint pass-rate MAE",
    )
    axes[2].grid(color=COLORS["light_gray"], linewidth=0.6)
    axes[0].legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.5)
    _save(fig, output)


def plot_partial_outer_recovery(
    per_model: pd.DataFrame,
    metrics: dict[str, Any],
    output: Path,
) -> None:
    needed = ["status", "outer_fold", "theta_reference", "theta_cat_mwle", "candidate_id"]
    _require_columns(per_model, needed, output)
    ok = per_model[per_model["status"] == "ok"].copy()
    not_run = per_model[per_model["status"] == "not_run_no_phase3_finalist"]
    _require(
        len(ok) == 32 and len(not_run) == 20,
        f"expected 32 scored and 20 unscored models, found {len(ok)} and {len(not_run)}",
    )
    folds = sorted(ok["outer_fold"].astype(int).unique())
    _require(folds == [0, 1, 4], f"unexpected scored folds: {folds}")
    diagnostics = {int(row["outer_fold"]): row for row in metrics["per_fold_finalist_diagnostics"]}
    _require(
        metrics.get("pooled_final_winner_metrics") is None,
        "pooled final-winner metrics unexpectedly exist",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharex=True, sharey=True)
    _title(
        fig,
        "Partial outer-fold CAT ability recovery",
        "INCOMPLETE: 32/52 models in 3/5 folds · folds 2 and 3 unscored · "
        "no pooled estimate · no final CAT policy",
        warning=True,
    )
    all_values = np.concatenate([ok["theta_reference"].to_numpy(), ok["theta_cat_mwle"].to_numpy()])
    low = math.floor(float(np.nanmin(all_values))) - 0.3
    high = math.ceil(float(np.nanmax(all_values))) + 0.3
    for fold, ax in zip(folds, axes, strict=True):
        subset = ok[ok["outer_fold"].astype(int) == fold]
        diag = diagnostics[fold]
        ax.scatter(
            subset["theta_reference"],
            subset["theta_cat_mwle"],
            s=42,
            color=COLORS["blue"],
            alpha=0.8,
        )
        ax.plot(
            [low, high],
            [low, high],
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1,
            label="perfect recovery",
        )
        slope, intercept = np.polyfit(subset["theta_reference"], subset["theta_cat_mwle"], 1)
        xs = np.array(
            [float(subset["theta_reference"].min()), float(subset["theta_reference"].max())]
        )
        ax.plot(xs, intercept + slope * xs, color=COLORS["orange"], linewidth=2, label="fold OLS")
        candidate = (
            str(subset["candidate_id"].iloc[0])
            .replace("floor_", "floor ")
            .replace("__se_", " · SE ")
            .replace("__selector_", " · ")
        )
        candidate = candidate.replace("0p", "0.")
        ax.set_title(f"Fold {fold} · n={len(subset)}\n{candidate}", fontsize=10.5)
        recovery_text = (
            f"r = {diag['recovery_correlation']:.3f}\n"
            f"95% lower r = {diag['recovery_correlation_lower_95_ci']:.3f}\n"
            f"slope = {diag['recovery_slope']:.3f}"
        )
        ax.text(
            0.04,
            0.96,
            recovery_text,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": COLORS["light_gray"],
            },
        )
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("reference theta (administration pool EAP)")
        ax.grid(color=COLORS["light_gray"], linewidth=0.6)
    axes[0].set_ylabel("CAT MWLE theta")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.4)
    _save(fig, output)


def plot_partial_cat_vs_random(
    metrics: dict[str, Any],
    prediction: pd.DataFrame,
    output: Path,
) -> None:
    rows = metrics.get("per_fold_finalist_diagnostics", [])
    _require(len(rows) == 3, f"expected three partial outer diagnostics, found {len(rows)}")
    folds = [int(row["outer_fold"]) for row in rows]
    _require(folds == [0, 1, 4], f"unexpected diagnostic folds: {folds}")
    _require_columns(prediction, ["fold", "mode", "log_loss", "pass_rate_mae"], output)
    _require(
        len(prediction) == 6, f"expected six CAT/baseline prediction rows, found {len(prediction)}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    _title(
        fig,
        "Partial outer-fold CAT versus random baseline",
        "INCOMPLETE: 32/52 models in folds 0, 1, and 4 only · per-fold diagnostics, "
        "not a pooled efficiency claim or final policy",
        warning=True,
    )
    x = np.arange(3)
    width = 0.34
    cat_counts = [float(row["mean_scenario_count"]) for row in rows]
    random_counts = [float(row["mean_random_scenario_count"]) for row in rows]
    axes[0].bar(x - width / 2, cat_counts, width, color=COLORS["blue"], label="CAT")
    axes[0].bar(x + width / 2, random_counts, width, color=COLORS["gray"], label="random baseline")
    axes[0].set_xticks(x, [f"Fold {fold}" for fold in folds])
    axes[0].set(title="Scenarios administered", ylabel="mean scenarios")
    axes[0].legend(fontsize=8.5)
    for idx, row in enumerate(rows):
        reduction_text = (
            f"{row['scenario_reduction_vs_random']:.0%} fewer\n"
            f"95% CI {row['scenario_reduction_lower_95_ci']:.0%}–"
            f"{row['scenario_reduction_upper_95_ci']:.0%}"
        )
        axes[0].text(
            idx,
            max(cat_counts[idx], random_counts[idx]) + 1,
            reduction_text,
            ha="center",
            fontsize=8,
        )
    axes[0].set_ylim(0, max(random_counts) * 1.28)
    axes[0].grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)

    for ax, metric, title, ylabel in [
        (axes[1], "log_loss", "Disjoint prediction log loss", "log loss (lower is better)"),
        (axes[2], "pass_rate_mae", "Disjoint pass-rate error", "pass-rate MAE (lower is better)"),
    ]:
        cat = prediction[prediction["mode"] == "cat"].set_index("fold").loc[folds]
        baseline = prediction[prediction["mode"] == "baseline"].set_index("fold").loc[folds]
        ax.bar(x - width / 2, cat[metric], width, color=COLORS["blue"], label="CAT")
        ax.bar(
            x + width / 2, baseline[metric], width, color=COLORS["gray"], label="random baseline"
        )
        ax.set_xticks(x, [f"Fold {fold}" for fold in folds])
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    axes[2].axhline(
        0.07, color=COLORS["red"], linestyle="--", linewidth=1.2, label="selection gate = 0.07"
    )
    axes[2].legend(fontsize=8.2)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.5)
    _save(fig, output)


def _write_guide(output_dir: Path) -> Path:
    lines = [
        "# InFoBench remediation figures",
        "",
        "These figures use only the completed remediation artifacts from "
        "`runs/calibration/InFoBench_remediation_v1_quadrature_resolution/`.",
        "",
        "## What is complete",
        "",
        "Phase 2 is complete: the 61-point fit grid passed common-cell equivalence, "
        "the 401-node normal-trapezoid EAP grid passed numerical checks, and ridge "
        "selection was run inside the nested cross-validation design.",
        "",
        "## What is incomplete",
        "",
        "Phase 3 did not produce finalists in folds 2 and 3. Outer CAT diagnostics "
        "therefore cover 32 of 52 models in only three folds. There is no pooled "
        "outer estimate, no frozen final CAT policy, and no Phase-4 parameter-"
        "bootstrap or order-stability result. The Qwen-on-InFoBench human audit also "
        "remains incomplete.",
        "",
        "## Figure-by-figure guide",
        "",
    ]
    for filename, title, description in FIGURE_SPECS:
        lines.extend([f"### `{filename}` — {title}", "", description, ""])
    lines.extend(
        [
            "## Deliberately not reproduced",
            "",
            "The obsolete run included total-SE, order/seed stability, final-policy "
            "recovery, complete CAT-versus-random, and model-leaderboard figures. "
            "Those require Phase 4 and/or a final policy and would be misleading for "
            "the current run, so they are not present here.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python scripts/plot_infobench_remediation_figures.py",
            "```",
            "",
            "`figure_source_manifest.json` records SHA-256 hashes for every source "
            "artifact and rendered PNG.",
            "",
        ]
    )
    path = output_dir / "FIGURE_GUIDE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_manifest(
    output_dir: Path,
    run_dir: Path,
    config_path: Path,
    sources: list[Path],
    figures: list[Path],
) -> Path:
    manifest = {
        "schema_version": "infobench-remediation-figure-manifest-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "renderer": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "renderer_sha256": _sha256(Path(__file__).resolve()),
        "run_dir": str(run_dir.resolve()),
        "phase2_status": "complete",
        "phase3_status": "incomplete_3_of_5_folds_with_finalists",
        "outer_model_support": {"scored": 32, "total": 52},
        "phase4_status": "not_run",
        "final_cat_policy_frozen": False,
        "source_files": {
            str(path.resolve().relative_to(REPO_ROOT)): _sha256(path) for path in sorted(sources)
        },
        "figure_files": {path.name: _sha256(path) for path in sorted(figures)},
        "config": {
            "path": str(config_path.resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(config_path),
        },
    }
    path = output_dir / "figure_source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render(run_dir: Path, config_path: Path, output_dir: Path) -> list[Path]:
    paths = {
        "items": run_dir / "fit_cache" / "grid_061" / "ridge_0p1" / "full" / "item_params.csv",
        "common": run_dir / "common_support" / "common_cell_gate.json",
        "primary": run_dir / "numerical_gates" / "primary_grid_gate.json",
        "primary_summary": run_dir / "numerical_gates" / "primary_grid_summary.csv",
        "bound": run_dir / "numerical_gates" / "bound_check.json",
        "cross": run_dir / "numerical_gates" / "cross_family_check.json",
        "ridge": run_dir / "nested_cat_cv" / "inner_ridge_results.csv",
        "inner": run_dir / "nested_cat_cv" / "inner_candidate_results.csv",
        "finalists": run_dir / "nested_cat_cv" / "outer_fold_finalists.json",
        "outer_rows": run_dir / "nested_cat_cv" / "outer_oof_per_model.csv",
        "outer_metrics": run_dir / "nested_cat_cv" / "outer_metrics.json",
        "prediction": run_dir / "nested_cat_cv" / "disjoint_prediction_metrics.csv",
    }
    missing = [path for path in [*paths.values(), config_path] if not path.is_file()]
    if missing:
        raise FigureInputError(f"missing required remediation artifacts: {missing}")

    items = _load_csv(paths["items"])
    common = _load_json(paths["common"])
    primary = _load_json(paths["primary"])
    primary_summary = _load_csv(paths["primary_summary"])
    bound = _load_json(paths["bound"])
    cross = _load_json(paths["cross"])
    ridge = _load_csv(paths["ridge"])
    inner = _load_csv(paths["inner"])
    finalists = _load_json(paths["finalists"])
    outer_rows = _load_csv(paths["outer_rows"])
    outer_metrics = _load_json(paths["outer_metrics"])
    prediction = _load_csv(paths["prediction"])
    config = _load_json(config_path)
    gates = config.get("selection_gates")
    _require(isinstance(gates, dict), "config lacks selection_gates")

    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / filename for filename, _, _ in FIGURE_SPECS]
    plot_item_parameters(items, expected[0])
    plot_common_cell_equivalence(common, expected[1])
    plot_numerical_stability(primary, primary_summary, bound, cross, expected[2])
    plot_nested_ridge(ridge, expected[3])
    plot_cat_gate_outcomes(inner, finalists, gates, expected[4])
    plot_tradeoff_landscape(inner, gates, expected[5])
    plot_floor_sweep(inner, gates, expected[6])
    plot_se_sweep(inner, gates, expected[7])
    plot_partial_outer_recovery(outer_rows, outer_metrics, expected[8])
    plot_partial_cat_vs_random(outer_metrics, prediction, expected[9])
    _require(all(path.is_file() for path in expected), "not every expected figure was rendered")
    _write_guide(output_dir)
    _write_manifest(output_dir, run_dir, config_path, list(paths.values()), expected)
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()
    figures = render(args.run_dir.resolve(), args.config.resolve(), args.output_dir.resolve())
    print(f"Rendered {len(figures)} remediation figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
