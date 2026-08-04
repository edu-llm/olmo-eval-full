#!/usr/bin/env python3
"""Validate ATLAS 3PL IRT params on models we scored that ATLAS never saw.

Loads the published ATLAS ARC item bank (3PL), maps numeric item indices to
ARC-Challenge test question_ids (identical order to our MCQ CSVs), runs a
Fisher-information CAT + p-IRT accuracy reconstruction on each held-out model,
and scatters diagnostic-predicted vs actual accuracy.
"""
from __future__ import annotations

import csv
import glob
import os
import zipfile

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(EXP)))
ATLAS = os.path.join(REPO, "AdaptiveTesting/Inputs/ATLAS")
MCQ = os.path.join(REPO, "AdaptiveTesting/Inputs/Open/LLM-Judge/mcq/arc_challenge")
OUT_FIG = os.path.join(EXP, "figures")
OUT_RES = os.path.join(EXP, "results")
OUT = OUT_RES
DATA_ZIP = os.path.join(ATLAS, "data/data.zip")
PARAMS = os.environ.get(
    "ATLAS_PARAMS",
    os.path.join(EXP, "calibration/irt_item_parameters_combined.csv"),
)
IDX_MAP = os.path.join(ATLAS, "arc/atlas_idx_to_question_id.csv")

SE_STOP = float(os.environ.get("ATLAS_SE_STOP", "0.3"))
OUT_TAG = os.environ.get("ATLAS_OUT_TAG", "") or f"atlas_arc_0p5_7b_heldout_se{SE_STOP:g}"
MIN_ITEMS = 8
MAX_ITEMS = 200

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()


def ensure_atlas_matrices() -> str:
    """Return path to ARC test response matrix (for ATLAS model-id roster)."""
    dest_dir = os.path.join(ATLAS, "data")
    test_csv = os.path.join(dest_dir, "gaussian_sampled_arc_response_matrix_test.csv")
    train_csv = os.path.join(dest_dir, "gaussian_sampled_arc_response_matrix_train.csv")
    if os.path.isfile(test_csv) and os.path.isfile(train_csv):
        return test_csv
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP) as zf:
        for name in (
            "data/gaussian_sampled_arc_response_matrix_test.csv",
            "data/gaussian_sampled_arc_response_matrix_train.csv",
        ):
            # zip members are data/<file>; extract into ATLAS/ so paths resolve
            zf.extract(name, ATLAS)
    assert os.path.isfile(test_csv), test_csv
    return test_csv


def load_atlas_models() -> set[str]:
    ensure_atlas_matrices()
    models: set[str] = set()
    for path in (
        os.path.join(ATLAS, "data/gaussian_sampled_arc_response_matrix_test.csv"),
        os.path.join(ATLAS, "data/gaussian_sampled_arc_response_matrix_train.csv"),
    ):
        with open(path) as fh:
            next(fh)
            for line in fh:
                models.add(line.split(",", 1)[0].strip('"'))
    return models


def load_atlas_idx_map() -> dict[int, str]:
    """ATLAS 1-based column index → ARC-Challenge question_id.

    Built from Open LLM Leaderboard ``harness_arc_challenge_25`` example order
    (see ``arc/atlas_idx_to_question_id.csv``). This is NOT HF test-split order.
    """
    if not os.path.isfile(IDX_MAP):
        raise SystemExit(
            f"missing {IDX_MAP}; regenerate via leaderboard details matching"
        )
    out: dict[int, str] = {}
    with open(IDX_MAP) as fh:
        for row in csv.DictReader(fh):
            qid = (row.get("question_id") or "").strip()
            if qid:
                out[int(row["atlas_idx"])] = qid
    return out


def load_atlas_item_bank(
    idx_map: dict[int, str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return (question_ids, a, b, c) for ATLAS-calibrated ARC items.

    ATLAS stores mirt-style (a1, d, g); convert with b = -d/a1, c = g.
    """
    qids: list[str] = []
    a_list: list[float] = []
    b_list: list[float] = []
    c_list: list[float] = []
    with open(PARAMS) as fh:
        for row in csv.DictReader(fh):
            k = int(str(row["X"]).lstrip("X"))
            a = float(row["a1"])
            d = float(row["d"])
            g = float(row["g"])
            if a <= 0 or not np.isfinite(a) or k not in idx_map:
                continue
            qids.append(idx_map[k])
            a_list.append(a)
            b_list.append(-d / a)
            c_list.append(float(np.clip(g, 0.0, 0.999)))
    return qids, np.asarray(a_list), np.asarray(b_list), np.asarray(c_list)


def load_our_responses(qids: list[str]) -> dict[str, np.ndarray]:
    """model_id -> binary response vector aligned to qids."""
    out: dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(os.path.join(MCQ, "*.csv"))):
        slug = os.path.basename(path)[:-4]
        mid = slug.replace("__", "/")
        by_q: dict[str, int] = {}
        with open(path) as fh:
            for row in csv.DictReader(fh):
                by_q[row["question_id"]] = (
                    1 if row["result"].strip().lower() == "correct" else 0
                )
        if not all(q in by_q for q in qids):
            continue
        out[mid] = np.asarray([by_q[q] for q in qids], dtype=float)
    return out


def prob(theta: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    z = np.clip(a * (theta - b), -30, 30)
    return np.clip(c + (1 - c) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


def eap_se(resp: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    m = float((NODES * w).sum())
    sd = float(np.sqrt(((NODES - m) ** 2 * w).sum()))
    return m, sd


def run_cat(
    resp_all: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[float, float, list[int]]:
    n = len(a)
    used = np.zeros(n, dtype=bool)
    theta, se, order = 0.0, 1.0, []
    for _ in range(min(MAX_ITEMS, n)):
        p = prob(theta, a, b, c)
        info = (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.asarray(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        if len(order) >= MIN_ITEMS and se <= SE_STOP:
            break
    return theta, se, order


def pirt_accuracy(
    resp_all: np.ndarray,
    order: list[int],
    theta: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """ATLAS p-IRT: weighted mix of observed subset + predicted unobserved."""
    n = len(a)
    subset = set(order)
    avg_obs = float(resp_all[np.asarray(order)].mean()) if order else 0.0
    unobs = [i for i in range(n) if i not in subset]
    if unobs:
        avg_pred = float(prob(theta, a[unobs], b[unobs], c[unobs]).mean())
    else:
        avg_pred = avg_obs
    w_obs = len(order) / n
    return w_obs * avg_obs + (1 - w_obs) * avg_pred


def main() -> None:
    idx_map = load_atlas_idx_map()
    qids, a, b, c = load_atlas_item_bank(idx_map)
    atlas_models = load_atlas_models()
    responses = load_our_responses(qids)

    held = sorted(m for m in responses if m not in atlas_models)
    overlap = sorted(m for m in responses if m in atlas_models)
    print(
        f"ATLAS ARC bank: {len(qids)} items (3PL, leaderboard order)  "
        f"SE_stop={SE_STOP}\n"
        f"params: {PARAMS}\n"
        f"NOTE: ATLAS params were fit on 25-shot leaderboard labels; "
        f"our CSVs are 0-shot loglik.\n"
        f"our scored models: {len(responses)}  "
        f"in ATLAS: {len(overlap)}  held-out: {len(held)}",
        flush=True,
    )
    if not held:
        raise SystemExit("no held-out models with responses")

    recs = []
    for mid in held:
        resp = responses[mid]
        actual = float(resp.mean())
        theta, se, order = run_cat(resp, a, b, c)
        pred = pirt_accuracy(resp, order, theta, a, b, c)
        mean_p = float(prob(theta, a, b, c).mean())
        recs.append((mid, actual, pred, mean_p, theta, se, len(order)))
        print(
            f"  {mid[:42]:42s} actual={actual:.3f} pirt={pred:.3f} "
            f"meanP={mean_p:.3f} θ={theta:+.2f} SE={se:.3f} items={len(order)}",
            flush=True,
        )

    acts = np.asarray([r[1] for r in recs])
    preds = np.asarray([r[2] for r in recs])
    r = float(np.corrcoef(preds, acts)[0, 1])
    mae = float(np.mean(np.abs(preds - acts)))
    rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))
    avg_items = float(np.mean([r[6] for r in recs]))
    print(
        f"\np-IRT vs actual: r={r:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
        f"avg_items={avg_items:.1f}",
        flush=True,
    )

    tag = OUT_TAG or f"atlas_arc_heldout_se{SE_STOP:g}"
    os.makedirs(OUT_RES, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    csv_path = os.path.join(OUT_RES, f"{tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model",
                "actual_atlas_bank",
                "pirt_pred",
                "mean_p_pred",
                "theta",
                "se",
                "n_items",
            ]
        )
        w.writerows(recs)

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(EXP, ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    lo = min(acts.min(), preds.min()) - 0.03
    hi = max(acts.max(), preds.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect (y=x)")
    ax.scatter(acts, preds, s=70, color="#2ca02c", edgecolor="black", zorder=3)
    for mid, act, pred, *_ in recs:
        ax.annotate(
            mid.split("/")[-1][:18],
            (act, pred),
            fontsize=6.2,
            xytext=(3, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("actual accuracy on ATLAS ARC bank (our 0-shot labels)")
    ax.set_ylabel(f"ATLAS p-IRT diagnostic (3PL CAT, SE≤{SE_STOP:g})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    bank_note = os.environ.get(
        "ATLAS_TITLE_NOTE",
        "0-shot responses, 25-shot bank",
    )
    ax.set_title(
        f"ATLAS 3PL params on {len(held)} held-out models (not in ATLAS)\n"
        f"Pearson r={r:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
        f"avg {avg_items:.0f} items  |  {bank_note}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    png = os.path.join(OUT_FIG, f"{tag}.png")
    fig.savefig(png, dpi=140)
    print(f"saved {csv_path}\nsaved {png}", flush=True)


if __name__ == "__main__":
    main()
