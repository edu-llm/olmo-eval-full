#!/usr/bin/env python3
"""Re-run the Pedagogy MCQ IRT/CAT feasibility experiment on the expanded model set.

We just scored 26 more models on the pedagogy MCQ benchmark, taking the calibration
pool from 52 to 78 models. This script re-runs, on BOTH the original 52-model set (OLD)
and the expanded 78-model set (NEW), the exact same three analyses the original
feasibility study used, so the two are directly comparable:

  1. 2PL bank  (girth twopl_mml)  held-out CAT diagnostic at SE<=0.3 and SE<=0.15
  2. 1PL/Rasch bank (girth rasch_mml) held-out CAT diagnostic at the same SE targets
  3. Linear recalibration (y = a*x + b) of raw CAT-predicted -> true pedagogy accuracy

Methodology is reused verbatim from ``scripts/mcq_diagnostic.py`` (2PL),
``rasch_1pl/run_pedagogy_1pl.py`` (1PL), and ``linear_calibration/run_linear_calibration.py``:
same ``load_benchmark`` / ``filter_items`` / ``fit_2pl``, same seed-7 held-out split with
``n_test = min(12, n_models // 3)``, same EAP-theta Fisher-information CAT with an 8-item
floor, same p-IRT mean-probability predictor, same least-squares linear map.

OLD is reproduced by subsetting the full 78-model matrix to the 52 originally-scored
models; because every model answers the identical 920 pedagogy items in the identical
order, that subset is bit-for-bit the matrix the original scripts saw, so the OLD column
here reproduces the published feasibility numbers as a self-check.

CPU-only, torch-free, ``uv run`` friendly, local ``.mplcache``. Reads per-item response
CSVs from ``_mcq_data/pedagogy/`` (downloaded read-only from S3). Writes every output
into this folder; nothing outside ``expanded_calibration/`` is touched.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
REPO = HERE.parents[5]  # .../olmo-eval-full
sys.path.insert(0, str(REPO / "eduLLM-Evals"))

from tutor_cat.mcq_irt.calibrate import fit_2pl  # noqa: E402
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402

MCQ_DIR = HERE / "_mcq_data"
BENCH = "pedagogy"
SEED = 7
N_TEST_CAP = 12
SE_TARGETS = [0.3, 0.15]
PRIMARY_SE = 0.3

# The 26 models scored on 2026-08-03 (S3 manifest date), as <org>__<model> file stems.
# A loaded model name m is "new" iff m.replace('/', '__') is in this set. This is robust
# to slash/underscore details and needs no separate name mapping.
NEW_STEMS = {
    "HuggingFaceH4__zephyr-7b-gemma-v0.1",
    "HuggingFaceTB__SmolLM-1.7B-Instruct",
    "HuggingFaceTB__SmolLM-360M-Instruct",
    "HuggingFaceTB__SmolLM2-1.7B-Instruct",
    "HuggingFaceTB__SmolLM2-360M-Instruct",
    "Intel__neural-chat-7b-v3-2",
    "Nexusflow__Starling-LM-7B-beta",
    "Open-Orca__Mistral-7B-OpenOrca",
    "Qwen__Qwen2-1.5B-Instruct",
    "Qwen__Qwen2-1.5B",
    "Qwen__Qwen2.5-1.5B-Instruct",
    "allenai__OLMo-2-1124-7B-Instruct",
    "argilla__notus-7b-v1",
    "bigscience__bloom-1b1",
    "bigscience__bloom-1b7",
    "bigscience__bloom-560m",
    "deepseek-ai__deepseek-coder-1.3b-base",
    "deepseek-ai__deepseek-llm-7b-chat",
    "facebook__opt-350m",
    "ibm__merlinite-7b",
    "kyutai__helium-1-preview-2b",
    "openai-community__gpt2-large",
    "openai-community__gpt2-medium",
    "openai-community__gpt2-xl",
    "princeton-nlp__Sheared-LLaMA-1.3B",
    "princeton-nlp__Sheared-LLaMA-2.7B",
}

# ---------------------------------------------------------------------------
# CAT / EAP machinery, copied verbatim from scripts/mcq_diagnostic.py.
# ---------------------------------------------------------------------------
NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()


def prob(theta, a, b):
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(a * (theta - b), -30, 30))), 1e-6, 1 - 1e-6)


def eap_se(resp, a, b):
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    m = float((NODES * w).sum())
    sd = float(np.sqrt(((NODES - m) ** 2 * w).sum()))
    return m, sd


def run_cat(resp_all, a, b, se_stop, max_items, min_items=8):
    n = len(a)
    used = np.zeros(n, bool)
    theta, se, order = 0.0, 1.0, []
    for _ in range(min(max_items, n)):
        p = prob(theta, a, b)
        info = (a**2) * p * (1 - p)  # Fisher information (a==1 -> 1PL)
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.array(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx])
        if len(order) >= min_items and se <= se_stop:
            break
    return theta, se, len(order)


def lsq_fit(x, y):
    """Least-squares a, b for y = a*x + b (true accuracy = a*predicted + b)."""
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def mae(pred, act):
    return float(np.mean(np.abs(pred - act)))


# ---------------------------------------------------------------------------
# One (model-set, method, SE) held-out CAT run.
# ---------------------------------------------------------------------------
def split_models(mat):
    """Reproduce the original seed-7 held-out split exactly."""
    models = list(mat.index)  # mat index is already sorted by load_benchmark
    rng = np.random.default_rng(SEED)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = min(N_TEST_CAP, len(models) // 3)
    test = sorted(order[:n_test])
    train = order[n_test:]
    return train, test


def predict_models(mat, model_list, items, a, b, se_stop, actual_full):
    preds, acts, nitems = [], [], []
    for m in model_list:
        resp = mat.loc[m, items].to_numpy(float)
        theta, se, n_adm = run_cat(resp, a, b, se_stop, max_items=len(items))
        preds.append(float(prob(theta, a, b).mean()))
        acts.append(float(actual_full[m]))
        nitems.append(n_adm)
    return np.array(preds), np.array(acts), np.array(nitems)


def run_config(full_mat, models, tag):
    """Run 2PL + 1PL held-out diagnostic and linear recalibration for one model set."""
    mat = full_mat.loc[sorted(models)]
    train, test = split_models(mat)
    actual_full = mat.mean(axis=1).to_dict()  # true accuracy over ALL 920 items
    kept_mat, report = filter_items(mat.loc[train], benchmark=BENCH)

    result = {
        "tag": tag,
        "n_models": len(mat),
        "n_train": len(train),
        "n_test": len(test),
        "n_items_raw": report.n_items_raw,
        "n_items_kept": report.n_items_kept,
        "test_models": test,
        "train_models": train,
        "actual_full": {m: float(actual_full[m]) for m in mat.index},
        "methods": {},
    }

    for model_name, method in [("2PL", "girth"), ("1PL", "rasch")]:
        cal = fit_2pl(kept_mat, method=method)
        items = cal.items
        a = np.asarray(cal.a, float)
        b = np.asarray(cal.b, float)
        mres = {"n_items_kept": len(items), "n_extreme_a": cal.n_extreme_a, "se": {}}
        for se in SE_TARGETS:
            xtr, ytr, _ = predict_models(mat, train, items, a, b, se, actual_full)
            xte, yte, nte = predict_models(mat, test, items, a, b, se, actual_full)
            r = float(np.corrcoef(xte, yte)[0, 1])
            mae_raw = mae(xte, yte)
            avg_items = float(np.mean(nte))
            pct = 100.0 * avg_items / len(items)

            a_tr, b_tr = lsq_fit(xtr, ytr)
            pred_cal = a_tr * xte + b_tr
            mae_train_fit = mae(pred_cal, yte)

            loo = np.empty_like(xte)
            for i in range(len(xte)):
                msk = np.arange(len(xte)) != i
                ai, bi = lsq_fit(xte[msk], yte[msk])
                loo[i] = ai * xte[i] + bi
            mae_loo = mae(loo, yte)

            a_in, b_in = lsq_fit(xte, yte)
            mae_in = mae(a_in * xte + b_in, yte)

            mres["se"][se] = {
                "r": r, "mae_raw": mae_raw, "avg_items": avg_items, "pct_items": pct,
                "slope_a": a_tr, "intercept_b": b_tr,
                "mae_train_fit": mae_train_fit, "mae_loo": mae_loo, "mae_insample": mae_in,
                "pct_mae_reduction": (100.0 * (mae_raw - mae_train_fit) / mae_raw
                                      if mae_raw > 0 else 0.0),
                "records": [
                    {"model": m, "actual_full": float(yte[i]),
                     "diagnostic_pred": float(xte[i]), "calibrated_pred": float(pred_cal[i]),
                     "n_items": int(nte[i])}
                    for i, m in enumerate(test)
                ],
                "acts": yte, "preds": xte, "pred_cal": pred_cal,
            }
        result["methods"][model_name] = mres
        print(
            f"[{tag}] {model_name}: kept={len(items)}/{report.n_items_raw} "
            f"SE0.3 r={mres['se'][0.3]['r']:.3f} items={mres['se'][0.3]['avg_items']:.1f} "
            f"MAE_raw={mres['se'][0.3]['mae_raw']:.4f} -> cal={mres['se'][0.3]['mae_train_fit']:.4f}",
            flush=True,
        )
    return result


def _metrics_block(mat, train_models, test_models, items, a, b, actual_full):
    """Per-SE raw + linear-recalibrated recovery metrics on a fixed test set."""
    se_res = {}
    for se in SE_TARGETS:
        xtr, ytr, _ = predict_models(mat, train_models, items, a, b, se, actual_full)
        xte, yte, nte = predict_models(mat, test_models, items, a, b, se, actual_full)
        r = float(np.corrcoef(xte, yte)[0, 1])
        mae_raw = mae(xte, yte)
        a_tr, b_tr = lsq_fit(xtr, ytr)
        pred_cal = a_tr * xte + b_tr
        mae_cal = mae(pred_cal, yte)
        se_res[se] = {
            "r": r, "mae_raw": mae_raw, "avg_items": float(np.mean(nte)),
            "pct_items": 100.0 * float(np.mean(nte)) / len(items),
            "slope_a": a_tr, "intercept_b": b_tr, "mae_train_fit": mae_cal,
            "pct_mae_reduction": (100.0 * (mae_raw - mae_cal) / mae_raw if mae_raw > 0 else 0.0),
            "pred_spread": float(xte.max() - xte.min()),
            "acts": yte, "preds": xte, "pred_cal": pred_cal,
        }
    return se_res


def run_controlled(full_mat, old_models, new_models):
    """Controlled comparison: FIX the eval set to the OLD seed-7 held-out 12 models, then
    compare a bank calibrated on the 40 OLD-train models vs a bank calibrated on those same
    40 plus the 26 newly-scored models (66 total). Same targets, same items axis -> isolates
    the pure effect of enlarging the calibration set."""
    old_mat = full_mat.loc[sorted(old_models)]
    old_train, old_test = split_models(old_mat)  # 40 train, 12 test (== faithful OLD split)
    actual_full = full_mat.mean(axis=1).to_dict()
    calib_sets = {
        "OLD_calib40": sorted(old_train),
        "NEW_calib66": sorted(list(old_train) + list(new_models)),
    }
    out = {"test_models": old_test, "n_test": len(old_test), "banks": {}}
    for label, train_models in calib_sets.items():
        kept_mat, report = filter_items(full_mat.loc[train_models], benchmark=BENCH)
        bank = {"n_train": len(train_models), "n_items_kept": report.n_items_kept, "methods": {}}
        for model_name, method in [("2PL", "girth"), ("1PL", "rasch")]:
            cal = fit_2pl(kept_mat, method=method)
            se_res = _metrics_block(full_mat, train_models, old_test,
                                    cal.items, np.asarray(cal.a, float),
                                    np.asarray(cal.b, float), actual_full)
            bank["methods"][model_name] = {"n_items_kept": len(cal.items), "se": se_res}
        out["banks"][label] = bank
        d = bank["methods"]["2PL"]["se"][PRIMARY_SE]
        print(f"[CONTROLLED {label}] train={len(train_models)} kept={report.n_items_kept} "
              f"2PL SE0.3 r={d['r']:.3f} items={d['avg_items']:.1f} MAE_raw={d['mae_raw']:.4f} "
              f"spread={d['pred_spread']:.3f}", flush=True)
    return out


def write_controlled_csv(ctrl):
    path = HERE / "controlled_comparison.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "model", "se_stop", "OLD_calib40", "NEW_calib66", "delta"])
        o = ctrl["banks"]["OLD_calib40"]; n = ctrl["banks"]["NEW_calib66"]
        for model_name in ["2PL", "1PL"]:
            for se in SE_TARGETS:
                od = o["methods"][model_name]["se"][se]
                nd = n["methods"][model_name]["se"][se]
                for metric in ["r", "mae_raw", "mae_train_fit", "avg_items", "pct_items",
                               "slope_a", "intercept_b", "pred_spread"]:
                    w.writerow([metric, model_name, se, round(od[metric], 4),
                                round(nd[metric], 4), round(nd[metric] - od[metric], 4)])
        w.writerow(["n_items_kept", "2PL", "-", o["methods"]["2PL"]["n_items_kept"],
                    n["methods"]["2PL"]["n_items_kept"],
                    n["methods"]["2PL"]["n_items_kept"] - o["methods"]["2PL"]["n_items_kept"]])
        w.writerow(["n_calib_models", "-", "-", o["n_train"], n["n_train"], n["n_train"] - o["n_train"]])
    return path


def make_controlled_fig(ctrl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = ctrl["banks"]["OLD_calib40"]; n = ctrl["banks"]["NEW_calib66"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, model_name in zip(axes, ["2PL", "1PL"]):
        od = o["methods"][model_name]["se"][PRIMARY_SE]
        nd = n["methods"][model_name]["se"][PRIMARY_SE]
        allv = np.concatenate([od["acts"], od["preds"], nd["preds"]])
        lo, hi = allv.min() - 0.03, allv.max() + 0.03
        ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect prediction (y=x)")
        ax.scatter(od["acts"], od["preds"], s=70, color="#ff7f0e", edgecolor="black",
                   zorder=3, label=f"calib 40 (r={od['r']:.3f})")
        ax.scatter(nd["acts"], nd["preds"], s=70, color="#1f77b4", marker="D", edgecolor="black",
                   zorder=3, label=f"calib 66 (r={nd['r']:.3f})")
        ax.set_xlabel(f"actual full {BENCH} accuracy")
        ax.set_ylabel(f"raw CAT-predicted accuracy (SE<={PRIMARY_SE:g})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{model_name}: same 12 held-out targets")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Controlled: calibrate on 40 (old) vs 66 (old+26 new), identical 12-model eval set")
    fig.tight_layout()
    p = HERE / f"controlled_calib40_vs_66_se{PRIMARY_SE:g}.png"
    fig.savefig(p, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Model inventory / degeneracy screen (reads the `predicted` column directly).
# ---------------------------------------------------------------------------
def model_inventory(full_mat):
    rows = []
    for f in sorted((MCQ_DIR / BENCH).glob("*.csv")):
        df = pd.read_csv(f, usecols=["question_id", "model", "predicted", "result"])
        m = str(df["model"].iloc[0])
        n_items = len(df)
        acc = float((df["result"] == "correct").mean())
        vc = df["predicted"].astype(str).value_counts(normalize=True)
        top_letter = str(vc.index[0])
        top_frac = float(vc.iloc[0])
        n_missing = int(df["predicted"].isna().sum())
        degenerate = top_frac >= 0.95  # predicts one option for ~all items
        rows.append({
            "model": m, "is_new": m.replace("/", "__") in NEW_STEMS,
            "n_items": n_items, "accuracy": round(acc, 4),
            "top_pred_letter": top_letter, "top_pred_frac": round(top_frac, 4),
            "n_missing_pred": n_missing, "degenerate_single_answer": degenerate,
            "in_matrix": m in set(full_mat.index),
        })
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------
def write_diag_csvs(res):
    for model_name, mres in res["methods"].items():
        for se, d in mres["se"].items():
            path = HERE / f"diag_{BENCH}_{model_name.lower()}_se{se:g}_{res['tag'].lower()}.csv"
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["model", "actual_full", "diagnostic_pred", "calibrated_pred", "n_items"])
                for rec in d["records"]:
                    w.writerow([rec["model"], rec["actual_full"], rec["diagnostic_pred"],
                                rec["calibrated_pred"], rec["n_items"]])


def write_summary_csv(results):
    path = HERE / "expanded_diagnostic_summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "model", "se_stop", "n_train", "n_test", "n_items_raw",
                    "n_items_kept", "corr", "mae_raw", "avg_cat_items", "pct_items"])
        for res in results:
            for model_name, mres in res["methods"].items():
                for se, d in mres["se"].items():
                    w.writerow([res["tag"], model_name, se, res["n_train"], res["n_test"],
                                res["n_items_raw"], mres["n_items_kept"], round(d["r"], 4),
                                round(d["mae_raw"], 4), round(d["avg_items"], 2),
                                round(d["pct_items"], 2)])
    return path


def write_linear_csv(results):
    path = HERE / "expanded_linear_calibration.csv"
    fields = ["set", "model", "se", "calibration_method", "r", "mae_raw", "mae_calibrated",
              "slope_a", "intercept_b", "pct_mae_reduction", "n_train", "n_test"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for res in results:
            for model_name, mres in res["methods"].items():
                for se, d in mres["se"].items():
                    base = {"set": res["tag"], "model": model_name, "se": se, "r": round(d["r"], 4),
                            "mae_raw": round(d["mae_raw"], 4)}
                    w.writerow({**base, "calibration_method": "train_fit",
                                "mae_calibrated": round(d["mae_train_fit"], 4),
                                "slope_a": round(d["slope_a"], 4), "intercept_b": round(d["intercept_b"], 4),
                                "pct_mae_reduction": round(d["pct_mae_reduction"], 1),
                                "n_train": res["n_train"], "n_test": res["n_test"]})
                    w.writerow({**base, "calibration_method": "loo_test",
                                "mae_calibrated": round(d["mae_loo"], 4),
                                "slope_a": float("nan"), "intercept_b": float("nan"),
                                "pct_mae_reduction": round(100.0 * (d["mae_raw"] - d["mae_loo"]) / d["mae_raw"], 1)
                                if d["mae_raw"] > 0 else 0.0,
                                "n_train": res["n_test"] - 1, "n_test": res["n_test"]})
                    w.writerow({**base, "calibration_method": "insample_naive",
                                "mae_calibrated": round(d["mae_insample"], 4),
                                "slope_a": float("nan"), "intercept_b": float("nan"),
                                "pct_mae_reduction": round(100.0 * (d["mae_raw"] - d["mae_insample"]) / d["mae_raw"], 1)
                                if d["mae_raw"] > 0 else 0.0,
                                "n_train": res["n_test"], "n_test": res["n_test"]})
    return path


def write_compare_csv(old, new):
    path = HERE / "old_vs_new_comparison.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "se_stop", "metric", "old", "new", "delta"])
        for model_name in ["2PL", "1PL"]:
            for se in SE_TARGETS:
                o = old["methods"][model_name]["se"][se]
                n = new["methods"][model_name]["se"][se]
                for metric, ov, nv in [
                    ("pearson_r", o["r"], n["r"]),
                    ("mae_raw", o["mae_raw"], n["mae_raw"]),
                    ("mae_calibrated_trainfit", o["mae_train_fit"], n["mae_train_fit"]),
                    ("avg_cat_items", o["avg_items"], n["avg_items"]),
                    ("pct_items", o["pct_items"], n["pct_items"]),
                    ("slope_a", o["slope_a"], n["slope_a"]),
                    ("intercept_b", o["intercept_b"], n["intercept_b"]),
                ]:
                    w.writerow([model_name, se, metric, round(ov, 4), round(nv, 4),
                                round(nv - ov, 4)])
        # bank size + true-accuracy range (set-level)
        for metric, ov, nv in [
            ("n_models", old["n_models"], new["n_models"]),
            ("n_items_kept_2pl", old["methods"]["2PL"]["n_items_kept"],
             new["methods"]["2PL"]["n_items_kept"]),
            ("n_items_kept_1pl", old["methods"]["1PL"]["n_items_kept"],
             new["methods"]["1PL"]["n_items_kept"]),
        ]:
            w.writerow(["-", "-", metric, ov, nv, nv - ov])
    return path


def acc_range(res):
    vals = np.array(list(res["actual_full"].values()))
    return float(vals.min()), float(vals.max()), float(vals.max() - vals.min())


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------
def make_figs(old, new):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) NEW recovery scatter, 2PL and 1PL, per SE.
    for model_name in ["2PL", "1PL"]:
        for se in SE_TARGETS:
            d = new["methods"][model_name]["se"][se]
            acts, preds = d["acts"], d["preds"]
            r, mae_raw = d["r"], d["mae_raw"]
            nkept = new["methods"][model_name]["n_items_kept"]
            avg_items, pct = d["avg_items"], d["pct_items"]
            fig, ax = plt.subplots(figsize=(7.2, 7))
            lo = min(acts.min(), preds.min()) - 0.03
            hi = max(acts.max(), preds.max()) + 0.03
            ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect prediction (y=x)")
            ax.scatter(acts, preds, s=70, color="#1f77b4", edgecolor="black", zorder=3)
            for rec in d["records"]:
                ax.annotate(rec["model"].split("/")[-1][:16],
                            (rec["actual_full"], rec["diagnostic_pred"]), fontsize=6.5,
                            xytext=(4, 3), textcoords="offset points")
            ax.set_xlabel(f"actual full {BENCH} accuracy (fraction correct, 920 items)")
            ax.set_ylabel(f"CAT-predicted accuracy ({model_name}, SE<={se:g})")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(
                f"{BENCH} {model_name} (NEW, {new['n_models']} models): "
                f"CAT-pred vs actual, {new['n_test']} held-out\n"
                f"Pearson r={r:.3f}, MAE={mae_raw:.3f}, "
                f"{avg_items:.1f} of {nkept} items ({pct:.0f}%)")
            ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(HERE / f"diag_{BENCH}_{model_name.lower()}_se{se:g}_new.png", dpi=130)
            plt.close(fig)

    # 2) Calibrated-vs-raw before/after (NEW), 2PL and 1PL, per SE.
    for model_name in ["2PL", "1PL"]:
        for se in SE_TARGETS:
            d = new["methods"][model_name]["se"][se]
            acts, preds, pred_cal = d["acts"], d["preds"], d["pred_cal"]
            r = d["r"]; a_tr, b_tr = d["slope_a"], d["intercept_b"]
            mae_raw, mae_cal = d["mae_raw"], d["mae_train_fit"]
            red = d["pct_mae_reduction"]
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
            lo = min(acts.min(), preds.min(), pred_cal.min()) - 0.03
            hi = max(acts.max(), preds.max(), pred_cal.max()) + 0.03
            axL = axes[0]
            axL.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect prediction (y=x)")
            axL.scatter(acts, preds, s=70, color="#1f77b4", edgecolor="black", zorder=3,
                        label="test models")
            xs = np.linspace(lo, hi, 100)
            axL.plot(a_tr * xs + b_tr, xs, "-", color="#d62728",
                     label=f"train fit: act={a_tr:.2f}*pred+{b_tr:.2f}")
            axL.set_xlabel(f"actual full {BENCH} accuracy (fraction correct)")
            axL.set_ylabel(f"raw CAT-predicted accuracy ({model_name}, SE<={se:g})")
            axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
            axL.set_title(f"Raw prediction: r={r:.3f}, MAE={mae_raw:.4f}")
            axL.grid(True, alpha=0.3); axL.legend(loc="upper left", fontsize=8)
            axR = axes[1]
            axR.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect prediction (y=x)")
            axR.scatter(acts, pred_cal, s=70, color="#2ca02c", edgecolor="black", zorder=3,
                        label="test models (calibrated)")
            axR.set_xlabel(f"actual full {BENCH} accuracy (fraction correct)")
            axR.set_ylabel(f"calibrated predicted accuracy ({model_name}, SE<={se:g})")
            axR.set_xlim(lo, hi); axR.set_ylim(lo, hi)
            axR.set_title(f"Calibrated (train fit): r={r:.3f}, MAE={mae_cal:.4f} ({red:.0f}% lower)")
            axR.grid(True, alpha=0.3); axR.legend(loc="upper left", fontsize=8)
            fig.suptitle(f"{BENCH} {model_name} linear calibration, NEW {new['n_models']} models "
                         f"({new['n_test']} held-out, fit on {new['n_train']} train)")
            fig.tight_layout()
            fig.savefig(HERE / f"calib_{BENCH}_{model_name.lower()}_se{se:g}_new.png", dpi=130)
            plt.close(fig)

    # 3) OLD vs NEW recovery overlay at the primary SE (2PL and 1PL side by side).
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, model_name in zip(axes, ["2PL", "1PL"]):
        do = old["methods"][model_name]["se"][PRIMARY_SE]
        dn = new["methods"][model_name]["se"][PRIMARY_SE]
        allv = np.concatenate([do["acts"], do["preds"], dn["acts"], dn["preds"]])
        lo, hi = allv.min() - 0.03, allv.max() + 0.03
        ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect prediction (y=x)")
        ax.scatter(do["acts"], do["preds"], s=64, color="#ff7f0e", edgecolor="black",
                   zorder=3, alpha=0.85, label=f"OLD {old['n_models']} (r={do['r']:.3f})")
        ax.scatter(dn["acts"], dn["preds"], s=64, color="#1f77b4", edgecolor="black",
                   marker="D", zorder=3, alpha=0.85, label=f"NEW {new['n_models']} (r={dn['r']:.3f})")
        ax.set_xlabel(f"actual full {BENCH} accuracy")
        ax.set_ylabel(f"CAT-predicted accuracy (SE<={PRIMARY_SE:g})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{model_name} recovery, held-out CAT (SE<={PRIMARY_SE:g})")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Pedagogy CAT recovery: OLD {old['n_models']} vs NEW {new['n_models']} "
                 f"calibration models")
    fig.tight_layout()
    fig.savefig(HERE / f"compare_old_vs_new_se{PRIMARY_SE:g}.png", dpi=130)
    plt.close(fig)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    full = load_benchmark(MCQ_DIR, BENCH).dropna(axis=0, how="any")
    all_models = list(full.index)
    new_models = [m for m in all_models if m.replace("/", "__") in NEW_STEMS]
    old_models = [m for m in all_models if m.replace("/", "__") not in NEW_STEMS]
    print(f"loaded {len(all_models)} models, {full.shape[1]} raw items; "
          f"OLD={len(old_models)} NEW_added={len(new_models)}", flush=True)
    assert len(new_models) == 26, f"expected 26 new, got {len(new_models)}"
    assert len(old_models) == 52, f"expected 52 old, got {len(old_models)}"

    inv = model_inventory(full)
    inv.to_csv(HERE / "model_inventory.csv", index=False)
    degen = inv[inv["degenerate_single_answer"]]
    missing = inv[inv["n_missing_pred"] > 0]
    print(f"inventory: {len(inv)} models; degenerate(single-answer>=95%)={len(degen)}; "
          f"with-missing-pred={len(missing)}", flush=True)
    if len(degen):
        print("  degenerate:", list(degen["model"]), flush=True)

    old = run_config(full, old_models, "OLD")
    new = run_config(full, all_models, "NEW")
    ctrl = run_controlled(full, old_models, new_models)

    write_diag_csvs(old)
    write_diag_csvs(new)
    summ = write_summary_csv([old, new])
    lin = write_linear_csv([old, new])
    cmp = write_compare_csv(old, new)
    ctrl_csv = write_controlled_csv(ctrl)
    make_figs(old, new)
    make_controlled_fig(ctrl)

    # A compact JSON snapshot for the report / README generation.
    snap = {"old": {}, "new": {}}
    for key, res in [("old", old), ("new", new)]:
        amin, amax, arng = acc_range(res)
        snap[key] = {
            "n_models": res["n_models"], "n_train": res["n_train"], "n_test": res["n_test"],
            "n_items_raw": res["n_items_raw"],
            "acc_min": round(amin, 4), "acc_max": round(amax, 4), "acc_range": round(arng, 4),
            "test_models": res["test_models"],
            "methods": {
                mn: {"n_items_kept": m["n_items_kept"],
                     "se": {str(se): {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in d.items()
                                      if k in ("r", "mae_raw", "avg_items", "pct_items",
                                               "slope_a", "intercept_b", "mae_train_fit",
                                               "mae_loo", "mae_insample", "pct_mae_reduction")}
                            for se, d in m["se"].items()}}
                for mn, m in res["methods"].items()
            },
        }
    snap["controlled"] = {
        "test_models": ctrl["test_models"], "n_test": ctrl["n_test"],
        "banks": {
            label: {"n_train": bank["n_train"],
                    "methods": {mn: {"n_items_kept": m["n_items_kept"],
                                     "se": {str(se): {k: round(v, 4) for k, v in d.items()
                                                      if isinstance(v, float)}
                                            for se, d in m["se"].items()}}
                                for mn, m in bank["methods"].items()}}
            for label, bank in ctrl["banks"].items()
        },
    }
    (HERE / "results_snapshot.json").write_text(json.dumps(snap, indent=2))

    print(f"\nwrote:\n  {summ}\n  {lin}\n  {cmp}\n  {ctrl_csv}\n  {HERE / 'model_inventory.csv'}\n"
          f"  {HERE / 'results_snapshot.json'}", flush=True)
    print("\n== CONTROLLED (fixed 12 eval, calib 40 vs 66, SE<=0.3) ==", flush=True)
    for mn in ["2PL", "1PL"]:
        od = ctrl["banks"]["OLD_calib40"]["methods"][mn]["se"][0.3]
        nd = ctrl["banks"]["NEW_calib66"]["methods"][mn]["se"][0.3]
        print(f"  {mn}: r {od['r']:.3f}->{nd['r']:.3f}  items {od['avg_items']:.1f}->{nd['avg_items']:.1f}  "
              f"MAE_raw {od['mae_raw']:.4f}->{nd['mae_raw']:.4f}  "
              f"pred_spread {od['pred_spread']:.3f}->{nd['pred_spread']:.3f}", flush=True)
    print("\n== OLD vs NEW (SE<=0.3) ==", flush=True)
    for mn in ["2PL", "1PL"]:
        o = old["methods"][mn]["se"][0.3]; n = new["methods"][mn]["se"][0.3]
        print(f"  {mn}: r {o['r']:.3f}->{n['r']:.3f}  items {o['avg_items']:.1f}->{n['avg_items']:.1f}  "
              f"MAE_raw {o['mae_raw']:.4f}->{n['mae_raw']:.4f}  "
              f"MAE_cal {o['mae_train_fit']:.4f}->{n['mae_train_fit']:.4f}", flush=True)


if __name__ == "__main__":
    main()
