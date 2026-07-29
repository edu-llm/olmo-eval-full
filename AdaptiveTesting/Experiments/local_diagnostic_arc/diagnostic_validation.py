#!/usr/bin/env python3
"""Held-out validation of the adaptive diagnostic.

Calibrate a 2PL bank on a train split, then run each held-out (test) model
through the Fisher-information CAT using its stored item responses, stopping at
SE < SE_STOP. Scatter each held-out model's actual full-benchmark score against
its diagnostic-predicted score, with the y=x line and Pearson correlation.
"""
import csv
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(EXP)))
sys.path.insert(0, os.path.join(REPO, "eduLLM-Evals"))
from tutor_cat.mcq_irt.matrix import filter_items  # noqa: E402

MCQ = os.path.join(REPO, "AdaptiveTesting/Inputs/Open/LLM-Judge/mcq")
OUT_FIG = os.path.join(EXP, "figures")
OUT_RES = os.path.join(EXP, "results")
OUT = OUT_RES  # CSVs land in results/; PNGs use OUT_FIG below

BENCH = sys.argv[1] if len(sys.argv) > 1 else "arc_challenge"
SE_STOP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
MODEL = sys.argv[3] if len(sys.argv) > 3 else "2pl"
SEED = 7
N_TEST = 13
N_TRAIN = 50
MIN_ITEMS = 8
TAG = f"{BENCH}_{MODEL}_se{SE_STOP:g}"

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()


def prob(theta, a, b, c):
    return np.clip(c + (1 - c) / (1.0 + np.exp(-np.clip(a * (theta - b), -30, 30))),
                   1e-6, 1 - 1e-6)


def eap_se(resp, a, b, c):
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    m = float((NODES * w).sum())
    sd = float(np.sqrt(((NODES - m) ** 2 * w).sum()))
    return m, sd


def run_cat(resp_all, a, b, c, se_stop, max_items):
    n = len(a)
    used = np.zeros(n, bool)
    theta, se, order = 0.0, 1.0, []
    for _ in range(min(max_items, n)):
        p = prob(theta, a, b, c)
        info = (a ** 2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2  # 3PL info
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.array(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        if len(order) >= MIN_ITEMS and se <= se_stop:
            break
    return theta, se, len(order)


def load_matrix(bench):
    rows = {}
    for p in sorted(glob.glob(os.path.join(MCQ, bench, "*.csv"))):
        m = os.path.basename(p)[:-4]
        d = {}
        for r in csv.DictReader(open(p)):
            d[r["question_id"]] = 1 if r["result"].strip().lower() == "correct" else 0
        rows[m] = pd.Series(d)
    return pd.DataFrame(rows).T.dropna(axis=0, how="any")


def fit_pyirt(kept, model_type):
    import pyro
    import torch
    from py_irt.config import IrtConfig
    from py_irt.training import IrtModelTrainer

    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    torch.manual_seed(0)

    items = list(kept.columns)
    tmp = os.path.join(OUT, f".diagval_{BENCH}.jsonlines")
    with open(tmp, "w") as f:
        for s in kept.index:
            f.write(json.dumps({"subject_id": str(s),
                                "responses": {it: int(kept.loc[s, it]) for it in items}}) + "\n")
    tr = IrtModelTrainer(config=IrtConfig(model_type=model_type, epochs=1500), data_path=tmp)
    tr.train(device="cpu")
    bp = tr.best_params
    ix = {v: k for k, v in bp["item_ids"].items()}
    disc = np.asarray(bp["disc"], float)
    diff = np.asarray(bp["diff"], float)
    lam = np.asarray(bp["lambdas"], float) if model_type == "3pl" else np.zeros_like(diff)
    os.unlink(tmp)
    a = np.array([disc[ix[it]] for it in items])
    b = np.array([diff[ix[it]] for it in items])
    c = np.array([lam[ix[it]] for it in items])
    return items, a, b, c


def main():
    mat = load_matrix(BENCH)
    models = list(mat.index)
    rng = np.random.default_rng(SEED)
    order = [models[i] for i in rng.permutation(len(models))]
    test = sorted(order[:N_TEST])
    train = order[N_TEST:N_TEST + N_TRAIN]
    actual_full = mat.mean(axis=1).to_dict()

    kept, _ = filter_items(mat.loc[train], benchmark=BENCH)
    items, a, b, c = fit_pyirt(kept, MODEL)
    print(f"{BENCH} [{MODEL}]: train={len(train)} test={len(test)} bank_items={len(items)} "
          f"SE_stop={SE_STOP}  median_c={np.median(c):.3f}", flush=True)

    recs = []
    for m in test:
        resp = mat.loc[m, items].to_numpy(float)
        theta, se, n_adm = run_cat(resp, a, b, c, SE_STOP, max_items=len(items))
        pred = float(prob(theta, a, b, c).mean())
        recs.append((m, float(actual_full[m]), pred, theta, se, n_adm))
        print(f"  {m[:34]:34s} actual={actual_full[m]:.3f} diag={pred:.3f} "
              f"theta={theta:+.2f} se={se:.3f} items={n_adm}", flush=True)

    acts = np.array([r[1] for r in recs])
    preds = np.array([r[2] for r in recs])
    r = float(np.corrcoef(preds, acts)[0, 1])
    mae = float(np.mean(np.abs(preds - acts)))
    avg_items = float(np.mean([r[5] for r in recs]))
    print(f"\ncorr={r:.3f}  MAE={mae:.3f}  avg_items_administered={avg_items:.0f}", flush=True)

    os.makedirs(OUT_RES, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    csv_path = os.path.join(OUT_RES, f"diag_validation_{TAG}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "actual_full", "diagnostic_pred", "theta", "se", "n_items"])
        w.writerows(recs)

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(EXP, ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 7))
    lo = min(acts.min(), preds.min()) - 0.03
    hi = max(acts.max(), preds.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect (y=x)")
    ax.scatter(acts, preds, s=70, color="#1f77b4", edgecolor="black", zorder=3)
    for m, act, pred, *_ in recs:
        ax.annotate(m.split("/")[-1][:16], (act, pred), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(f"actual full {BENCH} score (all items)")
    ax.set_ylabel(f"diagnostic-predicted score ({MODEL.upper()}, SE<{SE_STOP} CAT)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(f"{BENCH} [{MODEL.upper()} calibration]: diagnostic vs full score, "
                 f"{len(test)} held-out models\n"
                 f"Pearson r={r:.3f}  MAE={mae:.3f}  avg {avg_items:.0f} items/diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    png = os.path.join(OUT_FIG, f"diag_validation_{TAG}.png")
    fig.savefig(png, dpi=130)
    print(f"saved {csv_path}\nsaved {png}", flush=True)


if __name__ == "__main__":
    main()
