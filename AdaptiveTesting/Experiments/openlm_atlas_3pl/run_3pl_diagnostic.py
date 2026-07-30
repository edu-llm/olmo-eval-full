#!/usr/bin/env python3
"""ATLAS unidimensional 3PL k-subset calibration + held-out CAT/p-IRT for OpenLM.

For a benchmark under AdaptiveTesting/Inputs/OpenLM/<bench>/:
  1. Build response matrix, 90/10 model split
  2. Fit mirt 3PL on ~100-item chunks (capped EM)
  3. Mean–σ link with polarity flip when cor < 0
  4. Fisher CAT + p-IRT on held-out 10%; scatter vs full accuracy

Does not modify existing GPQA outputs under openlm_gpqa_atlas_3pl/.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
OPENLM = REPO / "AdaptiveTesting/Inputs/OpenLM"
ATLAS = REPO / "AdaptiveTesting/Inputs/ATLAS"

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()

FIT_R = EXP / "fit_chunk_3pl.r"
LINK_R = EXP / "link_chunks_polarity.r"


def write_helper_scripts() -> None:
    FIT_R.write_text(
        r"""
library(mirt)
args <- commandArgs(trailingOnly = TRUE)
kv <- list()
for (a in args) {
  if (grepl("^--", a)) {
    parts <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
    kv[[parts[1]]] <- parts[2]
  }
}
data_file <- kv$data_file
chunk_end <- as.integer(kv$chunk_end)
chunk_ends <- as.integer(strsplit(kv$chunk_ends, ",")[[1]])
outdir <- kv$outdir
ncycles <- as.integer(if (!is.null(kv$ncycles)) kv$ncycles else "500")

data <- read.csv(data_file)
if ("avg_score" %in% colnames(data)) data$avg_score <- NULL
data_clean <- na.omit(data)
data <- data_clean[, colSums(is.na(data_clean)) == 0]
constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data <- data[, !constant_cols]
constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data <- clean_data[!constant_rows, ]

chunk_idx <- which(chunk_ends == chunk_end)
start_col <- if (chunk_idx == 1L) 2L else chunk_ends[chunk_idx - 1L] + 1L
dat <- clean_data[, start_col:chunk_end]
cat("Fitting 3PL chunk", chunk_end, "items", ncol(dat), "ncycles", ncycles, "\n")
model <- mirt(dat, 1, itemtype = "3PL", method = "EM",
              technical = list(NCYCLES = ncycles))
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE,
                        full.scores.SE = TRUE, quadpts = 61)
item_params <- coef(model, simplify = TRUE)$items
m2 <- tryCatch(M2(model), error = function(e) data.frame(error = as.character(e)))
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
tag <- as.character(chunk_end)
write.csv(theta_scores, file.path(outdir, paste0("irt_person_scores_", tag, ".csv")), row.names = FALSE)
write.csv(item_params, file.path(outdir, paste0("irt_item_parameters_", tag, ".csv")), row.names = TRUE)
write.csv(m2, file.path(outdir, paste0("m2_", tag, ".csv")), row.names = TRUE)
cat("Saved 3PL chunk", tag, "\n")
"""
    )
    LINK_R.write_text(
        r"""
args <- commandArgs(trailingOnly = TRUE)
kv <- list()
for (a in args) {
  if (grepl("^--", a)) {
    parts <- strsplit(sub("^--", "", a), "=", fixed = TRUE)[[1]]
    kv[[parts[1]]] <- parts[2]
  }
}
outdir <- kv$outdir
chunk_indices <- as.integer(strsplit(kv$chunk_ends, ",")[[1]])
scores_list <- lapply(chunk_indices, function(i)
  read.csv(file.path(outdir, paste0("irt_person_scores_", i, ".csv"))))
reference_scores <- scores_list[[1]]
linked <- list(read.csv(file.path(outdir, paste0("irt_item_parameters_", chunk_indices[1], ".csv"))))

for (j in 2:length(scores_list)) {
  i <- chunk_indices[j]
  params_j <- read.csv(file.path(outdir, paste0("irt_item_parameters_", i, ".csv")))
  sc_ref <- as.numeric(reference_scores$F1)
  sc_j <- as.numeric(scores_list[[j]]$F1)
  A <- sd(sc_ref, na.rm = TRUE) / sd(sc_j, na.rm = TRUE)
  B <- mean(sc_ref, na.rm = TRUE) - A * mean(sc_j, na.rm = TRUE)
  r <- cor(sc_ref, A * sc_j + B, use = "pairwise.complete.obs")
  if (!is.na(r) && r < 0) {
    A <- -A
    B <- mean(sc_ref, na.rm = TRUE) - A * mean(sc_j, na.rm = TRUE)
    r2 <- cor(sc_ref, A * sc_j + B, use = "pairwise.complete.obs")
    cat("Chunk", i, "FLIPPED A=", A, "B=", B, "cor", r, "->", r2, "\n")
  } else {
    cat("Chunk", i, "A=", A, "B=", B, "cor", r, "\n")
  }
  a_orig <- params_j$a1
  params_j$a1 <- a_orig / A
  params_j$d <- A * params_j$d + B * a_orig
  if ("g" %in% colnames(params_j)) params_j$g <- params_j$g
  linked[[j]] <- params_j
}
combined <- do.call(rbind, linked)
out_csv <- file.path(outdir, "irt_item_parameters_combined.csv")
write.csv(combined, out_csv, row.names = FALSE)
cat("Saved", nrow(combined), "linked items to", out_csv, "\n")
"""
    )


def load_long(bench: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((OPENLM / bench).glob("*.csv")):
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                sub = r.get("subtask") or bench
                rows.append(
                    {
                        "model": r["model"],
                        "item": f"{sub}|{r['question_id']}",
                        "correct": 1 if r["result"].strip().lower() == "correct" else 0,
                    }
                )
    return pd.DataFrame(rows)


def prepare(bench: str, out: Path, seed: int, test_frac: float, chunk_size: int) -> list[int]:
    print(f"[{bench}] loading CSVs…", flush=True)
    mat = load_long(bench).pivot_table(
        index="model", columns="item", values="correct", aggfunc="max"
    )
    mat = mat.dropna(axis=0, how="any")
    mat = mat.loc[:, mat.nunique() > 1]
    mat = mat.loc[mat.nunique(axis=1) > 1].astype(int)
    items = sorted(mat.columns)
    models = list(mat.index)
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = max(1, int(round(len(order) * test_frac)))
    test_models, train_models = sorted(order[:n_test]), sorted(order[n_test:])
    train = mat.loc[train_models, items]
    train = train.loc[:, train.nunique() > 1]
    train = train.loc[train.nunique(axis=1) > 1]
    kept = list(train.columns)
    test = mat.loc[test_models, kept]
    print(
        f"[{bench}] train={len(train)} test={len(test)} items={len(kept)}",
        flush=True,
    )

    def write_atlas(m: pd.DataFrame, path: Path) -> None:
        df = m.copy()
        df.columns = list(range(1, len(kept) + 1))
        df.insert(0, "", df.index)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    data = out / "data"
    write_atlas(train, data / "response_matrix_train.csv")
    write_atlas(test, data / "response_matrix_test.csv")
    with (data / "item_id_map.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["atlas_idx", "item_id", "subtask", "question_id"])
        for i, item in enumerate(kept, 1):
            sub, qid = item.split("|", 1)
            w.writerow([i, item, sub, qid])
    with (data / "split_models.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split"])
        for m in train.index:
            w.writerow([m, "train"])
        for m in test_models:
            w.writerow([m, "test"])

    n_cols = 1 + len(kept)
    ends = list(range(1 + chunk_size, n_cols, chunk_size))
    if not ends or ends[-1] != n_cols:
        ends.append(n_cols)
    ends = [e for e in ends if e >= 2 + 20]
    if ends[-1] != n_cols:
        ends.append(n_cols)
    (data / "chunk_ends.txt").write_text(",".join(map(str, ends)) + "\n")
    print(f"[{bench}] chunks={len(ends)} ends={ends[:6]}{'…' if len(ends)>6 else ''}", flush=True)
    return ends


def fit_and_link(bench: str, out: Path, ends: list[int], ncycles: int, workers: int) -> None:
    train = out / "data" / "response_matrix_train.csv"
    calib = out / "calibration"
    calib.mkdir(parents=True, exist_ok=True)
    log = out / "logs" / "calibration.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    ends_s = ",".join(map(str, ends))

    def fit_one(end: int) -> tuple[int, int, float]:
        dest = calib / f"irt_item_parameters_{end}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            return end, 0, 0.0
        t0 = time.time()
        with log.open("a") as fh:
            r = subprocess.run(
                [
                    "Rscript",
                    str(FIT_R),
                    f"--data_file={train}",
                    f"--chunk_end={end}",
                    f"--chunk_ends={ends_s}",
                    f"--outdir={calib}",
                    f"--ncycles={ncycles}",
                ],
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        return end, r.returncode, time.time() - t0

    todo = [e for e in ends if not (calib / f"irt_item_parameters_{e}.csv").exists()]
    print(f"[{bench}] fitting {len(todo)}/{len(ends)} chunks with {workers} workers…", flush=True)
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fit_one, e) for e in todo]
        for fut in as_completed(futs):
            end, code, dt = fut.result()
            print(f"[{bench}] chunk {end} rc={code} {dt:.0f}s", flush=True)
            if code != 0:
                fail += 1
    if fail:
        raise SystemExit(f"[{bench}] {fail} chunk fits failed — see {log}")

    print(f"[{bench}] linking…", flush=True)
    r = subprocess.run(
        [
            "Rscript",
            str(LINK_R),
            f"--outdir={calib}",
            f"--chunk_ends={ends_s}",
        ],
        capture_output=True,
        text=True,
    )
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"[{bench}] linking failed")


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "model"}).set_index("model")
    df.columns = [int(c) for c in df.columns]
    return df.astype(float)


def load_bank(calib: Path, mat_cols: list[int]):
    by_idx: dict[int, tuple[float, float, float]] = {}
    pdf = pd.read_csv(calib / "irt_item_parameters_combined.csv")
    name_col = pdf.columns[0]
    for _, row in pdf.iterrows():
        key = str(row[name_col]).lstrip("X")
        try:
            idx = int(float(key))
        except ValueError:
            continue
        a = float(row["a1"])
        d = float(row["d"])
        g = float(row["g"]) if "g" in pdf.columns else 0.0
        if not np.isfinite(a) or a == 0:
            continue
        by_idx[idx] = (a, -d / a, float(np.clip(g, 0.0, 0.999)))
    idxs, a, b, c = [], [], [], []
    for idx in mat_cols:
        if idx not in by_idx:
            continue
        aa, bb, cc = by_idx[idx]
        idxs.append(idx)
        a.append(aa)
        b.append(bb)
        c.append(cc)
    return idxs, np.asarray(a), np.asarray(b), np.asarray(c)


def prob(theta, a, b, c):
    z = np.clip(a * (theta - b), -30, 30)
    return np.clip(c + (1 - c) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


def eap_se(resp, a, b, c):
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    m = float((NODES * w).sum())
    sd = float(np.sqrt(((NODES - m) ** 2 * w).sum()))
    return m, sd


def run_cat(resp_all, a, b, c, se_stop, min_items=8, max_items=200):
    n = len(a)
    used = np.zeros(n, dtype=bool)
    theta, se, order = 0.0, 1.0, []
    for _ in range(min(max_items, n)):
        p = prob(theta, a, b, c)
        info = (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.asarray(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        if len(order) >= min_items and se <= se_stop:
            break
    return theta, se, order


def pirt(resp_all, order, theta, a, b, c):
    n = len(a)
    avg_obs = float(resp_all[np.asarray(order)].mean()) if order else 0.0
    unobs = [i for i in range(n) if i not in set(order)]
    avg_pred = float(prob(theta, a[unobs], b[unobs], c[unobs]).mean()) if unobs else avg_obs
    w = len(order) / n
    return w * avg_obs + (1 - w) * avg_pred


def diagnostic(bench: str, out: Path, se_stop: float) -> dict:
    test = load_matrix(out / "data" / "response_matrix_test.csv")
    idxs, a, b, c = load_bank(out / "calibration", list(test.columns))
    print(f"[{bench}] CAT on {len(test)} held-out, bank={len(idxs)}, SE≤{se_stop}", flush=True)
    recs = []
    for mid in test.index:
        resp = test.loc[mid, idxs].to_numpy(float)
        actual = float(resp.mean())
        theta, se, order = run_cat(resp, a, b, c, se_stop)
        pred = pirt(resp, order, theta, a, b, c)
        recs.append((mid, actual, pred, theta, se, len(order)))

    acts = np.asarray([r[1] for r in recs])
    preds = np.asarray([r[2] for r in recs])
    r = float(np.corrcoef(preds, acts)[0, 1])
    mae = float(np.mean(np.abs(preds - acts)))
    rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))
    avg_items = float(np.mean([r[5] for r in recs]))
    print(f"[{bench}] r={r:.3f} MAE={mae:.3f} RMSE={rmse:.3f} avg_items={avg_items:.1f}", flush=True)

    res = out / "results"
    figdir = out / "figures"
    res.mkdir(exist_ok=True)
    figdir.mkdir(exist_ok=True)
    tag = f"{bench}_atlas3pl_se{se_stop:g}"
    csv_path = res / f"{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "actual_full", "pirt_pred", "theta", "se", "n_items"])
        w.writerows(recs)

    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    lo = min(acts.min(), preds.min()) - 0.03
    hi = max(acts.max(), preds.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect (y=x)")
    ax.scatter(acts, preds, s=28, color="#1f77b4", edgecolor="black", linewidths=0.25, zorder=3)
    ax.set_xlabel(f"actual full {bench} accuracy (all items)")
    ax.set_ylabel(f"ATLAS p-IRT diagnostic (3PL CAT, SE≤{se_stop:g})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"OpenLM {bench} — ATLAS 3PL k-subset (90% train) → {len(recs)} held-out (10%)\n"
        f"Pearson r={r:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  avg {avg_items:.0f} items/CAT"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    png = figdir / f"{tag}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[{bench}] saved {csv_path}\n[{bench}] saved {png}", flush=True)
    return {
        "benchmark": bench,
        "n_test": len(recs),
        "n_bank": len(idxs),
        "r": r,
        "mae": mae,
        "rmse": rmse,
        "avg_items": avg_items,
        "csv": str(csv_path),
        "png": str(png),
    }


def run_one(bench: str, args: argparse.Namespace) -> dict:
    out = EXP / bench
    out.mkdir(parents=True, exist_ok=True)
    ends = prepare(bench, out, args.seed, args.test_frac, args.chunk_size)
    fit_and_link(bench, out, ends, args.ncycles, args.workers)
    return diagnostic(bench, out, args.se_stop)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmarks",
        default="ifeval,math,musr,bbh",
        help="Comma-separated OpenLM benches (gpqa already done separately)",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--se-stop", type=float, default=0.3)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip benches that already have a results CSV")
    args = p.parse_args()
    write_helper_scripts()

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    summary = []
    for bench in benches:
        res_dir = EXP / bench / "results"
        existing = list(res_dir.glob(f"{bench}_atlas3pl_se*.csv")) if res_dir.exists() else []
        if args.skip_existing and existing:
            print(f"[{bench}] skip — found {existing[0].name}", flush=True)
            continue
        if not (OPENLM / bench).is_dir():
            print(f"[{bench}] missing OpenLM folder", flush=True)
            continue
        summary.append(run_one(bench, args))

    # Include GPQA prior result if present
    gpqa_csv = (
        REPO
        / "AdaptiveTesting/Experiments/openlm_gpqa_atlas_3pl/results/openlm_gpqa_atlas3pl_se0.3.csv"
    )
    if gpqa_csv.is_file():
        rows = list(csv.DictReader(gpqa_csv.open()))
        acts = np.array([float(r["actual_full"]) for r in rows])
        preds = np.array([float(r["pirt_pred"]) for r in rows])
        summary.append(
            {
                "benchmark": "gpqa",
                "n_test": len(rows),
                "n_bank": "",
                "r": float(np.corrcoef(preds, acts)[0, 1]),
                "mae": float(np.mean(np.abs(preds - acts))),
                "rmse": float(np.sqrt(np.mean((preds - acts) ** 2))),
                "avg_items": float(np.mean([float(r["n_items"]) for r in rows])),
                "csv": str(gpqa_csv),
                "png": str(
                    REPO
                    / "AdaptiveTesting/Experiments/openlm_gpqa_atlas_3pl/figures/openlm_gpqa_atlas3pl_se0.3.png"
                ),
            }
        )

    sum_path = EXP / "results_summary.csv"
    EXP.mkdir(exist_ok=True)
    with sum_path.open("w", newline="", encoding="utf-8") as fh:
        fields = ["benchmark", "n_test", "n_bank", "r", "mae", "rmse", "avg_items", "csv", "png"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in sorted(summary, key=lambda x: x["benchmark"]):
            w.writerow(row)
            print(
                f"SUMMARY {row['benchmark']}: r={row['r']:.3f} MAE={row['mae']:.3f} "
                f"avg_items={row['avg_items']:.1f}",
                flush=True,
            )
    print(f"wrote {sum_path}", flush=True)


if __name__ == "__main__":
    main()
