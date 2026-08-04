#!/usr/bin/env python3
"""Large-scale calibration-model-count (train-size) sweep for ATLAS-style adaptive
testing on the OpenLM per-question response data.

For one OpenLM benchmark we hold a FIXED held-out test set (the existing seed-7 10%
split, ~110 models) constant and sweep the number of calibration/train models in steps
of 100 (100, 200, ..., up to total_models - n_test). At EACH step we run exactly ONE
calibration (no reps), fitting the item bank with the SAME ATLAS methodology used by
`AdaptiveTesting/Experiments/openlm_atlas_3pl/run_3pl_diagnostic.py`: a k-subset /
chunked 3PL fit in R `mirt` (`fit_chunk_3pl.r`) followed by mean-sigma chunk linking with
polarity flip (`link_chunks_polarity.r`). We then run the ATLAS p-IRT Fisher-information
CAT on the fixed held-out models and record, per SE stopping target (0.3 primary, 0.2 if
requested): mean/median #items administered, % of bank, Pearson r of p-IRT-predicted vs
actual full-benchmark accuracy, and MAE.

Calibration model subsets are NESTED: we fix one seed-7 permutation of the train pool and
take the first n_train models at each step, so larger n_train supersets smaller ones.

Outputs (per benchmark) go under
  AdaptiveTesting/Research/01_MCQ_ATLAS/data/openlm_trainsize/
    <bench>_openlm_trainsize.csv         (n_train x se rows)
    <bench>_openlm_trainsize.png         (dual-axis corr & mean #items vs n_train, SE0.3)
and `--combine` writes openlm_trainsize_summary.csv + openlm_trainsize_corr_combined.png.

Usage:
  export PYTHONPATH=/Users/arhant/Documents/EDLM/olmo-eval-full/eduLLM-Evals
  uv run python openlm_trainsize_sweep.py --bench ifeval --step 100 --workers 6
  uv run python openlm_trainsize_sweep.py --bench bbh --step 200 --workers 8
  uv run python openlm_trainsize_sweep.py --combine
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

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
OPENLM = REPO / "AdaptiveTesting/Inputs/OpenLM"
OUT_ROOT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/openlm_trainsize"

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()
MIN_ITEMS = 8

# ---------------------------------------------------------------------------
# R helper scripts (identical methodology to openlm_atlas_3pl/run_3pl_diagnostic.py)
# ---------------------------------------------------------------------------
FIT_R_TEXT = r"""
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
fit_chunk <- function(genrand, seed) {
  if (genrand) set.seed(seed)
  model <- mirt(dat, 1, itemtype = "3PL", method = "EM",
                technical = list(NCYCLES = ncycles),
                GenRandomPars = genrand, verbose = FALSE)
  theta_scores <- fscores(model, method = "EAP", full.scores = TRUE,
                          full.scores.SE = TRUE, quadpts = 61)
  item_params <- coef(model, simplify = TRUE)$items
  list(theta = theta_scores, items = item_params)
}
# Deterministic default fit first (numerically identical to the plain
# mirt() call whenever the EM converges). Some model subsets drive the 3PL
# EM to a divergent Heywood solution whose EAP scoring (fscores) errors; in
# that case retry from random starting values (fixed seeds -> reproducible)
# to reach a well-behaved solution instead of aborting the whole fold.
res <- tryCatch(fit_chunk(FALSE, 0L), error = function(e) NULL)
if (is.null(res)) {
  for (sd_ in c(1L, 7L, 42L, 123L, 2024L)) {
    cat("Chunk", chunk_end, "default EM unstable; retry GenRandomPars seed", sd_, "\n")
    res <- tryCatch(fit_chunk(TRUE, sd_), error = function(e) NULL)
    if (!is.null(res)) break
  }
}
if (is.null(res)) stop(paste("chunk", chunk_end, "failed after GenRandomPars retries"))
theta_scores <- res$theta
item_params <- res$items
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
tag <- as.character(chunk_end)
write.csv(theta_scores, file.path(outdir, paste0("irt_person_scores_", tag, ".csv")),
          row.names = FALSE)
write.csv(item_params, file.path(outdir, paste0("irt_item_parameters_", tag, ".csv")),
          row.names = TRUE)
cat("Saved 3PL chunk", tag, "\n")
"""

LINK_R_TEXT = r"""
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
linked <- list(read.csv(
  file.path(outdir, paste0("irt_item_parameters_", chunk_indices[1], ".csv"))))

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
  linked[[j]] <- params_j
}
combined <- do.call(rbind, linked)
out_csv <- file.path(outdir, "irt_item_parameters_combined.csv")
write.csv(combined, out_csv, row.names = FALSE)
cat("Saved", nrow(combined), "linked items to", out_csv, "\n")
"""


def write_helper_scripts(work: Path) -> tuple[Path, Path]:
    work.mkdir(parents=True, exist_ok=True)
    fit_r = work / "fit_chunk_3pl.r"
    link_r = work / "link_chunks_polarity.r"
    fit_r.write_text(FIT_R_TEXT)
    link_r.write_text(LINK_R_TEXT)
    return fit_r, link_r


# ---------------------------------------------------------------------------
# Data loading (identical filtering to run_3pl_diagnostic.py)
# ---------------------------------------------------------------------------
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


def build_master(bench: str) -> pd.DataFrame:
    """Wide model x item 0/1 matrix, ATLAS-style cleaning (complete rows, non-constant)."""
    mat = load_long(bench).pivot_table(
        index="model", columns="item", values="correct", aggfunc="max"
    )
    mat = mat.dropna(axis=0, how="any")
    mat = mat.loc[:, mat.nunique() > 1]
    mat = mat.loc[mat.nunique(axis=1) > 1].astype(int)
    return mat


def split_models(mat: pd.DataFrame, seed: int, test_frac: float) -> tuple[list, list]:
    """Fixed seed permutation -> (test_models, train_pool) matching run_3pl_diagnostic."""
    models = list(mat.index)
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = max(1, int(round(len(order) * test_frac)))
    test_models = order[:n_test]
    train_pool = order[n_test:]  # nested subsets take a prefix of this fixed order
    return test_models, train_pool


def write_atlas(m: pd.DataFrame, kept: list[str], path: Path) -> None:
    df = m.loc[:, kept].copy()
    df.columns = list(range(1, len(kept) + 1))
    df.insert(0, "", df.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def chunk_ends(n_items: int, chunk_size: int, min_last: int = 20) -> list[int]:
    """Chunk boundaries (column indices; col 1 = model id). Merge a small trailing
    remainder into the previous chunk so mirt never receives a degenerate 1-2 item chunk."""
    n_cols = 1 + n_items
    ends = list(range(1 + chunk_size, n_cols, chunk_size))
    if not ends:
        return [n_cols]
    if ends[-1] != n_cols:
        if n_cols - ends[-1] < min_last:
            ends[-1] = n_cols  # extend last full chunk to absorb the small tail
        else:
            ends.append(n_cols)
    return ends


def prepare_step(
    mat: pd.DataFrame,
    train_models: list,
    test_models: list,
    step_dir: Path,
    chunk_size: int,
) -> tuple[list[int], int]:
    train = mat.loc[train_models]
    train = train.loc[:, train.nunique() > 1]
    train = train.loc[train.nunique(axis=1) > 1]
    kept = list(train.columns)
    test = mat.loc[test_models, kept]
    data = step_dir / "data"
    write_atlas(train, kept, data / "response_matrix_train.csv")
    write_atlas(test, kept, data / "response_matrix_test.csv")
    ends = chunk_ends(len(kept), chunk_size)
    (data / "chunk_ends.txt").write_text(",".join(map(str, ends)) + "\n")
    return ends, len(kept)


# ---------------------------------------------------------------------------
# Fit + link (R mirt), reused per step
# ---------------------------------------------------------------------------
def fit_and_link(
    step_dir: Path, ends: list[int], ncycles: int, workers: int, fit_r: Path, link_r: Path
) -> None:
    train = step_dir / "data" / "response_matrix_train.csv"
    calib = step_dir / "calibration"
    calib.mkdir(parents=True, exist_ok=True)
    log = step_dir / "calibration.log"
    ends_s = ",".join(map(str, ends))

    def fit_one(end: int):
        dest = calib / f"irt_item_parameters_{end}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            return end, 0
        with log.open("a") as fh:
            r = subprocess.run(
                ["Rscript", str(fit_r),
                 f"--data_file={train}", f"--chunk_end={end}",
                 f"--chunk_ends={ends_s}", f"--outdir={calib}", f"--ncycles={ncycles}"],
                stdout=fh, stderr=subprocess.STDOUT,
            )
        return end, r.returncode

    todo = [e for e in ends if not (calib / f"irt_item_parameters_{e}.csv").exists()]
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(fit_one, e) for e in todo]):
            _, code = fut.result()
            if code != 0:
                fail += 1
    if fail:
        raise RuntimeError(f"{fail} chunk fits failed - see {log}")

    r = subprocess.run(
        ["Rscript", str(link_r), f"--outdir={calib}", f"--chunk_ends={ends_s}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"linking failed: {r.stderr[-500:]}")


# ---------------------------------------------------------------------------
# Bank load + CAT (identical to run_3pl_diagnostic.py)
# ---------------------------------------------------------------------------
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
        # Require positive discrimination (ATLAS convention, matches the validated
        # openlm_gpqa_atlas_3pl/diagnostic_validation.py). Items that link to a<=0 are
        # mis-linked/degenerate and reverse the response curve, poisoning the CAT.
        if not np.isfinite(a) or a <= 0:
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


def pirt_pred(resp_all, order, theta, a, b, c):
    n = len(a)
    subset = set(order)
    avg_obs = float(resp_all[np.asarray(order)].mean()) if order else 0.0
    unobs = [i for i in range(n) if i not in subset]
    avg_pred = float(prob(theta, a[unobs], b[unobs], c[unobs]).mean()) if unobs else avg_obs
    w = len(order) / n
    return w * avg_obs + (1 - w) * avg_pred


def full_cat_traces(resp_all, a, b, c, se_floor=0.15):
    """One Fisher-info CAT to a tight floor; per-step (n_items, se, pirt_pred)."""
    n = len(a)
    used = np.zeros(n, bool)
    theta, order, steps = 0.0, [], []
    for _ in range(n):
        p = prob(theta, a, b, c)
        info = (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.asarray(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        steps.append((len(order), se, pirt_pred(resp_all, order, theta, a, b, c)))
        if se <= se_floor and len(order) >= MIN_ITEMS:
            break
    return steps


def diagnose(step_dir: Path, ses: list[float]):
    test = load_matrix(step_dir / "data" / "response_matrix_test.csv")
    idxs, a, b, c = load_bank(step_dir / "calibration", list(test.columns))
    n_bank = len(idxs)
    se_floor = min(min(ses), 0.15)
    traces, actual = {}, {}
    for mid in test.index:
        resp = test.loc[mid, idxs].to_numpy(float)
        actual[mid] = float(resp.mean())
        traces[mid] = full_cat_traces(resp, a, b, c, se_floor=se_floor)
    out = {}
    for se in ses:
        n_items, preds, acts = [], [], []
        for mid, steps in traces.items():
            chosen = None
            for (ni, s, pr) in steps:
                if ni >= MIN_ITEMS and s <= se:
                    chosen = (ni, pr)
                    break
            if chosen is None:
                chosen = (steps[-1][0], steps[-1][2])
            n_items.append(chosen[0])
            preds.append(chosen[1])
            acts.append(actual[mid])
        preds = np.asarray(preds)
        acts = np.asarray(acts)
        r = float(np.corrcoef(preds, acts)[0, 1]) if preds.std() > 0 else float("nan")
        out[se] = {
            "mean_items": round(float(np.mean(n_items)), 2),
            "median_items": float(np.median(n_items)),
            "pct_items": (round(100 * float(np.mean(n_items)) / n_bank, 2)
                          if n_bank else float("nan")),
            "corr": round(r, 4),
            "mae": round(float(np.mean(np.abs(preds - acts))), 4),
            "n_bank_items": n_bank,
        }
    return out


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def run_sweep(args: argparse.Namespace) -> None:
    bench = args.bench
    work = Path(args.work_root) / bench if args.work_root else OUT_ROOT / "_work" / bench
    fit_r, link_r = write_helper_scripts(work)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[{bench}] building master matrix...", flush=True)
    mat = build_master(bench)
    test_models, train_pool = split_models(mat, args.seed, args.test_frac)
    max_train = len(train_pool) if args.max_train <= 0 else min(args.max_train, len(train_pool))
    print(
        f"[{bench}] models={len(mat)} items={mat.shape[1]} "
        f"n_test={len(test_models)} train_pool={len(train_pool)} max_train={max_train}",
        flush=True,
    )

    steps = list(range(args.step, max_train + 1, args.step))
    if not steps or steps[-1] != max_train:
        steps.append(max_train)
    ses = [float(s) for s in args.se_list.split(",")]
    print(f"[{bench}] n_train steps: {steps}  SE targets: {ses}", flush=True)

    rows = []
    for n_train in steps:
        step_dir = work / f"n{n_train}"
        train_models = train_pool[:n_train]
        t0 = time.time()
        try:
            ends, n_kept = prepare_step(mat, train_models, test_models, step_dir, args.chunk_size)
            fit_and_link(step_dir, ends, args.ncycles, args.workers, fit_r, link_r)
            metrics = diagnose(step_dir, ses)
            calib_s = round(time.time() - t0, 1)
            for se in ses:
                m = metrics[se]
                rows.append({
                    "benchmark": bench, "n_train": n_train, "se_target": se,
                    "mean_items": m["mean_items"], "median_items": m["median_items"],
                    "pct_items": m["pct_items"], "corr": m["corr"], "mae": m["mae"],
                    "n_bank_items": m["n_bank_items"], "n_test": len(test_models),
                    "calib_s": calib_s,
                })
            main = metrics[ses[0]]
            print(
                f"[{bench}] n_train={n_train:<4} bank={main['n_bank_items']:<5} "
                f"SE{ses[0]}: r={main['corr']} items={main['mean_items']} "
                f"MAE={main['mae']} ({calib_s}s)",
                flush=True,
            )
        except Exception as exc:  # keep sweeping past a bad step (e.g. link failure)
            print(f"[{bench}] n_train={n_train} FAILED: {exc}", flush=True)
            for se in ses:
                rows.append({
                    "benchmark": bench, "n_train": n_train, "se_target": se,
                    "mean_items": float("nan"), "median_items": float("nan"),
                    "pct_items": float("nan"), "corr": float("nan"), "mae": float("nan"),
                    "n_bank_items": 0, "n_test": len(test_models),
                    "calib_s": round(time.time() - t0, 1),
                })
        # incremental save so partial progress survives
        _write_bench_csv(bench, rows)

    _write_bench_csv(bench, rows)
    _plot_bench(bench, rows, ses[0])
    print(f"[{bench}] DONE -> {OUT_ROOT / f'{bench}_openlm_trainsize.csv'}", flush=True)


FIELDS = ["benchmark", "n_train", "se_target", "mean_items", "median_items",
          "pct_items", "corr", "mae", "n_bank_items", "n_test", "calib_s"]


def _write_bench_csv(bench: str, rows: list[dict]) -> None:
    path = OUT_ROOT / f"{bench}_openlm_trainsize.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _mpl():
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_ROOT / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_bench(bench: str, rows: list[dict], se: float) -> None:
    sel = [r for r in rows if r["se_target"] == se and np.isfinite(r["corr"])]
    if not sel:
        return
    sel.sort(key=lambda r: r["n_train"])
    x = [r["n_train"] for r in sel]
    corr = [r["corr"] for r in sel]
    items = [r["mean_items"] for r in sel]
    n_bank = sel[-1]["n_bank_items"]
    plt = _mpl()
    fig, ax1 = plt.subplots(figsize=(7.8, 4.9))
    ax1.plot(x, corr, "o-", color="#1f77b4", label="Pearson r")
    ax1.set_xlabel("# calibration (train) models")
    ax1.set_ylabel("Pearson r (p-IRT pred vs actual)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(min(corr) - 0.05, 1.01)
    ax2 = ax1.twinx()
    ax2.plot(x, items, "s--", color="#d62728", label="mean # items")
    ax2.set_ylabel("mean # CAT items administered", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(
        f"OpenLM {bench}: correlation and test length vs train-set size (SE<={se:g})\n"
        f"(bank ~{n_bank} items, {sel[-1]['n_test']} fixed held-out models)"
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    png = OUT_ROOT / f"{bench}_openlm_trainsize.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)


def replot_bench(bench: str, ses: list[float]) -> None:
    """Re-render a per-bench figure from its existing CSV (no fits, numbers untouched)."""
    path = OUT_ROOT / f"{bench}_openlm_trainsize.csv"
    if not path.exists():
        print(f"[{bench}] no CSV at {path}; nothing to replot")
        return
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    se = ses[0] if ses[0] in set(df["se_target"]) else float(df["se_target"].iloc[0])
    _plot_bench(bench, rows, se)
    print(f"[{bench}] replotted {OUT_ROOT / f'{bench}_openlm_trainsize.png'} (SE{se})")


def combine() -> None:
    csvs = sorted(OUT_ROOT.glob("*_openlm_trainsize.csv"))
    if not csvs:
        print("no per-bench CSVs found")
        return
    frames = [pd.read_csv(p) for p in csvs]
    allrows = pd.concat(frames, ignore_index=True)
    summ = OUT_ROOT / "openlm_trainsize_summary.csv"
    allrows.to_csv(summ, index=False)
    print(f"wrote {summ} ({len(allrows)} rows)")

    uniq = sorted(allrows["se_target"].unique())
    primary = 0.3 if 0.3 in uniq else uniq[-1]  # SE<=0.3 is the primary reporting target
    df = allrows[(allrows["se_target"] == primary) & allrows["corr"].notna()]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = plt.cm.tab10.colors
    for i, (bench, g) in enumerate(df.groupby("benchmark")):
        g = g.sort_values("n_train")
        ax.plot(g["n_train"], g["corr"], "o-", color=colors[i % 10], label=bench)
    ax.set_xlabel("# calibration (train) models")
    ax.set_ylabel("Pearson r (p-IRT pred vs actual)")
    ax.set_title(
        f"OpenLM: correlation vs number of calibration models (SE<={primary:g})")
    ax.grid(True, alpha=0.3)
    ax.legend(title="benchmark")
    fig.tight_layout()
    png = OUT_ROOT / "openlm_trainsize_corr_combined.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"wrote {png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", help="single OpenLM benchmark (ifeval/gpqa/math/bbh/musr)")
    p.add_argument("--combine", action="store_true", help="combine per-bench CSVs + figure")
    p.add_argument("--replot", action="store_true",
                   help="re-render the per-bench figure from its existing CSV (no fits)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--max-train", type=int, default=0, help="cap n_train (0 = all available)")
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--ncycles", type=int, default=500)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--se-list", default="0.3,0.2")
    p.add_argument("--work-root", default="")
    args = p.parse_args()

    if args.combine:
        combine()
        return
    if not args.bench:
        p.error("--bench required (or use --combine)")
    if args.replot:
        replot_bench(args.bench, [float(s) for s in args.se_list.split(",")])
        return
    run_sweep(args)


if __name__ == "__main__":
    main()
