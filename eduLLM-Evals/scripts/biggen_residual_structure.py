"""Look for specific skill dimensions in BiGGen underneath the general-ability factor.

Raw capability scores correlate at ~0.92 because they are dominated by overall model
quality. This regresses each task on a model-level general-ability estimate and clusters
the residuals, so tasks group by what makes models over- or under-perform *relative to
their own baseline* rather than by overall competence.

Structure found this way is easy to fabricate from noise, so every result is computed
independently under two evaluators (GPT-4-Turbo and Claude) and only agreement between
them is treated as signal. Exploratory; see also scripts/biggen_factor_probe.py.
"""

import os
import re

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from huggingface_hub import HfFileSystem
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

CACHE = "biggen_results_cache.parquet"
COLS = ["capability", "task", "instance_idx", "model_name", "gpt4_04_turbo_score", "claude_score"]
EVALUATORS = ["gpt4_04_turbo_score", "claude_score"]
SIZE_CEILING = 7.0


def load() -> pd.DataFrame:
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    fs = HfFileSystem(token=False)
    files = [
        f
        for f in fs.glob("datasets/prometheus-eval/BiGGen-Bench-Results/**/*.parquet")
        if "llm_as_a_judge" in f and "multilingual" not in f
    ]
    df = ds.dataset(files, format="parquet", filesystem=fs).to_table(columns=COLS).to_pandas()
    df.to_parquet(CACHE, index=False)
    return df


def param_size(name: str) -> float | None:
    m = re.findall(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name.replace("_", "-"))
    if not m:
        return None
    if re.search(r"\d+\s*x\s*\d", name, re.I):  # Mixtral-style MoE: not a <=7B model
        return 99.0
    return max(float(x) for x in m)


def task_matrix(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    d = df.dropna(subset=[score_col]).copy()
    d["tkey"] = d["capability"] + "/" + d["task"]
    m = d.pivot_table(index="model_name", columns="tkey", values=score_col, aggfunc="mean")
    keep = [i for i in m.index if (s := param_size(i)) is not None and s <= SIZE_CEILING]
    return m.loc[keep].dropna(axis=1, how="any")


def residualize(mat: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Regress each task on model general ability; return residuals and variance removed."""
    g = mat.mean(axis=1)
    g = (g - g.mean()) / g.std()
    resid = {}
    for t in mat.columns:
        y = mat[t].values
        slope, intercept = np.polyfit(g.values, y, 1)
        resid[t] = y - (slope * g.values + intercept)
    r = pd.DataFrame(resid, index=mat.index)
    removed = 1 - r.values.var() / mat.sub(mat.mean(axis=0), axis=1).values.var()
    return r, removed


def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices_from(m, k=1)]


def main() -> None:
    df = load()
    mats, resids, corrs = {}, {}, {}
    for ev in EVALUATORS:
        m = task_matrix(df, ev)
        r, removed = residualize(m)
        mats[ev], resids[ev] = m, r
        corrs[ev] = np.corrcoef(r.values, rowvar=False)
        print(f"{ev}: {m.shape[0]} models x {m.shape[1]} tasks | "
              f"general factor removed {100 * removed:.0f}% of task variance")

    tasks = list(mats[EVALUATORS[0]].columns)
    assert tasks == list(mats[EVALUATORS[1]].columns)

    a, b = upper(corrs[EVALUATORS[0]]), upper(corrs[EVALUATORS[1]])
    rep = np.corrcoef(a, b)[0, 1]
    print("\n=== cross-evaluator replication of the residual correlation matrix ===")
    print(f"r between the two {len(tasks)}x{len(tasks)} residual matrices = {rep:.3f}")
    print("(near 0 => residuals are noise; substantial => real specific structure)")

    # Raw-score matrices as a reference point: how much of that agreement is just g?
    raw = [np.corrcoef(mats[e].values, rowvar=False) for e in EVALUATORS]
    print(f"same statistic on RAW (non-residualized) task correlations = "
          f"{np.corrcoef(upper(raw[0]), upper(raw[1]))[0, 1]:.3f}")

    # Cluster the evaluator-averaged residual correlations.
    mean_corr = (corrs[EVALUATORS[0]] + corrs[EVALUATORS[1]]) / 2
    dist = np.clip(1 - mean_corr, 0, 2)
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")

    for k in (2, 3, 4, 5, 6):
        lab = fcluster(link, k, criterion="maxclust")
        # stability: do the two evaluators induce the same partition independently?
        parts = []
        for ev in EVALUATORS:
            d = np.clip(1 - corrs[ev], 0, 2)
            np.fill_diagonal(d, 0.0)
            parts.append(fcluster(linkage(squareform(d, checks=False), "average"), k, "maxclust"))
        print(f"\nk={k}: cluster sizes {np.bincount(lab)[1:].tolist()} | "
              f"cross-evaluator agreement (ARI) = {ari(parts[0], parts[1]):.3f}")

    for k in (3, 4):
        lab = fcluster(link, k, criterion="maxclust")
        print(f"\n=== clusters at k={k} (evaluator-averaged residuals) ===")
        for c in range(1, k + 1):
            members = [t for t, cl in zip(tasks, lab, strict=True) if cl == c]
            caps = pd.Series([t.split("/")[0] for t in members]).value_counts().to_dict()
            print(f"\ncluster {c} ({len(members)} tasks) capability mix: {caps}")
            for t in members:
                print("   ", t)


def ari(x: np.ndarray, y: np.ndarray) -> float:
    """Adjusted Rand index between two partitions."""
    from itertools import combinations

    n = len(x)
    tp_fp = sum(1 for i, j in combinations(range(n), 2) if x[i] == x[j])
    tp_fn = sum(1 for i, j in combinations(range(n), 2) if y[i] == y[j])
    tp = sum(1 for i, j in combinations(range(n), 2) if x[i] == x[j] and y[i] == y[j])
    total = n * (n - 1) / 2
    exp = tp_fp * tp_fn / total
    mx = (tp_fp + tp_fn) / 2
    return (tp - exp) / (mx - exp) if mx != exp else 0.0


if __name__ == "__main__":
    main()
