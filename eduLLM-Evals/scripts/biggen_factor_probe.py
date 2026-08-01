"""Probe the latent structure of BiGGen using the authors' released judgments.

Reads only the score/label columns of prometheus-eval/BiGGen-Bench-Results
(llm_as_a_judge split), builds a models x instances score matrix, and reports
capability-level collinearity plus eigenvalues for candidate skill groupings.
Exploratory: not part of the ingest pipeline.
"""

import os
import re

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from huggingface_hub import HfFileSystem

CACHE = "biggen_results_cache.parquet"

COLS = ["capability", "task", "instance_idx", "model_name", "gpt4_04_turbo_score", "claude_score"]

GROUPS4 = {
    "input_fidelity": {"grounding", "instruction_following", "refinement"},
    "derivational_correctness": {"reasoning", "planning", "tool_usage"},
    "social_inference": {"theory_of_mind"},
    "normative_restraint": {"safety"},
}


def load() -> pd.DataFrame:
    if os.path.exists(CACHE):
        print(f"using cache {CACHE}")
        return pd.read_parquet(CACHE)
    fs = HfFileSystem(token=False)
    files = [
        f
        for f in fs.glob("datasets/prometheus-eval/BiGGen-Bench-Results/**/*.parquet")
        if "llm_as_a_judge" in f and "multilingual" not in f
    ]
    print(f"parquet shards: {len(files)}")
    for f in files:
        print("  ", f)
    dataset = ds.dataset(files, format="parquet", filesystem=fs)
    df = dataset.to_table(columns=COLS).to_pandas()
    df.to_parquet(CACHE, index=False)
    return df


def pivot(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[score_col]).copy()
    df["item"] = df["capability"] + "/" + df["task"] + "/" + df["instance_idx"].astype(str)
    return df.pivot_table(index="model_name", columns="item", values=score_col, aggfunc="mean")


def cap_of(item: str) -> str:
    return item.split("/")[0]


def report(mat: pd.DataFrame, label: str) -> None:
    caps = sorted({cap_of(c) for c in mat.columns})
    per_cap = pd.DataFrame(
        {c: mat[[x for x in mat.columns if cap_of(x) == c]].mean(axis=1) for c in caps}
    )
    print(f"\n===== {label}: models={mat.shape[0]} items={mat.shape[1]} =====")
    print("\n-- capability score correlation across models (8x8) --")
    corr = per_cap.corr()
    print(corr.round(2).to_string())
    off = corr.values[np.triu_indices_from(corr.values, k=1)]
    print(f"\nmean off-diagonal r = {off.mean():.3f}  min = {off.min():.3f}  "
          f"max = {off.max():.3f}")

    grp = pd.DataFrame(
        {
            g: mat[[x for x in mat.columns if cap_of(x) in caps_in]].mean(axis=1)
            for g, caps_in in GROUPS4.items()
        }
    )
    print("\n-- proposed 4-skill grouping: correlation across models --")
    gcorr = grp.corr()
    print(gcorr.round(3).to_string())
    goff = gcorr.values[np.triu_indices_from(gcorr.values, k=1)]
    print(f"mean off-diagonal r = {goff.mean():.3f}")

    # Dimensionality: eigenvalues of the item correlation matrix vs a parallel-analysis
    # baseline from permuted data (breaks item covariance, keeps marginals).
    z = mat.dropna(axis=1, how="any")
    z = z.loc[:, z.std() > 0]
    R = np.corrcoef(z.values, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    rng = np.random.default_rng(0)
    perm_ev = []
    for _ in range(20):
        p = np.column_stack([rng.permutation(z.values[:, j]) for j in range(z.shape[1])])
        Rp = np.nan_to_num(np.corrcoef(p, rowvar=False), nan=0.0)
        perm_ev.append(np.sort(np.linalg.eigvalsh(Rp))[::-1])
    perm_ev = np.mean(perm_ev, axis=0)
    keep = int(np.sum(ev > perm_ev))
    tot = ev.sum()
    print(f"\n-- dimensionality on {z.shape[1]} complete items --")
    print("observed eigenvalues :", np.round(ev[:8], 1))
    print("parallel-analysis    :", np.round(perm_ev[:8], 1))
    print(f"factors retained (parallel analysis): {keep}")
    print(f"variance explained by factor 1: {100 * ev[0] / tot:.1f}%   factors 1-4: "
          f"{100 * ev[:4].sum() / tot:.1f}%")


def param_size(name: str) -> float | None:
    """Parameter count in billions parsed from an HF model id, else None."""
    m = re.findall(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name.replace("_", "-"))
    if not m:
        return None
    # names like "Mixtral-8x7B" -> use the expert size, not the multiplier
    mo = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]", name, re.I)
    if mo:
        return float(mo.group(2)) * float(mo.group(1))
    return max(float(x) for x in m)


def size_band_report(mat: pd.DataFrame, ceiling: float = 7.0) -> None:
    """Correlation structure restricted to models in our deployment size range."""
    caps = sorted({cap_of(c) for c in mat.columns})
    sizes = {m: param_size(m) for m in mat.index}
    unknown = [m for m, s in sizes.items() if s is None]
    small = sorted([m for m, s in sizes.items() if s is not None and s <= ceiling],
                   key=lambda m: sizes[m])
    print(f"\n===== parameter-size band: <= {ceiling}B =====")
    print(f"models with unparseable size ({len(unknown)}): {unknown}")
    print(f"\nmodels <= {ceiling}B ({len(small)}):")
    for m in small:
        print(f"   {sizes[m]:5.1f}B  {m}")
    if len(small) < 8:
        print("too few models for a stable correlation matrix")
        return

    sub = mat.loc[small]
    per_cap = pd.DataFrame(
        {c: sub[[x for x in sub.columns if cap_of(x) == c]].mean(axis=1) for c in caps}
    )
    print(f"\nmean score in band: min {per_cap.mean(axis=1).min():.2f} "
          f"max {per_cap.mean(axis=1).max():.2f} sd {per_cap.mean(axis=1).std():.3f}")
    print("\n-- capability correlation matrix (<=7B models only) --")
    print(per_cap.corr().round(2).to_string())
    c = per_cap.corr().values
    off = c[np.triu_indices_from(c, k=1)]
    print(f"mean off-diagonal r = {off.mean():.3f}  min = {off.min():.3f}  max = {off.max():.3f}")

    grp = pd.DataFrame(
        {
            g: sub[[x for x in sub.columns if cap_of(x) in cin]].mean(axis=1)
            for g, cin in GROUPS4.items()
        }
    )
    print("\n-- proposed 4-skill correlation matrix (<=7B models only) --")
    print(grp.corr().round(3).to_string())
    gc = grp.corr().values
    goff = gc[np.triu_indices_from(gc, k=1)]
    print(f"mean off-diagonal r = {goff.mean():.3f}  min = {goff.min():.3f}")

    cc = per_cap.corr()
    pairs = [
        (cc.index[i], cc.columns[j], cc.values[i, j])
        for i in range(len(cc))
        for j in range(i + 1, len(cc))
    ]
    pairs.sort(key=lambda t: t[2])
    print("\n-- most separable capability pairs in band --")
    for a, b, v in pairs[:6]:
        print(f"   {a:22s} vs {b:22s} r={v:.3f}")


def task_structure(mat: pd.DataFrame, ceiling: float = 7.0) -> None:
    """Do tasks cluster by capability, or does something cut across it?

    If a task's nearest neighbours (by cross-model score correlation) mostly share its
    capability, a capability-aligned axis is the right one. If they cut across, the
    latent structure is organised by something other than capability -- which is what a
    demand-based axis would predict.
    """
    sizes = {m: param_size(m) for m in mat.index}
    small = [m for m, s in sizes.items() if s is not None and s <= ceiling]
    sub = mat.loc[small]
    tasks = sorted({"/".join(c.split("/")[:2]) for c in sub.columns})
    tm = pd.DataFrame(
        {t: sub[[c for c in sub.columns if c.startswith(t + "/")]].mean(axis=1) for t in tasks}
    )
    corr = tm.corr()
    print(f"\n===== task-level structure (<= {ceiling}B, {len(tasks)} tasks) =====")

    same, total, cross = 0, 0, []
    for t in tasks:
        cap = t.split("/")[0]
        nn = corr[t].drop(t).sort_values(ascending=False).head(3)
        hits = sum(1 for o in nn.index if o.split("/")[0] == cap)
        same += hits
        total += 3
        if hits == 0:
            cross.append((t, list(nn.index), nn.values))
    print(f"nearest-neighbour purity: {same}/{total} = {100 * same / total:.0f}% of each task's "
          f"top-3 most-correlated tasks share its capability")
    base = sum(
        3 * (len([x for x in tasks if x.split("/")[0] == t.split("/")[0]]) - 1) / (len(tasks) - 1)
        for t in tasks
    )
    print(f"chance baseline if structure were unrelated to capability: "
          f"{100 * base / total:.0f}%")
    print(f"\ntasks whose top-3 neighbours are ALL from other capabilities ({len(cross)}):")
    for t, nn, vals in cross[:14]:
        neighbours = ", ".join(f"{a} ({v:.2f})" for a, v in zip(nn, vals, strict=True))
        print(f"   {t:42s} -> {neighbours}")


def band_report(mat: pd.DataFrame) -> None:
    """Restriction-of-range check: does a narrower ability band separate better?"""
    caps = sorted({cap_of(c) for c in mat.columns})
    overall = mat.mean(axis=1).sort_values()
    print("\n===== restriction-of-range check (gpt4_04_turbo) =====")
    print(f"model mean score: min {overall.min():.2f} median {overall.median():.2f} "
          f"max {overall.max():.2f}")
    bands = {
        "all 99 models": overall.index,
        "top 50 by mean": overall.index[-50:],
        "top 30 by mean": overall.index[-30:],
        "middle 50": overall.index[25:75],
    }
    for name, idx in bands.items():
        sub = mat.loc[idx]
        per_cap = pd.DataFrame(
            {c: sub[[x for x in sub.columns if cap_of(x) == c]].mean(axis=1) for c in caps}
        )
        corr = per_cap.corr().values
        off = corr[np.triu_indices_from(corr, k=1)]
        grp = pd.DataFrame(
            {
                g: sub[[x for x in sub.columns if cap_of(x) in cin]].mean(axis=1)
                for g, cin in GROUPS4.items()
            }
        )
        gcorr = grp.corr().values
        goff = gcorr[np.triu_indices_from(gcorr, k=1)]
        spread = per_cap.mean(axis=1)
        print(f"  {name:16s} n={len(idx):3d} score sd={spread.std():.3f} | "
              f"8-cap mean r={off.mean():.3f} (min {off.min():.3f}) | "
              f"4-skill mean r={goff.mean():.3f} (min {goff.min():.3f})")
        if name == "top 50 by mean":
            print("\n  -- top-50 band: capability correlation matrix --")
            print(per_cap.corr().round(2).to_string())
            print("\n  -- top-50 band: 4-skill correlation matrix --")
            print(grp.corr().round(3).to_string())
            c = per_cap.corr()
            pairs = [
                (c.index[i], c.columns[j], c.values[i, j])
                for i in range(len(c))
                for j in range(i + 1, len(c))
            ]
            pairs.sort(key=lambda t: t[2])
            print("\n  -- most separable capability pairs (lowest r) --")
            for a, b, v in pairs[:6]:
                print(f"     {a:22s} vs {b:22s} r={v:.3f}")
            print()


def main() -> None:
    df = load()
    print("\nrows:", len(df), "| models:", df["model_name"].nunique())
    print("capabilities:", sorted(df["capability"].unique()))
    m = pivot(df, "gpt4_04_turbo_score")
    size_band_report(m, ceiling=7.0)
    task_structure(m, ceiling=7.0)


if __name__ == "__main__":
    main()
