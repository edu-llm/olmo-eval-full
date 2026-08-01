"""Tasks 2 + 3: org/family breakdown and flag-category counts with examples."""

from __future__ import annotations

import os
import sys

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "_scratch_atlas")
df = pd.read_csv(os.path.join(OUT, "candidates_flagged.csv")).fillna(
    {"family": "", "nonenglish_markers": "", "params_b": float("nan")}
)

W = df[(df["params_b"].notna()) & (df["params_b"] > 0) & (df["params_b"] <= 7.0)].copy()

print("=" * 78)
print(f"SCOPE: 0-7B window = {len(W)} candidates (of {len(df)} total unique)")
print("=" * 78)

# ------------------------------------------------------------------ TASK 2 orgs
print()
print("### TASK 2a - TOP 100 ORGS BY CANDIDATE COUNT (0-7B window)")
orgs = (
    W.groupby("org")
    .agg(
        n=("model", "size"),
        n_clean=("clean", "sum"),
        median_params=("params_b", "median"),
        mean_openlm_avg=("average", "mean"),
        reputable=("reputable_org", "max"),
    )
    .sort_values("n", ascending=False)
)
orgs["mean_openlm_avg"] = orgs["mean_openlm_avg"].round(2)
orgs["median_params"] = orgs["median_params"].round(2)
top100 = orgs.head(100).reset_index()
top100.index = range(1, len(top100) + 1)
print(top100.to_string())
print()
print(f"total distinct orgs in window: {W['org'].nunique()}")
print(f"orgs with exactly 1 candidate: {(orgs['n'] == 1).sum()}")
print(f"share of window from top-10 orgs: {orgs['n'].head(10).sum()} "
      f"({100 * orgs['n'].head(10).sum() / len(W):.1f}%)")
top100.to_csv(os.path.join(OUT, "top_orgs.csv"))

# ---------------------------------------------------------------- TASK 2 family
print()
print("### TASK 2b - CANONICAL FAMILY vs ANONYMOUS COMMUNITY MODEL (0-7B window)")
fam = (
    W[W["has_family"] == 1]
    .groupby("family")
    .agg(
        n=("model", "size"),
        n_reputable_org=("reputable_org", "sum"),
        n_clean=("clean", "sum"),
        median_params=("params_b", "median"),
    )
    .sort_values("n", ascending=False)
)
fam["median_params"] = fam["median_params"].round(2)
print(fam.to_string())
print()
print(f"recognizable canonical family : {int((W.has_family == 1).sum())} "
      f"({100 * (W.has_family == 1).mean():.1f}%)")
print(f"anonymous community model     : {int((W.has_family == 0).sum())} "
      f"({100 * (W.has_family == 0).mean():.1f}%)")
print(f"  of the family-matched, from a reputable/known lab org: "
      f"{int(((W.has_family == 1) & (W.reputable_org == 1)).sum())}")
print(f"  of the family-matched, from a community org (derivative finetune/merge): "
      f"{int(((W.has_family == 1) & (W.reputable_org == 0)).sum())}")
print()
print("20 examples of ANONYMOUS community models (no canonical family):")
for m in W[W["has_family"] == 0]["model"].head(20):
    print("   ", m)

# ------------------------------------------------------------------- TASK 3 flags
CATS = [
    ("f_nonenglish", "3a  Non-English / language-specific", "nonenglish_markers"),
    ("f_merge", "3b  Merges / SLERP / MoE-merge / frankenmodels", "merge_markers"),
    ("f_quant", "3c  Quantized / adapter-only / non-full-weight", "quant_markers"),
    ("f_rp_nsfw", "3d  Roleplay / NSFW / uncensored / character", "rp_nsfw_markers"),
    ("f_code", "3e-i   Domain: code", "code_markers"),
    ("f_math", "3e-ii  Domain: math", "math_markers"),
    ("f_domain_bmf", "3e-iii Domain: biomed / legal / finance", "domain_bmf_markers"),
    ("f_vision", "3e-iv  Domain: vision / multimodal / audio", "vision_markers"),
    ("f_reasoning", "3f  Reasoning / <think> models", "reasoning_markers"),
    ("f_degenerate", "3g  Degenerate / tiny (<0.15B) / debug", "degenerate_markers"),
]

print()
print("=" * 78)
print("### TASK 3 - PROBLEMATIC CATEGORY COUNTS (0-7B window)")
print("=" * 78)
summary = []
for col, label, _ in CATS:
    summary.append({"category": label, "n_in_window": int(W[col].sum()),
                    "pct_of_window": round(100 * W[col].mean(), 1),
                    "n_all_pools": int(df[col].sum())})
print(pd.DataFrame(summary).to_string(index=False))
print()
print(f"candidates with >=1 flag : {int((W.n_flags > 0).sum())} ({100*(W.n_flags>0).mean():.1f}%)")
print(f"candidates with 0 flags  : {int((W.n_flags == 0).sum())} ({100*(W.n_flags==0).mean():.1f}%)")
print()
print("flag-count histogram:")
print(W["n_flags"].value_counts().sort_index().to_string())

print()
print("ADDITIONAL QUALITY GATES (not part of 3a-3g, used for the Task-4 shortlist):")
for col, lab in (("f_ablation", "hyper-parameter / ablation-grid checkpoint dump"),
                 ("f_orgspam", "non-reputable org with >=10 near-identical uploads"),
                 ("f_mirror", "mirror / re-upload of someone else's weights")):
    sub = W[W[col] == 1]
    print(f"  {lab:<52} n={len(sub)}")
    print(f"      e.g. {', '.join(sub['model'].head(6))}")
print(f"  {'clean AND passes all quality gates':<52} "
      f"n={int(((W.clean == 1) & (W.quality_gate_ok == 1)).sum())}")

print()
print("NON-ENGLISH strength split (soft = CJK vendor whose base ckpts are English-benchmark staples):")
print(W["nonenglish_strength"].replace("", "(english-first)").value_counts().to_string())
print("  soft_cjk_vendor members kept in the shortlist:")
print("   ", ", ".join(sorted(W[(W.nonenglish_strength == "soft_cjk_vendor")
                                & (W.clean == 1) & (W.quality_gate_ok == 1)]["model"])[:25]))

n_ex = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for col, label, mcol in CATS:
    sub = W[W[col] == 1]
    print()
    print("-" * 78)
    print(f"{label}   COUNT = {len(sub)}")
    print("-" * 78)
    if col == "f_nonenglish":
        ex = sub["nonenglish_langs"].str.split(";").explode().value_counts()
        print("language breakdown:")
        print(ex.to_string())
        print()
    else:
        mk = sub[mcol].fillna("").str.split(";").explode()
        mk = mk[mk.str.len() > 0].str.split(":").str[0].value_counts()
        print("marker breakdown:", dict(mk))
        print()
    print(f"{min(n_ex, len(sub))} examples (model | params_b | marker):")
    for _, r in sub.head(n_ex).iterrows():
        print(f"   {r['model'][:62]:<62} {r['params_b']:>6.2f}  {str(r[mcol])[:56]}")
