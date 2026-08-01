"""Tasks 4 + 5: build the curated shortlist and its size-bin coverage.

Tier A = reputable/recognizable lab org, canonical family, no 3a-3g flag,
         passes the ablation/org-spam quality gate.
Tier B = same but from a community org (derivative finetune of a canonical base).

Writes _scratch_atlas/shortlist.csv (both tiers) and prints the report.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 230)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 4000)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "_scratch_atlas")
df = pd.read_csv(os.path.join(OUT, "candidates_flagged.csv"))

W = df[(df.params_b.notna()) & (df.params_b > 0) & (df.params_b <= 7.0)].copy()
surv = W[(W.clean == 1) & (W.has_family == 1) & (W.quality_gate_ok == 1)].copy()
surv["tier"] = np.where(surv.reputable_org == 1, "A_lab", "B_community")
surv = surv.sort_values(["tier", "family", "params_b", "model"])

COLS = ["tier", "family", "model", "params_b", "variant", "average", "org",
        "in_atlas", "in_openlm", "n_pools", "dl_status", "n_bench",
        "nonenglish_strength", "params_source"]
surv[COLS].to_csv(os.path.join(OUT, "shortlist.csv"), index=False)

A = surv[surv.tier == "A_lab"]
B = surv[surv.tier == "B_community"]

print("=" * 100)
print("### TASK 4 - SHORTLIST")
print("=" * 100)
print(f"0-7B window                                  : {len(W)}")
print(f"  minus 3a-3g flags (clean)                  : {int(W.clean.sum())}")
print(f"  minus no-canonical-family                  : {int(((W.clean==1)&(W.has_family==1)).sum())}")
print(f"  minus ablation-grid / org-spam dumps       : {len(surv)}")
print(f"     TIER A  recognizable lab org            : {len(A)}")
print(f"     TIER B  community tune of canonical base: {len(B)}")
print(f"full list written to: {os.path.join(OUT, 'shortlist.csv')}")

print()
print("-" * 100)
print("PER-FAMILY COUNTS")
print("-" * 100)
piv = (
    surv.pivot_table(index="family", columns="tier", values="model", aggfunc="count")
    .fillna(0).astype(int)
)
piv["total"] = piv.sum(axis=1)
piv = piv.sort_values("total", ascending=False)
print(piv.to_string())

print()
print("=" * 100)
print(f"TIER A - EVERY SURVIVING CANDIDATE ({len(A)} models), grouped by family, sorted by params_b")
print("=" * 100)
for fam, g in A.groupby("family", sort=False):
    g = g.sort_values(["params_b", "model"])
    print()
    print(f"--- {fam}  (n={len(g)}) ---")
    for _, r in g.iterrows():
        avg = f"{r['average']:6.2f}" if pd.notna(r["average"]) else "     -"
        soft = " [cjk-vendor]" if r["nonenglish_strength"] == "soft_cjk_vendor" else ""
        pools = ("ATLAS" if r["in_atlas"] else "") + ("+OpenLM" if r["in_openlm"] else "")
        print(f"    {r['model'][:56]:<56} {r['params_b']:>6.2f}  avg={avg}  {r['variant']:<8} {pools:<12}{soft}")

print()
print("=" * 100)
print(f"TIER B - community tunes of canonical bases ({len(B)} models): first 15 per family")
print("=" * 100)
for fam, g in B.groupby("family", sort=False):
    g = g.sort_values(["params_b", "model"])
    print()
    print(f"--- {fam}  (n={len(g)}, showing {min(15, len(g))}) ---")
    for _, r in g.head(15).iterrows():
        avg = f"{r['average']:6.2f}" if pd.notna(r["average"]) else "     -"
        pools = ("ATLAS" if r["in_atlas"] else "") + ("+OpenLM" if r["in_openlm"] else "")
        print(f"    {r['model'][:56]:<56} {r['params_b']:>6.2f}  avg={avg}  {r['variant']:<8} {pools}")

# ----------------------------------------------------------------- TASK 5 bins
print()
print("=" * 100)
print("### TASK 5 - SIZE-BIN COVERAGE OF THE SHORTLIST")
print("=" * 100)
EDGES = [0.15, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0001]
LABELS = ["[0.15-0.5)", "[0.5-1)", "[1-1.5)", "[1.5-2)", "[2-3)", "[3-4)", "[4-5)",
          "[5-6)", "[6-7]"]
for name, d in (("TIER A", A), ("TIER B", B), ("TIER A+B", surv),
                ("whole 0-7B window", W)):
    d = d.copy()
    d["bin"] = pd.cut(d.params_b, bins=EDGES, labels=LABELS, right=False)
    vc = d["bin"].value_counts().reindex(LABELS).fillna(0).astype(int)
    print(f"\n{name}  (n={len(d)})")
    for lab in LABELS:
        bar = "#" * int(vc[lab] / max(1, vc.max()) * 45)
        print(f"   {lab:<12} {vc[lab]:>5}  {bar}")

A2 = A.copy()
A2["bin"] = pd.cut(A2.params_b, bins=EDGES, labels=LABELS, right=False)
vcA = A2["bin"].value_counts().reindex(LABELS).fillna(0).astype(int)
S2 = surv.copy()
S2["bin"] = pd.cut(S2.params_b, bins=EDGES, labels=LABELS, right=False)
vcS = S2["bin"].value_counts().reindex(LABELS).fillna(0).astype(int)
print()
print("underpopulated bins (Tier A < 15 models):")
for lab in LABELS:
    if vcA[lab] < 15:
        print(f"   {lab:<12} TierA={vcA[lab]:<4} TierA+B={vcS[lab]}")
print()
print("Tier A bin table with example models:")
for lab in LABELS:
    ex = A2[A2["bin"] == lab].sort_values("params_b")["model"].head(4).tolist()
    print(f"   {lab:<12} n={vcA[lab]:<4} e.g. {', '.join(m.split('/')[-1] for m in ex)}")
