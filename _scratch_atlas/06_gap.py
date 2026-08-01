"""Task 6: gap analysis of the shortlist vs the two existing model rosters."""

from __future__ import annotations

import os

import pandas as pd
import yaml

pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 4000)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "_scratch_atlas")

df = pd.read_csv(os.path.join(OUT, "candidates_flagged.csv"))
short = pd.read_csv(os.path.join(OUT, "shortlist.csv"))

# ------------------------------------------------------------------- rosters
def load_roster(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    rows = []
    for m in doc.get("models", []):
        if isinstance(m, str):
            rows.append({"id": m, "params_b": None, "family": None})
        else:
            rows.append({"id": m.get("id"), "params_b": m.get("params_b"),
                         "family": m.get("family")})
    return pd.DataFrame(rows)


r200 = load_roster(os.path.join(ROOT, "AdaptiveTesting", "Inputs", "Models", "models_200.yaml"))
redu = load_roster(os.path.join(ROOT, "eduLLM-Evals", "models.yaml"))
r200["key"] = r200["id"].str.lower()
redu["key"] = redu["id"].str.lower()
have = set(r200["key"]) | set(redu["key"])

print("=" * 100)
print("### TASK 6 - GAP ANALYSIS")
print("=" * 100)
print(f"models_200.yaml (AdaptiveTesting/Inputs/Models) entries : {len(r200)}")
print(f"eduLLM-Evals/models.yaml entries                        : {len(redu)}")
print(f"union of both rosters (case-insensitive unique)         : {len(have)}")
print(f"overlap between the two rosters                         : "
      f"{len(set(r200['key']) & set(redu['key']))}")

# ------------------------------------------------- (a) shortlist NOT in rosters
short["key"] = short["model"].str.lower()
new = short[~short["key"].isin(have)].copy()
already = short[short["key"].isin(have)].copy()

print()
print("-" * 100)
print("(a) SHORTLIST MODELS NOT ALREADY IN EITHER ROSTER  = new additions available")
print("-" * 100)
print(f"shortlist total          : {len(short)}  (Tier A {int((short.tier=='A_lab').sum())} / "
      f"Tier B {int((short.tier=='B_community').sum())})")
print(f"  already in a roster    : {len(already)}  (Tier A {int((already.tier=='A_lab').sum())})")
print(f"  NEW (not in a roster)  : {len(new)}  (Tier A {int((new.tier=='A_lab').sum())} / "
      f"Tier B {int((new.tier=='B_community').sum())})")
newA = new[new.tier == "A_lab"].sort_values(["family", "params_b", "model"])
print()
print(f"TIER A NEW ADDITIONS ({len(newA)}), grouped by family:")
for fam, g in newA.groupby("family", sort=False):
    print(f"\n  --- {fam}  (n={len(g)}) ---")
    for _, r in g.iterrows():
        avg = f"{r['average']:6.2f}" if pd.notna(r["average"]) else "     -"
        print(f"      {r['model'][:58]:<58} {r['params_b']:>6.2f}  avg={avg}  {r['variant']}")

newB = new[new.tier == "B_community"]
print()
print(f"TIER B NEW ADDITIONS: {len(newB)} (per-family counts; full rows in new_additions.csv)")
print(newB.groupby("family").size().sort_values(ascending=False).to_string())
new.drop(columns=["key"]).to_csv(os.path.join(OUT, "new_additions.csv"), index=False)

print()
print("Shortlist models ALREADY in a roster (no new calibration value):")
for _, r in already.sort_values(["tier", "family", "params_b"]).iterrows():
    tag = "200" if r["key"] in set(r200["key"]) else ""
    tag += "/edu" if r["key"] in set(redu["key"]) else ""
    print(f"    [{r['tier']:<11}] {r['model'][:56]:<56} {r['params_b']:>6.2f}  in:{tag}")

# --------------------------------- (b) roster entries with NO external coverage
allkeys = set(df["key"])
print()
print("-" * 100)
print("(b) EXISTING 187-MODEL ROSTER ENTRIES WITH NO ATLAS/OpenLM CALIBRATION DATA")
print("-" * 100)
for label, roster in (("models_200.yaml", r200), ("eduLLM-Evals/models.yaml", redu)):
    covered = roster[roster["key"].isin(allkeys)]
    missing = roster[~roster["key"].isin(allkeys)]
    print()
    print(f"{label}: {len(roster)} entries -> covered by some pool: {len(covered)} "
          f"({100*len(covered)/len(roster):.1f}%), NOT covered: {len(missing)}")
    if len(covered):
        cov = df[df["key"].isin(set(covered["key"]))]
        print(f"    of the covered: in ATLAS {int(cov.in_atlas.sum())}, "
              f"in OpenLM {int(cov.in_openlm.sum())}, "
              f"with OpenLM average score {int(cov['average'].notna().sum())}")
        print("    covered entries:")
        for _, r in covered.sort_values("id").iterrows():
            row = df[df["key"] == r["key"]].iloc[0]
            pools = ("ATLAS" if row["in_atlas"] else "") + ("+OpenLM" if row["in_openlm"] else "")
            avg = f"{row['average']:.2f}" if pd.notna(row["average"]) else "-"
            print(f"        {r['id'][:58]:<58} {pools:<13} avg={avg}")
    print(f"    NOT covered ({len(missing)}) - no external response data exists:")
    miss = missing.copy()
    miss["fam"] = miss["family"].fillna("(unlabelled)")
    for fam, g in miss.groupby("fam", sort=True):
        ids = [x.split("/")[-1] for x in g["id"]]
        head = f"        {fam:<22} n={len(g):<3} "
        line = head
        for i, x in enumerate(ids):
            piece = x + (", " if i < len(ids) - 1 else "")
            if len(line) + len(piece) > 155:
                print(line)
                line = " " * len(head) + piece
            else:
                line += piece
        print(line)

print()
print("-" * 100)
print("(b-summary) COVERAGE OF THE COMBINED ROSTER UNION")
print("-" * 100)
union = pd.DataFrame({"key": sorted(have)})
union["covered"] = union["key"].isin(allkeys)
print(f"union entries                : {len(union)}")
print(f"  covered by some pool       : {int(union.covered.sum())} "
      f"({100*union.covered.mean():.1f}%)")
print(f"  NOT covered                : {int((~union.covered).sum())}")
cov_union = df[df["key"].isin(set(union[union.covered]['key']))]
print(f"  covered AND has an OpenLM average score : {int(cov_union['average'].notna().sum())}")
print(f"  covered but ATLAS-only (no OpenLM score): "
      f"{int(((cov_union.in_atlas == 1) & (cov_union.in_openlm == 0)).sum())}")

miss_all = r200[~r200["key"].isin(allkeys)]
miss_all.to_csv(os.path.join(OUT, "roster200_uncovered.csv"), index=False)
redu[~redu["key"].isin(allkeys)].to_csv(os.path.join(OUT, "rosteredu_uncovered.csv"), index=False)
print()
print(f"wrote {os.path.join(OUT, 'new_additions.csv')}, roster200_uncovered.csv, "
      f"rosteredu_uncovered.csv")
