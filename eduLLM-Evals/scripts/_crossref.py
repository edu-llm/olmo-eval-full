import json, sys, csv, glob, os
sys.stdout.reconfigure(encoding="utf-8")

sets = json.load(open("staging/_changed_sets.json", encoding="utf-8"))
text_modified = set(sets["text_modified"])
split_parents = set(sets["split_parents"])
merged_orphans = set(sets["merged_orphans"])
changed_all = set(sets["changed_all"])
split_children = sets["split_children"]

# ---- Sample (a): Q-matrix human review ----
man = json.load(open("qmatrix_human_review/coordinator_manifest.json", encoding="utf-8"))
sample_a = {c["criterion_id"] for c in man["criteria"]}
sample_a_scen = {c["scenario_id"] for c in man["criteria"]}

# ---- Sample (b): judge-selection human grades ----
rub = [json.loads(l) for l in open("grader_packets/sample_rubrics.jsonl", encoding="utf-8")]
sample_b = {r["criterion_id"] for r in rub}
sample_b_scen = {r["scenario_id"] for r in rub}
# also union criterion ids actually appearing in grader csvs
grader_ids = set()
for f in glob.glob("grader_packets/grader_*.csv"):
    with open(f, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = row.get("criterion_id")
            if cid:
                grader_ids.add(cid)
print("sample_b from rubrics:", len(sample_b), "| grader-csv ids:", len(grader_ids),
      "| union:", len(sample_b | grader_ids), "| in rubrics not csv:", len(sample_b-grader_ids),
      "| in csv not rubrics:", len(grader_ids-sample_b))
sample_b = sample_b | grader_ids
sample_b_scen |= {i.rsplit("_c",1)[0] for i in grader_ids}

print("\nsample_a (Q-matrix review) unique criteria:", len(sample_a))
print("sample_b (judge-selection) unique criteria:", len(sample_b), "scenarios:", len(sample_b_scen))

def breakdown(name, s):
    inter = changed_all & s
    tm = sorted(text_modified & s)
    sp = sorted(split_parents & s)
    mg = sorted(merged_orphans & s)
    print(f"\n=== {name}: changed ∩ sample = {len(inter)} ===")
    print(f"  text-modified ({len(tm)}): {tm}")
    print(f"  split parents ({len(sp)}): {sp}")
    print(f"  merged/presentation orphans ({len(mg)}): {mg}")
    return inter, tm, sp, mg

a_inter, a_tm, a_sp, a_mg = breakdown("Sample A (Q-matrix human review)", sample_a)
b_inter, b_tm, b_sp, b_mg = breakdown("Sample B (judge-selection human grades)", sample_b)

# unchanged-in-sample (no human action)
print("\nSample A changed but NOT (unchanged/renumbered):", sorted(a_inter))
print("Sample B changed but NOT (unchanged/renumbered):", sorted(b_inter))

# Added criteria falling into sample scenarios (candidates to add to human sets)
added = set(sets["added_curated"])
cur = {json.loads(l)["criterion_id"]: json.loads(l) for l in open("data/TutorBench/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")}
added_scen = {cid: cur[cid]["scenario_id"] for cid in added}
a_added = [cid for cid,scen in added_scen.items() if scen in sample_a_scen]
b_added = [cid for cid,scen in added_scen.items() if scen in sample_b_scen]
print(f"\nadded criteria total: {len(added)}")
print(f"  added in Sample-A scenarios: {len(a_added)} -> {sorted(a_added)}")
print(f"  added in Sample-B scenarios: {len(b_added)} -> {sorted(b_added)}")

# split children in sample-B scenarios (new atomic items with no human grade)
b_split_children = []
for parent in b_sp:
    b_split_children += split_children.get(parent, [])
print(f"\nsplit children created under Sample-B parents: {len(b_split_children)} -> {sorted(b_split_children)}")
a_split_children = []
for parent in a_sp:
    a_split_children += split_children.get(parent, [])
print(f"split children created under Sample-A parents: {len(a_split_children)} -> {sorted(a_split_children)}")

# Changed criteria in NEITHER human sample (no human action)
both = sample_a | sample_b
no_human = changed_all - both
print(f"\nCHANGED criteria in NEITHER human sample (NO human action): {len(no_human)} of {len(changed_all)} changed")
print(f"  breakdown: text-mod {len(text_modified-both)}, split {len(split_parents-both)}, merged {len(merged_orphans-both)}")

json.dump({
 "sample_a": sorted(sample_a), "sample_b": sorted(sample_b),
 "a_inter": sorted(a_inter), "b_inter": sorted(b_inter),
 "a_tm": a_tm, "a_sp": a_sp, "a_mg": a_mg,
 "b_tm": b_tm, "b_sp": b_sp, "b_mg": b_mg,
 "a_added": sorted(a_added), "b_added": sorted(b_added),
 "no_human_count": len(no_human),
}, open("staging/_crossref.json","w",encoding="utf-8"), indent=1)
