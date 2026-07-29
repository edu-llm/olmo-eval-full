import json, sys, csv, os
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")

src = [json.loads(l) for l in open("data/TutorBench/rubrics_qmatrix_final.jsonl", encoding="utf-8")]
cur = [json.loads(l) for l in open("data/TutorBench/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")]
src_by_id = {r["criterion_id"]: r for r in src}

# Build maps from original source id -> curated records
split_children = defaultdict(list)     # parent_src_id -> [curated ids]
consolidated_into = defaultdict(list)  # orphan_src_id -> [curated presentation id]
soften_map = {}                        # src_id -> curated id
rescope_map = {}                       # src_id -> curated id
renumbered_map = {}                    # src_id -> curated id (same text)
added = []                             # curated presentation_added ids

for r in cur:
    cu = r.get("curation") or {}
    op = cu.get("op")
    cid = r["criterion_id"]
    if op == "split":
        split_children[cu.get("from")].append(cid)
    elif op == "format_consolidated":
        for o in (cu.get("from") or []):
            consolidated_into[o].append(cid)
    elif op == "soften":
        soften_map[cu.get("original_criterion_id") or cu.get("from")] = cid
    elif op == "rescope_optional":
        rescope_map[cu.get("original_criterion_id") or cu.get("from")] = cid
    elif op == "renumbered":
        renumbered_map[cu.get("original_criterion_id")] = cid
    elif op == "presentation_added":
        added.append(cid)

rows = []  # original_id, curated_ids, scenario, change_type, note
counts = Counter()
changed_original_ids = set()   # text-modified + merged + split (+ removed)
textmod_ids = set()
merged_ids = set()
split_ids = set()
removed_ids = set()

for r in src:
    sid = r["criterion_id"]
    scen = r.get("scenario_id")
    if sid in split_children:
        kids = sorted(split_children[sid])
        rows.append((sid, ";".join(kids), scen, "split", f"1 parent -> {len(kids)} atomic children"))
        counts["split"] += 1
        split_ids.add(sid); changed_original_ids.add(sid)
    elif sid in consolidated_into:
        tgt = sorted(consolidated_into[sid])
        rows.append((sid, ";".join(tgt), scen, "merged", "style/presentation orphan (all-zero Q) consolidated into optional style_surface criterion"))
        counts["merged"] += 1
        merged_ids.add(sid); changed_original_ids.add(sid)
    elif sid in soften_map:
        rows.append((sid, soften_map[sid], scen, "text-modified", "soften: removed absolute-failure/reference-leak/prescribed phrasing (text changed, q_mapping unchanged)"))
        counts["text-modified"] += 1
        textmod_ids.add(sid); changed_original_ids.add(sid)
    elif sid in rescope_map:
        rows.append((sid, rescope_map[sid], scen, "text-modified", "rescope_optional: rewritten as conditional/optional (text changed, q_mapping unchanged)"))
        counts["text-modified"] += 1
        textmod_ids.add(sid); changed_original_ids.add(sid)
    elif sid in renumbered_map:
        cidr = renumbered_map[sid]
        note = "renumbered only (text & q_mapping identical)" if cidr != sid else "renumbered (id unchanged)"
        rows.append((sid, cidr, scen, "renumbered_unchanged", note))
        counts["renumbered_unchanged"] += 1
    else:
        # unchanged (op null) - keeps same id
        rows.append((sid, sid, scen, "unchanged", "no change"))
        counts["unchanged"] += 1

# added rows
cur_by_id = {r["criterion_id"]: r for r in cur}
for cid in added:
    scen = cur_by_id[cid].get("scenario_id")
    rows.append(("", cid, scen, "added", "new optional style_surface presentation criterion (no source, no prior human label)"))
    counts["added"] += 1

os.makedirs("staging", exist_ok=True)
with open("staging/curation_delta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["original_id", "curated_ids", "scenario", "change_type", "note"])
    w.writerows(rows)

print("counts:", dict(counts))
print("total source rows:", len(src), " total csv rows:", len(rows))
print("changed_original_ids (textmod+merged+split):", len(changed_original_ids))
print("  text-modified:", sorted(textmod_ids))
print("  split parents:", len(split_ids))
print("  merged orphans:", len(merged_ids))

# save the changed sets for task 3
json.dump({
  "text_modified": sorted(textmod_ids),
  "split_parents": sorted(split_ids),
  "merged_orphans": sorted(merged_ids),
  "changed_all": sorted(changed_original_ids),
  "added_curated": sorted(added),
  "split_children": {k: sorted(v) for k,v in split_children.items()},
}, open("staging/_changed_sets.json","w",encoding="utf-8"), indent=1)
print("wrote staging/curation_delta.csv and staging/_changed_sets.json")
