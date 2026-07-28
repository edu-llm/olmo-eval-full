"""Validate the per-agent tranche op files cover the frozen worklist exactly."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

wl = json.load(open("curation/tranche/worklist_remaining.json", encoding="utf-8"))
slices = {"agent1": (0, 108), "agent2": (108, 216), "agent3": (216, 324), "agent4": (324, 430)}

all_ids = set()
tot_split = tot_keep = 0
for name, (a, b) in slices.items():
    ids = set(wl[a:b])
    ops = json.load(open(f"curation/tranche/{name}.json", encoding="utf-8"))
    got = [o["id"] for o in ops]
    assert set(got) == ids, (name, "missing", ids - set(got), "extra", set(got) - ids)
    assert len(got) == len(set(got)) == (b - a), (name, "count", len(got))
    s = sum(o["op"] == "split" for o in ops)
    k = sum(o["op"] == "keep" for o in ops)
    other = [o["op"] for o in ops if o["op"] not in ("split", "keep")]
    assert not other, (name, "unexpected ops", other)
    tot_split += s
    tot_keep += k
    all_ids |= ids
    print(f"{name} OK: {s} splits, {k} keeps")

assert len(all_ids) == 430, len(all_ids)
print(f"TOTAL: {tot_split} splits, {tot_keep} keeps across {len(all_ids)} ids")
