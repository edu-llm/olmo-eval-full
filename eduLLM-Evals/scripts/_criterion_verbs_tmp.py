import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import re, csv
import pyarrow.parquet as pq
from collections import Counter

base = r"C:\Users\Avane\UT-Austin\AlphaIA\LLM-From-Scratch\olmo-eval-full\eduLLM-Evals\data\dr-sci"
O = base + r"\Dr_SCI_open-ended.parquet"

PREFIX = re.compile(r"^\s*(?:Essential|Important|Optional|Pitfall)\s+Criteria:\s*", re.I)
SCAFFOLD = re.compile(
    r"^(?:the\s+(?:response|answer|model|solution|explanation|reply|student|tutor|final answer)\s+"
    r"(?:must|should|shall|needs?\s+to|has\s+to|is\s+expected\s+to|ought\s+to|will|may|can|is|are)?\s*)",
    re.I)
NEG = re.compile(r"^(?:fails?\s+to\s+|does\s+not\s+|do\s+not\s+|doesn'?t\s+|did\s+not\s+|"
                 r"avoids?\s+|neglects?\s+to\s+|omits?\s+(?:to\s+)?|without\s+|refrains?\s+from\s+|no\s+)", re.I)
ADVERB = re.compile(r"^(?:correctly|clearly|explicitly|accurately|properly|appropriately|fully|"
                    r"optionally|ideally|briefly|concisely|successfully|adequately|precisely|"
                    r"explicit|proper|clear|accurate|thoroughly|logically|carefully|only|also|"
                    r"either|both|not|never|always|consistently|effectively|correct|specifically)\s+", re.I)
FIRST_ALPHA = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def lead_verb(desc):
    d = PREFIX.sub("", desc or "")
    for _ in range(3):
        nd = SCAFFOLD.sub("", d)
        nd = NEG.sub("", nd)
        nd = ADVERB.sub("", nd)
        if nd == d:
            break
        d = nd
    m = FIRST_ALPHA.search(d)
    return m.group(0).lower() if m else None


pf = pq.ParquetFile(O)
verb = Counter()
title_head = Counter()
ncrit = 0
for g in range(pf.metadata.num_row_groups):
    for r in pf.read_row_group(g, columns=["reward_model"]).to_pylist():
        for c in (r["reward_model"].get("rubric") or []):
            ncrit += 1
            v = lead_verb(c.get("description"))
            if v:
                verb[v] += 1
            t = c.get("title") or ""
            m = FIRST_ALPHA.search(t)
            if m:
                title_head[m.group(0).lower()] += 1

print("total criteria:", ncrit)
print("unique description lead-verbs:", len(verb))
print("unique title head-words:", len(title_head))

with open(base + r"\rubric_criterion_verbs.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["desc_lead_verb", "count"])
    for k, n in verb.most_common():
        w.writerow([k, n])
with open(base + r"\rubric_title_headwords.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["title_head_word", "count"])
    for k, n in title_head.most_common():
        w.writerow([k, n])

print("\n=== description lead-verbs with count >= 500 ===")
for k, n in verb.most_common():
    if n >= 500:
        print(f"  {k:22} {n}")
tail = sum(1 for k, n in verb.items() if n < 500)
print(f"  (+{tail} rarer verbs in the long tail; full list in rubric_criterion_verbs.csv)")

print("\n=== title head-words with count >= 1000 ===")
for k, n in title_head.most_common():
    if n >= 1000:
        print(f"  {k:22} {n}")
tail2 = sum(1 for k, n in title_head.items() if n < 1000)
print(f"  (+{tail2} rarer title words in the long tail; full list in rubric_title_headwords.csv)")
