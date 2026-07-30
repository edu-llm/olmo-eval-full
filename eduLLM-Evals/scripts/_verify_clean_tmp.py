import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pyarrow.parquet as pq, re
base = r"C:\Users\Avane\UT-Austin\AlphaIA\LLM-From-Scratch\olmo-eval-full\eduLLM-Evals\data\dr-sci"
V = base + r"\Dr_SCI_verifiable.parquet"
O = base + r"\Dr_SCI_open-ended.parquet"
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
BLOOD = re.compile(r"200\s*m(eters)?\b")

pf = pq.ParquetFile(V)
remaining = 0
examples = []
for g in range(pf.metadata.num_row_groups):
    for r in pf.read_row_group(g, columns=["reward_model"]).to_pylist():
        gt = r["reward_model"]["ground_truth"]
        if THOUSANDS.search(gt):
            remaining += 1
            if len(examples) < 8:
                examples.append(gt[:90])
print("VERIFIABLE: GTs still containing a thousands-comma:", remaining)
for e in examples:
    print("   still-comma:", repr(e))

# show previously-multi-value style GTs (comma-separated numbers) as now stored
shown = 0
for g in range(pf.metadata.num_row_groups):
    for r in pf.read_row_group(g, columns=["reward_model"]).to_pylist():
        gt = r["reward_model"]["ground_truth"]
        if re.match(r"^\s*-?\$?\d+(?:\.\d+)?(?:\s*[^\s,]{0,12})?,\s+", gt) and shown < 8:
            print("   multi-value now:", repr(gt[:90]))
            shown += 1
    if shown >= 8:
        break

pf = pq.ParquetFile(O)
blood = 0
for g in range(pf.metadata.num_row_groups):
    for r in pf.read_row_group(g, columns=["extra_info"]).to_pylist():
        q = r["extra_info"]["question"]
        if "blood" in q.lower() and BLOOD.search(q):
            blood += 1
print("OPEN-ENDED: blood-in-metres rows remaining:", blood)
print("final rows -> V:", pq.ParquetFile(V).metadata.num_rows, "O:", pq.ParquetFile(O).metadata.num_rows)
