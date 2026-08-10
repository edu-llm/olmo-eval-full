import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ids = ["biggen__0109","biggen__0128","biggen__0153","biggen__0159","biggen__0185",
       "biggen__0195","biggen__0201","biggen__0222","biggen__0232","biggen__0244"]
r = {json.loads(l)["gold_case_id"]: json.loads(l)
     for l in open("api_judge_pilot/gold/biggen_core/packets.jsonl", encoding="utf-8")}
for i in ids:
    x = r[i]
    labs = {n: p.get("label") for n, p in x["proposals"].items()}
    print(i, "|", x["stratum"], "| opus=", labs.get("opus-5"), "gpt=", labs.get("gpt-5.5"))
    print("   CRIT:", x["criterion"][:200])
