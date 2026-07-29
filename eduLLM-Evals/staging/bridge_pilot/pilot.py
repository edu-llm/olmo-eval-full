"""Bridge criterion-bank pilot: does the rubric actually measure anything?

A dress rehearsal for the real calibration run, sized at ~1.4% of it, whose only
job is to turn three design guesses into measurements:

  Q1 CEILING FROM THE SYSTEM PROMPT. The Bridge system prompt tells every tutor to
     "Identify the specific error", "guid[e] them toward the right approach rather
     than simply giving away the answer" and "Keep a supportive, encouraging tone" --
     which restates D1, P1 and A1 almost verbatim. If every model passes those, they
     carry no information, and D1/P1 are both `critical`.

  Q2 VACUOUS PASS FROM NEGATIVE WORDING. 19 codes are phrased "the response does not
     misuse X" so that silence passes rather than forcing an unfair fail. If nearly
     everything passes them, that fix traded a floor problem for a ceiling one.

  Q3 REDUNDANCY. A1-A4, P1-P4 and M1/M3/M4/M5 may be one construct wearing four hats.
     Per scenario, `strategy` alone draws ~7.9 criteria, all scored from ONE response.

Stages (each is resumable -- every API result is cached to disk, so re-running skips
completed work and only fills gaps):

    python pilot.py sample     stratified 100-scenario subset
    python pilot.py generate   8 tutors answer each scenario
    python pilot.py judge      one judge call per response, all its criteria at once
    python pilot.py analyze    pass rates, correlations, discrimination -> report.md

Everything routes through the TrueFoundry gateway (TFY_API_KEY), which speaks the
OpenAI format for every provider.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENARIOS = ROOT / "data" / "Bridge" / "scenarios.jsonl"
RUBRICS = ROOT / "data" / "Bridge" / "rubrics.jsonl"

SAMPLE = HERE / "sample.json"
RESP_DIR = HERE / "responses"
JUDGE_DIR = HERE / "judgments"
REPORT = HERE / "report.md"

BASE_URL = "https://tfy.promptlens.trilogy.com/api/llm/api/inference/openai"
N_SCENARIOS = 100
SEED = 42
WORKERS = 12

# Deliberately spans weak -> frontier. A pilot of only strong models cannot
# distinguish "this criterion is easy" from "these models are good".
TUTORS = [
    "openai-group/gpt-4.1-nano",
    "openai-group/gpt-5.4-nano",
    "llama3-3-70b",
    "qwen3-32b",
    "openai-group/gpt-4o",
    "mistral-large-3",
    "openai-group/gpt-5.5",
    "claude-group/claude-opus-4-8",
]
JUDGE_MODEL = "openai-group/gpt-5.5"

# Verbatim from tutor_cat/respgen/prompts.py SYSTEM_PROMPTS_BY_BENCHMARK["Bridge"].
# Q1 is only a valid test if the pilot uses the prompt the real run would use.
BRIDGE_SYSTEM_PROMPT = (
    "You are an AI math tutor. The student has just made a mistake in the "
    "conversation. Identify the specific error, then help the student correct it "
    "by guiding them toward the right approach rather than simply giving away the "
    "answer. Keep a supportive, encouraging tone."
)

_ROLE = {"student": "user", "tutor": "assistant"}


def client():
    from openai import OpenAI

    key = os.environ.get("TFY_API_KEY")
    if not key:
        raise SystemExit("TFY_API_KEY is not set")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=120)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def load_bank() -> tuple[list[dict], dict[str, list[dict]]]:
    scen = [json.loads(l) for l in SCENARIOS.open(encoding="utf-8")]
    rub: dict[str, list[dict]] = defaultdict(list)
    for line in RUBRICS.open(encoding="utf-8"):
        r = json.loads(line)
        rub[r["scenario_id"]].append(r)
    return scen, rub


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def stage_sample() -> None:
    """Proportional stratified draw so every module/domain/band is represented.

    A uniform random 100 would under-sample the small strata (imprecise=53,
    data_graphing=27) exactly where per-criterion estimates are already thinnest.
    """
    scen, _ = load_bank()
    rng = random.Random(SEED)

    def key(s: dict) -> tuple:
        return (s["error_module"], s["topic_domain"], s["grade_band"], s["visible_mistake"])

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for s in scen:
        strata[key(s)].append(s)

    # Largest-remainder allocation: guarantees every stratum gets >=1 slot and the
    # totals still sum to exactly N_SCENARIOS.
    total = len(scen)
    quota = {k: N_SCENARIOS * len(v) / total for k, v in strata.items()}
    alloc = {k: max(1, int(q)) for k, q in quota.items()}
    while sum(alloc.values()) > N_SCENARIOS:          # trim the largest first
        k = max(alloc, key=lambda k: (alloc[k], quota[k]))
        if alloc[k] > 1:
            alloc[k] -= 1
        else:
            break
    while sum(alloc.values()) < N_SCENARIOS:          # top up by largest remainder
        k = max(alloc, key=lambda k: quota[k] - alloc[k])
        alloc[k] += 1

    picked: list[dict] = []
    for k, rows in sorted(strata.items(), key=lambda kv: str(kv[0])):
        rows = sorted(rows, key=lambda s: s["scenario_id"])
        picked.extend(rng.sample(rows, min(alloc[k], len(rows))))
    picked = sorted(picked, key=lambda s: s["scenario_id"])[:N_SCENARIOS]

    SAMPLE.write_text(json.dumps([s["scenario_id"] for s in picked], indent=2), encoding="utf-8")
    print(f"sampled {len(picked)} scenarios from {len(strata)} strata -> {SAMPLE.name}")
    for f in ("error_module", "topic_domain", "grade_band", "visible_mistake"):
        got = Counter(s[f] for s in picked)
        print(f"  {f}: " + ", ".join(f"{k}={v}" for k, v in got.most_common()))


def sampled_scenarios() -> list[dict]:
    ids = set(json.loads(SAMPLE.read_text(encoding="utf-8")))
    scen, _ = load_bank()
    return [s for s in scen if s["scenario_id"] in ids]


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


# 476 of 642 Bridge scenarios (74%) open on a TUTOR turn, which maps to `assistant`
# and yields [system, assistant, ...]. AWS Bedrock rejects that outright ("A
# conversation must start with a user message"), so llama3-3-70b failed 71/100 before
# this adapter existed. tutor_cat/respgen/prompts.py has the same gap -- _coalesce()
# only merges ADJACENT same-role turns and never repairs a leading assistant turn --
# so the production run will hit this on every Bedrock-routed model.
# A synthetic opener is used rather than dropping the turns, which would silently
# discard real tutoring context.
_SESSION_OPENER = "(The tutoring session begins.)"


def messages_for(s: dict) -> list[dict]:
    msgs = [{"role": "system", "content": BRIDGE_SYSTEM_PROMPT}]
    for t in s["conversation_context"]:
        msgs.append({"role": _ROLE.get(t.get("role", "student"), "user"),
                     "content": t.get("content", "")})
    msgs.append({"role": "user", "content": s["prompt"]})
    if len(msgs) > 1 and msgs[1]["role"] == "assistant":
        msgs.insert(1, {"role": "user", "content": _SESSION_OPENER})
    return msgs


def stage_generate() -> None:
    cli = client()
    scen = sampled_scenarios()
    jobs = []
    for model in TUTORS:
        d = RESP_DIR / slug(model)
        d.mkdir(parents=True, exist_ok=True)
        for s in scen:
            p = d / f"{s['scenario_id']}.json"
            if not p.exists():
                jobs.append((model, s, p))
    print(f"generate: {len(jobs)} calls pending "
          f"({len(TUTORS)} models x {len(scen)} scenarios, cached ones skipped)")
    if not jobs:
        return

    def run(job):
        model, s, p = job
        try:
            r = cli.chat.completions.create(
                model=model, messages=messages_for(s), max_tokens=900, temperature=0.0)
            text = (r.choices[0].message.content or "").strip()
        except Exception as e:                       # some models reject temperature
            try:
                r = cli.chat.completions.create(
                    model=model, messages=messages_for(s), max_tokens=900)
                text = (r.choices[0].message.content or "").strip()
            except Exception as e2:
                p.write_text(json.dumps({"model": model, "scenario_id": s["scenario_id"],
                                         "response": "", "error": f"{type(e2).__name__}: {e2}"[:300]}),
                             encoding="utf-8")
                return 0
        p.write_text(json.dumps({"model": model, "scenario_id": s["scenario_id"],
                                 "response": text}), encoding="utf-8")
        return 1 if text else 0

    with ThreadPoolExecutor(WORKERS) as ex:
        done = list(ex.map(run, jobs))
    print(f"  wrote {len(done)} ({sum(done)} non-empty)")


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You grade an AI math tutor's reply against a checklist. For each numbered "
    "criterion return pass or fail. Judge ONLY the tutor reply shown, against the "
    "criterion as literally written. A criterion phrased as 'the response does not "
    "do X' PASSES when the response simply never does X. Do not reward or penalise "
    "length or style beyond what a criterion asks. Never answer N/A -- every "
    "criterion gets pass or fail. Return ONLY a JSON object of the form "
    '{"verdicts": [{"n": 1, "v": "pass"}, ...]} with one entry per criterion, no prose.'
)


def judge_prompt(s: dict, crits: list[dict], response: str) -> str:
    ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in s["conversation_context"])
    lines = "\n".join(f"{i}. {c['criterion']}" for i, c in enumerate(crits, 1))
    return (
        f"<conversation_so_far>\n{ctx}\n</conversation_so_far>\n\n"
        f"<student_final_turn>\n{s['prompt']}\n</student_final_turn>\n\n"
        f"<expert_reference_reply>\n{s['reference_solution']}\n</expert_reference_reply>\n\n"
        f"<tutor_reply_to_grade>\n{response}\n</tutor_reply_to_grade>\n\n"
        f"<checklist>\n{lines}\n</checklist>\n\n"
        f"Return {len(crits)} verdicts."
    )


def parse_verdicts(text: str, n: int) -> list[str] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = ["fail"] * n
    seen = 0
    for v in obj.get("verdicts", []):
        try:
            i = int(v["n"]) - 1
        except Exception:
            continue
        if 0 <= i < n:
            out[i] = "pass" if str(v.get("v", "")).lower().startswith("p") else "fail"
            seen += 1
    return out if seen == n else None


def stage_judge() -> None:
    cli = client()
    scen = {s["scenario_id"]: s for s in sampled_scenarios()}
    _, rub = load_bank()
    jobs = []
    for model in TUTORS:
        rd, jd = RESP_DIR / slug(model), JUDGE_DIR / slug(model)
        jd.mkdir(parents=True, exist_ok=True)
        for sid in scen:
            rp, jp = rd / f"{sid}.json", jd / f"{sid}.json"
            if jp.exists() or not rp.exists():
                continue
            resp = json.loads(rp.read_text(encoding="utf-8")).get("response", "")
            if resp:
                jobs.append((model, sid, resp, jp))
    print(f"judge: {len(jobs)} calls pending")
    if not jobs:
        return

    def run(job):
        model, sid, resp, jp = job
        s = scen[sid]
        crits = sorted(rub[sid], key=lambda c: c["criterion_id"])
        for attempt in range(3):
            try:
                r = cli.chat.completions.create(
                    model=JUDGE_MODEL, max_tokens=3000,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": judge_prompt(s, crits, resp)}])
                v = parse_verdicts(r.choices[0].message.content or "", len(crits))
                if v:
                    jp.write_text(json.dumps({
                        "model": model, "scenario_id": sid,
                        "verdicts": {c["criterion_code"]: y for c, y in zip(crits, v)},
                    }), encoding="utf-8")
                    return 1
            except Exception:
                pass
        return 0

    with ThreadPoolExecutor(WORKERS) as ex:
        ok = list(ex.map(run, jobs))
    print(f"  judged {sum(ok)}/{len(jobs)} (failures are simply left uncached and retried next run)")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def phi(a: list[int], b: list[int]) -> float | None:
    """Phi coefficient -- Pearson r for two binary vectors."""
    n = len(a)
    if n == 0:
        return None
    sa, sb = sum(a), sum(b)
    if sa in (0, n) or sb in (0, n):
        return None                                   # no variance -> undefined
    sab = sum(x * y for x, y in zip(a, b))
    num = n * sab - sa * sb
    den = (sa * (n - sa) * sb * (n - sb)) ** 0.5
    return num / den if den else None


def stage_analyze() -> None:
    _, rub = load_bank()
    code_meta = {}
    for rows in rub.values():
        for r in rows:
            code_meta.setdefault(r["criterion_code"], r)

    # cell[code][(model, sid)] = 0/1
    cell: dict[str, dict[tuple, int]] = defaultdict(dict)
    per_model_total: dict[str, list[int]] = defaultdict(list)
    n_resp = 0
    for model in TUTORS:
        jd = JUDGE_DIR / slug(model)
        if not jd.exists():
            continue
        for p in sorted(jd.glob("*.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            n_resp += 1
            got = 0
            for code, v in j["verdicts"].items():
                y = 1 if v == "pass" else 0
                cell[code][(model, j["scenario_id"])] = y
                got += y
            per_model_total[model].append(got / max(1, len(j["verdicts"])))
    if not n_resp:
        raise SystemExit("no judgments found -- run the judge stage first")

    # Overall ability per (model, scenario): the share of ITS criteria passed.
    # Used as the "total score" a criterion's discrimination is measured against.
    totals: dict[tuple, float] = {}
    keys = {k for d in cell.values() for k in d}
    for k in keys:
        vals = [d[k] for d in cell.values() if k in d]
        totals[k] = sum(vals) / len(vals)

    rows = []
    for code, d in cell.items():
        ks = sorted(d)
        y = [d[k] for k in ks]
        rate = sum(y) / len(y)
        rest = []
        for k in ks:                                  # corrected item-total: exclude self
            others = [dd[k] for c2, dd in cell.items() if c2 != code and k in dd]
            rest.append(sum(others) / len(others) if others else 0.0)
        my, mr = rate, sum(rest) / len(rest)
        num = sum((a - my) * (b - mr) for a, b in zip(y, rest))
        den = (sum((a - my) ** 2 for a in y) * sum((b - mr) ** 2 for b in rest)) ** 0.5
        rows.append({
            "code": code, "n": len(y), "pass_rate": rate,
            "disc": (num / den) if den else None,
            "crit": code_meta[code]["criticality"],
            "neg": " does not " in code_meta[code]["criterion"],
            "expl": code_meta[code]["explicitness"],
        })
    rows.sort(key=lambda r: -r["pass_rate"])

    def fam(prefix):
        cs = sorted(c for c in cell if c.startswith(prefix) and c[1:].isdigit())
        out = []
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                shared = sorted(set(cell[a]) & set(cell[b]))
                if len(shared) >= 30:
                    r = phi([cell[a][k] for k in shared], [cell[b][k] for k in shared])
                    if r is not None:
                        out.append((a, b, r, len(shared)))
        return sorted(out, key=lambda t: -t[2])

    L = []
    L.append("# Bridge criterion-bank pilot — results\n")
    L.append(f"{len(TUTORS)} tutors x {N_SCENARIOS} scenarios; **{n_resp} graded responses**, "
             f"{sum(len(d) for d in cell.values()):,} criterion judgments. "
             f"Judge: `{JUDGE_MODEL}`. Tutor system prompt: the real Bridge one.\n")

    L.append("\n## Model spread (sanity check)\n")
    L.append("If the tutors do not differ, nothing below can be read as a property of the criteria.\n")
    L.append("\n| model | mean pass rate |\n|---|---|")
    for m in TUTORS:
        v = per_model_total.get(m, [])
        if v:
            L.append(f"| `{m}` | {sum(v)/len(v):.3f} |")

    L.append("\n\n## Q1 — are D1 / P1 / A1 at ceiling?\n")
    L.append("The system prompt names these three almost verbatim.\n")
    L.append("\n| code | pass rate | discrimination | criticality |\n|---|---|---|---|")
    for c in ("D1", "P1", "A1"):
        r = next((x for x in rows if x["code"] == c), None)
        if r:
            d = f"{r['disc']:.3f}" if r["disc"] is not None else "n/a"
            L.append(f"| **{c}** | {r['pass_rate']:.3f} | {d} | {r['crit']} |")

    L.append("\n\n## Q2 — do the 19 negative-form criteria vacuously pass?\n")
    neg = [r for r in rows if r["neg"]]
    pos = [r for r in rows if not r["neg"]]
    if neg and pos:
        L.append(f"\n- negative-form ({len(neg)} codes): mean pass rate "
                 f"**{sum(r['pass_rate'] for r in neg)/len(neg):.3f}**")
        L.append(f"- positive-form ({len(pos)} codes): mean pass rate "
                 f"**{sum(r['pass_rate'] for r in pos)/len(pos):.3f}**")

    L.append("\n\n## Q3 — redundancy within a skill family\n")
    L.append("Phi correlation between sibling criteria on the same responses. "
             "Above ~0.7 they are close to one criterion asked twice.\n")
    for pre, name in (("A", "affective"), ("P", "strategy"), ("M", "math"), ("C", "communication")):
        pairs = fam(pre)
        if pairs:
            L.append(f"\n**{name}**\n")
            L.append("| pair | phi | n |\n|---|---|---|")
            for a, b, r, n in pairs:
                L.append(f"| {a} – {b} | {r:+.3f} | {n} |")

    L.append("\n\n## Dead weight — criteria carrying little or no information\n")
    L.append("Pass rate above 0.95 or below 0.05 means almost no variance to measure; "
             "discrimination near zero means the criterion does not track overall quality.\n")
    L.append("\n| code | pass rate | disc | flags |\n|---|---|---|---|")
    for r in rows:
        flags = []
        if r["pass_rate"] >= 0.95:
            flags.append("CEILING")
        if r["pass_rate"] <= 0.05:
            flags.append("FLOOR")
        if r["disc"] is not None and r["disc"] < 0.10:
            flags.append("no-disc")
        if flags:
            d = f"{r['disc']:.3f}" if r["disc"] is not None else "n/a"
            L.append(f"| {r['code']} | {r['pass_rate']:.3f} | {d} | {' '.join(flags)} |")

    L.append("\n\n## All criteria\n")
    L.append("\n| code | pass rate | disc | crit | form | explicitness |\n|---|---|---|---|---|---|")
    for r in rows:
        d = f"{r['disc']:.3f}" if r["disc"] is not None else "n/a"
        L.append(f"| {r['code']} | {r['pass_rate']:.3f} | {d} | {r['crit']} | "
                 f"{'neg' if r['neg'] else 'pos'} | {r['expl']} |")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")
    print("\n".join(L[:14]))


STAGES = {"sample": stage_sample, "generate": stage_generate,
          "judge": stage_judge, "analyze": stage_analyze}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: pilot.py [{'|'.join(STAGES)}]")
    STAGES[sys.argv[1]]()
