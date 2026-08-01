"""Read every Bridge scenario for what it actually asks, and check its labels against that.

Two label defects have already been found by reading items rather than trusting their
metadata, both the same shape: a field describes the SESSION and was used as though it
described the graded TURN.

  * `data_graphing` (retired) gated on `lesson_topic`, so B1 -- "did the tutor misread the
    scale, key or axis?" -- was attached to a response to "12 - 1", inside a lesson merely
    titled "Bar Graphs".
  * 353 scenarios were cut because the problem statement lived on the session whiteboard and
    the transcript records only the conversation around it.

Both were found by hand, on samples. This does it for all 289 at once. Per scenario the
classifier is asked to state what the turn asks and what the student got wrong, then judge
the assigned labels against that:

    asks_for              what the graded turn actually asks (free text)
    student_error         what the student got wrong (free text)
    student_actually_wrong  is there an error at all? Bridge asserts one exists, but the
                          exclusion of free-text "no mistake" annotations only caught the
                          cases an annotator said so in words.
    gradeable             can a tutor diagnose it from the visible text? (residual
                          missing-context after the two cuts)
    topic / topic_ok      best-fit topic_domain from the seven, vs the keyword gate's guess
    error_fits            does the canonical error_module match the visible evidence?
    concern               anything else that would make this a bad item

The reference reply is shown, because what the task WAS is exactly what it is evidence of.
`gradeable` is the one field it must not be used for -- an error only the expert could see is
not gradeable -- and the prompt says so.

    python scripts/audit_bridge_items.py sweep     # one call per scenario, cached
    python scripts/audit_bridge_items.py report    # aggregate -> report.md

Resumable: every verdict is cached, so re-running fills gaps only.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "data" / "Bridge" / "scenarios.jsonl"
RUBRICS = ROOT / "data" / "Bridge" / "rubrics.jsonl"
OUT_DIR = ROOT / "staging" / "bridge_item_audit"
VERDICT_DIR = OUT_DIR / "verdicts"
REPORT = OUT_DIR / "report.md"

BASE_URL = "https://tfy.promptlens.trilogy.com/api/llm/api/inference/openai"
MODEL = "openai-group/gpt-5.5"
WORKERS = 12

# The seven live topic domains and the criterion each one attaches, so the classifier judges
# fit against what the criterion actually asks rather than against the label's name.
TOPIC_MEANING = {
    "geometry_spatial": "shapes, area, perimeter, volume, angles, symmetry, figures "
    "(attaches V1/V2: invite a visual representation; name the figure or unit precisely)",
    "operations_arithmetic": "add/subtract/multiply/divide, the meaning of an operation "
    "(attaches O1: connect the operation to its meaning, not just the procedure)",
    "place_value_number": "place value, rounding, decimals, comparing/ordering, factors "
    "(attaches N1: do not describe or use place value incorrectly)",
    "measurement_conversion": "unit conversion, time, money, standard measurement "
    "(attaches U1: do not confuse or omit the units the problem turns on)",
    "fractions": "fractions and mixed numbers "
    "(attaches F1: treat the fraction as a quantity, not two numbers to manipulate)",
    "proportional_reasoning": "ratio, rate, percent, proportion, speed "
    "(attaches Z1: identify which two quantities the rate compares)",
    "algebra_expressions": "expressions, equations, variables, order of operations, functions "
    "(attaches X1: draw attention to the structure -- what the unknown stands for)",
}

ERROR_MEANING = {
    "conceptual": "the student's reasoning or understanding is itself wrong "
    "(attaches D3/D4/D5: address the underlying misconception)",
    "guess": "the student produced an answer with no visible reasoning "
    "(attaches G1/G2: do not assert an unevidenced misconception; give an entry point)",
    "right_idea": "the approach is sound but execution or completion is not "
    "(attaches R1/R2: preserve and build on the correct part)",
    "careless": "the student knows the method but slipped "
    "(attaches S1/S2: do not re-teach the whole concept; let them find the slip)",
    "imprecise": "on the right track but imprecise, unlabelled or incomplete "
    "(attaches I1/I2: do not call it entirely wrong; model precise notation)",
}

SYSTEM = """You audit items in a math-tutoring benchmark. Each item is a real tutoring
transcript cut at a student mistake; a tutor model must write the next tutor turn.

Read the item for what the GRADED TURN actually asks and what the student actually did. Then
judge the item's stored labels against that reading. Labels were assigned by keyword rules
over session-level metadata, so they are frequently about the lesson rather than this turn:
a lesson titled "Bar Graphs" can contain a plain subtraction question.

Fields:

asks_for -- what is being asked of the student at this point, in one clause. If the problem
  is not stated in the text, say what it appears to be and prefix "UNSTATED: ".

student_error -- what the student got wrong, in one clause. Empty if nothing is wrong.

student_actually_wrong -- true if the student's final turn is genuinely incorrect or
  incomplete. FALSE if the student is actually right, is merely acknowledging ("ok", "yes"),
  is asking a question, or is ending the session. Do not assume an error exists because the
  dataset says one does.

gradeable -- true if a tutor could identify the error from the VISIBLE TEXT alone. Judge this
  on the visible text only: if the error is knowable only from the expert reply, answer false.

topic -- which ONE domain best fits what this turn is mathematically about. Choose from the
  list given. Use "none_fit" if the turn is not about mathematics at all (session setup,
  tool talk, pure encouragement).

error_fits -- true if the item's stored error_module matches the visible evidence. A student
  who showed working is not a "guess"; a student whose method is sound but arithmetic slipped
  is "careless" or "right_idea", not "conceptual".

concern -- anything else making this a bad benchmark item: the tutor (not the student) is
  wrong, the transcript is incoherent, the student answered in another language, the "error"
  is a typo, the question is unanswerable as posed. Empty string if the item is sound.

Return ONLY:
{"asks_for": "...", "student_error": "...", "student_actually_wrong": true|false,
 "gradeable": true|false, "topic": "<domain or none_fit>", "error_fits": true|false,
 "error_suggest": "<module or empty>", "concern": "...", "confidence": "high"|"medium"|"low"}"""


def client():
    from openai import OpenAI

    key = os.environ.get("TFY_API_KEY")
    if not key:
        raise SystemExit("TFY_API_KEY is not set")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=180)


def scenarios() -> list[dict]:
    return [json.loads(line) for line in SCENARIOS.open(encoding="utf-8")]


def user_prompt(s: dict) -> str:
    ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in s["conversation_context"])
    topics = "\n".join(f"  {k}: {v}" for k, v in TOPIC_MEANING.items())
    errors = "\n".join(f"  {k}: {v}" for k, v in ERROR_MEANING.items())
    return (
        f"<visible_to_the_tutor_model>\n{ctx}\n[student] {s['prompt']}\n"
        f"</visible_to_the_tutor_model>\n\n"
        f"<expert_reply_evidence_of_what_the_task_was_do_not_use_for_gradeable>\n"
        f"{s['reference_solution']}\n</expert_reply_evidence_of_what_the_task_was_do_not_use_for_gradeable>\n\n"
        f"<stored_labels>\n"
        f"  lesson_topic (session-level, may not describe this turn): {s.get('lesson_topic')}\n"
        f"  topic_domain (assigned by keyword on lesson_topic): {s['topic_domain']}\n"
        f"  error_module (expert annotation, canonical per conversation): {s['error_module']}\n"
        f"  grade_band: {s['grade_band']}\n</stored_labels>\n\n"
        f"<valid_topic_domains>\n{topics}\n</valid_topic_domains>\n\n"
        f"<valid_error_modules>\n{errors}\n</valid_error_modules>\n\n"
        "Audit this item."
    )


FIELDS = ("asks_for", "student_error", "student_actually_wrong", "gradeable", "topic",
          "error_fits", "error_suggest", "concern", "confidence")


def parse(text: str) -> dict | None:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    for b in ("student_actually_wrong", "gradeable", "error_fits"):
        if not isinstance(obj.get(b), bool):
            return None
    if not obj.get("topic"):
        return None
    out = {k: obj.get(k) for k in FIELDS}
    for k in ("asks_for", "student_error", "topic", "error_suggest", "concern"):
        out[k] = str(out[k] or "")[:400]
    out["confidence"] = str(out["confidence"] or "medium").lower()
    return out


def stage_sweep() -> None:
    cli = client()
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (s, VERDICT_DIR / f"{s['scenario_id']}.json")
        for s in scenarios()
        if not (VERDICT_DIR / f"{s['scenario_id']}.json").exists()
    ]
    print(f"sweep: {len(jobs)} pending (cached ones skipped)", flush=True)
    if not jobs:
        return

    def run(job):
        s, path = job
        # A reasoning model can spend the whole budget thinking and emit nothing; escalate
        # rather than lose the scenario to an empty completion.
        for tokens in (1400, 4000, 9000):
            try:
                r = cli.chat.completions.create(
                    model=MODEL,
                    max_tokens=tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_prompt(s)},
                    ],
                )
                v = parse(r.choices[0].message.content or "")
                if v:
                    path.write_text(
                        json.dumps({"scenario_id": s["scenario_id"], **v}), encoding="utf-8"
                    )
                    return 1
            except Exception:
                pass
        return 0

    with ThreadPoolExecutor(WORKERS) as ex:
        ok = list(ex.map(run, jobs))
    print(f"  swept {sum(ok)}/{len(jobs)} (failures left uncached, retried next run)")


def stage_report() -> None:
    scen = {s["scenario_id"]: s for s in scenarios()}
    crit_count: dict[str, int] = Counter()
    for line in RUBRICS.open(encoding="utf-8"):
        crit_count[json.loads(line)["scenario_id"]] += 1
    v = {}
    for p in sorted(VERDICT_DIR.glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row["scenario_id"] in scen:
            v[row["scenario_id"]] = row
    if not v:
        raise SystemExit("no verdicts -- run the sweep stage first")

    not_wrong = [k for k in v if not v[k]["student_actually_wrong"]]
    ungradeable = [k for k in v if not v[k]["gradeable"]]
    topic_bad = [k for k in v if v[k]["topic"] not in (scen[k]["topic_domain"], "")]
    none_fit = [k for k in v if v[k]["topic"] == "none_fit"]
    err_bad = [k for k in v if not v[k]["error_fits"]]
    concerns = [k for k in v if v[k]["concern"].strip()]
    unstated = [k for k in v if v[k]["asks_for"].startswith("UNSTATED")]

    def block(name, ids, note=""):
        L = [f"\n### {name} — {len(ids)} of {len(v)} ({len(ids) / len(v):.0%})\n"]
        if note:
            L.append(f"{note}\n")
        return L

    L = ["# Bridge per-item audit\n"]
    L.append(
        f"Classifier `{MODEL}`, one call per scenario, {len(v)}/{len(scen)} scenarios read for "
        f"what the graded turn actually asks. Labels judged against that reading.\n"
    )
    L.append("\n## Headline\n")
    L.append("\n| finding | scenarios | share |\n|---|---:|---:|")
    for name, ids in (
        ("student is not actually wrong", not_wrong),
        ("not gradeable from visible text", ungradeable),
        ("topic_domain mismatched", topic_bad),
        ("...of which not about maths at all (`none_fit`)", none_fit),
        ("error_module contradicted by the evidence", err_bad),
        ("problem statement still unstated", unstated),
        ("other concern flagged", concerns),
    ):
        L.append(f"| {name} | {len(ids)} | {len(ids) / len(v):.0%} |")
    clean = [k for k in v if k not in set(not_wrong) | set(ungradeable) | set(topic_bad)
             | set(err_bad) | set(concerns)]
    L.append(f"| **clean on every check** | **{len(clean)}** | **{len(clean) / len(v):.0%}** |")

    L.append("\n\n## topic_domain: assigned vs read\n")
    L.append("\n| assigned | classifier's reading | n |\n|---|---|---:|")
    pairs = Counter((scen[k]["topic_domain"], v[k]["topic"]) for k in topic_bad)
    for (a, b), n in pairs.most_common():
        L.append(f"| {a} | {b} | {n} |")

    L.append("\n\n## error_module: assigned vs suggested\n")
    L.append("\n| assigned | suggested | n |\n|---|---|---:|")
    ep = Counter((scen[k]["error_module"], v[k]["error_suggest"] or "?") for k in err_bad)
    for (a, b), n in ep.most_common():
        L.append(f"| {a} | {b} | {n} |")

    L.extend(block("Students who are not actually wrong", not_wrong,
                   "D1/D2 ask the tutor to identify an error. Where there is none they are "
                   "unpassable, and both are `critical`."))
    for k in not_wrong[:20]:
        L.append(f"- **{k}** · student said `{scen[k]['prompt'][:40]}` — {v[k]['asks_for'][:110]}")

    L.extend(block("Still not gradeable", ungradeable,
                   "Residual missing context the two earlier cuts did not catch."))
    for k in ungradeable[:20]:
        L.append(f"- **{k}** · student said `{scen[k]['prompt'][:40]}` — {v[k]['asks_for'][:110]}")

    L.extend(block("Other concerns", concerns))
    for k in concerns[:30]:
        L.append(f"- **{k}** — {v[k]['concern'][:170]}")

    L.append("\n\n## Criteria at stake\n")
    bad = set(not_wrong) | set(ungradeable)
    L.append(
        f"\n{len(bad)} scenarios fail the two checks that make an item unanswerable "
        f"(no error present, or not gradeable), carrying "
        f"**{sum(crit_count[k] for k in bad)} criteria** and "
        f"**{len(bad) * 7} critical judgments**.\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    decision = OUT_DIR / "findings.json"
    with decision.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            {
                "_comment": "Per-item audit of the Bridge bank: what each graded turn actually "
                "asks, and whether its stored labels match. Groups are not mutually exclusive.",
                "classifier": MODEL,
                "student_not_actually_wrong": not_wrong,
                "not_gradeable": ungradeable,
                "topic_domain_mismatch": {k: v[k]["topic"] for k in topic_bad},
                "error_module_mismatch": {k: v[k]["error_suggest"] for k in err_bad},
                "concerns": {k: v[k]["concern"] for k in concerns},
                "verdicts": v,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
        fh.write("\n")
    print(f"wrote {REPORT.relative_to(ROOT)} and {decision.relative_to(ROOT)}")
    print("\n".join(L[:22]))


STAGES = {"sweep": stage_sweep, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: audit_bridge_items.py [{'|'.join(STAGES)}]")
    STAGES[sys.argv[1]]()
