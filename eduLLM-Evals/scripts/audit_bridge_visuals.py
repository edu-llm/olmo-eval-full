"""Classify every Bridge scenario by whether answering it needs a visual the text lacks.

Bridge is text-only (verified against the HuggingFace repo: 9 columns, all text, and the
only .png in the repo is the paper's own figure). But the conversations are transcripts of
live sessions held over a shared whiteboard, so tutors point at things the dataset never
captured -- "the pink rectangle", "have a look at the whiteboard", and in 16 cases a pasted
image URL. A tutor model is then asked to remediate an error it cannot see, and D1/D2 (both
critical) are unpassable on those items for a reason that has nothing to do with tutoring
ability.

A regex screen finds 117 scenarios but is a floor, not a ceiling: the hardest cases are
silent, e.g. "What is the area?" where the dimensions only ever existed in a figure and no
word in the transcript says so. This asks a model instead, one scenario at a time.

The classifier sees the expert's reply as well as the visible context. That is deliberate:
the expert reply is evidence about what the item required, and the worst cases are the ones
where the model sees no clue at all but the expert clearly did.

    python scripts/audit_bridge_visuals.py classify   # one call per scenario, cached
    python scripts/audit_bridge_visuals.py report     # aggregate -> report.md

Resumable: every verdict is cached to disk, so re-running fills gaps only.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "data" / "Bridge" / "scenarios.jsonl"
OUT_DIR = ROOT / "staging" / "bridge_visual_audit"
VERDICT_DIR = OUT_DIR / "verdicts"
REPORT = OUT_DIR / "report.md"

BASE_URL = "https://tfy.promptlens.trilogy.com/api/llm/api/inference/openai"
MODEL = "openai-group/gpt-5.5"
WORKERS = 12

SYSTEM = """You audit a math-tutoring dataset for items that cannot be answered from text alone.

These are transcripts of real online tutoring sessions. The student and tutor shared a
whiteboard showing a worksheet. Only the chat text was saved -- any figure, diagram, chart,
number line or problem image is GONE. You are deciding whether this specific item still
makes sense without it.

A tutor model will be shown ONLY the conversation and the student's final turn, and must
write the next tutor message: identify the student's error and guide them to fix it.

Mark requires_visual = true when a competent tutor could NOT identify the student's error
without seeing something absent from the text. Typical cases:
  - an explicit pointer to a shared artifact ("the pink rectangle", "look at the whiteboard",
    "this chart", a pasted image link)
  - the question's data lives only in a figure -- e.g. asking for an area, a count of sides,
    a graph reading, or a point on a number line, where no dimensions/values ever appear in
    the text
  - the student's answer refers to a labelled option ("b", "picture 3") that only a figure
    defines

Mark requires_visual = false when the text is self-sufficient, even if it sounds visual.
A mention of a shape or a "let's look at this" figure of speech is NOT enough: ask whether
the numbers and the student's error are recoverable from the words alone. Arithmetic stated
in text ("15 goes into 45"), or a question whose values are all written out, is fine.

Return ONLY a JSON object:
{"requires_visual": true|false,
 "confidence": "high"|"medium"|"low",
 "kind": "explicit_pointer"|"missing_data"|"labelled_option"|"none",
 "evidence": "<=15 words quoted or paraphrased from the item, or empty>"}"""


def client():
    from openai import OpenAI

    key = os.environ.get("TFY_API_KEY")
    if not key:
        raise SystemExit("TFY_API_KEY is not set")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=120)


def scenarios() -> list[dict]:
    return [json.loads(line) for line in SCENARIOS.open(encoding="utf-8")]


def user_prompt(s: dict) -> str:
    ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in s["conversation_context"])
    return (
        f"<lesson_topic>{s.get('lesson_topic')}</lesson_topic>\n\n"
        f"<conversation_shown_to_the_tutor_model>\n{ctx}\n"
        f"[student] {s['prompt']}\n</conversation_shown_to_the_tutor_model>\n\n"
        f"<expert_tutor_reply_evidence_only>\n{s['reference_solution']}\n"
        f"</expert_tutor_reply_evidence_only>\n\n"
        "Does answering this require a visual the text does not contain?"
    )


def parse(text: str) -> dict | None:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(obj.get("requires_visual"), bool):
        return None
    return {
        "requires_visual": obj["requires_visual"],
        "confidence": str(obj.get("confidence", "")).lower() or "medium",
        "kind": str(obj.get("kind", "none")),
        "evidence": str(obj.get("evidence", ""))[:200],
    }


def stage_classify() -> None:
    cli = client()
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (s, VERDICT_DIR / f"{s['scenario_id']}.json")
        for s in scenarios()
        if not (VERDICT_DIR / f"{s['scenario_id']}.json").exists()
    ]
    print(f"classify: {len(jobs)} pending (cached ones skipped)", flush=True)
    if not jobs:
        return

    def run(job):
        s, path = job
        for _ in range(3):
            try:
                r = cli.chat.completions.create(
                    model=MODEL,
                    max_tokens=400,
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
    print(f"  classified {sum(ok)}/{len(jobs)} (failures left uncached, retried next run)")


def stage_report() -> None:
    """Aggregate the cached verdicts against whatever the bank currently holds.

    Verdicts outlive the bank: once an audited scenario is excluded it is gone from
    scenarios.jsonl but its verdict stays on disk, which is what lets this report show how
    much of the flagged set has already been acted on. Breakdowns that need scenario
    metadata cover only the scenarios still present.
    """
    scen = {s["scenario_id"]: s for s in scenarios()}
    verdicts = {}
    for p in sorted(VERDICT_DIR.glob("*.json")):
        v = json.loads(p.read_text(encoding="utf-8"))
        verdicts[v["scenario_id"]] = v
    if not verdicts:
        raise SystemExit("no verdicts -- run the classify stage first")

    removed = {k for k in verdicts if k not in scen}
    verdicts = {k: v for k, v in verdicts.items() if k in scen}
    if not verdicts:
        raise SystemExit(
            f"all {len(removed)} audited scenarios have been excluded from the bank; "
            "nothing left to report on"
        )

    flagged = {k for k, v in verdicts.items() if v["requires_visual"]}
    high = {k for k in flagged if verdicts[k]["confidence"] == "high"}
    crit = sum(len(scen[k]["criterion_ids"]) for k in flagged)
    convs = {scen[k]["source_id"] for k in flagged}

    L = ["# Bridge visual-dependency audit\n"]
    L.append(
        f"Classifier `{MODEL}`, one call per scenario, {len(verdicts)}/{len(scen)} scenarios "
        f"in the current bank judged. Question: can a tutor identify the student's error "
        f"from the text alone?\n"
    )
    if removed:
        L.append(
            f"\n> **{len(removed)} audited scenarios have since been excluded from the bank** "
            f"and are not counted below. The audit as it stood before that cut is in "
            f"[`report_precut_642.md`](report_precut_642.md).\n"
        )
    L.append("\n## Headline\n")
    L.append("\n| | scenarios | conversations | criteria |\n|---|---:|---:|---:|")
    L.append(f"| requires a visual | {len(flagged)} | {len(convs)} | {crit} |")
    L.append(
        f"| ...of those, high confidence | {len(high)} | "
        f"{len({scen[k]['source_id'] for k in high})} | "
        f"{sum(len(scen[k]['criterion_ids']) for k in high)} |"
    )
    L.append(f"| self-sufficient (would remain) | {len(scen) - len(flagged)} | — | — |")
    L.append(f"\nShare of the bank flagged: **{len(flagged) / len(scen):.1%}**\n")

    L.append("\n## Confidence\n")
    L.append("\n| confidence | flagged |\n|---|---:|")
    for c, n in Counter(verdicts[k]["confidence"] for k in flagged).most_common():
        L.append(f"| {c} | {n} |")

    L.append("\n\n## Why they fail\n")
    L.append("\n| kind | scenarios |\n|---|---:|")
    for k, n in Counter(verdicts[i]["kind"] for i in flagged).most_common():
        L.append(f"| `{k}` | {n} |")

    L.append("\n\n## Damage by topic domain\n")
    L.append("\n| topic_domain | flagged | total | share |\n|---|---:|---:|---:|")
    tot = Counter(s["topic_domain"] for s in scen.values())
    hit = Counter(scen[k]["topic_domain"] for k in flagged)
    for d, n in sorted(tot.items(), key=lambda kv: -hit[kv[0]] / kv[1]):
        L.append(f"| {d} | {hit[d]} | {n} | {hit[d] / n:.0%} |")

    L.append("\n\n## Damage by grade band\n")
    L.append("\n| grade_band | flagged | total | share |\n|---|---:|---:|---:|")
    tb = Counter(s["grade_band"] for s in scen.values())
    hb = Counter(scen[k]["grade_band"] for k in flagged)
    for b, n in sorted(tb.items()):
        L.append(f"| {b} | {hb[b]} | {n} | {hb[b] / n:.0%} |")

    L.append("\n\n## Interaction with `visible_mistake`\n")
    L.append(
        "`visible_mistake: false` already flags scenarios with no remediable error. If the "
        "two overlap little, this audit is finding a genuinely separate defect.\n"
    )
    L.append("\n| | flagged visual | not flagged |\n|---|---:|---:|")
    for vm in (True, False):
        row = [s for s in scen.values() if s["visible_mistake"] is vm]
        L.append(
            f"| visible_mistake={vm} | {sum(1 for s in row if s['scenario_id'] in flagged)} "
            f"| {sum(1 for s in row if s['scenario_id'] not in flagged)} |"
        )

    L.append("\n\n## Conversation-level consistency\n")
    L.append(
        "Repeats of one conversation share a stimulus, so they should be classified alike. "
        "Splits are classifier noise and mark the borderline cases.\n"
    )
    by_conv: dict[str, set[bool]] = defaultdict(set)
    for k, v in verdicts.items():
        by_conv[scen[k]["source_id"]].add(v["requires_visual"])
    split = [c for c, vals in by_conv.items() if len(vals) > 1]
    per_conv = Counter(scen[k]["source_id"] for k in verdicts)
    multi = [c for c in by_conv if per_conv[c] > 1]
    L.append(f"\n- conversations with >1 scenario: **{len(multi)}**")
    L.append(f"- of those, classified inconsistently: **{len(split)}**")

    L.append("\n\n## Sample of flagged items\n")
    for k in sorted(high)[:12]:
        s, v = scen[k], verdicts[k]
        L.append(f"\n**{k}** · `{s['lesson_topic']}` · {v['kind']}")
        L.append(f"> {v['evidence']}")
        L.append(f"> student's final turn: `{s['prompt'][:60]}`")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    print("\n".join(L[:16]))


# ---------------------------------------------------------------------------
# reaudit -- a second, sharper pass over what the first pass flagged
# ---------------------------------------------------------------------------
# The classify stage asks "does this need a visual?", which for the missing-problem-statement
# rows is the same as "is this answerable?" -- the question is simply absent. For the rows
# that merely POINT at a figure the two come apart: a student who answers "4 m" to "what is
# the area?" has given a linear unit for an area, and that is diagnosable without ever seeing
# the rectangle. Blanket-cutting those would discard real items, so they get asked the
# question that actually decides inclusion.
REAUDIT_DIR = OUT_DIR / "reaudit"

REAUDIT_SYSTEM = """You decide whether a math-tutoring item is gradeable.

Context: these are transcripts of online tutoring over a shared whiteboard. The worksheet
was not saved, so a figure the conversation refers to is gone. A tutor model sees ONLY the
text below and must write the next tutor turn.

The question is NOT whether a figure is mentioned. It is whether a competent tutor could,
from the visible text alone, identify something specific the student got wrong and give a
useful corrective next turn.

Answer diagnosable = true when the student's response carries the error on its face. For
example:
  - a unit or type mismatch -- "4 m" answering a question about an AREA, "180 hours" for a
    number of days, a fraction where a whole number was asked for
  - an answer contradicted by something stated earlier in the text
  - an arithmetic claim, stated in the text, that is simply wrong
  - the student names a concept incorrectly in words you can read
A tutor can act on any of these without seeing the figure.

Answer diagnosable = false when the response is only meaningful against the missing figure:
  - a bare assent ("yes", "done", "ok") with no answer to evaluate
  - a bare option label ("A", "b", "3") whose meaning only the figure defines
  - a bare number whose correctness cannot be checked because the problem is not stated
Do NOT use the expert reply to rescue an item: if the error is only knowable from the
expert's text and not from the student's, it is false.

Return ONLY:
{"diagnosable": true|false,
 "confidence": "high"|"medium"|"low",
 "the_error": "<what a tutor could tell is wrong, from the visible text; empty if none>"}"""


def reaudit_prompt(s: dict) -> str:
    ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in s["conversation_context"])
    return (
        f"<lesson_topic>{s.get('lesson_topic')}</lesson_topic>\n\n"
        f"<visible_to_the_tutor_model>\n{ctx}\n[student] {s['prompt']}\n"
        f"</visible_to_the_tutor_model>\n\n"
        f"<expert_reply_do_not_use_to_rescue_the_item>\n{s['reference_solution']}\n"
        f"</expert_reply_do_not_use_to_rescue_the_item>\n\n"
        "Is the student's error diagnosable from the visible text alone?"
    )


def reaudit_parse(text: str) -> dict | None:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(obj.get("diagnosable"), bool):
        return None
    return {
        "diagnosable": obj["diagnosable"],
        "confidence": str(obj.get("confidence", "")).lower() or "medium",
        "the_error": str(obj.get("the_error", ""))[:240],
    }


def _still_flagged() -> list[dict]:
    """Scenarios still in the bank that the first pass flagged as needing a visual."""
    scen = {s["scenario_id"]: s for s in scenarios()}
    out = []
    for p in sorted(VERDICT_DIR.glob("*.json")):
        v = json.loads(p.read_text(encoding="utf-8"))
        if v["requires_visual"] and v["scenario_id"] in scen:
            out.append(scen[v["scenario_id"]])
    return out


def stage_reaudit() -> None:
    cli = client()
    REAUDIT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (s, REAUDIT_DIR / f"{s['scenario_id']}.json")
        for s in _still_flagged()
        if not (REAUDIT_DIR / f"{s['scenario_id']}.json").exists()
    ]
    print(f"reaudit: {len(jobs)} pending", flush=True)
    if not jobs:
        return

    def run(job):
        s, path = job
        # 400 tokens is enough for the verdict but not always for the reasoning that
        # precedes it; escalate rather than lose the scenario to an empty completion.
        for tokens in (700, 2000, 6000):
            try:
                r = cli.chat.completions.create(
                    model=MODEL,
                    max_tokens=tokens,
                    messages=[
                        {"role": "system", "content": REAUDIT_SYSTEM},
                        {"role": "user", "content": reaudit_prompt(s)},
                    ],
                )
                v = reaudit_parse(r.choices[0].message.content or "")
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
    print(f"  reaudited {sum(ok)}/{len(jobs)}")


def stage_reaudit_report() -> None:
    scen = {s["scenario_id"]: s for s in scenarios()}
    first = {}
    for p in sorted(VERDICT_DIR.glob("*.json")):
        v = json.loads(p.read_text(encoding="utf-8"))
        first[v["scenario_id"]] = v
    second = {}
    for p in sorted(REAUDIT_DIR.glob("*.json")):
        v = json.loads(p.read_text(encoding="utf-8"))
        second[v["scenario_id"]] = v
    if not second:
        raise SystemExit("no reaudit verdicts -- run the reaudit stage first")

    keep = sorted(k for k, v in second.items() if v["diagnosable"])
    cut = sorted(k for k, v in second.items() if not v["diagnosable"])
    print(f"reaudited {len(second)} of the still-flagged scenarios")
    print(f"  diagnosable from text alone (SALVAGED): {len(keep)}")
    print(f"  not diagnosable (cut candidates):       {len(cut)}")

    print("\nby first-pass kind:")
    for kind in ("explicit_pointer", "labelled_option", "missing_data"):
        ids = [k for k in second if first[k]["kind"] == kind]
        if ids:
            n = sum(1 for k in ids if second[k]["diagnosable"])
            print(f"  {kind:<18} {len(ids):>3} judged -> {n:>3} salvaged, {len(ids) - n:>3} cut")

    print("\nconfidence on the salvaged:",
          dict(Counter(second[k]["confidence"] for k in keep)))
    print("\nsample of SALVAGED items (error visible without the figure):")
    for k in keep[:8]:
        print(f"  [{k}] student said {scen[k]['prompt']!r}")
        print(f"      -> {second[k]['the_error'][:120]}")
    print("\nsample of CUT items:")
    for k in cut[:5]:
        print(f"  [{k}] student said {scen[k]['prompt']!r}  ({first[k]['kind']})")

    out = OUT_DIR / "reaudit_decision.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            {
                "_comment": "Second pass over scenarios the first audit flagged. Asks whether "
                "the student's error is diagnosable from the visible text alone, which is the "
                "question that decides inclusion; the first pass asked only whether a figure "
                "was referenced.",
                "classifier": MODEL,
                "salvage": [{"scenario_id": k, **second[k]} for k in keep],
                "cut": [{"scenario_id": k, **second[k]} for k in cut],
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
        fh.write("\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


STAGES = {
    "classify": stage_classify,
    "report": stage_report,
    "reaudit": stage_reaudit,
    "reaudit_report": stage_reaudit_report,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: audit_bridge_visuals.py [{'|'.join(STAGES)}]")
    STAGES[sys.argv[1]]()
