"""Adversarially try to DISQUALIFY every scenario the earlier audits kept.

Every pass so far has hunted for bad items and then acted on what it found, which measures
precision and says nothing about recall: the items that survived were never themselves put on
trial. Three times now a deeper read has found a defect the previous read missed, so the
remaining question is not "what did we flag?" but "what is still wrong with what we kept?".

So this inverts the burden of proof. The prompt does not ask whether an item is usable; it
instructs the model to find the strongest concrete reason the item must NOT be used, and to
resolve genuine uncertainty toward disqualification. A single confirm-shaped reader agreeing
with a previous confirm-shaped reader is worth very little; a reader told to attack it and
failing is worth something.

Three independent attempts run per scenario with different attack lenses, and a scenario is
reported as suspect when at least two of the three disqualify it. Majority-of-lenses is used
rather than any-single-lens because an adversarial prompt will always manufacture some
objection, and the cost of over-cutting a benchmark this small is now high.

    python scripts/verify_bridge_items.py attack     # 3 calls per scenario, cached
    python scripts/verify_bridge_items.py report     # majority verdicts -> report.md

Resumable per (scenario, lens): every attempt is cached, so re-running fills gaps only.
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
RUBRICS = ROOT / "data" / "Bridge" / "rubrics.jsonl"
OUT_DIR = ROOT / "staging" / "bridge_item_verify"
ATTACK_DIR = OUT_DIR / "attacks"
REPORT = OUT_DIR / "report.md"

BASE_URL = "https://tfy.promptlens.trilogy.com/api/llm/api/inference/openai"
MODEL = "openai-group/gpt-5.5"
WORKERS = 12

# Distinct attack surfaces. Redundant skeptics find redundant problems; each lens is given a
# different failure mode to hunt so the three attempts are not one attempt repeated.
LENSES = {
    "answerability": """Attack ANSWERABILITY. Can a competent tutor, seeing only the visible
text, name something specific the student got wrong? Hunt for: the problem statement absent
so the student's answer cannot be checked; an answer that is a bare label or assent; a
student who is actually correct, or merely acknowledging, asking a question or leaving; an
error knowable only from the expert reply. If the tutor would have to guess what was asked,
disqualify.""",
    "criteria_fit": """Attack CRITERION FIT. The item will be graded on these criteria. Hunt
for criteria that cannot be answered pass/fail against any plausible response to THIS item,
or that ask about content the item cannot contain: a criterion about units where no unit is
involved, about place value where none appears, about a visual representation where nothing
is visual, about fractions where none occur, about building on the student's partial work
where the student showed none. If an attached criterion is unanswerable or beside the point,
disqualify.""",
    "gold_soundness": """Attack the GOLD KEY and the tutoring premise. The expert reply is the
reference a judge is shown. Hunt for: the expert reply not actually addressing the student's
turn; the expert affirming an answer the item treats as an error, or correcting one that is
right; the TUTOR being the one who is wrong earlier in the transcript; the transcript being
incoherent or truncated mid-thought; the "error" being a typo or a language artifact rather
than a maths mistake. If the gold key would mislead a judge, disqualify.""",
}

SYSTEM = """You are removing unfit items from a math-tutoring benchmark. Assume the item is
unfit and try to prove it.

The items are real tutoring transcripts cut at a student mistake. A tutor model sees the
conversation and the student's final turn, and must write the next tutor turn: identify the
error and guide the student to fix it. It is then graded pass/fail on the attached criteria.

Your job is not to assess quality. It is to find the strongest concrete, specific reason this
item must not be used, quoting the text that proves it. Resolve genuine uncertainty toward
disqualification: say so if you cannot find a real defect, but do not invent one, and do not
disqualify an item merely for being terse, informal, containing typos in the student's
writing, or being easy.

Two adjudicated points, already settled on the whole bank. Do not re-open them:

  * A REFERENCED BUT ABSENT FIGURE IS NOT AUTOMATICALLY FATAL. Items whose error was legible
    only from the missing figure have already been removed. What remains are items where the
    student's answer is wrong ON ITS FACE, independently of the figure -- a linear unit
    answering a question about an area ("4 m" for an area, which cannot be an area whatever
    the rectangle measures), a shape name that is not a shape, a measurement type given where
    a unit was asked for. Disqualify only if you cannot state the error without the figure.
  * A BRIEF EXPERT REPLY IS NOT AN UNSOUND GOLD KEY. Bridge's expert tutors write ~83
    characters and routinely open "Great try!" as politeness before correcting, so an
    affirming opener does not mean the student was right. Handing one question back to the
    student is competent tutoring, not an incomplete answer.

{lens}

Return ONLY:
{{"disqualify": true|false,
 "reason": "<the specific defect, with the text that proves it; empty if none>",
 "severity": "fatal"|"serious"|"minor",
 "confidence": "high"|"medium"|"low"}}"""


def client():
    from openai import OpenAI

    key = os.environ.get("TFY_API_KEY")
    if not key:
        raise SystemExit("TFY_API_KEY is not set")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=180)


def load() -> tuple[list[dict], dict[str, list[dict]]]:
    scen = [json.loads(line) for line in SCENARIOS.open(encoding="utf-8")]
    rub: dict[str, list[dict]] = defaultdict(list)
    for line in RUBRICS.open(encoding="utf-8"):
        r = json.loads(line)
        rub[r["scenario_id"]].append(r)
    return scen, rub


def user_prompt(s: dict, crits: list[dict]) -> str:
    ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in s["conversation_context"])
    # Only the conditional criteria are listed: the 16 core ones attach to every scenario by
    # design, so a complaint about them is a complaint about the bank, not about this item.
    cond = [c for c in crits if c["applicability"] != "core"]
    lines = "\n".join(
        f"  [{c['criterion_code']} · {c['applicability']} · {c['criticality']}] {c['criterion']}"
        for c in sorted(cond, key=lambda c: c["criterion_id"])
    )
    return (
        f"<visible_to_the_tutor_model>\n{ctx}\n[student] {s['prompt']}\n"
        f"</visible_to_the_tutor_model>\n\n"
        f"<expert_reply_the_gold_key_shown_to_the_judge>\n{s['reference_solution']}\n"
        f"</expert_reply_the_gold_key_shown_to_the_judge>\n\n"
        f"<conditional_criteria_attached_to_this_item>\n{lines}\n"
        f"</conditional_criteria_attached_to_this_item>\n\n"
        f"<labels>lesson_topic={s.get('lesson_topic')} · topic_domain={s['topic_domain']} · "
        f"error_module={s['error_module']} · grade_band={s['grade_band']}</labels>\n\n"
        "Prove this item is unfit, or report that you could not."
    )


def parse(text: str) -> dict | None:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(obj.get("disqualify"), bool):
        return None
    return {
        "disqualify": obj["disqualify"],
        "reason": str(obj.get("reason") or "")[:500],
        "severity": str(obj.get("severity") or "minor").lower(),
        "confidence": str(obj.get("confidence") or "medium").lower(),
    }


def stage_attack() -> None:
    cli = client()
    scen, rub = load()
    ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in scen:
        for lens in LENSES:
            p = ATTACK_DIR / f"{s['scenario_id']}__{lens}.json"
            if not p.exists():
                jobs.append((s, rub[s["scenario_id"]], lens, p))
    print(f"attack: {len(jobs)} pending ({len(scen)} scenarios x {len(LENSES)} lenses)", flush=True)
    if not jobs:
        return

    def run(job):
        s, crits, lens, path = job
        system = SYSTEM.format(lens=LENSES[lens])
        for tokens in (1200, 3500, 8000):
            try:
                r = cli.chat.completions.create(
                    model=MODEL,
                    max_tokens=tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt(s, crits)},
                    ],
                )
                v = parse(r.choices[0].message.content or "")
                if v:
                    path.write_text(
                        json.dumps({"scenario_id": s["scenario_id"], "lens": lens, **v}),
                        encoding="utf-8",
                    )
                    return 1
            except Exception:
                pass
        return 0

    with ThreadPoolExecutor(WORKERS) as ex:
        ok = list(ex.map(run, jobs))
    print(f"  attacked {sum(ok)}/{len(jobs)} (failures left uncached, retried next run)")


def stage_report() -> None:
    scen = {s["scenario_id"]: s for s in load()[0]}
    per: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in sorted(ATTACK_DIR.glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        if row["scenario_id"] in scen:
            per[row["scenario_id"]][row["lens"]] = row
    complete = {k: v for k, v in per.items() if len(v) == len(LENSES)}
    if not complete:
        raise SystemExit("no complete attack sets -- run the attack stage first")

    votes = {k: sum(1 for a in v.values() if a["disqualify"]) for k, v in complete.items()}
    suspect = sorted([k for k, n in votes.items() if n >= 2], key=lambda k: -votes[k])
    unanimous = [k for k in suspect if votes[k] == 3]
    survived = [k for k, n in votes.items() if n == 0]

    L = ["# Bridge adversarial verification — trying to disqualify what we kept\n"]
    L.append(
        f"`{MODEL}`, {len(LENSES)} independent attack lenses per scenario "
        f"({len(complete)}/{len(scen)} scenarios with a complete set). Each attempt is told to "
        f"assume the item is unfit and prove it, resolving uncertainty toward disqualification. "
        f"A scenario is suspect on a majority (>=2 of 3), since an adversarial prompt will "
        f"always manufacture some objection.\n"
    )
    L.append("\n## Headline\n")
    L.append("\n| | scenarios | share |\n|---|---:|---:|")
    L.append(f"| survived all 3 attacks | {len(survived)} | {len(survived) / len(complete):.0%} |")
    lone = sum(1 for n in votes.values() if n == 1)
    L.append(f"| 1 of 3 disqualified (noise floor) | {lone} | {lone / len(complete):.0%} |")
    L.append(f"| **suspect (>=2 of 3)** | **{len(suspect)}** "
             f"| **{len(suspect) / len(complete):.0%}** |")
    L.append(f"| unanimous (3 of 3) | {len(unanimous)} | {len(unanimous) / len(complete):.0%} |")

    L.append("\n\n## Which lens disqualifies most\n")
    L.append("\nA lens firing far more than the others is likelier to be over-eager than right.\n")
    L.append("\n| lens | disqualified | of complete |\n|---|---:|---:|")
    for lens in LENSES:
        n = sum(1 for k in complete if complete[k][lens]["disqualify"])
        L.append(f"| `{lens}` | {n} | {n / len(complete):.0%} |")

    L.append("\n\n## Severity among suspect items\n")
    L.append("\n| worst severity claimed | scenarios |\n|---|---:|")
    rank = {"fatal": 3, "serious": 2, "minor": 1}
    worst = {k: max((a["severity"] for a in complete[k].values() if a["disqualify"]),
                    key=lambda s: rank.get(s, 0), default="minor") for k in suspect}
    for sev, n in Counter(worst.values()).most_common():
        L.append(f"| {sev} | {n} |")

    L.append(f"\n\n## Unanimously disqualified — {len(unanimous)}\n")
    for k in unanimous[:40]:
        s = complete[k]
        L.append(f"\n**{k}** · student said `{scen[k]['prompt'][:44]}` · {worst[k]}")
        for lens in LENSES:
            if s[lens]["disqualify"]:
                L.append(f"- *{lens}*: {s[lens]['reason'][:230]}")

    L.append(f"\n\n## Majority (2 of 3) — {len(suspect) - len(unanimous)}\n")
    for k in [x for x in suspect if votes[x] == 2][:40]:
        s = complete[k]
        reasons = " · ".join(
            f"{ln}: {s[ln]['reason'][:110]}" for ln in LENSES if s[ln]["disqualify"]
        )
        L.append(f"- **{k}** `{scen[k]['prompt'][:34]}` — {reasons}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    out = OUT_DIR / "suspect.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            {
                "_comment": "Scenarios a majority of adversarial lenses would disqualify. NOT an "
                "exclusion list: an adversarial prompt over-fires by construction, so these are "
                "candidates for review, not decisions.",
                "classifier": MODEL,
                "lenses": list(LENSES),
                "unanimous": unanimous,
                "majority": [k for k in suspect if votes[k] == 2],
                "survived_all": survived,
                "detail": {k: complete[k] for k in suspect},
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
        fh.write("\n")
    print(f"wrote {REPORT.relative_to(ROOT)} and {out.relative_to(ROOT)}")
    print("\n".join(L[:18]))


STAGES = {"attack": stage_attack, "report": stage_report}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: verify_bridge_items.py [{'|'.join(STAGES)}]")
    STAGES[sys.argv[1]]()
