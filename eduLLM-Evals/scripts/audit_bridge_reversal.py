"""Argue the opposite case, then adjudicate proposed Bridge restorations blind.

The audited cutting passes removed 470 of the 642 rows that survived deterministic checks;
the other 58 source rows were dropped before ids were assigned. Every audit was looking for
reasons to remove, so this script measured false-positive cuts by arguing the other side.

So this inverts the burden again. The prompt assumes the item is USABLE and asks the model to
prove it, resolving uncertainty toward keeping. A rescue is a claim that a competent tutor
could diagnose the student's error from the visible text -- the same bar the cuts were made
against, argued from the other direction.

Sampling is per-reason rather than proportional, so each removal reason gets its own
false-positive estimate instead of one pooled number dominated by the largest category. Rates
are therefore NOT comparable to the overall 470 audited cuts; read them per row.

    python scripts/audit_bridge_reversal.py sample    # stratified draw from the 470 audited cuts
    python scripts/audit_bridge_reversal.py rescue    # one call per sampled scenario
    python scripts/audit_bridge_reversal.py report    # per-reason false-positive rates
    python scripts/audit_bridge_reversal.py blind     # 3 tutor-visible-only votes/candidate
    python scripts/audit_bridge_reversal.py blind_report
    python scripts/audit_bridge_reversal.py publish   # committed compact decision ledger

Resumable: every verdict is cached, so re-running fills gaps only.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "staging" / "bridge_reversal"
SAMPLE = OUT_DIR / "sample.json"
VERDICT_DIR = OUT_DIR / "verdicts"
REPORT = OUT_DIR / "report.md"
# The reinstatement list this audit produced, read back by the blind stage to re-check it.
RESTORE_PATH = ROOT / "data" / "Bridge" / "restored.json"
BLIND_DECISION_PATH = ROOT / "data" / "Bridge" / "blind_readjudication.json"
EXCLUSION_PATHS = (
    ROOT / "data" / "Bridge" / "visual_exclusions.json",
    ROOT / "data" / "Bridge" / "item_exclusions.json",
    ROOT / "data" / "Bridge" / "verify_exclusions.json",
)

BASE_URL = "https://tfy.promptlens.trilogy.com/api/llm/api/inference/openai"
MODEL = "openai-group/gpt-5.5"
WORKERS = 12
# Per-stratum cap. 25 gives each reason its own false-positive estimate cheaply;
# BRIDGE_REVERSAL_N=all reviews every removal, which is what you need once an estimate has
# shown a category is unsafe and the question becomes which individual items to restore.
_PR = os.environ.get("BRIDGE_REVERSAL_N", "25")
PER_REASON = 10**9 if _PR == "all" else int(_PR)
SEED = 7

SYSTEM = """You are reviewing items that were CUT from a math-tutoring benchmark, deciding
whether each was cut in error. Assume the item is usable and try to prove it.

The items are real tutoring transcripts cut at a student mistake. A tutor model sees only the
conversation and the student's final turn, and must write the next tutor turn: identify the
student's error and guide them to fix it. These transcripts come from live sessions held over
a shared whiteboard that was never saved, so anything the tutor and student were looking at
is absent from the text.

The bar the item was cut against, argued from your side: could a competent tutor, seeing ONLY
the visible text, name something specific the student got wrong and give a useful corrective
next turn? If yes, the cut was a mistake and you should say so.

Argue for keeping. Resolve genuine uncertainty toward keeping. But do not rescue an item by:
  - guessing what the missing problem probably was, or inferring it from the expert reply --
    the tutor model never sees that reply, so an error only knowable from it does not count
  - treating "the tutor could ask a clarifying question" as a diagnosis; every unusable item
    would pass that test, so it cannot distinguish anything
  - calling a bare assent ("yes", "done", "ok") or a bare option label ("b", "3") an answer
  - relying on the student being wrong when the visible text does not show what was asked

State plainly if the item really is unusable. A wrong rescue is worse than a missed one here.

Return ONLY:
{"keep": true|false,
 "the_error": "<the specific error a tutor could name from the visible text; empty if none>",
 "confidence": "high"|"medium"|"low"}"""


def _ingester():
    spec = importlib.util.spec_from_file_location("ib", ROOT / "scripts" / "ingest_bridge.py")
    ib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ib)
    return ib


def original_exclusion_reasons() -> dict[str, str]:
    """Reasons from immutable audit ledgers, including subsequently restored ids."""
    reasons = {}
    for path in EXCLUSION_PATHS:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc.get("excluded", []):
            reasons.setdefault(row["scenario_id"], row["reason"])
    return reasons


def all_scenarios_with_reasons() -> tuple[dict[str, dict], dict[str, str]]:
    """Rebuild the bank with no exclusions, so removed scenarios are recoverable in full.

    dropped.jsonl records why each was cut but not its text; the ingester is deterministic,
    so rebuilding with EXCLUSION_PATHS emptied reproduces every removed scenario exactly.
    """
    ib = _ingester()
    ib.EXCLUSION_PATHS = ()
    scen, _rub, _dropped = ib.build()
    by_id = {s["scenario_id"]: s for s in scen}
    # Read the immutable source ledgers, not dropped.jsonl: the latter is the *final*
    # 450-row partition and no longer includes the 78 exclusions this audit overturned.
    return by_id, original_exclusion_reasons()


def client():
    from openai import OpenAI

    key = os.environ.get("TFY_API_KEY")
    if not key:
        raise SystemExit("TFY_API_KEY is not set")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=180)


def stage_sample() -> None:
    _by_id, reasons = all_scenarios_with_reasons()
    by_reason: dict[str, list[str]] = defaultdict(list)
    for sid, r in reasons.items():
        by_reason[r].append(sid)
    rng = random.Random(SEED)
    picked: dict[str, list[str]] = {}
    for r, ids in sorted(by_reason.items()):
        ids = sorted(ids)
        picked[r] = sorted(rng.sample(ids, min(PER_REASON, len(ids))))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text(json.dumps(picked, indent=2), encoding="utf-8")
    total = sum(len(v) for v in picked.values())
    print(f"sampled {total} of {len(reasons)} removed scenarios, up to {PER_REASON} per reason")
    for r, ids in picked.items():
        print(f"  {r:<36} {len(ids):>3} of {len(by_reason[r])}")


def user_prompt(s: dict) -> str:
    ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in s["conversation_context"])
    return (
        f"<visible_to_the_tutor_model>\n{ctx}\n[student] {s['prompt']}\n"
        f"</visible_to_the_tutor_model>\n\n"
        f"<expert_reply_NOT_visible_to_the_tutor_model>\n{s['reference_solution']}\n"
        f"</expert_reply_NOT_visible_to_the_tutor_model>\n\n"
        f"<lesson_topic>{s.get('lesson_topic')}</lesson_topic>\n\n"
        "Was this item cut in error? Argue for keeping it."
    )


def parse(text: str) -> dict | None:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(obj.get("keep"), bool):
        return None
    return {
        "keep": obj["keep"],
        "the_error": str(obj.get("the_error") or "")[:300],
        "confidence": str(obj.get("confidence") or "medium").lower(),
    }


def stage_rescue() -> None:
    cli = client()
    by_id, _ = all_scenarios_with_reasons()
    picked = json.loads(SAMPLE.read_text(encoding="utf-8"))
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for reason, ids in picked.items():
        for sid in ids:
            p = VERDICT_DIR / f"{sid}.json"
            if not p.exists() and sid in by_id:
                jobs.append((by_id[sid], reason, p))
    print(f"rescue: {len(jobs)} pending", flush=True)
    if not jobs:
        return

    def run(job):
        s, reason, path = job
        for tokens in (1200, 3500, 8000):
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
                        json.dumps({"scenario_id": s["scenario_id"], "reason": reason, **v}),
                        encoding="utf-8",
                    )
                    return 1
            except Exception:
                pass
        return 0

    with ThreadPoolExecutor(WORKERS) as ex:
        ok = list(ex.map(run, jobs))
    print(f"  judged {sum(ok)}/{len(jobs)}")


def stage_report() -> None:
    by_id, _ = all_scenarios_with_reasons()
    v = {}
    for p in sorted(VERDICT_DIR.glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        v[row["scenario_id"]] = row
    if not v:
        raise SystemExit("no verdicts -- run the rescue stage first")

    by_reason: dict[str, list[dict]] = defaultdict(list)
    for row in v.values():
        by_reason[row["reason"]].append(row)

    L = ["# Bridge removal-reversal audit — arguing the cuts were wrong\n"]
    L.append(
        f"`{MODEL}`, one call per sampled scenario, {len(v)} of the 470 audited cuts "
        f"reviewed. The prompt assumes each item is usable and asks the model to prove it, "
        f"resolving uncertainty toward keeping — the inverse of every pass that made the cuts. "
        f"A rescue means the model claims a tutor could diagnose the error from the visible "
        f"text alone.\n"
    )
    L.append(
        "\nSampling is per-reason (up to 25 each), not proportional, so each row is its own "
        "estimate and the rates do NOT aggregate to a figure for all 470 audited cuts.\n"
    )
    L.append("\n## False-positive rate by removal reason\n")
    L.append("\n| reason | sampled | rescued | rate | high-conf |"
             "\n|---|---:|---:|---:|---:|")
    order = sorted(by_reason, key=lambda r: -len(by_reason[r]))
    for r in order:
        rows = by_reason[r]
        k = [x for x in rows if x["keep"]]
        hi = [x for x in k if x["confidence"] == "high"]
        L.append(f"| `{r}` | {len(rows)} | {len(k)} | {len(k) / len(rows):.0%} | {len(hi)} |")
    allk = [x for x in v.values() if x["keep"]]
    nhi = sum(1 for x in allk if x["confidence"] == "high")
    L.append(f"| **sampled total** | **{len(v)}** | **{len(allk)}** "
             f"| **{len(allk) / len(v):.0%}** | **{nhi}** |")

    L.append("\n\n## Rescued items — each is a cut to re-examine\n")
    for r in order:
        k = [x for x in by_reason[r] if x["keep"]]
        if not k:
            continue
        L.append(f"\n**`{r}`** — {len(k)} rescued\n")
        for x in sorted(k, key=lambda x: x["scenario_id"]):
            s = by_id[x["scenario_id"]]
            L.append(f"- **{x['scenario_id']}** [{x['confidence']}] student said "
                     f"`{s['prompt'][:40]}` — {x['the_error'][:170]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    out = OUT_DIR / "rescued.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            {
                "_comment": "Scenarios a keep-biased reviewer would restore. Candidates for "
                "re-examination, not an automatic restore list: this prompt is biased toward "
                "keeping by construction, exactly as the cutting passes were biased the other "
                "way.",
                "classifier": MODEL,
                "sampled": len(v),
                "rescued": {x["scenario_id"]: x for x in allk},
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
        fh.write("\n")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    print("\n".join(L[:14]))


# ---------------------------------------------------------------------------
# blind -- re-adjudicate reinstatements with the leaky context REMOVED
# ---------------------------------------------------------------------------
# The rescue stage above showed the reviewer the expert reply and the lesson_topic, with an
# instruction not to infer the problem from them. That instruction did not hold. Three of the
# reinstatements it produced are provably leaked: bridge_0419 ("nhw") was rescued because the
# answer "cannot be a multiple of 10", and bridge_0246 ("6,198") because it "cannot be the
# product of two two-digit numbers" -- neither fact appears anywhere a tutor model can see.
# bridge_0364 is worse: the verdict text states that the only visible question is whether to
# enable audio, and still returns keep.
#
# So the fix is to remove the information rather than forbid its use. This stage shows ONLY
# what a tutor model sees, and votes three times: a reinstatement stands on a majority.
BLIND_DIR = OUT_DIR / "blind"
BLIND_VOTES = 3

BLIND_SYSTEM = """You are checking whether one item belongs in a math-tutoring benchmark.

You see a tutoring conversation and the student's final turn. That is ALL a tutor model will
see. There is no figure, no worksheet, no problem statement beyond what is written here --
these are transcripts of sessions held over a whiteboard that was never saved.

Answer one question: can you name a SPECIFIC thing the student got wrong, using only the text
shown?

Say yes only if you can state the error concretely. The error may be legible from the form of
the answer rather than its value -- a linear unit given for an area, a single number where an
ordered pair was asked for, a measurement type where a unit was asked for, an answer that
contradicts something the student themselves said earlier. Those count.

Say no if you would have to know the problem, see a figure, or guess. In particular say no
when:
  - the student's turn is a bare assent, a bare option label, or gibberish
  - the answer is a plain number and nothing shown says what was asked
  - the visible question is not a maths question at all
  - naming the error requires a fact that does not appear above

Do not speculate about what the problem probably was. If you find yourself supplying the
question from your own knowledge of what such lessons usually cover, the answer is no.

Return ONLY:
{"diagnosable": true|false, "the_error": "<the specific error, quoting the text; empty if none>"}"""


def blind_prompt(s: dict) -> str:
    """Only the tutor-visible turns. No reference, no lesson_topic, no labels."""
    ctx = "\n".join(f"[{t['role']}] {t['content']}" for t in s["conversation_context"])
    return (
        f"<conversation>\n{ctx}\n[student] {s['prompt']}\n</conversation>\n\n"
        "Can you name a specific error the student made, from this text alone?"
    )


def blind_parse(text: str) -> dict | None:
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(obj.get("diagnosable"), bool):
        return None
    return {"diagnosable": obj["diagnosable"], "the_error": str(obj.get("the_error") or "")[:300]}


def _adjudication_set() -> list[str]:
    """Every id previously proposed/salvaged while leaky context was visible.

    The committed decision ledger is authoritative because restored.json intentionally keeps
    only the 78 upheld exclusions and cannot reconstruct the 49 overturned candidates. The
    fallback exists only for bootstrapping the first blind run.
    """
    if BLIND_DECISION_PATH.is_file():
        doc = json.loads(BLIND_DECISION_PATH.read_text(encoding="utf-8"))
        ids = [row["scenario_id"] for row in doc.get("candidates", [])]
        if ids:
            return sorted(set(ids))

    ids = set()
    if RESTORE_PATH.is_file():
        ids |= {r["scenario_id"] for r in
                json.loads(RESTORE_PATH.read_text(encoding="utf-8")).get("restored", [])}
    dec = ROOT / "staging" / "bridge_visual_audit" / "reaudit_decision.json"
    if dec.is_file():
        ids |= {r["scenario_id"] for r in
                json.loads(dec.read_text(encoding="utf-8")).get("salvage", [])}
    return sorted(ids)


def stage_blind() -> None:
    cli = client()
    by_id, _ = all_scenarios_with_reasons()
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sid in _adjudication_set():
        if sid not in by_id:
            continue
        for vote in range(BLIND_VOTES):
            p = BLIND_DIR / f"{sid}__v{vote}.json"
            if not p.exists():
                jobs.append((by_id[sid], vote, p))
    print(f"blind: {len(jobs)} pending "
          f"({len(_adjudication_set())} ids x {BLIND_VOTES} votes)", flush=True)
    if not jobs:
        return

    def run(job):
        s, vote, path = job
        for tokens in (900, 2500, 6000):
            try:
                r = cli.chat.completions.create(
                    model=MODEL,
                    max_tokens=tokens,
                    messages=[
                        {"role": "system", "content": BLIND_SYSTEM},
                        {"role": "user", "content": blind_prompt(s)},
                    ],
                )
                v = blind_parse(r.choices[0].message.content or "")
                if v:
                    path.write_text(
                        json.dumps({"scenario_id": s["scenario_id"], "vote": vote, **v}),
                        encoding="utf-8",
                    )
                    return 1
            except Exception:
                pass
        return 0

    with ThreadPoolExecutor(WORKERS) as ex:
        ok = list(ex.map(run, jobs))
    print(f"  judged {sum(ok)}/{len(jobs)}")


def stage_blind_report() -> None:
    by_id, reasons = all_scenarios_with_reasons()
    votes: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(BLIND_DIR.glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        votes[row["scenario_id"]].append(row)
    complete = {k: v for k, v in votes.items() if len(v) >= BLIND_VOTES}
    if not complete:
        raise SystemExit("no complete blind vote sets -- run the blind stage first")

    yes = {k: sum(1 for x in v if x["diagnosable"]) for k, v in complete.items()}
    upheld = sorted([k for k, n in yes.items() if n >= 2])
    overturned = sorted([k for k, n in yes.items() if n < 2])

    print(f"blind re-adjudication of {len(complete)} reinstated/salvaged ids, "
          f"{BLIND_VOTES} votes each, majority rules")
    print(f"  UPHELD (still diagnosable blind):  {len(upheld)}")
    print(f"  OVERTURNED (was leakage):          {len(overturned)}")
    print(f"  unanimous upheld: {sum(1 for k in upheld if yes[k] == BLIND_VOTES)}"
          f"   unanimous overturned: {sum(1 for k in overturned if yes[k] == 0)}")
    print("\n  by the reason each was originally cut for:")
    up = set(upheld)
    for r in sorted({reasons.get(k, "round2_salvage") for k in complete}):
        ids = [k for k in complete if reasons.get(k, "round2_salvage") == r]
        print(f"    {r:<34} {len([k for k in ids if k in up]):>3}/{len(ids):<4} upheld")
    decision = {
        "_comment": "Blind re-adjudication of every id reinstated or salvaged while the "
        "expert reply and lesson_topic were visible. Those fields are removed here and "
        "each id gets three independent votes; a majority upholds. `overturned` are "
        "reinstatements that rested on leaked context and should go back out.",
        "classifier": MODEL,
        "votes_per_item": BLIND_VOTES,
        "upheld": upheld,
        "overturned": overturned,
        "detail": {
            k: {
                "yes_votes": yes[k],
                "errors": [x["the_error"] for x in complete[k] if x["diagnosable"]],
            }
            for k in complete
        },
    }
    out = OUT_DIR / "blind_decision.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(decision, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    _publish_blind_decision(decision)


def _publish_blind_decision(decision: dict) -> None:
    """Write a compact, committed ledger from the ignored staging decision."""
    reasons = original_exclusion_reasons()
    upheld = set(decision.get("upheld", []))
    overturned = set(decision.get("overturned", []))
    candidate_ids = sorted(upheld | overturned)
    details = decision.get("detail", {})
    controls = sorted(upheld - set(reasons))
    candidates = []
    for sid in candidate_ids:
        detail = details.get(sid, {})
        candidates.append({
            "scenario_id": sid,
            "origin": "audited_exclusion" if sid in reasons else "round2_salvage_control",
            "was_cut_for": reasons.get(sid),
            "yes_votes": int(detail.get("yes_votes", 0)),
            "decision": "upheld" if sid in upheld else "overturned",
            "blind_errors": list(detail.get("errors", [])),
        })
    payload = {
        "_comment": "Committed evidence for Bridge v8 blind re-adjudication. The expert "
        "reply, lesson_topic, and labels were removed; three independent votes saw only "
        "tutor-visible text. Majority (at least 2/3) upheld. Six upheld rows were already-kept "
        "round-2 salvage controls, so 84 upheld decisions produced 78 actual restorations.",
        "generated_by": "scripts/audit_bridge_reversal.py blind_report/publish",
        "classifier": decision.get("classifier", MODEL),
        "votes_per_item": int(decision.get("votes_per_item", BLIND_VOTES)),
        "candidate_count": len(candidate_ids),
        "upheld_count": len(upheld),
        "overturned_count": len(overturned),
        "restored_count": len(upheld & set(reasons)),
        "already_kept_control_count": len(controls),
        "already_kept_controls": controls,
        "candidates": candidates,
    }
    BLIND_DECISION_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {BLIND_DECISION_PATH.relative_to(ROOT)}")


def stage_publish() -> None:
    """Publish an existing ignored blind decision without making any API calls."""
    source = OUT_DIR / "blind_decision.json"
    if not source.is_file():
        raise SystemExit(f"missing {source}; run blind_report first")
    _publish_blind_decision(json.loads(source.read_text(encoding="utf-8")))


STAGES = {"sample": stage_sample, "rescue": stage_rescue, "report": stage_report,
          "blind": stage_blind, "blind_report": stage_blind_report,
          "publish": stage_publish}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: audit_bridge_reversal.py [{'|'.join(STAGES)}]")
    STAGES[sys.argv[1]]()
