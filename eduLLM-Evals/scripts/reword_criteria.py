"""LLM-assisted rewording of flagged TutorEval criteria into judge-ready form.

The binary LLM judge (``tutor_cat/judge.py``) sees ONLY the scenario prompt +
conversation context + tutor response + one criterion string. A criterion must
therefore be a self-contained, positively framed, checkable requirement ON THE
TUTOR'S RESPONSE. This script scans ``rubrics_final.jsonl`` with the same
heuristics as ``scan_criteria_wording.py``, and for every flagged criterion asks
Claude to propose a ``decision`` (reword / split / flag_for_human / keep), the
rewritten text (``after``; a list of atomic rewrites when splitting), and a short
``rationale``. It emits a HUMAN-REVIEW SHEET in the exact shape the
``reframe_review.tsv`` / ``build_final_rubrics.py`` workflow consumes
(``bucket, criterion_id, decision, before, after, rationale``); the human edits
``decision``/``after`` and then ``build_final_rubrics.py`` assembles the reworded
bank.

The model never removes a criterion: unsalvageable or factually suspect criteria get
``flag_for_human`` (empty ``after`` + rationale) for a human to resolve. It does NOT
relabel the q-matrix and does NOT touch IRT/MIRT. Rewords/splits (and any human-approved
drops) desync the q-matrix labels for the touched IDs; re-run ``generate_qmatrix.py`` /
``verify_qmatrix.py`` on those IDs afterward (see the module README / handoff).

Routing (TrueFoundry AI Gateway) -- identical to ``generate_qmatrix.py``:
- ``ANTHROPIC_BASE_URL`` = gateway root (SDK appends ``/v1/messages``).
- ``ANTHROPIC_AUTH_TOKEN`` = TrueFoundry user key, sent as ``Authorization: Bearer``.
- ``--model`` = TrueFoundry catalog slug (default ``claude-group/claude-opus-4-8``).

Usage
-----
.. code-block:: bash

    export ANTHROPIC_BASE_URL="https://tfy.promptlens.trilogy.com"
    export ANTHROPIC_AUTH_TOKEN="<TrueFoundry user key>"
    unset ANTHROPIC_API_KEY

    # Validate the prompt/schema/plumbing on a few items first:
    python scripts/reword_criteria.py --sample 10
    # Full pass over every flagged criterion (~1.3k calls -- budget-aware!):
    python scripts/reword_criteria.py --full
    # Re-call only the blank/failed rows and merge them back into the sheet:
    python scripts/reword_criteria.py --retry-failed

Outputs
-------
- The review sheet (default ``data/TutorEval/wording_review.tsv``): one row per
  flagged criterion, ``after`` prefilled with the model's proposal for the human
  to edit. Split rewrites are joined in the ``after`` cell with `` SPLIT_DELIM``.
- ``reword_logs/<mode>/``: per-criterion request+response JSON, the system prompt,
  and ``manifest.json``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse the generator's gateway/plumbing helpers and the scanner's heuristics so the
# rewriter sees byte-identical scenario context and the identical flag set.
import generate_qmatrix as gq
import scan_criteria_wording as scan

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUBRICS = ROOT / "data" / "TutorEval" / "rubrics_final.jsonl"
DEFAULT_SCENARIOS = ROOT / "data" / "TutorEval" / "scenarios_final.jsonl"
DEFAULT_OUT_TSV = ROOT / "data" / "TutorEval" / "wording_review.tsv"
LOG_DIR = ROOT / "reword_logs"

DEFAULT_MODEL = gq.DEFAULT_MODEL
MAX_TOKENS = 1500
DEFAULT_CONCURRENCY = gq.DEFAULT_CONCURRENCY

# v2: model flags unsalvageable criteria (flag_for_human) instead of auto-dropping.
PROMPT_VERSION = "reword-v2-tutoreval-flag-not-drop"

# Delimiter used in the review sheet's `after` cell to join an atomic split into one
# TSV field. MUST match build_final_rubrics.py's SPLIT_DELIM.
SPLIT_DELIM = " ||| "

DECISIONS = ("reword", "split", "flag_for_human", "keep")

REVIEW_COLUMNS = ["bucket", "criterion_id", "decision", "before", "after", "rationale"]


# ---------------------------------------------------------------------------
# Structured output schema (only sent with --strict-schema)
# ---------------------------------------------------------------------------

REWORD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        # Always a list. reword/keep -> one string; split -> >=2 atomic strings;
        # flag_for_human -> empty list.
        "after": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["decision", "after", "rationale"],
    "additionalProperties": False,
}

OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": REWORD_JSON_SCHEMA}}


# ---------------------------------------------------------------------------
# Prompt (versioned; logged verbatim with every run)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You rewrite rubric criteria for an automated TUTORING judge. The judge grades a
single criterion BINARY (pass/fail) and sees ONLY: the scenario prompt, the prior
conversation, and the tutor's response. It NEVER sees the reference solution or any
"expected evidence". So every criterion must be a self-contained, positively framed,
checkable requirement ON THE TUTOR'S RESPONSE.

You are given one criterion, the scenario it belongs to, and the heuristic buckets a
scanner flagged it under. Decide how to fix it and return the fix.

DECISIONS
- "reword": keep one requirement, rewrite it into judge-ready form (the common case).
- "split": the criterion bundles two or more independent requirements -> return the
  atomic requirements as separate strings (one requirement each).
- "flag_for_human": the criterion is not a checkable requirement and cannot be salvaged
  by rewording (e.g. a truly optional "could" nicety, or a bare restatement of the
  student's question with nothing to grade), OR its embedded content appears factually
  wrong. Do NOT rewrite it and do NOT delete it -- leave "after" empty and explain in
  "rationale" so a human decides.
- "keep": the criterion is already fine as written (use sparingly -- it was flagged).

STYLE GUIDE (apply when rewording/splitting)
1. Frame it as a requirement on the response. Lead with a 3rd-person action verb:
   States / Explains / Identifies / Confirms / Provides / Acknowledges / Corrects /
   Points out / Recognizes / Describes / Does not ... (for must-not / withhold rules).
   Turn bare facts and observations about the student into a requirement on the tutor.
     Bad:  "Positive symptoms are symptoms most people without schizophrenia lack."
     Good: "States that positive symptoms are ones most people without schizophrenia
            do not experience."
2. Convert verdicts/observations into a "confirms X" requirement plus the content.
     Bad:  "Yes, this is correct." / "It is O(n)."
     Good: "Confirms that the student's claim is correct."
            "States that the time complexity is O(n)."
3. Remove hedging (may / might / could / not necessarily / ideally / often). If the
   hedged action is truly OPTIONAL, "flag_for_human"; otherwise state the firm requirement.
4. One requirement per criterion. If two independent tutor actions are bundled, "split".
   Do NOT split a single mathematical argument or a code snippet that merely contains
   semicolons or clauses.
5. Replace vague qualifiers (good / well / clearly / properly / appropriately) with an
   observable property, or remove the qualifier if it adds nothing gradeable.
6. Keep it reference-free: never say "matches the reference / golden / model answer".
   State the substantive requirement directly.
7. PRESERVE THE ORIGINAL MEANING. Do NOT invent new requirements, add scope, or change
   any scientific / mathematical content. Rewording changes FORM, not substance.
8. Keep it concise: a single clause, typically 5-40 words.

OUTPUT FORMAT
Return ONLY one JSON object -- no prose, no markdown, no code fences:
{
  "decision": "reword" | "split" | "flag_for_human" | "keep",
  "after": ["<rewritten requirement>"],   // one string for reword/keep,
                                           // >=2 atomic strings for split,
                                           // [] for flag_for_human
  "rationale": "one concise sentence explaining the fix"
}
For "keep", set "after" to the original text unchanged. For "flag_for_human", set "after" to []."""

SYSTEM_BLOCKS = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]


# ---------------------------------------------------------------------------
# Per-criterion user message
# ---------------------------------------------------------------------------


def build_user_content(criterion: dict, scenario: dict | None, buckets: list[str]) -> str:
    """Render scenario context + the flagged criterion + its scanner buckets."""
    lines: list[str] = []
    if scenario is not None:
        lines.append(f"SUBJECT: {scenario.get('subject')}")
        lines.append(f"USE CASE: {scenario.get('use_case')}")
        turns = scenario.get("conversation_context") or []
        if turns:
            lines.append("CONVERSATION SO FAR:")
            for turn in turns:
                lines.append(f"  [{turn.get('role')}] {turn.get('content')}")
        lines.append("TURN THE TUTOR IS EVALUATED ON (the tutor must respond to this):")
        lines.append(f"  {scenario.get('prompt')}")
    else:
        lines.append("(scenario context unavailable)")

    lines.append("")
    lines.append(f"SCANNER BUCKETS: {', '.join(buckets)}")
    lines.append("CRITERION TO FIX:")
    lines.append(f'  "{criterion.get("criterion")}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_proposal(obj: object) -> dict:
    """Validate a decoded reword proposal; raise ValueError on any shape violation."""
    if not isinstance(obj, dict):
        raise ValueError("proposal is not a JSON object")
    decision = obj.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"'decision' must be one of {DECISIONS}, got {decision!r}")
    after = obj.get("after")
    # Lenient coercion: without --strict-schema the model occasionally returns a bare
    # string (or null) instead of a list. Coerce to the list contract before validating.
    if after is None:
        after = []
    elif isinstance(after, str):
        after = [after]
    if not isinstance(after, list) or not all(isinstance(x, str) for x in after):
        raise ValueError("'after' must be a list of strings")
    after = [a.strip() for a in after if a.strip()]
    if decision == "reword" and len(after) != 1:
        raise ValueError(f"'reword' requires exactly one 'after' string, got {len(after)}")
    if decision == "split" and len(after) < 2:
        raise ValueError(f"'split' requires >=2 'after' strings, got {len(after)}")
    if decision == "keep" and len(after) != 1:
        raise ValueError(f"'keep' requires the original text in 'after', got {len(after)}")
    if decision == "flag_for_human" and after:
        raise ValueError("'flag_for_human' must have an empty 'after'")
    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("'rationale' is missing or empty")
    return {"decision": decision, "after": after, "rationale": rationale.strip()}


def after_cell(proposal: dict) -> str:
    """Render the proposal's ``after`` list into a single TSV cell."""
    return SPLIT_DELIM.join(proposal["after"])


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_dir_for(mode: str) -> Path:
    d = LOG_DIR / mode
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_item_log(mode: str, criterion_id: str, payload: dict) -> None:
    import json

    with open(log_dir_for(mode) / f"{criterion_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Reword one criterion
# ---------------------------------------------------------------------------


def reword_one(
    client, model: str, mode: str, strict_schema: bool, item: dict, scenario: dict | None
) -> tuple[dict, dict | None]:
    """
    Reword one flagged criterion via a single ``/v1/messages`` call.

    :param item: a rubric record augmented with a ``buckets`` list.
    :returns: ``(row, failure)`` -- ``row`` is the review-sheet dict (``after`` empty on
        failure); ``failure`` is ``None`` on success or ``{"criterion_id", "error"}``.
    """
    import anthropic

    cid = item["criterion_id"]
    buckets = item["buckets"]
    before = item.get("criterion", "")
    user_content = build_user_content(item, scenario, buckets)

    row = {
        "bucket": ";".join(buckets),
        "criterion_id": cid,
        "decision": "",
        "before": before,
        "after": "",
        "rationale": "",
    }

    kwargs = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_BLOCKS,
        messages=[{"role": "user", "content": user_content}],
    )
    if strict_schema:
        kwargs["output_config"] = OUTPUT_CONFIG

    try:
        response = client.messages.create(**kwargs)
        text = gq.extract_text(response)
        proposal = validate_proposal(gq.parse_json_object(text))
        row["decision"] = proposal["decision"]
        row["after"] = after_cell(proposal)
        row["rationale"] = proposal["rationale"]
        write_item_log(
            mode,
            cid,
            {
                "criterion_id": cid,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "strict_schema": strict_schema,
                "buckets": buckets,
                "request": {"system": SYSTEM_PROMPT, "user": user_content},
                "response": gq.safe_to_dict(response),
                "proposal": proposal,
            },
        )
        return row, None
    except (anthropic.APIError, ValueError) as e:
        write_item_log(
            mode,
            cid,
            {
                "criterion_id": cid,
                "model": model,
                "buckets": buckets,
                "error": str(e),
                "request": {"system": SYSTEM_PROMPT, "user": user_content},
            },
        )
        return row, {"criterion_id": cid, "error": str(e)}


# ---------------------------------------------------------------------------
# Concurrent runner
# ---------------------------------------------------------------------------


def run_items(
    client,
    model: str,
    mode: str,
    strict_schema: bool,
    concurrency: int,
    items: list[dict],
    scenarios: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Reword ``items`` with a bounded thread pool, preserving order."""
    results: dict[int, dict] = {}
    failures: list[dict] = []
    lock = threading.Lock()
    total = len(items)

    def work(idx: int, it: dict):
        scenario = scenarios.get(it.get("scenario_id"))
        return idx, reword_one(client, model, mode, strict_schema, it, scenario)

    print(
        f"{mode} mode: rewording {total} flagged criteria with {model} "
        f"(concurrency={concurrency}, strict_schema={strict_schema})."
    )

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, i, it) for i, it in enumerate(items)]
        for fut in as_completed(futures):
            idx, (row, failure) = fut.result()
            with lock:
                results[idx] = row
                if failure is not None:
                    failures.append(failure)
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  {done}/{total} done ({len(failures)} failed)")

    ordered = [results[i] for i in range(total)]
    return ordered, failures


# ---------------------------------------------------------------------------
# Flagged-item selection
# ---------------------------------------------------------------------------


def collect_flagged(rubrics: list[dict]) -> list[dict]:
    """Return rubric records that trip >=1 wording bucket, augmented with ``buckets``."""
    flagged: list[dict] = []
    for r in rubrics:
        buckets = scan.classify((r.get("criterion") or "").strip())
        if buckets:
            flagged.append({**r, "buckets": buckets})
    return flagged


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def read_review_sheet(path: Path) -> list[dict]:
    """Read an existing review sheet back into row dicts (for --retry-failed merges)."""
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_review_sheet(path: Path, rows: list[dict]) -> None:
    """Write the human-review sheet (reframe_review.tsv shape + rationale)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            clean = {
                k: (r.get(k, "") or "").replace("\t", " ").replace("\n", " ")
                for k in REVIEW_COLUMNS
            }
            w.writerow(clean)


def finish(
    mode: str,
    model: str,
    strict_schema: bool,
    rows: list[dict],
    failures: list[dict],
    out_path: Path,
) -> None:
    """Write the review sheet, the run manifest, the system prompt, and a summary."""
    import json

    write_review_sheet(out_path, rows)

    decisions = collections.Counter(r["decision"] for r in rows if r["decision"])
    bucket_hits = collections.Counter()
    for r in rows:
        for b in r["bucket"].split(";"):
            if b:
                bucket_hits[b] += 1

    log_dir = log_dir_for(mode)
    with open(log_dir / f"system_prompt_{PROMPT_VERSION}.txt", "w", encoding="utf-8") as f:
        f.write(SYSTEM_PROMPT + "\n")

    manifest = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "mode": mode,
        "strict_schema": strict_schema,
        "timestamp_utc": gq.now_iso(),
        "total": len(rows),
        "succeeded": len(rows) - len(failures),
        "failed": len(failures),
        "decision_counts": dict(decisions),
        "bucket_hits": dict(bucket_hits),
        "split_delim": SPLIT_DELIM,
        "failures": failures,
    }
    with open(log_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(rows)} review rows -> {out_path}")
    print(f"Succeeded {len(rows) - len(failures)}, failed {len(failures)}.")
    print("Decision counts: " + ", ".join(f"{k}={v}" for k, v in decisions.most_common()))
    print(f"Logs + manifest: {log_dir}")

    if mode == "sample":
        print("\n--- example proposals ---")
        shown = 0
        for r in rows:
            if not r["decision"]:
                continue
            print(f"[{r['criterion_id']}] ({r['bucket']})")
            print(f"  before: {r['before']}")
            print(f"  {r['decision']}: {r['after']}")
            print(f"  rationale: {r['rationale']}")
            print()
            shown += 1
            if shown >= 5:
                break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reword flagged TutorEval criteria into judge-ready form via the "
        "TrueFoundry AI Gateway; emit a human-review sheet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Reword N evenly-spaced flagged criteria (prompt validation).",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Reword every flagged criterion via a bounded thread pool.",
    )
    group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-call only the blank/failed rows in the existing --out sheet and merge "
        "the results back in (leaves succeeded rows untouched).",
    )
    parser.add_argument(
        "--rubrics",
        type=Path,
        default=DEFAULT_RUBRICS,
        help="Input rubric JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=DEFAULT_SCENARIOS,
        help="Input scenarios JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_TSV,
        help="Review-sheet TSV to write (default: %(default)s).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="TrueFoundry catalog slug (default: %(default)s)."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Gateway base URL (else ANTHROPIC_BASE_URL env). SDK appends "
        "/v1/messages, so use the root.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max concurrent requests (default: %(default)s).",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Send output_config json_schema to enforce the response shape.",
    )
    args = parser.parse_args()

    import os

    import anthropic

    rubrics = gq.read_jsonl(args.rubrics)
    scenarios = gq.index_scenarios(gq.read_jsonl(args.scenarios))
    flagged = collect_flagged(rubrics)
    print(f"{len(flagged)}/{len(rubrics)} criteria flagged by the wording scanner.")

    existing_rows: list[dict] = []
    if args.retry_failed:
        mode = "retry"
        if not args.out.exists():
            parser.error(f"--retry-failed needs an existing sheet at {args.out}")
        existing_rows = read_review_sheet(args.out)
        failed_ids = [
            r["criterion_id"] for r in existing_rows if not (r.get("decision") or "").strip()
        ]
        by_id = {it["criterion_id"]: it for it in flagged}
        items = [by_id[c] for c in failed_ids if c in by_id]
        print(f"retrying {len(items)} blank/failed rows from {args.out}.")
        if not items:
            print("nothing to retry.")
            return
    elif args.sample is not None:
        items = gq.stride_sample(flagged, args.sample)
        mode = "sample"
    else:
        items = flagged
        mode = "full"

    client_kwargs = {"max_retries": 5, "timeout": 120.0}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not auth_token:
        parser.error("set ANTHROPIC_AUTH_TOKEN to your TrueFoundry user key")
    os.environ.pop("ANTHROPIC_API_KEY", None)
    client_kwargs["auth_token"] = auth_token
    client_kwargs["api_key"] = None

    client = anthropic.Anthropic(**client_kwargs)

    rows, failures = run_items(
        client, args.model, mode, args.strict_schema, args.concurrency, items, scenarios
    )
    if args.retry_failed:
        new_by_id = {r["criterion_id"]: r for r in rows}
        rows = [new_by_id.get(r["criterion_id"], r) for r in existing_rows]
    finish(mode, args.model, args.strict_schema, rows, failures, args.out)


if __name__ == "__main__":
    main()
