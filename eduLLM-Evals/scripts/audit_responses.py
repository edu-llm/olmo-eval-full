"""Audit tutor-model response shards for systemic generation failures.

Reproduces every figure in RESPONSE-QA-FINDINGS.md and doubles as the regression
check after the fixes land.

    uv run python eduLLM-Evals/TutorModelRun8-9F/audit_responses.py <shard-root>

<shard-root> is scanned recursively for `*.jsonl`; the parent directory of each
file is taken as the benchmark name and the file stem as the model name.

Pass condition after fixes: no `output_empty`, no `lost_system_header`,
no `transcript_leak`, and `loop_severe` in low single digits for tiny base models.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORD = re.compile(r"\S+")
ROLE_LEAK = re.compile(r"\n\s*(Student|System|Tutor|User|Assistant|Human)\s*:", re.I)
SYS_PREFIX = "System:"
GEN_CUE = "Tutor:"
END_PUNCT = re.compile(r"[.?!:;\"'\)\]}\u00bb\u201d`]\s*$")
MIDWORD = re.compile(r"[A-Za-z]{2}\s*$")

# Declared context windows, for cross-checking what the run recorded. Extend as
# the roster grows; a model absent here is simply not checked.
REAL_WINDOW = {
    "openai-community_gpt2-large": 1024,
    "openai-community_gpt2-xl": 1024,
    "EleutherAI_pythia-410m": 2048,
    "EleutherAI_pythia-1b": 2048,
    "EleutherAI_pythia-1.4b": 2048,
    "bigscience_bloom-1b1": 2048,
    "bigscience_bloom-1b7": 2048,
    "bigscience_bloomz-1b7": 2048,
    "facebook_opt-1.3b": 2048,
    "HuggingFaceTB_SmolLM2-135M": 8192,
    "HuggingFaceTB_SmolLM2-360M": 8192,
    "HuggingFaceTB_SmolLM2-1.7B": 8192,
    "Qwen_Qwen2.5-0.5B": 32768,
    "Qwen_Qwen2.5-1.5B": 32768,
    "Qwen_Qwen2.5-3B": 32768,
}


def ngrams(ws, n):
    return [tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)]


def rep_ratio(ws, n=4):
    g = ngrams(ws, n)
    return 1.0 - len(set(g)) / len(g) if g else 0.0


def top_ngram_count(ws, n=10):
    g = ngrams(ws, n)
    return Counter(g).most_common(1)[0][1] if g else 0


def tail_cycle(ws, window=160, max_period=40):
    tail = ws[-window:]
    if len(tail) < 40:
        return None
    for p in range(1, max_period + 1):
        if len(tail) < 3 * p:
            break
        if all(tail[i] == tail[i % p] for i in range(len(tail))):
            return p
    return None


def prompt_body(p: str) -> str:
    """Prompt minus the trailing generation cue."""
    i = p.rfind(GEN_CUE)
    return p[:i] if i >= 0 else p


def last_word(p: str) -> str | None:
    b = prompt_body(p).rstrip()
    if not b or END_PUNCT.search(b) or not MIDWORD.search(b):
        return None
    m = re.search(r"([A-Za-z]+)\s*$", b)
    return m.group(1).lower() if m else None


def ends_midword(p: str, vocab: set[str] | None = None) -> bool:
    """ADVISORY: the prompt ends without terminal punctuation on a token not seen
    completed elsewhere in the corpus.

    This over-reports and is not a pass/fail signal. Plenty of prompts legitimately
    end on a bare word — BiGGen's field style (`Phenomenon: Increased intensity of
    hurricanes`) and list endings (`...Notre Dame Cathedral`) both do — and short
    prompts give the corpus no second occurrence to vouch for the final token.
    Distinguishing those from a genuine clip (`...the tree's balanc`) needs a real
    dictionary. Treat the flagged rows as a list to eyeball, not as a count.
    """
    w = last_word(p)
    if w is None:
        return False
    if vocab is None:
        return True
    return w not in vocab


COMPLETE_WORD = re.compile(r"([A-Za-z]{2,})(?=[\s.,;:!?'\")\]}])")


def build_vocab(all_prompts) -> set[str]:
    """Words observed *completed* (followed by space or punctuation) anywhere in
    the corpus. Used to tell a clipped fragment from a legitimate final word.

    Each prompt's own trailing token is dropped first: a fragment like `chang` is
    followed by a newline, so harvesting it would let it vouch for itself.
    """
    vocab: set[str] = set()
    for p in all_prompts:
        b = re.sub(r"[A-Za-z]+$", "", prompt_body(p).rstrip())
        vocab.update(m.group(1).lower() for m in COMPLETE_WORD.finditer(b))
    return vocab


def audit_row(d: dict, vocab: set[str] | None = None) -> tuple[set[str], dict]:
    flags: set[str] = set()
    out = d.get("Output")
    fin = (d.get("Finish Reason") or "").lower()
    pt, ot, mml = d.get("Prompt Tokens"), d.get("Output Tokens"), d.get("Max Model Len")
    mnt = (d.get("Generation Params") or {}).get("max_new_tokens")

    ws = WORD.findall(out) if out else []
    if out is None:
        flags.add("output_null")
    elif not out.strip():
        flags.add("output_empty")
    elif len(out.strip()) < 15:
        flags.add("output_tiny")

    if "error" in fin:
        flags.add("finish_error")
    elif fin == "length":
        flags.add("finish_length")

    if d.get("Truncated"):
        flags.add("truncated_flag")
        # P0-1: does the truncated prompt still carry the task instruction?
        if not (d.get("Rendered Prompt") or "").startswith(SYS_PREFIX):
            flags.add("lost_system_header")
    if d.get("Issue"):
        flags.add("issue_flag")

    # P0-2: exactly one token means the model emitted EOS immediately
    if isinstance(ot, int):
        if ot <= 1:
            flags.add("single_token_output")
        if isinstance(mnt, int) and mnt and ot >= mnt:
            flags.add("output_hit_cap")
    if isinstance(pt, int) and pt <= 2:
        flags.add("prompt_collapsed")
    if isinstance(pt, int) and isinstance(mml, int) and mml and pt >= mml:
        flags.add("prompt_at_window")

    if len(ws) >= 60:
        rr = rep_ratio(ws)
        if rr >= 0.7:
            flags.add("loop_severe")
        elif rr >= 0.45:
            flags.add("loop_moderate")
        if top_ngram_count(ws) >= 8:
            flags.add("ngram10_repeat")
    if tail_cycle(ws) is not None:
        flags.add("tail_cycle")
    if out and ROLE_LEAK.search(out):
        flags.add("transcript_leak")
    if (d.get("Rendered Prompt") or "") and ends_midword(d["Rendered Prompt"], vocab):
        flags.add("prompt_end_advisory")

    return flags, {"rr": rep_ratio(ws), "nwords": len(ws)}


FLAG_ORDER = [
    "finish_error",
    "output_null",
    "output_empty",
    "single_token_output",
    "output_tiny",
    "issue_flag",
    "truncated_flag",
    "lost_system_header",
    "prompt_collapsed",
    "prompt_at_window",
    "finish_length",
    "output_hit_cap",
    "loop_severe",
    "loop_moderate",
    "ngram10_repeat",
    "tail_cycle",
    "transcript_leak",
    "prompt_end_advisory",
]
# Flags that must be zero for a run to be considered clean.
BLOCKING = [
    "finish_error",
    "output_null",
    "output_empty",
    "single_token_output",
    "lost_system_header",
    "prompt_collapsed",
    "transcript_leak",
]


def load(p: Path) -> list[dict]:
    rows, bad = [], 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--examples", type=int, default=2, help="examples per pathology")
    args = ap.parse_args()

    shards = sorted(args.root.rglob("*.jsonl"))
    if not shards:
        print(f"no *.jsonl under {args.root}")
        return 2

    data = {}
    scen_union = defaultdict(set)
    loaded = {}
    for p in shards:
        loaded[p] = load(p)
    # One vocabulary over every prompt in the run, so a clipped final token can be
    # told apart from a word that simply ends without punctuation.
    vocab = build_vocab(d.get("Rendered Prompt") or "" for rows, _ in loaded.values() for d in rows)

    for p in shards:
        bench, model = p.parent.name, p.stem
        rows, bad = loaded[p]
        flags = Counter()
        examples = defaultdict(list)
        scen = []
        for d in rows:
            scen.append(d.get("Scenario"))
            fl, det = audit_row(d, vocab)
            for f in fl:
                flags[f] += 1
                if len(examples[f]) < args.examples:
                    examples[f].append((d, det))
        scen_union[bench].update(x for x in scen if x)
        data[(bench, model)] = dict(
            rows=rows,
            bad=bad,
            flags=flags,
            examples=examples,
            scen=set(x for x in scen if x),
            dupes=len(scen) - len(set(scen)),
        )

    print("=" * 120)
    print("COVERAGE")
    print("=" * 120)
    for bench, union in sorted(scen_union.items()):
        print(f"\n{bench}: {len(union)} distinct scenarios")
        print(f"  {'model':<44}{'rows':>7}{'dupes':>7}{'missing':>9}{'bad json':>10}")
        for (b, m), s in sorted(data.items()):
            if b == bench:
                print(
                    f"  {m:<44}{len(s['rows']):>7}{s['dupes']:>7}"
                    f"{len(union - s['scen']):>9}{s['bad']:>10}"
                )

    print()
    print("=" * 120)
    print("FLAG MATRIX (rows affected, % of shard)")
    print("=" * 120)
    present = [f for f in FLAG_ORDER if any(s["flags"].get(f) for s in data.values())]
    print(f"{'shard':<46}" + "".join(f"{f[:14]:>16}" for f in present))
    for (b, m), s in sorted(data.items()):
        line = f"{b + '/' + m:<46}"
        for f in present:
            v = s["flags"].get(f, 0)
            line += f"{(str(v) + f' ({100 * v / max(1, len(s["rows"])):.0f}%)') if v else '-':>16}"
        print(line)

    print()
    print("=" * 120)
    print("CONFIG / TOKEN STATS")
    print("=" * 120)
    print(
        f"{'shard':<46}{'chatT':>8}{'window':>9}{'real':>8}{'ptok med/max':>16}"
        f"{'otok med/max':>16}{'rep4':>8}"
    )
    window_problems = []
    for (b, m), s in sorted(data.items()):
        rows = s["rows"]
        pts = sorted(
            d["Prompt Tokens"] for d in rows if isinstance(d.get("Prompt Tokens"), int)
        ) or [0]
        ots = sorted(
            d["Output Tokens"] for d in rows if isinstance(d.get("Output Tokens"), int)
        ) or [0]
        mmls = Counter(d.get("Max Model Len") for d in rows)
        ct = Counter(d.get("Chat Template Applied") for d in rows)
        rr = st.mean([rep_ratio(WORD.findall(d.get("Output") or "")) for d in rows]) if rows else 0
        rec = list(mmls)[0] if len(mmls) == 1 else "MIXED"
        real = REAL_WINDOW.get(m)
        if isinstance(rec, int) and real and rec > real:
            window_problems.append((f"{b}/{m}", rec, real))
        print(
            f"{b + '/' + m:<46}{','.join(map(str, ct)):>8}{str(rec):>9}"
            f"{str(real or '?'):>8}{f'{pts[len(pts) // 2]}/{pts[-1]}':>16}"
            f"{f'{ots[len(ots) // 2]}/{ots[-1]}':>16}{rr:>8.3f}"
        )
        gps = Counter(json.dumps(d.get("Generation Params") or {}, sort_keys=True) for d in rows)
        if len(gps) > 1:
            mnts = sorted({json.loads(k).get("max_new_tokens") for k in gps})
            print(
                f"    note: per-row max_new_tokens varies over {mnts[:6]}"
                f"{' ...' if len(mnts) > 6 else ''}  (min={mnts[0]}, max={mnts[-1]})"
            )

    if window_problems:
        print("\n  *** CONTEXT WINDOW OVER-DECLARED (P1-2) ***")
        for name, rec, real in window_problems:
            print(f"    {name:<52} recorded={rec} but real window={real}")

    print()
    print("=" * 120)
    print("USABLE-ROW ESTIMATE  (excludes empty, truncated, severe loop, <15 chars)")
    print("=" * 120)
    print(f"{'shard':<46}{'rows':>8}{'unusable':>10}{'usable':>9}{'usable %':>10}")
    for (b, m), s in sorted(data.items()):
        bad = 0
        for d in s["rows"]:
            out = (d.get("Output") or "").strip()
            ws = WORD.findall(out)
            if (
                not out
                or d.get("Truncated")
                or len(out) < 15
                or (len(ws) >= 60 and rep_ratio(ws) >= 0.7)
            ):
                bad += 1
        n = len(s["rows"])
        print(f"{b + '/' + m:<46}{n:>8}{bad:>10}{n - bad:>9}{100 * (n - bad) / max(1, n):>9.0f}%")

    print()
    print("=" * 120)
    print("EXAMPLES")
    print("=" * 120)
    for want in (
        "output_empty",
        "single_token_output",
        "lost_system_header",
        "transcript_leak",
        "loop_severe",
        "prompt_end_advisory",
    ):
        printed = 0
        for (b, m), s in sorted(data.items()):
            for d, det in s["examples"].get(want, []):
                if printed >= args.examples * 2:
                    break
                out = d.get("Output") or ""
                print(
                    f"\n[{want}] {b}/{m} {d.get('Scenario')}  "
                    f"ptok={d.get('Prompt Tokens')} otok={d.get('Output Tokens')} "
                    f"fin={d.get('Finish Reason')!r} trunc={d.get('Truncated')} "
                    f"issue={d.get('Issue')} rep4={det['rr']:.2f}"
                )
                print(f"   prompt head: {(d.get('Rendered Prompt') or '')[:150]!r}")
                print(f"   prompt tail: {(d.get('Rendered Prompt') or '')[-110:]!r}")
                print(f"   output     : {out[:200]!r}")
                printed += 1
            if printed >= args.examples * 2:
                break

    print()
    print("=" * 120)
    total = {f: sum(s["flags"].get(f, 0) for s in data.values()) for f in BLOCKING}
    failing = {f: n for f, n in total.items() if n}
    if failing:
        print("RESULT: FAIL — blocking defects present")
        for f, n in sorted(failing.items(), key=lambda kv: -kv[1]):
            print(f"  {f:<26} {n} rows")
        return 1
    print("RESULT: PASS — no blocking defects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
