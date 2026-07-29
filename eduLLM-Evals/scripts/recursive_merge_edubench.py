"""Deterministic half of the recursive greedy skill-merge clustering.

    reads   ingest_edubench.METRICS (the current, already-augmented Table 7) --
            no HuggingFace fetch, no scenario/rubric generation
    writes  data/EduBench/augmented_qmat/merged/recursive_generations.json (a log,
            appended to by save_log(); nothing else)

Groups are `frozenset[str]` over the 9 original slugs. A group's applicability to a
metric is "any member's slug was in that metric's ORIGINAL applies set" -- the same
union rule build_edubench_merged.py already uses, computed directly from ie.METRICS
rather than by materializing intermediate merged-METRICS objects, since it's the same
arithmetic either way and this needs to run many times per generation.

The tie-break votes themselves are NOT here -- they need an actual Agent-tool subagent,
which only the orchestrating conversation can dispatch. This module only ever *reports*
a tie (as a list of candidate edges) and *applies* whatever winner it's given; see
run_generation()'s `decisions` parameter.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest_edubench as ie

ROOT = ie.ROOT
LOG_PATH = ROOT / "data" / "EduBench" / "augmented_qmat" / "merged" / "recursive_generations.json"

SLUG_ORDER = [t["slug"] for t in ie.TASKS]
STEM_OF = {t["slug"]: t["stem"] for t in ie.TASKS}

# Paper's Appendix A, one paragraph per task type -- cached verbatim from
# c:\tmp\edubench.txt:920-938 (arXiv 2505.16160, Appendix A.1/A.2).
DESCRIPTIONS = {
    "answering_questions": "Problem Solving: The ability of an AI system to accurately "
        "solve questions posed by students across various subjects and difficulty levels.",
    "error_correction": "Error Correction: The capacity to identify and correct student "
        "errors in assignments, exams, or daily exercises. Errors can range from obvious "
        "mistakes to subtle issues such as variable misuse in code or logical flaws in "
        "mathematical reasoning. Evaluation focuses on the accuracy of error detection and "
        "the quality of correction.",
    "idea_provision": "Idea Provision: This includes answering student queries about "
        "knowledge points, homework guidance, or exam preparation. It is subdivided into "
        "basic factual explanations, step-by-step solution analysis, and general academic "
        "advice. Responses are evaluated for accuracy, clarity, and informativeness.",
    "learning_support": "Personalized Learning Support: Based on student profiles (e.g., "
        "skill level, learning goals), the system recommends learning paths, exercises, or "
        "reading materials tailored to individual needs. Effectiveness is judged by the "
        "relevance, difficulty alignment, and usefulness of the recommendations.",
    "mental_health": "Emotional Support: This involves detecting a student's emotional "
        "state (e.g., anxiety before exams) from text and offering appropriate supportive "
        "feedback or suggestions. Scenarios include pre-exam stress, post-exam frustration, "
        "or social isolation. Evaluation metrics include emotion classification accuracy, "
        "specificity of emotional cues, and quality of suggestions.",
    "question_generation": "Question Generation: Generating questions based on specified "
        "topics, difficulty levels, and knowledge scopes. This includes both single-topic "
        "and multi-topic (comprehensive) question generation. Advanced requirements involve "
        "generating explanations and formatting full exams. Evaluation focuses on question "
        "quality, relevance, and structural coherence.",
    "grading": "Automatic Grading: Supporting grading of objective questions (e.g., "
        "multiple-choice, fill-in-the-blank) and subjective tasks (e.g., project reports) "
        "based on scoring rubrics. Feedback generation is also supported. Metrics include "
        "scoring accuracy, reasonableness, and feedback informativeness.",
    "material_generation": "Teaching Material Generation: Automatically generating "
        "educational content such as slides, teaching plans, and lecture notes. This "
        "includes content structuring and supplementing with relevant external materials "
        "like images or references.",
    "personalized_content_creation": "Personalized Content Creation: Generating "
        "differentiated content for students based on their learning levels or personal "
        "profiles. This includes both individualized assignments and tiered content design "
        "(e.g., differentiated learning objectives, teaching strategies, and assessments for "
        "varying student levels). Evaluation focuses on the internal validity of each item "
        "and cross-tier consistency.",
}
assert set(DESCRIPTIONS) == set(SLUG_ORDER), "DESCRIPTIONS must cover exactly the 9 slugs"


def order(group: frozenset[str]) -> list[str]:
    return sorted(group, key=SLUG_ORDER.index)


def label(group: frozenset[str]) -> str:
    return "+".join(STEM_OF[s] for s in order(group))


def describe(group: frozenset[str]) -> str:
    return " || ".join(DESCRIPTIONS[s] for s in order(group))


def initial_partition() -> list[frozenset[str]]:
    return [frozenset({t["slug"]}) for t in ie.TASKS]


def node_weight(a: frozenset[str]) -> int:
    return sum(1 for m in ie.METRICS if m["applies"] & a)


def edge_weight(a: frozenset[str], b: frozenset[str]) -> int:
    """Shared-criteria count -- the original weight. Unaffected by --weight-mode."""
    return sum(1 for m in ie.METRICS if (m["applies"] & a) and (m["applies"] & b))


def edge_weight_percentage(a: frozenset[str], b: frozenset[str]) -> Fraction:
    """min(shared / |A|, shared / |B|) -- the overlap coefficient. Exact (Fraction, not
    float) so tie detection between two ratios of small integers is never a rounding bug.
    Undefined (returns Fraction(0)) if either node has zero applicable criteria, which
    cannot happen here since every real metric applies to at least one of the 9 slugs."""
    shared = edge_weight(a, b)
    wa, wb = node_weight(a), node_weight(b)
    if not wa or not wb:
        return Fraction(0)
    return min(Fraction(shared, wa), Fraction(shared, wb))


def all_edges(partition: list[frozenset[str]], mode: str = "count"):
    weigh = edge_weight if mode == "count" else edge_weight_percentage
    return [(a, b, weigh(a, b)) for a, b in combinations(partition, 2)]


def _touches(e, other) -> bool:
    return e is not other and (e[0] in other[:2] or e[1] in other[:2])


def _split_tier(tier: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    """(auto-applicable edges, genuinely tied edges) within one weight tier. An edge is
    auto-applicable iff it shares no node with any OTHER edge still in this tier."""
    auto = [e for e in tier if not any(_touches(e, o) for o in tier)]
    tied = [e for e in tier if e not in auto]
    return auto, tied


def _tie_key(tied: list[tuple]) -> str:
    """Stable id for a specific tie cluster, for matching against `decisions`."""
    return "|".join(sorted(f"{label(a)}~{label(b)}" for a, b, _ in tied))


def run_generation(partition: list[frozenset[str]],
                   decisions: dict[str, tuple[str, str]] | None = None,
                   mode: str = "count") -> dict:
    """One generation, replayed deterministically from `partition`. `decisions` maps a
    tie-cluster key (see _tie_key) to the winning pair's (label(a), label(b)), for ties
    already resolved by a previous (paused) call. Returns one of:

      {"status": "base_case_a", "n": n}
      {"status": "tie", "candidates": [...], "key": ..., "partial_merges": [...]}
      {"status": "generation_done", "n": n, "merges": [...], "new_partition": [...]}

    mode="count" (default, original behavior): weight is the shared-criteria count;
    qualifying range is {n-1, n}; a generation may apply several non-conflicting merges.

    mode="percentage": weight is min(shared/|A|, shared/|B|) (edge_weight_percentage);
    a generation applies every non-conflicting edge at the CURRENT top weight (same
    conflict rule as count mode -- a tie means sharing a node, not just an equal value);
    unlike count mode there is no second (n-1) tier, since percentage weight has no
    integer "range" to fall back a step within.
    """
    if mode == "percentage":
        return _run_generation_percentage(partition, decisions)
    return _run_generation_count(partition, decisions)


def _run_generation_count(partition: list[frozenset[str]],
                          decisions: dict[str, tuple[str, str]] | None = None) -> dict:
    decisions = decisions or {}
    n = len(partition)
    lo, hi = n - 1, n
    edges = [e for e in all_edges(partition, mode="count") if lo <= e[2] <= hi]
    if not edges:
        return {"status": "base_case_a", "n": n}

    available = list(partition)
    formed: list[frozenset[str]] = []
    merges = []

    for w in (hi, lo):
        progress = True
        while progress:
            progress = False
            tier = [e for e in edges if e[2] == w and e[0] in available and e[1] in available]
            if not tier:
                break
            auto, tied = _split_tier(tier)
            if auto:
                for a, b, ww in auto:
                    available.remove(a)
                    available.remove(b)
                    merged = a | b
                    formed.append(merged)
                    merges.append({"a": label(a), "b": label(b), "weight": ww})
                progress = True
                continue
            if tied:
                key = _tie_key(tied)
                if key in decisions:
                    wa, wb = decisions[key]
                    winner = next(e for e in tied if {label(e[0]), label(e[1])} == {wa, wb})
                    a, b, ww = winner
                    available.remove(a)
                    available.remove(b)
                    formed.append(a | b)
                    merges.append({"a": label(a), "b": label(b), "weight": ww,
                                  "tie_resolved_among": [f"{label(x)}~{label(y)}" for x, y, _ in tied]})
                    progress = True
                    continue
                return {
                    "status": "tie",
                    "tier_weight": w,
                    "candidates": [{"a": label(a), "b": label(b), "weight": ww,
                                   "desc_a": describe(a), "desc_b": describe(b)}
                                  for a, b, ww in tied],
                    "key": key,
                    "partial_merges": merges,
                }

    new_partition = available + formed
    return {"status": "generation_done", "n": n, "merges": merges,
           "new_partition": [order(g) for g in new_partition]}


def _fmt_weight(w: Fraction) -> dict:
    """Fraction isn't JSON-serializable; keep both the exact ratio and a float."""
    return {"exact": str(w), "value": float(w)}


def _run_generation_percentage(partition: list[frozenset[str]],
                               decisions: dict[str, tuple[str, str]] | None = None) -> dict:
    """No cap on merges per generation: every non-conflicting edge at the CURRENT top
    weight is applied (mirroring count mode's _split_tier -- a tie is specifically edges
    that share a node, not just an equal weight); genuine conflicts go to a subagent
    vote. Unlike count mode there is no second (n-1) tier -- only the single highest
    weight value is in play, since percentage weight has no natural "range" analog."""
    decisions = decisions or {}
    n = len(partition)
    edges = [e for e in all_edges(partition, mode="percentage") if e[2] > 0]
    if not edges:
        return {"status": "base_case_a", "n": n}

    top = max(e[2] for e in edges)
    available = list(partition)
    formed: list[frozenset[str]] = []
    merges = []

    progress = True
    while progress:
        progress = False
        tier = [e for e in edges if e[2] == top and e[0] in available and e[1] in available]
        if not tier:
            break
        auto, tied = _split_tier(tier)
        if auto:
            for a, b, ww in auto:
                available.remove(a)
                available.remove(b)
                formed.append(a | b)
                merges.append({"a": label(a), "b": label(b), "weight": _fmt_weight(ww)})
            progress = True
            continue
        if tied:
            key = _tie_key(tied)
            if key in decisions:
                wa, wb = decisions[key]
                winner = next(e for e in tied if {label(e[0]), label(e[1])} == {wa, wb})
                a, b, ww = winner
                available.remove(a)
                available.remove(b)
                formed.append(a | b)
                merges.append({"a": label(a), "b": label(b), "weight": _fmt_weight(ww),
                              "tie_resolved_among": [f"{label(x)}~{label(y)}" for x, y, _ in tied]})
                progress = True
                continue
            return {
                "status": "tie",
                "candidates": [{"a": label(a), "b": label(b), "weight": _fmt_weight(ww),
                               "desc_a": describe(a), "desc_b": describe(b)}
                              for a, b, ww in tied],
                "key": key,
                "partial_merges": merges,
            }

    new_partition = available + formed
    return {"status": "generation_done", "n": n, "merges": merges,
           "new_partition": [order(g) for g in new_partition]}


def partition_from_groups(groups: list[list[str]]) -> list[frozenset[str]]:
    return [frozenset(g) for g in groups]


def load_log(path: Path = LOG_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_log(entries: list[dict], path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


if __name__ == "__main__":
    # Smoke test: run from the original 9 singletons, print what generation 1 sees.
    p = initial_partition()
    print(f"n={len(p)}: {[label(g) for g in p]}")
    result = run_generation(p)
    print(json.dumps(result, indent=2))
