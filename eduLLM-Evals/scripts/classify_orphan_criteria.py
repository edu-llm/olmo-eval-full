"""Orphan-criteria probe: partition the TutorBench rubric bank by Q-pattern and
classify the currently-UNMODELED criteria (all-zero + scaffolding-only) into
candidate "delivery/affect/communication" buckets.

Purpose
-------
Empirically referee whether the tutor signal that our 3-skill instrument does NOT
model (content / diagnosis / scaffolding) is best described as a NARROW
"motivation/affect" axis, a BROADER "communication/delivery" axis, or neither --
and whether either would stay distinct from ``scaffolding`` or re-collide with it
(the way content<->diagnosis collapsed at latent r=0.945 and adaptability
re-collided with scaffolding). This script is READ-ONLY over the data: it never
writes to ``data/rubrics_qmatrix_final.jsonl`` or any skill definition. It emits a
report under ``staging/`` and prints a summary. It DECIDES NOTHING -- the powered
call is the full-matrix EFA/MIRT test pre-registered in the taxonomy memo.

What it does
------------
1. PARTITION the rubric bank by exact Q-pattern (content, diagnosis, scaffolding).
   The two candidate "unmodeled" pools are:
     * all-zero  (0,0,0) -- tone/affect/formatting/conversational orphans (~1107).
     * scaff-only(0,0,1) -- pure pedagogical-structuring items (~402).

2. CLASSIFY every criterion's ``criterion`` text with a TRANSPARENT keyword/regex
   heuristic into candidate buckets:
     * affect_motivation      -- encouragement, reassurance, empathy, tone,
                                 "supportive", acknowledge feelings.
     * communication_clarity  -- clear/plain/student-friendly LANGUAGE, concise,
                                 easy to follow, avoid jargon, readable.
     * organization_structure -- headings, bullets, numbered/ordered layout,
                                 bold, markdown, sections, formatting.
     * metacognitive          -- reflect, self-check, plan/monitor your approach.
     * socratic_questioning   -- ask a guiding question, prompt the student to
                                 consider/think (withhold the answer).
     * other_uncertain        -- no bucket matched.
   A criterion may hit SEVERAL buckets (esp. affect+communication, which are
   frequently co-authored in one criterion). We record every hit as a boolean
   flag AND assign a single ``bucket`` by a fixed precedence so the counts sum
   cleanly. ALL per-item labels are dumped to CSV for a hand audit.

3. VARIANCE CHECK (DIRECTIONAL, N~=28) -- for each bucket, over the graded models
   in ``staging/response_matrix.csv``, compute the per-model pass rate on that
   bucket's items and then the mean / std / variance ACROSS models, to flag
   near-ceiling (mean>~0.9) or low-variance (std<~0.1) buckets (weak
   discrimination) vs. buckets with real spread.

*** EVERY number here is a HEURISTIC or UNDERPOWERED. The keyword classifier
needs a hand audit (use the CSV). The variance block is N~=28 on a biased
small-model subsample -- DIRECTIONAL ONLY. The powered read is the full-matrix
M2PL latent-correlation + EFA (scripts/calibrate_mirt.py). ***

Usage
-----
    python scripts/classify_orphan_criteria.py
    python scripts/classify_orphan_criteria.py --structural-only   # skip variance
    python scripts/classify_orphan_criteria.py \
        --matrix staging/response_matrix.csv \
        --rubrics data/rubrics_qmatrix_final.jsonl --out-dir staging
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the calibrators' loader/read_jsonl verbatim so IO stays identical to the
# other analysis scripts (analyze_collinearity.py does the same).
_CP_PATH = ROOT / "scripts" / "calibrate_partial.py"
_spec = importlib.util.spec_from_file_location("calibrate_partial", _CP_PATH)
assert _spec and _spec.loader
cp = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("calibrate_partial", cp)
_spec.loader.exec_module(cp)

SKILLS = ("content", "diagnosis", "scaffolding")

DEFAULT_MATRIX = ROOT / "staging" / "response_matrix.csv"
DEFAULT_RUBRICS = ROOT / "data" / "rubrics_qmatrix_final.jsonl"
DEFAULT_OUT_DIR = ROOT / "staging"

REPORT_CSV = "orphan_criteria.csv"
REPORT_JSON = "orphan_criteria.json"

# Buckets in fixed PRECEDENCE order (first match wins for the single ``bucket``
# label). Precedence puts the most specific / least ambiguous constructs first.
# Rationale for order:
#   socratic_questioning  -- "guiding question / withhold the answer" is a sharp,
#                            near-definitional scaffolding signature; check first.
#   metacognitive         -- reflect/self-check/plan is specific.
#   affect_motivation     -- encourage/empathy/tone is specific & domain-free.
#   organization_structure-- headings/bullets/markdown is very distinctive.
#   communication_clarity -- fuzziest (bare "clear" over-catches CONTENT items,
#                            so we keep it narrow AND lowest before "other").
BUCKETS = (
    "socratic_questioning",
    "metacognitive",
    "affect_motivation",
    "organization_structure",
    "communication_clarity",
    "other_uncertain",
)

# --- keyword banks (lowercased, regex; matched against the criterion text) -----
# NOTE: bare "clear/clearly" is DELIBERATELY excluded from communication_clarity
# because, from a sample read, "clearly explain that <domain fact>" attaches to
# CONTENT items far more often than to genuine clarity-of-language items. We
# require a language/readability object to fire communication_clarity.
_AFFECT_PATTERNS = [
    r"encourag\w+",
    r"reassur\w+",
    r"empath\w+",
    r"supportive",
    r"motivat\w+",
    r"praise\w*|praising",
    r"\bwarm\b",
    r"\bkind\b|\bkindly\b",
    r"positive\s+(?:and\s+\w+\s+)?tone|encouraging\s+tone|validating\s+tone|"
    r"empathetic\s+tone|friendly\s+tone|warm\s+tone|supportive\s+tone",
    r"\btone\b",
    r"acknowledg\w+\s+(?:the\s+)?(?:student'?s?\s+)?"
    r"(?:feeling|feelings|confusion|frustration|effort|struggle|anxiety|emotion|concern)",
    r"validat\w+\s+(?:the\s+)?(?:student'?s?\s+)?"
    r"(?:feeling|feelings|effort|confusion|struggle|concern)",
    r"growth\s+mindset",
    r"don'?t\s+be\s+discouraged|not\s+be\s+discouraged",
    r"boost\w*\s+(?:the\s+student'?s?\s+)?confidence|build\s+confidence",
    r"anxiet\w+|frustrat\w+",
    r"emotional\s+(?:state|support)",
    r"rapport",
]
_COMMUNICATION_PATTERNS = [
    r"concise\w*|concisely",
    r"\bclarity\b",
    r"easy\s+to\s+(?:follow|understand|read)",
    r"(?:clear|plain|simple|accessible|student-friendly|jargon-free|"
    r"age-appropriate|layman'?s?|everyday|understandable)\s+"
    r"(?:and\s+\w+\s+)?(?:language|wording|terms|vocabulary|words|explanation)",
    r"avoid\w*\s+(?:overly\s+|unnecessary\s+|technical\s+)?jargon|"
    r"without\s+jargon|minimal\s+jargon|free\s+of\s+jargon|no\s+jargon",
    r"readable|readability",
    r"clear\s+and\s+(?:concise|accessible|understandable|simple|easy)",
    r"not\s+too\s+(?:long|verbose|wordy|technical)|avoid\w*\s+(?:being\s+)?"
    r"(?:verbose|wordy|overly\s+long)",
    r"succinct\w*",
    r"digestible|approachable",
]
_ORGANIZATION_PATTERNS = [
    r"heading\w*|subheading\w*",
    r"bullet\w*",
    r"numbered\s+(?:list|steps?)|ordered\s+list|numbered\s+points?",
    r"\bbold\b|bolding|bolded|italic\w*",
    r"markdown",
    r"\blatex\b",
    r"\bsection\w*\b",
    r"paragraph\w*",
    r"format\w*|formatting",
    r"well[-\s]organiz\w+|logically\s+organiz\w+|organiz\w+\s+(?:the\s+)?"
    r"(?:response|explanation|ideas|content|answer)",
    r"structured\s+(?:layout|formatting)|hierarchical\s+formatting",
    r"\btable\b|tabular",
    r"visual(?:ly)?\s+(?:separate|demarcate|distinct)|demarcat\w+",
]
_METACOGNITIVE_PATTERNS = [
    r"reflect\w*",
    r"self[-\s]check\w*|self[-\s]assess\w*|self[-\s]correct\w*|self[-\s]monitor\w*",
    r"check\s+(?:their|your)\s+(?:own\s+)?(?:work|answer|understanding|reasoning)",
    r"plan\s+(?:their|your)\s+(?:own\s+)?(?:approach|solution|next|strategy)",
    r"monitor\s+(?:their|your)\s+(?:own\s+)?(?:progress|understanding|thinking)",
    r"think\s+about\s+how\s+(?:they|you)\s+(?:approach\w*|solv\w*|reason\w*)",
    r"metacognit\w+",
    r"re-?check\s+(?:their|your)",
    r"verify\s+(?:their|your|the)\s+(?:own\s+)?(?:answer|work|solution|result)",
]
_SOCRATIC_PATTERNS = [
    r"guiding\s+question\w*|guiding\s+prompt\w*",
    r"socratic",
    r"ask\w*\s+(?:the\s+student\s+)?a\s+(?:guiding\s+|leading\s+|probing\s+)?question",
    r"pose\w*\s+a\s+(?:guiding\s+|leading\s+)?question",
    r"rhetorical\s+question",
    r"prompt\w*\s+the\s+student\s+to\s+"
    r"(?:consider|think|determine|identify|write|derive|recall|explore|reason|figure)",
    r"ask\w*\s+the\s+student\s+to\s+(?:consider|think|determine|reflect|explore)",
    r"invite\w*\s+the\s+student\s+to\s+"
    r"(?:consider|think|explore|try|verify|attempt|derive|check|test)",
    r"without\s+(?:explicitly\s+)?(?:telling|revealing|giving|stating)",
    r"lead\w*\s+the\s+student\s+to|guide\w*\s+the\s+student\s+to",
    r"hint\b|hints\b",
    r"elicit\w+",
]

_BANKS = {
    "affect_motivation": re.compile("|".join(_AFFECT_PATTERNS), re.IGNORECASE),
    "communication_clarity": re.compile("|".join(_COMMUNICATION_PATTERNS), re.IGNORECASE),
    "organization_structure": re.compile("|".join(_ORGANIZATION_PATTERNS), re.IGNORECASE),
    "metacognitive": re.compile("|".join(_METACOGNITIVE_PATTERNS), re.IGNORECASE),
    "socratic_questioning": re.compile("|".join(_SOCRATIC_PATTERNS), re.IGNORECASE),
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def q_row(rec: dict) -> tuple[int, int, int]:
    qm = rec.get("q_mapping") or {}
    return (int(qm.get("content", 0)), int(qm.get("diagnosis", 0)),
            int(qm.get("scaffolding", 0)))


def q_pattern_name(c: int, d: int, s: int) -> str:
    return {
        (1, 0, 0): "content_only",
        (1, 1, 0): "content_diagnosis",
        (1, 0, 1): "content_scaffolding",
        (1, 1, 1): "content_diagnosis_scaffolding",
        (0, 1, 0): "diagnosis_only",
        (0, 1, 1): "diagnosis_scaffolding",
        (0, 0, 1): "scaffolding_only",
        (0, 0, 0): "all_zero",
    }[(c, d, s)]


def classify_text(text: str) -> tuple[str, dict[str, int]]:
    """Return (single_bucket_by_precedence, per-bucket hit flags 0/1)."""
    hits = {b: int(bool(_BANKS[b].search(text))) for b in _BANKS}
    single = "other_uncertain"
    for b in BUCKETS:
        if b == "other_uncertain":
            break
        if hits.get(b):
            single = b
            break
    return single, hits


def load_rubrics(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"rubric bank not found: {path}")
    return cp.read_jsonl(path)


def build_per_item(records: list[dict]) -> list[dict]:
    per_item: list[dict] = []
    for rec in records:
        if not isinstance(rec.get("q_mapping"), dict):
            continue
        c, d, s = q_row(rec)
        crit = rec.get("criterion") or ""
        single, hits = classify_text(crit)
        per_item.append({
            "criterion_id": rec.get("criterion_id"),
            "scenario_id": rec.get("scenario_id"),
            "content": c, "diagnosis": d, "scaffolding": s,
            "q_pattern": q_pattern_name(c, d, s),
            "bucket": single,
            "hit_affect_motivation": hits["affect_motivation"],
            "hit_communication_clarity": hits["communication_clarity"],
            "hit_organization_structure": hits["organization_structure"],
            "hit_metacognitive": hits["metacognitive"],
            "hit_socratic_questioning": hits["socratic_questioning"],
            "n_buckets_hit": sum(hits.values()),
            "criticality": rec.get("criticality"),
            "criterion": crit.replace("\n", " ").strip(),
        })
    return per_item


def _bucket_tally(items: list[dict]) -> dict[str, int]:
    t = {b: 0 for b in BUCKETS}
    for it in items:
        t[it["bucket"]] += 1
    return t


def _multihit_tally(items: list[dict]) -> dict[str, int]:
    """Count criteria matching each bucket regardless of precedence (multi-label)."""
    keys = {
        "affect_motivation": "hit_affect_motivation",
        "communication_clarity": "hit_communication_clarity",
        "organization_structure": "hit_organization_structure",
        "metacognitive": "hit_metacognitive",
        "socratic_questioning": "hit_socratic_questioning",
    }
    return {b: sum(it[k] for it in items) for b, k in keys.items()}


def structural_and_buckets(per_item: list[dict]) -> dict:
    pattern_counts: dict[str, int] = {}
    for it in per_item:
        pattern_counts[it["q_pattern"]] = pattern_counts.get(it["q_pattern"], 0) + 1

    all_zero = [it for it in per_item if it["q_pattern"] == "all_zero"]
    scaff_only = [it for it in per_item if it["q_pattern"] == "scaffolding_only"]

    # Co-authoring probe: within the all-zero pool, how often do affect and
    # communication fire TOGETHER on the same criterion?
    az_affect_and_comm = sum(
        1 for it in all_zero
        if it["hit_affect_motivation"] and it["hit_communication_clarity"])
    az_affect = sum(it["hit_affect_motivation"] for it in all_zero)
    az_comm = sum(it["hit_communication_clarity"] for it in all_zero)

    return {
        "n_criteria": len(per_item),
        "q_pattern_counts": pattern_counts,
        "candidate_pools": {
            "all_zero_n": len(all_zero),
            "scaffolding_only_n": len(scaff_only),
        },
        "bucket_precedence_counts": {
            "overall": _bucket_tally(per_item),
            "within_all_zero": _bucket_tally(all_zero),
            "within_scaffolding_only": _bucket_tally(scaff_only),
        },
        "bucket_multilabel_counts": {
            "overall": _multihit_tally(per_item),
            "within_all_zero": _multihit_tally(all_zero),
            "within_scaffolding_only": _multihit_tally(scaff_only),
        },
        "affect_communication_coauthoring_within_all_zero": {
            "affect_hits": az_affect,
            "communication_hits": az_comm,
            "both_hit_same_criterion": az_affect_and_comm,
            "note": (
                "criteria in the all-zero pool that fire BOTH affect and "
                "communication -- high overlap => a broad 'delivery' axis would "
                "merge them rather than keep affect narrow."
            ),
        },
    }


# ---------------------------------------------------------------------------
# variance check (directional, N~=28)
# ---------------------------------------------------------------------------


def variance_by_bucket(mat: pd.DataFrame, per_item: list[dict],
                       subset: str | None, min_items: int) -> dict:
    """Per-model pass rate on each bucket's items, then mean/std ACROSS models.

    ``subset`` restricts to a q_pattern ('all_zero'/'scaffolding_only') or None
    (all criteria). A model contributes to a bucket only if it has >=``min_items``
    observed items in that bucket.
    """
    items = per_item if subset is None else [it for it in per_item
                                             if it["q_pattern"] == subset]
    out: dict[str, dict] = {}
    for b in BUCKETS:
        cols = [it["criterion_id"] for it in items
                if it["bucket"] == b and it["criterion_id"] in mat.columns]
        n_items_present = len(cols)
        if n_items_present == 0:
            out[b] = {"n_items_present": 0, "n_models_used": 0,
                      "mean_passrate": None, "std_passrate": None,
                      "var_passrate": None, "min": None, "max": None,
                      "flag": "no_gradeable_items"}
            continue
        block = mat[cols]
        rates = []
        for model in mat.index:
            row = block.loc[model]
            if int(row.notna().sum()) >= min_items:
                rates.append(float(row.mean(skipna=True)))
        if len(rates) < 3:
            out[b] = {"n_items_present": n_items_present,
                      "n_models_used": len(rates),
                      "mean_passrate": round(float(np.mean(rates)), 4) if rates else None,
                      "std_passrate": None, "var_passrate": None,
                      "min": None, "max": None,
                      "flag": "too_few_models"}
            continue
        arr = np.array(rates, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        flags = []
        if mean >= 0.9:
            flags.append("near_ceiling")
        if mean <= 0.1:
            flags.append("near_floor")
        if std < 0.1:
            flags.append("low_variance")
        out[b] = {
            "n_items_present": n_items_present,
            "n_models_used": len(rates),
            "mean_passrate": round(mean, 4),
            "std_passrate": round(std, 4),
            "var_passrate": round(float(arr.var(ddof=1)), 5),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "flag": "|".join(flags) if flags else "has_spread",
        }
    return out


def variance_check(mat: pd.DataFrame, per_item: list[dict], min_items: int) -> dict:
    return {
        "CAVEAT": (
            "N~=28 biased small-model subsample on a partial matrix. DIRECTIONAL "
            "ONLY -- flags weak-discrimination buckets, does NOT decide anything. "
            "Powered read = full-matrix M2PL latent corr + EFA (calibrate_mirt.py)."
        ),
        "n_models_rows": int(mat.shape[0]),
        "min_items_per_bucket_per_model": min_items,
        "overall": variance_by_bucket(mat, per_item, None, min_items),
        "within_all_zero": variance_by_bucket(mat, per_item, "all_zero", min_items),
        "within_scaffolding_only": variance_by_bucket(
            mat, per_item, "scaffolding_only", min_items),
    }


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def write_reports(struct: dict, variance: dict | None, per_item: list[dict],
                  out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPORT_JSON
    report = {
        "generated_at": _utcnow(),
        "note": (
            "HEURISTIC keyword classification over criterion text; needs a hand "
            "audit (see CSV). Bucket precedence: "
            "socratic>metacognitive>affect>organization>communication>other."
        ),
        "structural_and_buckets": struct,
        "variance_check_directional_only": variance,
        "provenance": {"script": "scripts/classify_orphan_criteria.py",
                       "argv": sys.argv[1:]},
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = out_dir / REPORT_CSV
    pd.DataFrame(per_item).to_csv(csv_path, index=False)
    return csv_path, json_path


def print_summary(struct: dict, variance: dict | None) -> None:
    pc = struct["q_pattern_counts"]
    print("=" * 74)
    print("Q-PATTERN PARTITION (exact, full pass over the rubric bank)")
    print("=" * 74)
    print(f"criteria total       : {struct['n_criteria']}")
    for k in ("content_only", "content_diagnosis", "content_scaffolding",
              "content_diagnosis_scaffolding", "diagnosis_only",
              "diagnosis_scaffolding", "scaffolding_only", "all_zero"):
        print(f"  {k:32s}: {pc.get(k, 0)}")
    cp_ = struct["candidate_pools"]
    print(f"\ncandidate pools: all-zero={cp_['all_zero_n']}  "
          f"scaffolding-only={cp_['scaffolding_only_n']}")

    print("\n" + "=" * 74)
    print("BUCKET COUNTS (single label by precedence)  [HEURISTIC -- audit CSV]")
    print("=" * 74)
    bp = struct["bucket_precedence_counts"]
    hdr = f"{'bucket':26s}{'overall':>10s}{'all_zero':>12s}{'scaff_only':>12s}"
    print(hdr)
    for b in BUCKETS:
        print(f"{b:26s}{bp['overall'][b]:>10d}"
              f"{bp['within_all_zero'][b]:>12d}"
              f"{bp['within_scaffolding_only'][b]:>12d}")

    print("\nMULTI-LABEL hits (a criterion can match several buckets):")
    bm = struct["bucket_multilabel_counts"]
    for b, k in (("affect_motivation", "affect_motivation"),
                 ("communication_clarity", "communication_clarity"),
                 ("organization_structure", "organization_structure"),
                 ("metacognitive", "metacognitive"),
                 ("socratic_questioning", "socratic_questioning")):
        print(f"  {b:26s} overall={bm['overall'][k]:>5d}  "
              f"all_zero={bm['within_all_zero'][k]:>5d}  "
              f"scaff_only={bm['within_scaffolding_only'][k]:>5d}")

    ca = struct["affect_communication_coauthoring_within_all_zero"]
    print(f"\naffect<->communication co-authoring (all-zero pool): "
          f"affect={ca['affect_hits']} comm={ca['communication_hits']} "
          f"BOTH_same_criterion={ca['both_hit_same_criterion']}")

    if variance is not None:
        print("\n" + "=" * 74)
        print("VARIANCE CHECK (DIRECTIONAL ONLY -- N~=28, biased subsample)")
        print("=" * 74)
        for scope in ("within_all_zero", "within_scaffolding_only", "overall"):
            print(f"\n[{scope}]")
            v = variance[scope]
            print(f"  {'bucket':26s}{'items':>6s}{'models':>7s}"
                  f"{'mean':>8s}{'std':>7s}{'flag':>22s}")
            for b in BUCKETS:
                r = v[b]
                print(f"  {b:26s}{r['n_items_present']:>6d}"
                      f"{r['n_models_used']:>7d}"
                      f"{(r['mean_passrate'] if r['mean_passrate'] is not None else float('nan')):>8.3f}"
                      f"{(r['std_passrate'] if r['std_passrate'] is not None else float('nan')):>7.3f}"
                      f"{r['flag']:>22s}")
        print("\nreminder: DIRECTIONAL ONLY. Decide from the full-matrix EFA/MIRT.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX,
                   help=f"response matrix CSV (default: {DEFAULT_MATRIX}).")
    p.add_argument("--rubrics", type=Path, default=DEFAULT_RUBRICS,
                   help=f"rubric bank JSONL (default: {DEFAULT_RUBRICS}).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"output directory (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--structural-only", action="store_true",
                   help="skip the variance (response-matrix) block.")
    p.add_argument("--min-items", type=int, default=3,
                   help="variance: min observed items in a bucket for a model to "
                        "contribute (default 3).")
    args = p.parse_args()

    try:
        records = load_rubrics(args.rubrics)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    per_item = build_per_item(records)
    struct = structural_and_buckets(per_item)

    variance = None
    if not args.structural_only:
        try:
            mat = cp.load_matrix(args.matrix)
            variance = variance_check(mat, per_item, args.min_items)
        except FileNotFoundError as e:
            print(f"WARNING: variance block skipped (matrix not found): {e}",
                  file=sys.stderr)

    print_summary(struct, variance)
    csv_path, json_path = write_reports(struct, variance, per_item, args.out_dir)
    print(f"\nwrote report JSON -> {json_path}")
    print(f"wrote per-item labels CSV -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
