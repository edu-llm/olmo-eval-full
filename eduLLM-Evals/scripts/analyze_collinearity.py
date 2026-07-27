"""Content<->diagnosis collinearity analysis for the TutorBench Q-matrix.

Purpose
-------
Inform (NOT decide, and NOT mutate) the question of whether the ``content`` and
``diagnosis`` skills are separable, or whether they should be collapsed / the
``diagnosis`` definition narrowed. This script is READ-ONLY over the data: it never
writes to ``data/rubrics_qmatrix_final.jsonl`` or any skill definition. It emits a
small report under ``staging/`` and prints a summary.

It produces THREE blocks:

1. STRUCTURAL -- recomputes the Q-matrix marginals, the 8 skill combos, the pairwise
   co-occurrences, and the conditional probabilities P(content|diagnosis) /
   P(diagnosis|content). This is exact (a full pass over the rubric bank) and is the
   authoritative source for the ~82% overlap figure.

2. CAUSAL-ATTRIBUTION HEURISTIC -- over every diagnosis-loaded item, a keyword pass
   over the ``diagnosis`` justification block of ``q_rationale`` classifies the item
   as one of:
     * specific   -- names/points at the student's SPECIFIC error, value, step,
                     misconception, or knowledge gap (genuine diagnosis).
     * broad_rule -- generic "correct/address THE STUDENT'S error" with no specific
                     misconception spelled out (the v2 broad "address-the-error" rule).
     * ack_affect -- acknowledgement / validation / empathy of the student's feeling
                     or state (under strict v2 this leans toward all-zero).
   This ESTIMATES how much of the content+diagnosis overlap is attributable to the
   broad rule (i.e. how much would dissolve if diagnosis were narrowed to "explicit
   specific misconception only"). It is a HEURISTIC over free text -- treat the split
   as directional, not exact. Every criterion's label is dumped to the CSV so the
   classification can be audited by hand.

3. EMPIRICAL (DIRECTIONAL ONLY) -- over the partial ``staging/response_matrix.csv``
   (~20 graded, small-model-biased persons per item) it computes:
     * per-skill-group item pass rates,
     * a per-MODEL pass-rate correlation between the content-loaded and
       diagnosis-loaded item sets (the most robust read at this N),
     * an item-level pass-pattern correlation (mean pairwise Pearson/phi and an
       approximate tetrachoric) between content-PRIMARY and diagnosis-PRIMARY items,
       compared against the within-group baselines.

   *** N ~= 20 and the graded fleet is a biased small-model subsample. STRUCTURAL
   overlap != EMPIRICAL collinearity, and none of the block-3 numbers are powered.
   They are DIRECTIONAL ONLY -- do not make the skill-definition decision from them.
   The powered read is the full-data M2PL latent correlation + EFA + AIC/BIC from
   scripts/calibrate_mirt.py. ***

Usage
-----
    python scripts/analyze_collinearity.py
    python scripts/analyze_collinearity.py --matrix staging/response_matrix.csv \
        --rubrics data/rubrics_qmatrix_final.jsonl --out-dir staging
    python scripts/analyze_collinearity.py --structural-only   # skip block 3
    python scripts/analyze_collinearity.py --min-overlap 8      # item-corr pair floor
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

# Reuse the calibrators' loader/read_jsonl verbatim so IO stays identical.
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

REPORT_CSV = "collinearity_report.csv"
REPORT_JSON = "collinearity_report.json"

# --- causal-attribution heuristic keyword banks (lowercased, regex) ------------
# Order of precedence in classify_diagnosis(): specific > broad_rule > ack_affect.
# SPECIFIC: the diagnosis text points at a concrete, particular student error.
_SPECIFIC_PATTERNS = [
    r"identif\w+ that the student",
    r"the student incorrectly",
    r"the student\s+(?:wrote|writes|said|says|states?|stated|uses?|used|proposed|"
    r"claims?|claimed|assumes?|assumed|treats?|treated|computed|calculated|added|"
    r"subtracted|multiplied|divided|flipped|confus\w+|mislabel\w+|conflat\w+|thinks?|"
    r"thought|believes?|believed)",
    r"student's\s+(?:answer|value|step|reasoning|solution|work|error|mistake|"
    r"misconception|proposal|proposed|calculation|expression|wording|phrasing|"
    r"terminology|response)",
    r"specific\s+(?:error|mistake|misconception|misunderstanding|confusion)",
    r"pinpoint",
    r"missing\s+(?:key\s+)?background",
    r"knowledge gap",
    r"the student's specific",
    r"where the student went wrong",
    r"proposed by the student",
    r"the student got\b",
]
# BROAD_RULE: generic "correct/address the student's error" w/o a named misconception.
_BROAD_PATTERNS = [
    r"correct\w*\s+the student'?s?\s+(?:error|mistake|misconception|misunderstanding)",
    r"address\w*\s+the student'?s?\s+(?:error|mistake|misconception|misunderstanding|"
    r"confusion)",
    r"correct\w*\s+the error",
    r"engag\w+\s+the student'?s?\s+error",
]
# ACK/AFFECT: acknowledgement/validation/empathy of a feeling/state.
_ACK_PATTERNS = [
    r"acknowledg\w+",
    r"validat\w+",
    r"empath\w+",
    r"reassur\w+",
    r"affirm\w+",
    r"recogniz\w+ (?:and )?(?:affirm|acknowledg|validat)",
]

_SPECIFIC_RE = re.compile("|".join(_SPECIFIC_PATTERNS), re.IGNORECASE)
_BROAD_RE = re.compile("|".join(_BROAD_PATTERNS), re.IGNORECASE)
_ACK_RE = re.compile("|".join(_ACK_PATTERNS), re.IGNORECASE)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# block 1: structural
# ---------------------------------------------------------------------------


def load_rubrics(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"rubric bank not found: {path}")
    return cp.read_jsonl(path)


def q_row(rec: dict) -> tuple[int, int, int]:
    qm = rec.get("q_mapping") or {}
    return (int(qm.get("content", 0)), int(qm.get("diagnosis", 0)),
            int(qm.get("scaffolding", 0)))


def structural(records: list[dict]) -> dict:
    combo_counts: dict[str, int] = {}
    marg = {"content": 0, "diagnosis": 0, "scaffolding": 0}
    cooc = {"content_diagnosis": 0, "content_scaffolding": 0, "diagnosis_scaffolding": 0}
    n = 0
    for rec in records:
        if not isinstance(rec.get("q_mapping"), dict):
            continue
        n += 1
        c, d, s = q_row(rec)
        key = f"{c}{d}{s}"
        combo_counts[key] = combo_counts.get(key, 0) + 1
        marg["content"] += c
        marg["diagnosis"] += d
        marg["scaffolding"] += s
        cooc["content_diagnosis"] += int(c and d)
        cooc["content_scaffolding"] += int(c and s)
        cooc["diagnosis_scaffolding"] += int(d and s)

    def cget(c, d, s) -> int:
        return combo_counts.get(f"{c}{d}{s}", 0)

    diagnosis_wo_content = cget(0, 1, 0) + cget(0, 1, 1)
    p_c_given_d = cooc["content_diagnosis"] / marg["diagnosis"] if marg["diagnosis"] else 0.0
    p_d_given_c = cooc["content_diagnosis"] / marg["content"] if marg["content"] else 0.0
    return {
        "n_criteria": n,
        "marginals": marg,
        "combos": {
            "content_only": cget(1, 0, 0),
            "content_diagnosis": cget(1, 1, 0),
            "content_scaffolding": cget(1, 0, 1),
            "content_diagnosis_scaffolding": cget(1, 1, 1),
            "diagnosis_only": cget(0, 1, 0),
            "diagnosis_scaffolding": cget(0, 1, 1),
            "scaffolding_only": cget(0, 0, 1),
            "none": cget(0, 0, 0),
        },
        "co_occurrence": cooc,
        "P_content_given_diagnosis": round(p_c_given_d, 4),
        "P_diagnosis_given_content": round(p_d_given_c, 4),
        "diagnosis_without_content": diagnosis_wo_content,
        "diagnosis_without_content_frac_of_diagnosis": round(
            diagnosis_wo_content / marg["diagnosis"], 4) if marg["diagnosis"] else 0.0,
    }


# ---------------------------------------------------------------------------
# block 2: causal-attribution heuristic
# ---------------------------------------------------------------------------


def _diagnosis_justification(rec: dict) -> str:
    """Extract the '- diagnosis:' justification block from q_rationale (fallback: all
    of q_rationale + criterion)."""
    rationale = rec.get("q_rationale") or ""
    text = rationale
    # Isolate the diagnosis bullet if present (from '- diagnosis:' to next '- ' bullet).
    m = re.search(r"-\s*diagnosis:\s*(.*?)(?:\n-\s*\w+:|\Z)", rationale, re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1)
    return f"{rec.get('criterion', '')}\n{text}"


def classify_diagnosis(rec: dict) -> str:
    """Heuristic label for a diagnosis-loaded item. Precedence: specific > broad > ack.

    A 'specific' hit dominates because a criterion that both names the specific error
    AND uses generic 'correct the student's error' phrasing is genuinely specific.
    """
    text = _diagnosis_justification(rec)
    if _SPECIFIC_RE.search(text):
        return "specific"
    if _BROAD_RE.search(text):
        return "broad_rule"
    if _ACK_RE.search(text):
        return "ack_affect"
    return "unclassified"


def causal_attribution(records: list[dict]) -> tuple[dict, list[dict]]:
    """Classify every diagnosis-loaded item; break out the content+diagnosis subset."""
    per_item: list[dict] = []
    tally = {k: {"specific": 0, "broad_rule": 0, "ack_affect": 0, "unclassified": 0}
             for k in ("all_diagnosis", "content_and_diagnosis", "diagnosis_without_content")}
    for rec in records:
        if not isinstance(rec.get("q_mapping"), dict):
            continue
        c, d, s = q_row(rec)
        if not d:
            continue
        label = classify_diagnosis(rec)
        tally["all_diagnosis"][label] += 1
        bucket = "content_and_diagnosis" if c else "diagnosis_without_content"
        tally[bucket][label] += 1
        per_item.append({
            "criterion_id": rec.get("criterion_id"),
            "content": c, "diagnosis": d, "scaffolding": s,
            "diagnosis_class": label,
        })

    cd = tally["content_and_diagnosis"]
    n_cd = sum(cd.values())
    # Overlap that would DISSOLVE if diagnosis were narrowed to specific-misconception
    # only = content+diagnosis items whose diagnosis is NOT specific (broad or ack).
    dissolve = cd["broad_rule"] + cd["ack_affect"]
    summary = {
        "note": (
            "HEURISTIC keyword classification over free-text q_rationale; directional, "
            "not exact. Precedence specific>broad_rule>ack_affect. Audit per-item labels "
            "in the CSV."
        ),
        "tally": tally,
        "content_and_diagnosis_total": n_cd,
        "overlap_dissolve_if_specific_only": {
            "count": dissolve,
            "frac_of_content_and_diagnosis": round(dissolve / n_cd, 4) if n_cd else 0.0,
            "definition": "content+diagnosis items whose diagnosis is broad_rule or ack_affect",
        },
        "estimated_genuine_specific_overlap": {
            "count": cd["specific"],
            "frac_of_content_and_diagnosis": round(cd["specific"] / n_cd, 4) if n_cd else 0.0,
        },
    }
    return summary, per_item


# ---------------------------------------------------------------------------
# block 3: empirical (directional only)
# ---------------------------------------------------------------------------


def item_skill_sets(records: list[dict]) -> dict[str, set[str]]:
    """Partition criterion_ids into skill-group sets used by the empirical block."""
    sets = {
        "content_loaded": set(), "diagnosis_loaded": set(), "scaffolding_loaded": set(),
        "content_only": set(), "diagnosis_only": set(),
        "content_primary": set(), "diagnosis_primary": set(),
    }
    for rec in records:
        if not isinstance(rec.get("q_mapping"), dict):
            continue
        cid = rec.get("criterion_id")
        c, d, s = q_row(rec)
        if c:
            sets["content_loaded"].add(cid)
        if d:
            sets["diagnosis_loaded"].add(cid)
        if s:
            sets["scaffolding_loaded"].add(cid)
        if c and not d and not s:
            sets["content_only"].add(cid)
        if d and not c and not s:
            sets["diagnosis_only"].add(cid)
        prim = rec.get("primary_skill")
        if prim == "content":
            sets["content_primary"].add(cid)
        elif prim == "diagnosis":
            sets["diagnosis_primary"].add(cid)
    return sets


def _phi_and_tetrachoric(x: np.ndarray, y: np.ndarray, min_overlap: int) -> tuple[float, float, int]:
    """Pearson (phi) and approximate tetrachoric correlation between two binary item
    vectors, over pairwise-complete (both observed) rows only.

    Tetrachoric via the Digby/Bonett-Price cosine approximation
        r_tet ~= cos( pi / (1 + OR^0.75) )
    with a 0.5 continuity correction on the 2x2 cell counts. Returns (phi, r_tet, n).
    Returns (nan, nan, n) when overlap < min_overlap or a margin is degenerate.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < min_overlap:
        return np.nan, np.nan, n
    xi, yi = x[mask], y[mask]
    if xi.std() == 0 or yi.std() == 0:
        return np.nan, np.nan, n
    phi = float(np.corrcoef(xi, yi)[0, 1])
    a = float(np.sum((xi == 1) & (yi == 1))) + 0.5
    b = float(np.sum((xi == 1) & (yi == 0))) + 0.5
    c = float(np.sum((xi == 0) & (yi == 1))) + 0.5
    dd = float(np.sum((xi == 0) & (yi == 0))) + 0.5
    orat = (a * dd) / (b * c)
    r_tet = float(np.cos(np.pi / (1.0 + orat ** 0.75)))
    return phi, r_tet, n


def mean_cross_corr(mat: pd.DataFrame, cols_a: list[str], cols_b: list[str],
                    min_overlap: int, max_pairs: int, rng: np.random.Generator,
                    same: bool = False) -> dict:
    """Mean pairwise phi / approx-tetrachoric across item pairs (a x b).

    Pairs with < ``min_overlap`` common respondents are skipped. To bound cost the
    pair list is subsampled to ``max_pairs`` when larger.
    """
    arr = {c: mat[c].to_numpy(dtype=float) for c in set(cols_a) | set(cols_b) if c in mat.columns}
    a = [c for c in cols_a if c in arr]
    b = [c for c in cols_b if c in arr]
    pairs: list[tuple[str, str]] = []
    if same:
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                pairs.append((a[i], a[j]))
    else:
        for ca in a:
            for cb in b:
                pairs.append((ca, cb))
    n_all = len(pairs)
    if n_all > max_pairs:
        idx = rng.choice(n_all, size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    phis, tets, ns = [], [], []
    for ca, cb in pairs:
        phi, tet, n = _phi_and_tetrachoric(arr[ca], arr[cb], min_overlap)
        if np.isfinite(phi):
            phis.append(phi)
            ns.append(n)
        if np.isfinite(tet):
            tets.append(tet)
    return {
        "n_pairs_total": n_all,
        "n_pairs_evaluated": len(pairs),
        "n_pairs_usable": len(phis),
        "mean_phi": round(float(np.mean(phis)), 4) if phis else None,
        "median_phi": round(float(np.median(phis)), 4) if phis else None,
        "mean_tetrachoric_approx": round(float(np.mean(tets)), 4) if tets else None,
        "median_overlap_n": int(np.median(ns)) if ns else 0,
    }


def per_model_group_passrates(mat: pd.DataFrame, cols_a: list[str], cols_b: list[str],
                              min_items: int) -> dict:
    """Per-model pass rate on set A vs set B, then Pearson correlation across models.

    A model contributes only if it has >= ``min_items`` observed items in BOTH sets.
    This is the most robust empirical read at N~=20 (one point per model, not per pair).
    """
    a = [c for c in cols_a if c in mat.columns]
    b = [c for c in cols_b if c in mat.columns]
    A = mat[a]
    B = mat[b]
    rate_a, rate_b, kept = [], [], []
    for model in mat.index:
        ra = A.loc[model]
        rb = B.loc[model]
        na = int(ra.notna().sum())
        nb = int(rb.notna().sum())
        if na >= min_items and nb >= min_items:
            rate_a.append(float(ra.mean(skipna=True)))
            rate_b.append(float(rb.mean(skipna=True)))
            kept.append(model)
    out = {
        "n_models_used": len(kept),
        "min_items_per_group": min_items,
        "mean_passrate_a": round(float(np.mean(rate_a)), 4) if rate_a else None,
        "mean_passrate_b": round(float(np.mean(rate_b)), 4) if rate_b else None,
        "pearson_r": None,
        "spearman_r": None,
    }
    if len(kept) >= 3 and np.std(rate_a) > 0 and np.std(rate_b) > 0:
        ra = pd.Series(rate_a)
        rb = pd.Series(rate_b)
        out["pearson_r"] = round(float(ra.corr(rb)), 4)
        out["spearman_r"] = round(float(ra.rank().corr(rb.rank())), 4)
    return out


def group_pass_rates(mat: pd.DataFrame, sets: dict[str, set[str]]) -> dict:
    out = {}
    for name, cols in sets.items():
        present = [c for c in cols if c in mat.columns]
        if not present:
            out[name] = {"n_items_present": 0, "pass_rate": None, "n_obs_cells": 0}
            continue
        block = mat[present].to_numpy(dtype=float)
        obs = np.isfinite(block)
        n_obs = int(obs.sum())
        pr = float(np.nansum(block) / n_obs) if n_obs else None
        out[name] = {
            "n_items_present": len(present),
            "pass_rate": round(pr, 4) if pr is not None else None,
            "n_obs_cells": n_obs,
        }
    return out


def empirical(mat: pd.DataFrame, sets: dict[str, set[str]], args) -> dict:
    rng = np.random.default_rng(args.seed)
    pass_rates = group_pass_rates(mat, sets)

    per_model = {
        "content_loaded_vs_diagnosis_loaded": per_model_group_passrates(
            mat, list(sets["content_loaded"]), list(sets["diagnosis_loaded"]), args.min_items),
        "content_only_vs_diagnosis_only": per_model_group_passrates(
            mat, list(sets["content_only"]), list(sets["diagnosis_only"]), args.min_items),
    }

    item_corr = {
        "content_primary_x_diagnosis_primary": mean_cross_corr(
            mat, list(sets["content_primary"]), list(sets["diagnosis_primary"]),
            args.min_overlap, args.max_pairs, rng),
        "within_content_primary": mean_cross_corr(
            mat, list(sets["content_primary"]), list(sets["content_primary"]),
            args.min_overlap, args.max_pairs, rng, same=True),
        "within_diagnosis_primary": mean_cross_corr(
            mat, list(sets["diagnosis_primary"]), list(sets["diagnosis_primary"]),
            args.min_overlap, args.max_pairs, rng, same=True),
    }
    return {
        "CAVEAT": (
            "N~=20 biased small-model subsample; partial matrix. DIRECTIONAL ONLY. "
            "Structural overlap != empirical collinearity. Powered read = full-data "
            "M2PL latent corr + EFA + AIC/BIC (scripts/calibrate_mirt.py)."
        ),
        "n_models_rows": int(mat.shape[0]),
        "n_criteria_cols": int(mat.shape[1]),
        "group_pass_rates": pass_rates,
        "per_model_passrate_correlation": per_model,
        "item_level_pass_pattern_correlation": item_corr,
    }


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def write_reports(struct: dict, causal: dict, per_item: list[dict], emp: dict | None,
                  out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPORT_JSON
    report = {
        "generated_at": _utcnow(),
        "structural": struct,
        "causal_attribution_heuristic": causal,
        "empirical_directional_only": emp,
        "provenance": {"script": "scripts/analyze_collinearity.py", "argv": sys.argv[1:]},
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Long CSV: per-diagnosis-item heuristic labels (auditable).
    csv_path = out_dir / REPORT_CSV
    pd.DataFrame(per_item).to_csv(csv_path, index=False)
    return csv_path, json_path


def print_summary(struct: dict, causal: dict, emp: dict | None) -> None:
    m = struct["marginals"]
    print("=" * 72)
    print("STRUCTURAL (exact, full pass over the rubric bank)")
    print("=" * 72)
    print(f"criteria           : {struct['n_criteria']}")
    print(f"marginals          : content {m['content']}  diagnosis {m['diagnosis']}  "
          f"scaffolding {m['scaffolding']}")
    print(f"content^diagnosis  : {struct['co_occurrence']['content_diagnosis']}")
    print(f"P(content|diag)    : {struct['P_content_given_diagnosis'] * 100:.1f}%")
    print(f"P(diag|content)    : {struct['P_diagnosis_given_content'] * 100:.1f}%")
    print(f"diagnosis w/o content: {struct['diagnosis_without_content']} "
          f"({struct['diagnosis_without_content_frac_of_diagnosis'] * 100:.1f}% of diagnosis)")

    print("\n" + "=" * 72)
    print("CAUSAL-ATTRIBUTION HEURISTIC (directional; audit CSV)")
    print("=" * 72)
    cd = causal["tally"]["content_and_diagnosis"]
    print(f"content+diagnosis items    : {causal['content_and_diagnosis_total']}")
    print(f"  specific-misconception   : {cd['specific']}")
    print(f"  broad address-the-error  : {cd['broad_rule']}")
    print(f"  acknowledgement/affect   : {cd['ack_affect']}")
    print(f"  unclassified             : {cd['unclassified']}")
    od = causal["overlap_dissolve_if_specific_only"]
    print(f"overlap that DISSOLVES if narrowed to specific-only: {od['count']} "
          f"({od['frac_of_content_and_diagnosis'] * 100:.1f}% of content+diagnosis)")

    if emp is not None:
        print("\n" + "=" * 72)
        print("EMPIRICAL (DIRECTIONAL ONLY -- N~=20, biased subsample)")
        print("=" * 72)
        pm = emp["per_model_passrate_correlation"]["content_loaded_vs_diagnosis_loaded"]
        pm2 = emp["per_model_passrate_correlation"]["content_only_vs_diagnosis_only"]
        print(f"per-model r (content-loaded vs diagnosis-loaded): {pm['pearson_r']} "
              f"(n_models={pm['n_models_used']})")
        print(f"per-model r (content-only   vs diagnosis-only)  : {pm2['pearson_r']} "
              f"(n_models={pm2['n_models_used']})")
        ic = emp["item_level_pass_pattern_correlation"]
        x = ic["content_primary_x_diagnosis_primary"]
        wc = ic["within_content_primary"]
        wd = ic["within_diagnosis_primary"]
        print(f"item-corr cross (C-prim x D-prim): mean phi={x['mean_phi']} "
              f"tet~={x['mean_tetrachoric_approx']} (usable pairs={x['n_pairs_usable']})")
        print(f"item-corr within content-primary : mean phi={wc['mean_phi']}")
        print(f"item-corr within diagnosis-primary: mean phi={wd['mean_phi']}")
        print("\nreminder: DIRECTIONAL ONLY. Decide from the full-data M2PL fit.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


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
                   help="skip the empirical (response-matrix) block.")
    p.add_argument("--min-items", type=int, default=3,
                   help="per-model: min observed items in EACH group to use the model "
                        "(default 3).")
    p.add_argument("--min-overlap", type=int, default=8,
                   help="item-corr: min common respondents for a usable item pair "
                        "(default 8).")
    p.add_argument("--max-pairs", type=int, default=20000,
                   help="item-corr: cap on evaluated item pairs per group (default 20000).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for pair subsampling.")
    args = p.parse_args()

    try:
        records = load_rubrics(args.rubrics)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    struct = structural(records)
    causal, per_item = causal_attribution(records)

    emp = None
    if not args.structural_only:
        try:
            mat = cp.load_matrix(args.matrix)
            sets = item_skill_sets(records)
            emp = empirical(mat, sets, args)
        except FileNotFoundError as e:
            print(f"WARNING: empirical block skipped (matrix not found): {e}",
                  file=sys.stderr)

    print_summary(struct, causal, emp)
    csv_path, json_path = write_reports(struct, causal, per_item, emp, args.out_dir)
    print(f"\nwrote report JSON -> {json_path}")
    print(f"wrote per-item heuristic labels CSV -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
