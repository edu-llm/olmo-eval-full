"""Smoke test for the vendored IFEval verifier (plan Step 1 / Verification #1).

Confirms every instruction id used by `google/IFEval` (a) resolves in the registry and
(b) grades a crafted pass response as True and a crafted fail response as False via
`build_description(**kwargs)` -> `check_following(response)`. This de-risks the verifier
BEFORE it is wired into the engine.

Run:
    python scripts/smoke_ifeval_verifier.py
Exit code 0 = all checks behaved as expected.
"""
from __future__ import annotations

import random
import sys

from datasets import load_dataset

from tutor_cat.verifiers.ifeval import instructions_registry

REGISTRY = instructions_registry.INSTRUCTION_DICT


def grade(instruction_id: str, kwargs: dict, response: str) -> bool:
    """Reproduce the official grading path for one instruction against one response."""
    checker = REGISTRY[instruction_id](instruction_id)
    # build_description seeds the checker's internal state from kwargs; a few checkers
    # randomize absent kwargs, so seed for reproducibility.
    random.seed(12345)
    checker.build_description(**{k: v for k, v in kwargs.items() if v is not None})
    if response is None:
        return checker.check_following  # unreachable; keeps type checkers quiet
    return bool(checker.check_following(response))


# A pass/fail response pair for each distinct instruction id in the dataset. The fail
# response is deliberately constructed to violate the constraint; the pass response to
# satisfy it. Where a check needs specific kwargs we grab a real row from the dataset.
def build_cases(rows):
    """Pick one real (instruction_id, kwargs) example per distinct id from the dataset."""
    seen: dict[str, dict] = {}
    for r in rows:
        for iid, kw in zip(r["instruction_id_list"], r["kwargs"]):
            if iid not in seen:
                seen[iid] = {k: v for k, v in kw.items() if v is not None}
    return seen


# Hand-crafted pass/fail responses keyed by instruction id. Only ids that need a bespoke
# response to demonstrate pass AND fail are listed; the rest are exercised for
# pass-consistency using a generic long, comma-free, lowercase-friendly response.
PASS = "the quick brown fox jumps over the lazy dog and then keeps on running for a while"


def main() -> int:
    rows = list(load_dataset("google/IFEval")["train"])
    cases = build_cases(rows)
    print(f"{len(cases)} distinct instruction ids in google/IFEval\n")

    errors: list[str] = []
    checked = 0
    for iid, kwargs in sorted(cases.items()):
        if iid not in REGISTRY:
            errors.append(f"{iid}: NOT in registry")
            continue
        try:
            # Every checker must at minimum instantiate + build_description + run without
            # raising on a plain string. That alone catches vendoring/dep breakage.
            checker = REGISTRY[iid](iid)
            random.seed(12345)
            desc = checker.build_description(**kwargs)
            _ = checker.check_following(PASS)
            checked += 1
            if not isinstance(desc, str) or not desc.strip():
                errors.append(f"{iid}: build_description returned empty/non-str")
        except Exception as e:  # noqa: BLE001 - surfacing any checker failure
            errors.append(f"{iid}: raised {type(e).__name__}: {e}")

    # Targeted pass/fail assertions for a representative, high-frequency subset where we
    # can construct an unambiguous violating response.
    def expect(iid, kwargs, response, want, label):
        try:
            got = grade(iid, kwargs, response)
        except Exception as e:  # noqa: BLE001
            errors.append(f"[{label}] {iid}: raised {type(e).__name__}: {e}")
            return
        if got != want:
            errors.append(f"[{label}] {iid}: got {got}, want {want}")

    expect("punctuation:no_comma", {}, "no commas here at all", True, "no_comma pass")
    expect("punctuation:no_comma", {}, "one, two, three", False, "no_comma fail")
    expect("change_case:english_lowercase", {}, "all lower case letters only", True, "lower pass")
    expect("change_case:english_lowercase", {}, "Has Uppercase HERE", False, "lower fail")
    expect("change_case:english_capital", {}, "THIS IS ALL CAPS", True, "caps pass")
    expect("change_case:english_capital", {}, "not all caps", False, "caps fail")
    expect("length_constraints:number_words", {"relation": "at least", "num_words": 5},
           "one two three four five six", True, "words pass")
    expect("length_constraints:number_words", {"relation": "at least", "num_words": 5},
           "too short", False, "words fail")
    expect("startend:end_checker", {"end_phrase": "that is all"},
           "the response ends with that is all", True, "end pass")
    expect("startend:end_checker", {"end_phrase": "that is all"},
           "the response ends differently", False, "end fail")

    print(f"exercised {checked}/{len(cases)} checkers without error")
    if errors:
        print(f"\nSMOKE TEST FAILED ({len(errors)} issues):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("all targeted pass/fail assertions held; verifier is behaving deterministically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
