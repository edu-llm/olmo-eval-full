"""Verify SQuADExactMatchScorer expectations against the real normalizer.

pytest and numpy are unavailable here, and importing olmo_eval pulls numpy, so
the normalization helpers are lifted straight out of the source with ast and
exercised directly. This tests the real code, not a reimplementation.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO

src_path = REPO / "src/olmo_eval/common/scorers/base.py"
src = src_path.read_text(encoding="utf-8")

ns = {}
wanted = {"_squad_normalize_answer", "_compute_squad_f1"}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)

norm = ns["_squad_normalize_answer"]
f1 = ns["_compute_squad_f1"]


def em(pred, refs):
    p = norm(pred)
    return 1.0 if any(p == norm(r) for r in refs) else 0.0


cases = [
    ("Shakespeare", ["Shakespeare"], 1.0, "identical match"),
    ("Dickens", ["Shakespeare"], 0.0, "different answer"),
    ("Great Depression", ["the Great Depression"], 1.0, "articles ignored"),
    ("washington dc", ["Washington, D.C."], 1.0, "punctuation + case ignored"),
    ("  New   York  ", ["New York"], 1.0, "whitespace collapsed"),
    ("Shakespeare", ["William Shakespeare"], 0.0, "no partial credit"),
    ("Bard", ["William Shakespeare", "the Bard"], 1.0, "multi-reference alias hit"),
    ("Marlowe", ["William Shakespeare", "the Bard"], 0.0, "multi-reference miss"),
]

failed = []
for pred, refs, expected, label in cases:
    got = em(pred, refs)
    ok = got == expected
    if not ok:
        failed.append(label)
    print(f"  {'OK  ' if ok else 'FAIL'} {label}")

partial = f1("Shakespeare", "William Shakespeare")
perfect = f1("Great Depression.", "the Great Depression")
print(f"  {'OK  ' if abs(partial - 2 / 3) < 0.01 else 'FAIL'} F1 gives partial credit ({partial:.3f}, expect 0.667)")
print(f"  {'OK  ' if perfect == 1.0 else 'FAIL'} F1 perfect where EM matches ({perfect:.3f})")
if abs(partial - 2 / 3) >= 0.01 or perfect != 1.0:
    failed.append("f1 behaviour")

print()
print("ALL SCORER CHECKS PASSED" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
