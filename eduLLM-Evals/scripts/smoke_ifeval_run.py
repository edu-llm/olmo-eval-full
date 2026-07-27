"""End-to-end tiny run over the 9-axis IFEval bank (plan Verification step 3).

Drives the real `run_evaluation` loop with the real `IFEvalVerifier` and a STUB tutor
(fixed response, no network) so the check is self-contained. Confirms:
  * the 9-category IFEval bank loads under a 9-slug SKILLS axis,
  * the deterministic verifier grades every criterion (binary, no LLM/endpoint),
  * the MIRT loop updates theta/SE over 9 axes without error,
  * judge_results.jsonl rows carry judge_model == "ifeval-verifier".

Requires tutor_cat.SKILLS already flipped to the 9 IFEval slugs (the caller edits
tutor_cat/__init__.py transiently and restores it). Run via the wrapper below, not directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from tutor_cat import SKILLS  # noqa: E402
from tutor_cat.dataio import load_bank  # noqa: E402
from tutor_cat.engine import RunConfig, run_evaluation  # noqa: E402
from tutor_cat.ifeval_verifier import IFEvalVerifier  # noqa: E402

EXPECTED_AXIS = (
    "keywords", "detectable_format", "length_constraints", "change_case",
    "startend", "punctuation", "combination", "detectable_content", "language",
)


class StubTutor:
    """Fixed-response tutor -- no network. Response is crafted to satisfy some IFEval
    constraints and violate others, so we see both pass and fail verdicts."""

    name = "stub"
    model = "stub-fixed-response"

    def respond(self, scenario) -> str:  # noqa: ANN001
        return (
            "here is a plain lowercase answer with no capital letters and no commas "
            "so that a few of the mechanical instruction checks will actually pass "
            "while most others will not this keeps the run honest about both outcomes"
        )


def main() -> int:
    assert tuple(SKILLS) == EXPECTED_AXIS, (
        f"SKILLS is not the 9 IFEval slugs (got {SKILLS!r}); "
        "edit tutor_cat/__init__.py before running this smoke test"
    )

    bank, report = load_bank("data/IFEval/scenarios.jsonl", "data/IFEval/rubrics.jsonl")
    print(f"bank loaded: ok={report.ok} scenarios={len(bank.scenarios)} "
          f"errors={len(report.errors)} warnings={len(report.warnings)}")
    for e in report.errors[:5]:
        print("  ERROR:", e)
    if not report.ok:
        print("FAIL: bank did not validate")
        return 1

    cfg = RunConfig(
        seed=42,
        top_n=5,
        theta_init=[0.0] * 9,
        u_init_diag=[1.0] * 9,
        max_se={s: 0.30 for s in SKILLS},
        min_evals_per_skill=1,
        max_scenarios=10,
        output_dir="runs",
        data_scenarios="data/IFEval/scenarios.jsonl",
        data_rubrics="data/IFEval/rubrics.jsonl",
        unmapped_criteria="judge",
    )
    judge = IFEvalVerifier(seed=42)
    final = run_evaluation(bank, StubTutor(), judge, cfg, mode="cat", run_id="ifeval_smoke")

    print(f"run finished: stop_reason={final['stop_reason']} "
          f"scenarios_administered={final['scenarios_administered']}")

    # --- inspect judge_results.jsonl ---
    jr_path = Path("runs/ifeval_smoke/judge_results.jsonl")
    rows = [json.loads(l) for l in jr_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_pass = sum(1 for r in rows if r["verdict"] == "pass")
    n_fail = sum(1 for r in rows if r["verdict"] == "fail")
    models = {r["judge_model"] for r in rows}
    pversions = {r["judge_prompt_version"] for r in rows}
    scores_binary = all(r["score"] in (0, 1) for r in rows)
    unscorable = [r for r in rows if r["unscorable_reason"]]

    print(f"judge_results: rows={len(rows)} pass={n_pass} fail={n_fail} "
          f"judge_model={models} prompt_version={pversions} "
          f"scores_binary={scores_binary} unscorable={len(unscorable)}")
    if unscorable:
        for r in unscorable[:5]:
            print("  UNSCORABLE:", r["criterion_id"], r["unscorable_reason"])

    print("final theta:", final["theta"])
    print("final se:   ", final["se"])
    print("counts:     ", final["scorable_evaluations"])

    # --- assertions ---
    ok = True
    if models != {"ifeval-verifier"}:
        print("FAIL: judge_model is not exactly {'ifeval-verifier'}"); ok = False
    if pversions != {"ifeval-v1"}:
        print("FAIL: judge_prompt_version is not exactly {'ifeval-v1'}"); ok = False
    if not scores_binary:
        print("FAIL: non-binary score found"); ok = False
    if unscorable:
        print("FAIL: some criteria were unscorable (missing/unknown verifier spec)"); ok = False
    if len(rows) == 0:
        print("FAIL: no judge rows written"); ok = False
    # theta/se must be finite over all 9 axes
    import math
    if not all(math.isfinite(v) for v in final["theta"].values()):
        print("FAIL: non-finite theta"); ok = False
    if not all(math.isfinite(v) for v in final["se"].values()):
        print("FAIL: non-finite se"); ok = False

    print("SMOKE RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
