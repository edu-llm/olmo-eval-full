"""Phase 2a step 1: apply the CAT-pool exclusion mask (A3 + extreme_a).

Locked decision: exclude A3 (criterion_code "A3", the per-scenario affective safety gate,
c15 slot) and the extreme_a criteria (|a| > cm.EXTREME_A or non-finite in the locked 1D
fit) from CAT ADMINISTRATION and ability SCORING, while KEEPING them in the calibration
fit and reporting. This writes a masked CAT-pool bank (the full 1D fitted bank minus the
excluded criteria) plus ``exclusion_mask.json`` documenting exactly what was removed and
why, and reports how many criteria/scenarios remain administrable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_scenario_lib as L  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = L.ROOT / "bridge_calibration"
    p.add_argument("--rubrics", type=Path, default=L.DEFAULT_RUBRICS)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--fitted-1d", type=Path, default=base / "bridge_scenario_fitted_1d.jsonl")
    p.add_argument("--out-bank", type=Path,
                   default=base / "bridge_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--out-mask", type=Path, default=base / "exclusion_mask.json")
    args = p.parse_args()

    maps = L.load_maps(args.rubrics, args.scenarios)
    mask = L.build_exclusion_mask(args.rubrics, args.fitted_1d)
    excluded = set(mask["excluded_ids"])

    n_kept = L.write_masked_bank(args.fitted_1d, args.out_bank, excluded)

    # administrable criteria / scenarios (over the masked CAT pool)
    kept_ids = []
    for line in args.fitted_1d.open(encoding="utf-8"):
        line = line.strip()
        if line:
            cid = json.loads(line)["criterion_id"]
            if cid not in excluded:
                kept_ids.append(cid)
    scen_admin = Counter(maps["c2s"][c] for c in kept_ids)
    n_scen_admin = len([s for s in scen_admin if scen_admin[s] > 0])

    # for reference: scenarios present in the pre-mask (ZV-only) fitted bank
    fitted_ids = [json.loads(l)["criterion_id"]
                  for l in args.fitted_1d.open(encoding="utf-8") if l.strip()]
    scen_fitted = len(set(maps["c2s"][c] for c in fitted_ids))

    doc = {
        "purpose": "CAT-pool exclusion mask (A3 safety gate + extreme_a). Excluded from "
                   "administration + theta scoring ONLY; retained in calibration + reporting.",
        "convention_all_dropped_scenarios": (
            "TutorBench convention (scenario_cat_lib.bank_for_model / "
            "offline_engine_driver.bank_for_model): a scenario with no administrable "
            "criteria is omitted from the per-model ItemBank ('if not cids: continue') -> "
            "un-administrable. No dedup. Bridge's 22 all-zero-variance scenarios have no "
            "fittable criteria and so never enter the CAT bank, matching this convention."),
        **{k: v for k, v in mask.items() if k not in ("excluded_ids", "a3_ids", "extreme_a_ids")},
        "counts": {
            "n_criteria_fitted_post_zv": len(fitted_ids),
            "n_excluded": len(excluded),
            "n_administrable_criteria": n_kept,
            "n_scenarios_post_zv": scen_fitted,
            "n_administrable_scenarios": n_scen_admin,
            "n_scenarios_all_dropped": 250 - n_scen_admin,
        },
        "note_extreme_a": (
            f"{mask['n_extreme_a']} extreme_a criteria in the locked 1D fit (ridge 1e-2). "
            "The '~44' figure in the brief was the item-level count; the scenario-level 1D "
            "fit has far fewer extreme loadings."),
        "excluded_ids": mask["excluded_ids"],
        "a3_ids": mask["a3_ids"],
        "extreme_a_ids": mask["extreme_a_ids"],
    }
    args.out_mask.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("=" * 84)
    print("CAT-POOL EXCLUSION MASK (A3 + extreme_a)")
    print("=" * 84)
    print(f"A3 (criterion_code) total       : {mask['n_a3_total']} "
          f"(in fitted post-ZV: {mask['n_a3_in_fitted']})")
    print(f"extreme_a (|a|>{mask['extreme_a_threshold']}) in 1D fit : {mask['n_extreme_a']}")
    print(f"excluded (A3 | extreme_a)        : {mask['n_excluded_in_fitted']}")
    print(f"criteria fitted post-ZV          : {len(fitted_ids)}")
    print(f"administrable criteria (CAT pool): {n_kept}")
    print(f"administrable scenarios          : {n_scen_admin}  "
          f"(all-dropped: {250 - n_scen_admin})")
    print(f"\nwrote masked CAT-pool bank -> {args.out_bank} ({n_kept} criteria)")
    print(f"wrote exclusion mask       -> {args.out_mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
