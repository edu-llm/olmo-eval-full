"""Phase 2a step 1: apply the CAT-pool exclusion mask (extreme_a) for WildBench.

PROVISIONAL decision: exclude the extreme_a criteria (|a| > cm.EXTREME_A or non-finite in
the locked 1D fit at ridge 1e-2) from CAT ADMINISTRATION and ability SCORING, while KEEPING
them in the calibration fit and reporting. WildBench has NO A3 safety-gate (unlike Bridge),
so extreme_a is the only exclusion. Writes a masked CAT-pool bank + ``exclusion_mask.json``
and reports how many criteria/scenarios remain administrable. The extreme_a set may shift
once ridge is swept in Phase 2b.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wildbench_scenario_lib as L  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = L.ROOT / "wildbench_calibration"
    p.add_argument("--rubrics", type=Path, default=L.DEFAULT_RUBRICS)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--fitted-1d", type=Path, default=base / "wildbench_scenario_fitted_1d.jsonl")
    p.add_argument("--out-bank", type=Path,
                   default=base / "wildbench_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--out-mask", type=Path, default=base / "exclusion_mask.json")
    args = p.parse_args()

    maps = L.load_maps(args.rubrics, args.scenarios)
    mask = L.build_exclusion_mask_extreme_a(args.fitted_1d)
    excluded = set(mask["excluded_ids"])

    n_kept = L.write_masked_bank(args.fitted_1d, args.out_bank, excluded)

    fitted_ids = [json.loads(l)["criterion_id"]
                  for l in args.fitted_1d.open(encoding="utf-8") if l.strip()]
    scen_fitted = len(set(maps["c2s"][c] for c in fitted_ids))
    kept_ids = [c for c in fitted_ids if c not in excluded]
    scen_admin = Counter(maps["c2s"][c] for c in kept_ids)
    n_scen_admin = len([s for s in scen_admin if scen_admin[s] > 0])

    doc = {
        "purpose": "CAT-pool exclusion mask (extreme_a only; WildBench has no A3 safety "
                   "gate). Excluded from administration + theta scoring ONLY; retained in "
                   "calibration + reporting.",
        "provisional": "extreme_a set is from the ridge=1e-2 1D fit; it may shrink when "
                       "ridge is swept in Phase 2b.",
        "convention_all_dropped_scenarios": (
            "TutorBench convention (scenario_cat_lib.bank_for_model): a scenario with no "
            "administrable criteria is omitted from the per-model ItemBank -> "
            "un-administrable. No dedup."),
        "extreme_a_threshold": mask["extreme_a_threshold"],
        "n_fitted": mask["n_fitted"],
        "n_extreme_a": mask["n_extreme_a"],
        "counts": {
            "n_criteria_fitted_post_zv": len(fitted_ids),
            "n_excluded": len(excluded),
            "n_administrable_criteria": n_kept,
            "n_scenarios_post_zv": scen_fitted,
            "n_administrable_scenarios": n_scen_admin,
            "n_scenarios_all_dropped": scen_fitted - n_scen_admin,
        },
        "excluded_ids": mask["excluded_ids"],
        "extreme_a_ids": mask["extreme_a_ids"],
    }
    args.out_mask.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("=" * 84)
    print("CAT-POOL EXCLUSION MASK (extreme_a; no A3)")
    print("=" * 84)
    print(f"extreme_a (|a|>{mask['extreme_a_threshold']}) in 1D fit : {mask['n_extreme_a']}")
    print(f"criteria fitted post-ZV          : {len(fitted_ids)}")
    print(f"administrable criteria (CAT pool): {n_kept}")
    print(f"administrable scenarios          : {n_scen_admin}  "
          f"(all-dropped: {scen_fitted - n_scen_admin})")
    print(f"\nwrote masked CAT-pool bank -> {args.out_bank} ({n_kept} criteria)")
    print(f"wrote exclusion mask       -> {args.out_mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
