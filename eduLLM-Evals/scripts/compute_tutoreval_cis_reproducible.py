"""Recompute the TutorEval calibration CIs from the reproducible k-fold export.

Thin wrapper around ``compute_tutoreval_cis`` that repoints its inputs at the canonical
re-export under ``regenerated_figures/scenario_level/kfold_reproducible/`` (produced with
``--scenarios data/TutorEval/scenarios_final.jsonl``, which reproduces the published
k-fold numbers exactly) and writes to a separate
``tutoreval_confidence_intervals_reproducible.csv`` so the original CI table is left
untouched. The statistical method (Fisher-z for r, model bootstrap B=2000 seed 20260803
for slope/means/gap, criteria bootstrap for Fleiss kappa) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import compute_tutoreval_cis as cci

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "regenerated_figures" / "scenario_level" / "kfold_reproducible"


def main() -> int:
    cci.KFOLD = REPRO
    cci.REGEN = REPRO
    cci.OUT_CSV = (
        ROOT
        / "staging"
        / "tutoreval_calibration"
        / "tutoreval_confidence_intervals_reproducible.csv"
    )
    return cci.main()


if __name__ == "__main__":
    raise SystemExit(main())
