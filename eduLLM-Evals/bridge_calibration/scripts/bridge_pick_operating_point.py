"""Stamp the LOCKED CAT operating point into experiments/06_floor_se_grid/best.json.

The sweep's generic auto-pick (max r at full convergence) is not the committed decision:
the locked point trades a hair of recovery for far fewer criteria at 100% convergence.
This writes a reconciled best.json from sweep_results.csv so the provenance is
reproducible (not a manual edit): the locked selection, the conservative max-r alternative,
and the explicitly-rejected sub-100%-convergence pick.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows_from(path: Path) -> list[dict]:
    out = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            out.append({k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()})
    return out


def find(rows, se, floor):
    for r in rows:
        if abs(r["se_target"] - se) < 1e-9 and abs(r["floor"] - floor) < 1e-9:
            return r
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=Path, required=True, help="experiments/06_floor_se_grid/sweep_results.csv")
    p.add_argument("--out", type=Path, required=True, help="best.json to write")
    p.add_argument("--se", type=float, default=0.15)
    p.add_argument("--floor", type=float, default=20)
    args = p.parse_args()

    rows = rows_from(args.sweep)
    sel = find(rows, args.se, args.floor)
    if sel is None:
        raise SystemExit(f"locked point SE {args.se}/floor {args.floor} not found in {args.sweep}")

    conv = [r for r in rows if r["conv_rate"] >= 0.999]
    max_r_conv = None
    if conv:
        rmax = max(r["r"] for r in conv)
        max_r_conv = min([r for r in conv if r["r"] >= rmax - 1e-9], key=lambda r: r["mean_items"])
    glob_max = max(rows, key=lambda r: r["r"])
    rejected = glob_max if glob_max["conv_rate"] < 0.999 else None

    def cell(r):
        return None if r is None else {
            "se_target": r["se_target"], "floor": int(r["floor"]), "r": round(r["r"], 4),
            "slope": round(r["slope"], 4), "theta_mae": round(r["theta_mae"], 4),
            "mean_items": round(r["mean_items"], 1), "conv_rate": round(r["conv_rate"], 3)}

    doc = {
        "selected": cell(sel),
        "selection_rule": ("knee of the efficiency frontier at 100% convergence: near-max OOS "
                           "recovery with far fewer criteria than the tightest SE; efficiency-"
                           "favored over the strict max-r point. Convergence-first: only points "
                           "where 100% of models reach the SE target are eligible."),
        "max_r_at_full_convergence": cell(max_r_conv),
    }
    if max_r_conv is not None:
        doc["max_r_at_full_convergence"]["note"] = (
            f"conservative alternative: +{max(0.0, max_r_conv['r'] - sel['r']):.3f} r for "
            f"~{max(0.0, max_r_conv['mean_items'] - sel['mean_items']):.0f} more criteria")
    if rejected is not None:
        doc["rejected_higher_r_pick"] = {
            **cell(rejected),
            "reason": (f"higher r ({rejected['r']:.3f}) but only {rejected['conv_rate']*100:.0f}% "
                       "convergence (some models never reach that SE); violates the convergence-"
                       "first rule, so it is NOT selected.")}
    ses = sorted({r["se_target"] for r in rows}); floors = sorted({int(r["floor"]) for r in rows})
    doc["grid"] = {"se_targets": ses, "floors": floors}
    doc["source"] = str(args.sweep)

    args.out.write_text(json.dumps(doc, indent=2))
    print(f"locked operating point -> SE {args.se}/floor {int(args.floor)} "
          f"(r={sel['r']:.4f}, mean_items={sel['mean_items']:.1f}, conv={sel['conv_rate']:.2f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
