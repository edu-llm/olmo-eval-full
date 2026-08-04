#!/usr/bin/env python3
"""Split the 56 unscored-on-pedagogy models into L4 vs L40S GPU groups by size.

Inputs (derived from LIVE S3 state, staged - pedagogy_scored):
  * failed_26.txt  - staged, in full200/done/, no pedagogy marker (re-run needs
                     the stale done/<cd> cleared first).
  * never_30.txt   - staged, not scored, never in full200/done/.

Routing rule (memory-safe; keeps every large / long-context model off the 24 GB
L4 so nothing OOMs, per the run brief):
  L40S (g6e.xlarge, 48 GB)  if params_b >= 3.5
                            OR (params_b >= 3.0 AND context >= 32768)
  L4   (g6.xlarge, 24 GB)   otherwise

Outputs (roster YAMLs the driver accepts + a routing manifest):
  Inputs/Models/pedagogy_rerun_56.yaml
  Inputs/Models/pedagogy_l4.yaml
  Inputs/Models/pedagogy_l40s.yaml
  pedagogy_rerun/routing.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
INF = HERE.parent  # AdaptiveTesting/Test/Inference
if str(INF) not in sys.path:
    sys.path.insert(0, str(INF))

from models_registry import known_context_window  # noqa: E402

ADAPTIVE_ROOT = INF.parent.parent
MODELS_DIR = ADAPTIVE_ROOT / "Inputs" / "Models"
ROSTER = MODELS_DIR / "models_200.yaml"

L40S_MIN_PARAMS = 3.5
LONGCTX_MIN_PARAMS = 3.0
LONGCTX_THRESHOLD = 32768


def read_ids(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def route(params_b: float, ctx: int) -> str:
    if params_b >= L40S_MIN_PARAMS:
        return "l40s"
    if params_b >= LONGCTX_MIN_PARAMS and ctx >= LONGCTX_THRESHOLD:
        return "l40s"
    return "l4"


def main() -> int:
    raw = yaml.safe_load(ROSTER.read_text())
    defaults = raw.get("defaults", {}) or {}
    by_id = {e["id"]: e for e in raw["models"]}

    failed = read_ids(HERE / "failed_26.txt")
    never = read_ids(HERE / "never_30.txt")
    source = {mid: "failed" for mid in failed}
    for mid in never:
        source[mid] = "never"

    all_ids = failed + never
    # sanity
    assert len(failed) == 26, f"expected 26 failed, got {len(failed)}"
    assert len(never) == 30, f"expected 30 never, got {len(never)}"
    assert len(set(all_ids)) == 56, f"expected 56 unique, got {len(set(all_ids))}"
    missing = [m for m in all_ids if m not in by_id]
    if missing:
        raise SystemExit(f"NOT in models_200.yaml: {missing}")

    # keep models_200.yaml order for determinism
    ordered = [e["id"] for e in raw["models"] if e["id"] in set(all_ids)]
    assert len(ordered) == 56, f"ordered={len(ordered)}"

    rows = []
    groups: dict[str, list[dict]] = {"l4": [], "l40s": []}
    for mid in ordered:
        entry = by_id[mid]
        p = float(entry["params_b"])
        ctx = known_context_window(mid) or 0
        gpu = route(p, ctx)
        groups[gpu].append(entry)
        rows.append(
            {"id": mid, "params_b": p, "ctx_offline": ctx, "gpu": gpu, "source": source[mid]}
        )

    def dump(path: Path, entries: list[dict]) -> None:
        path.write_text(
            yaml.safe_dump(
                {"defaults": defaults, "models": entries},
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        )

    dump(MODELS_DIR / "pedagogy_rerun_56.yaml", [by_id[m] for m in ordered])
    dump(MODELS_DIR / "pedagogy_l4.yaml", groups["l4"])
    dump(MODELS_DIR / "pedagogy_l40s.yaml", groups["l40s"])
    (HERE / "routing.json").write_text(json.dumps(rows, indent=2))

    n_l4, n_l40s = len(groups["l4"]), len(groups["l40s"])
    print(f"total={len(ordered)}  L4={n_l4}  L40S={n_l40s}")
    print(f"  failed={len(failed)}  never={len(never)}")
    print("\n=== L40S (g6e.xlarge, 48GB) ===")
    for r in rows:
        if r["gpu"] == "l40s":
            print(f"  {r['id']:52s} {r['params_b']:>5}B ctx={r['ctx_offline']:>6} [{r['source']}]")
    print("\n=== L4 (g6.xlarge, 24GB) ===")
    for r in rows:
        if r["gpu"] == "l4":
            print(f"  {r['id']:52s} {r['params_b']:>5}B ctx={r['ctx_offline']:>6} [{r['source']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
