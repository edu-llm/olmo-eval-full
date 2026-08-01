"""Run the REAL production CAT engine offline against a saved response matrix.

This is a standalone driver. It does NOT reimplement any CAT math and it does not modify
``tutor_cat/engine.py``, ``tutor_cat/selector.py``, ``tutor_cat/mirt.py``,
``tutor_cat/schemas.py``, ``tutor_cat/dataio.py``, or the simplified harness
(``scripts/regen_cat_figures.py``). It only supplies the two things production normally
gets from the network -- a tutor and a judge -- from precomputed data, and then calls
``tutor_cat.engine.run_evaluation`` verbatim.

Why this works: ``engine.run_evaluation`` already takes ``tutor`` and ``judge`` as
injected Protocols (``TutorLike`` / ``JudgeClient``). Swapping in matrix-backed
implementations makes the run offline while leaving selection (``selector.select_next``,
targeting ``argmax(se)``), the M2PL updates (``mirt.update``, PRD Eqs 1-3), and the
stopping rule (per-skill max SE + min scorable evaluations + max scenarios) untouched.

    MatrixTutor  -- respond() returns a placeholder; the response text is irrelevant
                    because the verdict is already recorded.
    MatrixJudge  -- evaluate() looks up the recorded 0/1 for (model, criterion_id).

Bank handling
-------------
The ``*_fitted`` banks key ``discrimination`` and ``q_modeled`` by the MODELED skill names
(e.g. correctness / scaffolding / presentation) rather than the fixed
``tutor_cat.SKILLS`` slot names, and they contain only calibrated criteria. This driver
therefore builds ``Rubric`` / ``Scenario`` / ``ItemBank`` objects directly, in modeled-skill
order, instead of going through ``dataio.load_bank``. Production's engine is agnostic to
the names -- it only needs len(SKILLS) vectors -- so the math is unaffected; the driver
remaps names for reporting.

Per-model banks
---------------
The matrix is ~98% filled. Rather than inventing verdicts for ungraded cells, each model
gets a bank whose scenario ``criterion_ids`` are filtered to criteria that (a) exist in the
fitted bank and (b) have a recorded verdict for that model. Nothing is fabricated and the
engine never sees a cell it cannot score.

Estimators
----------
The engine reports production's own online sequential theta. Because the engine logs every
administered criterion to ``criterion_updates.jsonl``, this driver additionally re-estimates
each model's ability from that exact administered set by batch EAP and by multidimensional
WLE, reusing the implementations in ``scripts/regen_cat_figures.py``. Selection is identical
in all three -- only the final estimator differs.

Usage
-----
    python scripts/offline_engine_driver.py \
        --bank data/TutorBench/rubrics_qmatrix_calibrated_3skill_fitted.jsonl \
        --matrix staging/response_matrix_full.csv \
        --out-dir regenerated_figures/production_engine/3_skills
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor_cat import SKILLS  # noqa: E402
from tutor_cat.dataio import ItemBank  # noqa: E402
from tutor_cat.engine import RunConfig, run_evaluation  # noqa: E402
from tutor_cat.schemas import JudgeVerdict, Rubric, Scenario  # noqa: E402

# Reuse the reference-ability and alternative-estimator math already written and verified.
_spec = importlib.util.spec_from_file_location("regen", ROOT / "scripts" / "regen_cat_figures.py")
regen = importlib.util.module_from_spec(_spec)
sys.modules["regen"] = regen
_spec.loader.exec_module(regen)


# ---------------------------------------------------------------------------
# bank loading (fitted schema: modeled-skill keys + q_modeled)
# ---------------------------------------------------------------------------


def load_fitted_bank(path: Path, negative_policy: str):
    """Parse a ``*_fitted`` rubric bank.

    Returns (records, dims, stats). ``records`` are raw dicts; ``dims`` is the modeled
    skill order taken from the file itself.
    """
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no records in {path}")

    dims = list(records[0]["discrimination"].keys())
    for r in records:
        if list(r["discrimination"].keys()) != dims:
            raise SystemExit(f"inconsistent discrimination keys at {r['criterion_id']}")
        if list(r["q_modeled"].keys()) != dims:
            raise SystemExit(f"q_modeled keys != discrimination keys at {r['criterion_id']}")

    if not (1 <= len(dims) <= len(SKILLS)):
        raise SystemExit(
            f"bank models {len(dims)} skills {dims}; the engine supports 1..{len(SKILLS)} "
            f"latent dimensions. The run passes these skills to RunConfig(skills=...), so "
            f"any count in range works without touching the core math."
        )

    n_neg = sum(1 for r in records
                for d in dims if float(r["discrimination"][d] or 0.0) < 0)
    n_neg_items = sum(1 for r in records
                      if any(float(r["discrimination"][d] or 0.0) < 0 for d in dims))
    if negative_policy == "clamp":
        for r in records:
            for d in dims:
                if float(r["discrimination"][d] or 0.0) < 0:
                    r["discrimination"][d] = 0.0
    elif negative_policy == "drop":
        records = [r for r in records
                   if not any(float(r["discrimination"][d] or 0.0) < 0 for d in dims)]

    stats = {"n_negative_cells": n_neg, "n_negative_items": n_neg_items,
             "negative_policy": negative_policy, "n_records_after": len(records)}
    return records, dims, stats


def build_rubrics(records: list[dict], dims: list[str]) -> dict[str, Rubric]:
    """Rubric objects with q/a as len(SKILLS) vectors in MODELED-skill order."""
    out: dict[str, Rubric] = {}
    for r in records:
        out[r["criterion_id"]] = Rubric(
            criterion_id=r["criterion_id"],
            scenario_id=r["scenario_id"],
            criterion=r.get("criterion", ""),
            q=np.array([int(r["q_modeled"][d]) for d in dims], dtype=int),
            a=np.array([float(r["discrimination"][d] or 0.0) for d in dims], dtype=float),
            b=float(r["difficulty"]),
            primary_skill=r.get("primary_skill", ""),
            scoring_type=r.get("scoring_type", "binary"),
            criticality=r.get("criticality", "standard"),
            calibration_version=str((r.get("irt_params") or {}).get("source", "")),
            status=r.get("status", "approved"),
        )
    return out


def load_scenarios(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                obj = json.loads(line)
                out[obj["scenario_id"]] = obj
    return out


# ---------------------------------------------------------------------------
# offline tutor + judge (the only things that differ from a live run)
# ---------------------------------------------------------------------------


class MatrixTutor:
    """TutorLike backed by precomputed responses; the text itself is never scored."""

    def __init__(self, model: str):
        self.model = model
        self.name = model.replace("/", "_")

    def respond(self, scenario: Scenario) -> str:
        return f"[offline] recorded response for {self.model} on {scenario.scenario_id}"


class MatrixJudge:
    """JudgeClient that returns the verdict already recorded in the response matrix."""

    def __init__(self, row: pd.Series, model: str):
        self._row = row
        self.name = "offline-matrix"
        self.prompt_version = "recorded"
        self.seed = 0
        self.model = model
        self.n_lookups = 0
        self.n_missing = 0

    def evaluate(self, scenario: Scenario, rubric: Rubric, response: str) -> JudgeVerdict:
        self.n_lookups += 1
        val = self._row.get(rubric.criterion_id, np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            # Should not happen: banks are pre-filtered to graded cells for this model.
            self.n_missing += 1
            return JudgeVerdict(verdict="fail", unscorable_reason="not_in_matrix")
        return JudgeVerdict(verdict="pass" if int(val) == 1 else "fail",
                            evidence="recorded", rationale="offline matrix lookup")


def bank_for_model(rubrics: dict[str, Rubric], scen_raw: dict[str, dict],
                   row: pd.Series) -> ItemBank:
    """Bank restricted to criteria this model actually has a recorded verdict for."""
    graded = {cid for cid in rubrics if cid in row.index and not pd.isna(row[cid])}
    scenarios: dict[str, Scenario] = {}
    for sid, obj in scen_raw.items():
        cids = [c for c in obj["criterion_ids"] if c in graded]
        if not cids:
            continue
        s = Scenario.from_json(obj)
        s.criterion_ids = sorted(cids)
        scenarios[sid] = s
    kept = {cid: rubrics[cid] for s in scenarios.values() for cid in s.criterion_ids}
    return ItemBank(scenarios, kept)


# ---------------------------------------------------------------------------
# post-hoc estimators over the engine's own administered set
# ---------------------------------------------------------------------------


def administered_from_log(run_dir: Path) -> list[str]:
    """Criterion ids the engine actually administered, in update order."""
    out: list[str] = []
    p = run_dir / "criterion_updates.jsonl"
    if not p.is_file():
        return out
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line)["criterion_id"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "data" / "TutorBench" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, default=ROOT / "staging" / "engine_runs")
    p.add_argument("--negative-policy", choices=("clamp", "keep", "drop"), default="clamp")
    p.add_argument("--models", type=int, default=None, help="limit to first N models (smoke).")
    # production RunConfig knobs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--max-se", type=float, default=0.30)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--max-scenarios", type=int, default=50)
    p.add_argument("--unmapped-criteria", choices=("judge", "skip"), default="judge")
    # reference / post-hoc estimator grid
    p.add_argument("--grid", type=int, default=61)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--keep-run-logs", action="store_true",
                   help="keep the per-model engine run directories (large).")
    args = p.parse_args()

    print("=" * 96)
    print("OFFLINE DRIVER FOR THE PRODUCTION CAT ENGINE")
    print("=" * 96)

    mat_sha = hashlib.sha256(args.matrix.read_bytes()).hexdigest()
    records, dims, neg_stats = load_fitted_bank(args.bank, args.negative_policy)
    prov = (records[0].get("irt_params") or {}).get("provenance") or {}
    print(f"bank    : {args.bank.name}  ({len(records)} criteria, dims={dims})")
    print(f"          negatives: {neg_stats['n_negative_cells']} cells over "
          f"{neg_stats['n_negative_items']} criteria -> policy '{args.negative_policy}'")
    print(f"matrix  : {args.matrix.name}  sha256={mat_sha[:24]}...")
    print(f"          bank expects {prov.get('matrix_csv')} sha={str(prov.get('matrix_sha256'))[:24]}...")
    aligned = prov.get("matrix_sha256") == mat_sha
    print(f"          PROVENANCE {'MATCH' if aligned else '*** MISMATCH ***'}")

    matrix = pd.read_csv(args.matrix, index_col=0)
    scen_raw = load_scenarios(args.scenarios)
    rubrics = build_rubrics(records, dims)
    models = list(matrix.index)[: args.models] if args.models else list(matrix.index)
    print(f"models  : {len(models)}")
    print(f"engine  : max_se={args.max_se} min_evals_per_skill={args.min_evals_per_skill} "
          f"max_scenarios={args.max_scenarios} top_n={args.top_n} seed={args.seed}")

    # Full-bank reference ability, computed on the SAME modeled-skill basis.
    ids = [r["criterion_id"] for r in records]
    A = np.array([[float(r["discrimination"][d] or 0.0) for d in dims] for r in records])
    b = np.array([float(r["difficulty"]) for r in records])
    sub = matrix.reindex(columns=ids)
    Yraw = sub.to_numpy(dtype=float)
    mask = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    col = {c: i for i, c in enumerate(ids)}
    grid, log_prior = regen.build_grid(len(dims), args.grid, "uniform", args.range)
    print(f"\ncomputing full-bank EAP reference ({grid.shape[0]:,} quadrature nodes) ...")
    row_of = {m: i for i, m in enumerate(matrix.index)}
    theta_full = regen.eap_all_models(Y, mask, A, b, grid, log_prior, 4096)

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(
        seed=args.seed, top_n=args.top_n,
        max_se={s: args.max_se for s in dims},
        min_evals_per_skill=args.min_evals_per_skill,
        max_scenarios=args.max_scenarios,
        output_dir=str(args.runs_dir),
        data_scenarios=str(args.scenarios), data_rubrics=str(args.bank),
        unmapped_criteria=args.unmapped_criteria,
        skills=tuple(dims),
    )

    print(f"running the production engine for {len(models)} models ...")
    rows = []
    stop_reasons: dict[str, int] = {}
    for i, model in enumerate(models, 1):
        r = row_of[model]
        row = matrix.loc[model]
        bank = bank_for_model(rubrics, scen_raw, row)
        tutor = MatrixTutor(model)
        judge = MatrixJudge(row, model)
        run_id = f"offline_{tutor.name}"
        final = run_evaluation(bank, tutor, judge, cfg, mode="cat", run_id=run_id)
        stop_reasons[final["stop_reason"]] = stop_reasons.get(final["stop_reason"], 0) + 1

        run_dir = args.runs_dir / run_id
        order = [c for c in administered_from_log(run_dir) if c in col]
        idx = np.array([col[c] for c in order], dtype=int)

        # production's own online estimate, in modeled-dim order
        th_online = np.array([final["theta"][s] for s in dims], dtype=float)
        se_online = np.array([final["se"][s] for s in dims], dtype=float)
        th_batch = (regen.eap_subset(Y[r], idx, A, b, grid, log_prior)
                    if idx.size else th_online.copy())
        th_mwle, ok = (regen.mwle_subset(Y[r], idx, A, b, th_batch)
                       if idx.size else (th_batch.copy(), True))

        obs = np.where(mask[r])[0]
        rec = {
            "model": model,
            "scenarios_administered": final["scenarios_administered"],
            "criteria_administered": len(order),
            "stop_reason": final["stop_reason"],
            "precision_reached": final["precision_reached"],
            "judge_lookups": judge.n_lookups,
            "judge_missing": judge.n_missing,
            "obs_acc": float(Y[r][obs].mean()) if obs.size else np.nan,
        }
        for k, d in enumerate(dims):
            rec[f"theta_full_{d}"] = float(theta_full[r, k])
            rec[f"theta_cat_{d}"] = float(th_online[k])
            rec[f"theta_batch_{d}"] = float(th_batch[k])
            rec[f"theta_mwle_{d}"] = float(th_mwle[k])
            rec[f"final_se_{d}"] = float(se_online[k])
            rec[f"scorable_evals_{d}"] = int(final["scorable_evaluations"][dims[k]])
        rows.append(rec)
        if not args.keep_run_logs:
            shutil.rmtree(run_dir, ignore_errors=True)
        if i % 10 == 0 or i == len(models):
            print(f"   {i}/{len(models)} models")

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "cat_per_model.csv", index=False)

    # ---- metrics + figures, one panel-set per estimator -------------------
    def stats_for(prefix: str) -> dict:
        out = {}
        for d in dims:
            x = df[f"theta_full_{d}"].to_numpy()
            y = df[f"{prefix}_{d}"].to_numpy()
            w = np.argsort(x)[:12]
            out[d] = {"slope": float(np.polyfit(x, y, 1)[0]),
                      "r": float(np.corrcoef(x, y)[0, 1]),
                      "gap_worst12": float(np.mean(y[w] - x[w])),
                      "n_distinct_x": int(np.unique(np.round(x, 3)).size)}
        return out

    estimators = {"theta_cat": "production online", "theta_batch": "batch EAP",
                  "theta_mwle": "batch EAP + MWLE"}
    agg = {p: stats_for(p) for p in estimators}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for prefix, label in estimators.items():
        for d in dims:
            x = df[f"theta_full_{d}"].to_numpy()
            y = df[f"{prefix}_{d}"].to_numpy()
            s = agg[prefix][d]
            fig, ax = plt.subplots(figsize=(4.6, 4.4))
            ax.scatter(x, y, s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
            lo, hi = min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3
            ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(f"full-bank EAP ability ({d})")
            ax.set_ylabel(f"CAT ability ({d})")
            ax.set_title(f"Production engine CAT recovery: {d}\n{label} -- "
                         f"r = {s['r']:.3f}, slope = {s['slope']:.3f} (n = {len(df)})")
            ax.legend(loc="upper left", fontsize=9)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{prefix}_recovery_{d}.png", dpi=130)
            plt.close(fig)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "driver": "scripts/offline_engine_driver.py",
        "engine": "tutor_cat.engine.run_evaluation (production, unmodified)",
        "selection": "tutor_cat.selector.select_next targeting argmax(se)",
        "bank": str(args.bank), "matrix": str(args.matrix),
        "matrix_sha256": mat_sha, "provenance_aligned": bool(aligned),
        "dims": dims, "negatives": neg_stats,
        "config": {"seed": args.seed, "top_n": args.top_n, "max_se": args.max_se,
                   "min_evals_per_skill": args.min_evals_per_skill,
                   "max_scenarios": args.max_scenarios,
                   "unmapped_criteria": args.unmapped_criteria,
                   "grid_nodes_per_dim": args.grid, "range": args.range},
        "n_models": int(len(df)),
        "stop_reasons": stop_reasons,
        "scenarios_administered": {
            "mean": float(df.scenarios_administered.mean()),
            "median": float(df.scenarios_administered.median()),
            "max": int(df.scenarios_administered.max())},
        "criteria_administered": {
            "mean": float(df.criteria_administered.mean()),
            "median": float(df.criteria_administered.median()),
            "max": int(df.criteria_administered.max())},
        "precision_reached": int(df.precision_reached.sum()),
        "recovery": agg,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n" + "=" * 96)
    print("RESULTS")
    print("=" * 96)
    print(f"stop reasons        : {stop_reasons}")
    print(f"precision reached   : {int(df.precision_reached.sum())}/{len(df)}")
    print(f"scenarios/model     : mean={df.scenarios_administered.mean():.1f} "
          f"median={df.scenarios_administered.median():.0f} max={df.scenarios_administered.max()}")
    print(f"criteria/model      : mean={df.criteria_administered.mean():.1f} "
          f"median={df.criteria_administered.median():.0f}")
    print(f"judge cells missing : {int(df.judge_missing.sum())}")
    print()
    hdr = f"{'dim':14s}" + "".join(f"{estimators[p]:>22s}" for p in estimators)
    print(hdr)
    for d in dims:
        print(f"{d:14s}" + "".join(
            f"  r={agg[p][d]['r']:.3f} m={agg[p][d]['slope']:.3f}" for p in estimators))
    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
