# Plan — `flow/mirt-frq`: the multidimensional (MIRT) FRQ CAT style

Branch goal: add ONE registered style, `mirt_frq`, under
`diagnostics/frq_cat/styles/mirt_frq/`, that runs a **multidimensional** (2-skill
M2PL) adaptive free-response diagnostic on a checkpoint. Same disjoint-directory
discipline — the frozen `diagnostics/frq_cat/` scaffolding is not edited.

Identical two-stage machinery to `uni_frq` (shared frozen respgen + shared frozen
generic-binary judge); the only differences are the **bank shape** (per-skill
discrimination vectors + a real `q_modeled`) and the **CAT script** (a vector-θ
M2PL estimator + a multidimensional selector). Per the FRQ design, MIRT is "a
different payload + a MIRT estimator/selector, not a different contract."

## Source materials

### The graduated release — `origin/frq/tutorbench` (target `flow/mirt-frq`)

Read via `git show origin/frq/tutorbench:eduLLM-Evals/<path>`:

| Artifact | Path (on `origin/frq/tutorbench`) | Notes |
|---|---|---|
| `FLOW_PACKAGE.md` | `eduLLM-Evals/data/TutorBench/FLOW_PACKAGE.md` | tag `tutorbench-2skill-115models`; N=115 |
| Fitted bank (canonical) | `eduLLM-Evals/data/TutorBench/rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl` | 3,666 criteria; 2-skill M2PL |
| Scenarios | `eduLLM-Evals/data/TutorBench/scenarios.jsonl` | 662 scenarios (multi-turn: populated `conversation_context`) |
| Judge config | `eduLLM-Evals/judge_frozen.yaml` | same frozen Qwen3.5-9B `generic-binary` judge |
| Evidence | TutorBench cat_eval reports + skillfit artifacts on branch | why 2 skills (`correctness`, `scaffolding`) |

Skills: **`correctness`, `scaffolding`** (the earlier content+diagnosis skills were
collapsed). Bank record (per-skill discrimination vector keyed by modeled skill):

```json
{"criterion_id": "...", "scenario_id": "...",
 "difficulty": 1.8928,
 "discrimination": {"correctness": 2.11, "scaffolding": 0.34},
 "q_modeled": {"correctness": 1, "scaffolding": 1}}
```

> Alternative reference: TutorEval also has a *non-shipped* 2-skill variant
> `origin/frq/tutoreval:eduLLM-Evals/data/TutorEval/rubrics_qmatrix_final_2skill_fitted.jsonl`
> (`conceptual_understanding`, `quantitative_procedural`). TutorBench is the canonical
> MIRT graduation; TutorEval-2skill is a fallback if a single-scenario-schema bank is
> preferred. Pick one in `FLOW_PACKAGE.md`.

Multi-turn scenarios: `conversation_context` carries `[{role: student|tutor,
content}]` turns; the frozen `common/respgen`/`build_tutor_messages` maps
`student→user`, `tutor→assistant`, then appends `prompt` as the final user turn.

### Shared judge + tutor scripts (same as `flow/uni-frq`)

The frozen respgen + judge are shared; port from `stash@{0}:diagnostics/frq_cat/
{checkpoint.py, judge.py, config/judge_frozen.yaml}` and the working-tree
`full_bank_{respgen,judge}.py`. See `Plan/flows/uni_frq/README.md` for the exact
mapping — it is identical here. MIRT changes nothing about grading.

### The MIRT CAT engine to port

Reference multidimensional engine (adaptive-evals / frq-lab):
- `tutor_cat/mirt.py` — M2PL likelihood + Bayesian MAP with a K×K posterior
  covariance `U` (the reference is 3-skill; use K=2 here).
- `scripts/cat_eval_tutorbench_multiskill.py` — TutorBench 2-skill CAT eval (selection
  + per-skill recovery), the closest existing driver to replicate.
- `tutor_cat/{engine,selector}.py` — the adaptive loop + item/scenario selection.

## What this branch adds (all under `diagnostics/frq_cat/styles/mirt_frq/`)

```
diagnostics/frq_cat/styles/mirt_frq/
  __init__.py        # @register("mirt_frq")
  style.py           # MIRT CAT script: vector M2PL estimate_ability + multidim selector + stopping + report
  config.yaml        # dimensions=2, skills, judge selection, per-skill se policy, max_items
  bank/
    params.jsonl     # graduated 2-skill bank (vector discrimination + q_modeled), ported to common/irt_params.py
    scenarios.jsonl  # graduated multi-turn scenarios
  judge_frozen.yaml  # shared frozen judge selection (NOT forked)
  FLOW_PACKAGE.md    # tag tutorbench-2skill-115models, n_persons=115, sha256, skill names, negative-loading policy
  evidence/          # dimensionality evidence (why 2 skills) + per-skill loading figures
  tests/             # schema-load + tiny 2-skill CAT smoke (stub respgen + stub judge)
```

### Port the bank → `bank/params.jsonl`

Conform TutorBench's `rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl` to the
frozen `common/irt_params.py` schema. The loader normalizes a **mapping-valued**
`discrimination` (skill-name → value) into a fixed-order vector and sets
`dimensions = len(skills)` — so keep `discrimination`/`q_modeled` keyed by skill
name; do not flatten to the legacy 3-slot `{content, diagnosis, scaffolding}` layout
(the TutorBench FLOW_PACKAGE calls this out explicitly).

### The CAT script — `style.py` (multidimensional)

- `download_bank` / `load_irt_params` → load scenarios + the 2-skill bank
  (`dimensions=2`).
- `estimate_ability` → **vector θ** = (correctness, scaffolding) via M2PL MAP with a
  2×2 posterior covariance; return `AbilityEstimate(theta=tuple, standard_error=tuple)`
  (mirror `tutor_cat/mirt.py`).
- `select_next_item` → multidimensional Fisher information over un-administered
  criteria (D-optimality `det(I)` or trace) at current θ; honor per-scenario respgen
  caching in the frozen `cat_loop`.
- `stopping_rule` → stop when **both** skills' SE < threshold (or `max_items`); MIN_ITEMS floor.
- `report` → per-skill θ + SE vector + administered criteria/verdicts.

### `config.yaml`

```yaml
style: mirt_frq
fit_family: m2pl
benchmark: tutorbench
skills: [correctness, scaffolding]
dimensions: 2
judge_config: judge_frozen.yaml
se_threshold: 0.3        # applied per-skill
max_items: 60
min_items: 12
selection: d_optimality
```

## How it runs (unchanged seam)

```bash
DIAGNOSTIC_MODALITY=frq_cat CAT_STYLE=mirt_frq \
  CHECKPOINT=s3://.../step_1000 \
  TUTOR_ENDPOINT=http://localhost:8000/v1 \
  JUDGE_ENDPOINT=http://localhost:8001/v1 \
  JUDGE_CONFIG=diagnostics/frq_cat/styles/mirt_frq/judge_frozen.yaml \
  tests/aws/run_checkpoint_diag.sh
```

## Verification before PR

1. `ruff` + `ty` on `diagnostics/frq_cat/styles/mirt_frq`.
2. `--list-styles` shows `mirt_frq`; loader reports `dimensions == 2`; `--dry-run` resolves it.
3. `tests/`: schema-load asserts vector/skill-keyed discrimination + `q_modeled` over
   2 skills (3,666 criteria / 662 scenarios); tiny 2-skill CAT smoke (both per-skill
   SEs decrease and cross threshold; report serializes θ vector).
4. Frozen-judge sanity: `adapter == generic-binary`,
   `revision == c202236235762e1c871ad0ccb60c8ee5ba337b9a`, `enable_thinking is False`.
5. `run_checkpoint_diag.sh --dry-run` for `CAT_STYLE=mirt_frq` prints both endpoints + judge config.

## Merge-conflict discipline

- Touch **only** `diagnostics/frq_cat/styles/mirt_frq/**` and this `Plan/flows/mirt_frq/`.
- Do NOT fork the judge or respgen; MIRT reuses them unchanged and selects the judge
  via `config.yaml`.
- The frozen `IRTBank`/loader already model vector discrimination + `dimensions`; any
  new contract need is fixed once on `CheckpointFlows`, not here.
- Register exactly once, in `styles/mirt_frq/__init__.py`.
