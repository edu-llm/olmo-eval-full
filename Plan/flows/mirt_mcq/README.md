# Plan — `flow/mirt-mcq`: the multidimensional (MIRT) MCQ CAT style

Branch goal: add ONE registered style, `mirt_mcq`, under
`diagnostics/mcq_cat/styles/mirt_mcq/`, that runs a **multidimensional** adaptive
multiple-choice diagnostic (per-skill ability vector) on a checkpoint. Same
disjoint-directory discipline as the other flows — nothing in the frozen
scaffolding is edited.

Like `uni_mcq`, MCQ grading is **in-process log-likelihood scoring** (no LLM judge,
no served endpoint). The frozen scorer in `diagnostics/mcq_cat/common/inference.py`
already produces the per-item correct/incorrect signal; the MIRT-specific work is
(a) a **multidimensional CAT script** and (b) a **calibrated multidimensional bank**.

## Reality check: the MIRT-MCQ bank must be calibrated (it does not exist yet)

Every calibrated MCQ bank in the adaptive-evals work is **unidimensional** — all
ATLAS / Research fits call `mirt(dat, 1, ...)` (single latent axis). There is **no**
MCQ discrimination-vector / q-matrix bank on disk. The only multidimensional IRT in
the repo is FRQ-side (`tutor_cat/mirt.py`, 3 skills), which is a modeling *pattern*
to port, not MCQ data to reuse.

So this branch has an explicit calibration prerequisite. The frozen loader already
supports the target shape: `diagnostics/mcq_cat/common/irt_params.py` reads a
**list-valued `discrimination`** as MIRT and sets `dimensions = len(list)`.

### Calibration inputs that already exist

| Artifact | Path | Use |
|---|---|---|
| Per-model 0/1 response matrices | `AdaptiveTesting/Inputs/Open/LLM-Judge/mcq/<bench>/<model>.csv` (`question_id,model,benchmark,predicted,gold,result,...`) | rows = models, cols = items → the calibration matrix |
| OpenLM matrices | `AdaptiveTesting/Inputs/OpenLM/<bench>/<model>.csv` | additional persons/benchmarks |
| Item stems | `AdaptiveTesting/Outputs/arc_easy_questions.json`, `AdaptiveTesting/Inputs/MCQ/Benchmarks/*.jsonl` | the administered items |

### Two viable calibration routes (pick one in `FLOW_PACKAGE.md`)

1. **Exploratory multidim 2PL** on a single benchmark: fit `mirt(dat, 2)` (or 3) over
   the model×item response matrix; ship the rotated per-item discrimination vector +
   scalar difficulty. Dimensionality is data-driven (no authored q-matrix).
2. **Confirmatory M2PL with a domain q-matrix**: treat *benchmarks/skills* as the
   axes (e.g. reasoning vs knowledge vs math) by pooling `arc/gsm8k/hellaswag/...`
   response matrices into one person×item matrix with a q-matrix marking each item's
   skill(s); fit confirmatory M2PL (mirror the FRQ `scripts/calibrate_mirt.py`
   pattern). This yields interpretable per-skill θ.

The calibration itself is a lab step (kept in `evidence/` as a one-shot script +
logs); only the fitted bank graduates.

## What this branch adds (all under `diagnostics/mcq_cat/styles/mirt_mcq/`)

```
diagnostics/mcq_cat/styles/mirt_mcq/
  __init__.py        # @register("mirt_mcq")
  style.py           # multidim CAT script: vector estimate_ability + multidim selector + stopping
  config.yaml        # dimensions, skill names, se policy (per-dim or aggregate), max_items
  bank/
    params.jsonl     # fitted MIRT bank: discrimination VECTOR + q_modeled + difficulty
    items.jsonl      # stems + choices + gold_index
  FLOW_PACKAGE.md    # calibration route, n_persons, n_dims, skill names, matrix sha256, rotation
  evidence/          # dimensionality justification (scree/parallel analysis, per-skill loadings) + fit script
  tests/
```

### Bank schema (frozen loader, MIRT form)

```json
{"item_id": "Mercury_417466", "difficulty": -0.12,
 "discrimination": [1.10, 0.42], "guessing": 0.0,
 "q_modeled": {"reasoning": 1, "knowledge": 1}}
```

`discrimination` is a list (→ `dimensions = 2`); `q_modeled` names the axes. All items
in the bank must share the same dimensionality (the loader enforces this).

### The CAT script — `style.py` (multidimensional)

- `estimate_ability` → **vector θ** with per-dimension SE. Simplest robust choice:
  Bayesian MAP with a K×K posterior-covariance update (mirror `tutor_cat/mirt.py`'s
  3×3 `U` update, generalized to K dims); return `AbilityEstimate(theta=tuple,
  standard_error=tuple)`. (Full multidim EAP on a K-D grid is an option for K≤2.)
- `select_next_item` → **multidimensional Fisher information**: pick the item
  maximizing a scalarization of the Fisher information matrix at current θ
  (D-optimality `det(I)` or trace), among un-administered items.
- `stopping_rule` → stop when **every** modeled skill's SE < its threshold (or
  `max_items`); expose per-skill thresholds in `config.yaml`.
- `report` → per-skill θ + SE vector in `CATReport`.

### `config.yaml`

```yaml
style: mirt_mcq
dimensions: 2
skills: [reasoning, knowledge]
se_threshold: 0.3          # applied per-skill
max_items: 60
min_items: 12
selection: d_optimality    # d_optimality | trace
```

## How it runs (unchanged seam)

```bash
DIAGNOSTIC_MODALITY=mcq_cat CAT_STYLE=mirt_mcq \
  CHECKPOINT=s3://.../step_1000 tests/aws/run_checkpoint_diag.sh
```

## Sequence for this branch

1. Decide calibration route (exploratory vs confirmatory) and record it in
   `FLOW_PACKAGE.md`.
2. Write the one-shot fit script (`evidence/fit_mirt_mcq.py`) over the response
   matrices; emit `bank/params.jsonl` (vector `a`, `q_modeled`) + dimensionality
   evidence figures.
3. Implement `style.py` (vector estimator + multidim selector + per-skill stopping).
4. Port items → `bank/items.jsonl` (ids aligned to params).
5. Verify (below), then PR into `CheckpointFlows`.

## Verification before PR

1. `ruff` + `ty` on `diagnostics/mcq_cat/styles/mirt_mcq`.
2. `--list-styles` shows `mirt_mcq`; loader reports `dimensions == len(skills)`.
3. `tests/`: schema-load asserts vector discrimination + consistent dims; tiny 2-D
   CAT smoke on a toy bank (per-skill SE decreases and both cross threshold).
4. `run_checkpoint_diag.sh --dry-run` for `CAT_STYLE=mirt_mcq`.

## Merge-conflict discipline

- Touch **only** `diagnostics/mcq_cat/styles/mirt_mcq/**` and this `Plan/flows/mirt_mcq/`.
- Do not edit shared scaffolding. The frozen loader/`base.py` already model vector
  discrimination + `dimensions`; if a genuinely new contract field is needed, change
  it once on `CheckpointFlows` for all styles rather than here.
- Register exactly once, in `styles/mirt_mcq/__init__.py`.
