# Plan — `flow/uni-mcq`: the unidimensional MCQ CAT style

Branch goal: add ONE registered style, `uni_mcq`, under
`diagnostics/mcq_cat/styles/uni_mcq/`, that runs a unidimensional (1PL/2PL/3PL)
adaptive multiple-choice diagnostic on a checkpoint and lands a `cat_report.json`
in S3. Everything lands under one disjoint directory so this branch PRs back into
`CheckpointFlows` without conflicts. Nothing in the frozen scaffolding
(`diagnostics/mcq_cat/{base,registry,runner}.py`, `common/*`) is edited.

MCQ grading is **in-process log-likelihood scoring** — there is *no* LLM judge and
*no* served endpoint. So on this branch "judging script" == the in-process scorer
already frozen in `diagnostics/mcq_cat/common/inference.py`; the real work is the
**CAT script** (IRT model + selector + stopping) plus **porting a calibrated IRT
bank** into the frozen loader schema.

## Source materials (already in the tree / adaptive-evals work)

Canonical calibrated banks (`AdaptiveTesting/`, from the adaptive-evals work):

| Artifact | Path | Schema |
|---|---|---|
| ATLAS 3PL params (arc, gsm8k, hellaswag, winogrande, truthfulqa, ifeval, math) | `AdaptiveTesting/Inputs/ATLAS/<bench>/irt_item_parameters_combined.csv` | `X,a1,d,g,u` |
| ATLAS 2PL replications | `AdaptiveTesting/Inputs/ATLAS/experiments/<bench>_2pl/irt_item_parameters_combined.csv` | `X,a1,d,g,u` (`g=0`) |
| ATLAS idx→id bridge | `AdaptiveTesting/Inputs/ATLAS/<bench>/atlas_idx_to_question_id.csv` | `atlas_idx,question_id,question` |
| Local py-irt 2PL cat banks | `AdaptiveTesting/Outputs/cat_bank_arc_easy.csv`, `cat_bank_arc_challenge_60.csv` | `item,a,b,gold` |
| Item stems/choices | `AdaptiveTesting/Outputs/arc_easy_questions.json` (`{id:{stem,options,answerKey}}`), `AdaptiveTesting/Inputs/MCQ/Benchmarks/*.jsonl` (`qid,prompt,options,gold_index`) |

Reference CAT engine (adaptive-evals): `tutor_cat.mcq_irt` — `ability.eap` (EAP on a
normal grid), `cat.run_cat` (max Fisher-information selection, SE-threshold stop).
Reference drivers: `AdaptiveTesting/Outputs/{cat_mcq_olmo.py, diagnostic_validation.py,
atlas_diagnostic_validation.py, cat_kfold_holdout.py}`.

**ATLAS → standard IRT conversion** (from `atlas_diagnostic_validation.load_atlas_item_bank`):
`a = a1`, `b = -d / a1`, `c = clip(g, 0, 0.999)`. 2PL banks have `g = 0` → `c = 0`.

## What this branch adds (all under `diagnostics/mcq_cat/styles/uni_mcq/`)

```
diagnostics/mcq_cat/styles/uni_mcq/
  __init__.py        # @register("uni_mcq") — the ONLY registration point
  style.py           # the CAT script: estimate_ability + select_next_item + stopping_rule + report + loaders
  config.yaml        # fit family (1pl|2pl|3pl), default benchmark, se_threshold, max_items, grid
  bank/
    params.jsonl     # graduated IRT bank, ported to the frozen loader schema
    items.jsonl      # question stems + choices + gold_index (the administered items)
  FLOW_PACKAGE.md    # provenance: source CSV, benchmark, param family, conversion, item count, sha256
  evidence/          # calibration provenance (fit logs / validation scatter from diagnostic_validation.py)
  tests/             # schema-load test + a tiny CAT smoke over a 5-item toy bank
```

### 1. Port the IRT bank → `bank/params.jsonl` (frozen loader schema)

Target schema is `diagnostics/mcq_cat/common/irt_params.py` (`item_id`, `difficulty` (`b`),
`discrimination` (`a`, scalar for unidim), `guessing` (`c`)):

```json
{"item_id": "Mercury_417466", "difficulty": -0.119675, "discrimination": 1.098289, "guessing": 0.0}
```

Write a one-shot, offline port script (kept in `evidence/` or `tests/`, not imported
at runtime) that reads the chosen source CSV, applies the ATLAS→(a,b,c) conversion,
joins `atlas_idx_to_question_id.csv` for stable `item_id`s, and emits `params.jsonl`.
Start with **arc_challenge/arc_easy** (smallest, both param families available); the
same script generalizes to the other ATLAS benchmarks.

### 2. Port the items → `bank/items.jsonl`

Conform to the frozen `BenchmarkItem` (`item_id`, `question`, `choices`, `gold_index`).
Source `arc_easy_questions.json` (map `answerKey` letter → `gold_index`) or the
`MCQ/Benchmarks/*.jsonl` shape (already `qid/prompt/options/gold_index`). `item_id`
must match `params.jsonl` so the CAT loop can join params↔item.

### 3. The CAT script — `style.py` (implements `CatStyle` from `base.py`)

- `download_benchmark(benchmark)` → load `bank/items.jsonl` into a `BenchmarkBank`.
- `load_irt_params(source)` → delegate to `common/irt_params.load_irt_params` (scalar
  discrimination → `dimensions=1`).
- `estimate_ability(bank, responses, previous)` → **EAP** on a normal grid
  (61–81 nodes, θ∈[-4,4]); returns `AbilityEstimate(theta, standard_error)`. Port the
  math from `tutor_cat.mcq_irt.ability.eap`; use the 3PL likelihood when `guessing>0`.
- `select_next_item(bank, state)` → **max Fisher information** at current θ among
  un-administered items (2PL: `a²·P·(1-P)`; 3PL: `a²·((1-P)/P)·((P-c)/(1-c))²`).
- `stopping_rule(state)` → `state.ability.standard_error < se_threshold` (default 0.3;
  honor `state.se_threshold`/`state.max_items` and a `MIN_ITEMS` floor of ~8).
- `report(state)` → `CATReport` (theta, SE, administered items); optionally include a
  bank-mean model-implied accuracy prediction (`prob(theta).mean()`) as metadata.

Grading itself is the frozen in-process scorer (`common/inference.py`, log-likelihood
over choices) driven by `common/cat_loop.run_cat` — this branch does not touch it.

### 4. `config.yaml`

```yaml
style: uni_mcq
fit_family: 2pl        # 1pl | 2pl | 3pl (drives the likelihood/Fisher form)
benchmark: arc_challenge
se_threshold: 0.3
max_items: 50
min_items: 8
grid: {n_nodes: 81, theta_min: -4.0, theta_max: 4.0}
```

## How it runs (unchanged seam)

```bash
DIAGNOSTIC_MODALITY=mcq_cat CAT_STYLE=uni_mcq \
  CHECKPOINT=s3://.../step_1000 CHECKPOINT_KIND=hf \
  tests/aws/run_checkpoint_diag.sh
# → uv run python -m diagnostics.mcq_cat.runner --cat-style uni_mcq --checkpoint ... --s3-out ...
```

MCQ needs no tutor/judge endpoints; the box loads the checkpoint and scores in-process.

## Verification before PR

1. `uv run ruff check diagnostics/mcq_cat/styles/uni_mcq && uv run ty check ...`
2. `python -m diagnostics.mcq_cat.runner --list-styles` shows `uni_mcq`.
3. `--dry-run` resolves the style and prints the plan.
4. `tests/`: schema-load of `params.jsonl`/`items.jsonl` + a tiny CAT smoke on a toy
   5-item bank with a stub scorer (SE decreases, terminates, report serializes).
5. One real `--dry-run` of `run_checkpoint_diag.sh` for `CAT_STYLE=uni_mcq`.

## Merge-conflict discipline

- Touch **only** `diagnostics/mcq_cat/styles/uni_mcq/**` and this `Plan/flows/uni_mcq/`.
- Never edit `base.py`, `registry.py`, `runner.py`, `common/*`, or the seam. If the
  frozen loader can't represent a needed field, raise it on `CheckpointFlows` (change
  the contract once, for all styles) rather than editing shared code here.
- Registration happens exactly once, in `styles/uni_mcq/__init__.py`.
