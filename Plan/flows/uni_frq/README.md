# Plan — `flow/uni-frq`: the unidimensional FRQ CAT style

Branch goal: add ONE registered style, `uni_frq`, under
`diagnostics/frq_cat/styles/uni_frq/`, that runs a unidimensional (2PL) adaptive
**free-response** diagnostic on a checkpoint and lands a `cat_report.json` in S3.
Same disjoint-directory discipline — nothing in the frozen `diagnostics/frq_cat/`
scaffolding (`base`, `registry`, `runner`, `common/*`) is edited.

FRQ is two-stage and both stages are **shared** (frozen in `common/`): the
checkpoint under test generates a response per scenario (respgen / tutor), then a
frozen LLM-as-a-judge grades each rubric criterion pass/fail. Each **criterion** is
the IRT item. So this branch delivers: (1) the graduated TutorEval calibration
package, (2) a thin `style.py` (the CAT script), and it *depends on* the frozen
`common/respgen.py` + `common/judge.py` being finished from the source scripts below.

## Source materials

### The graduated release — `origin/frq/tutoreval` (target `flow/uni-frq`)

Read via `git show origin/frq/tutoreval:eduLLM-Evals/<path>`:

| Artifact | Path (on `origin/frq/tutoreval`) | Notes |
|---|---|---|
| `FLOW_PACKAGE.md` | `eduLLM-Evals/data/TutorEval/FLOW_PACKAGE.md` | tag `tutoreval-unidim-52models`; PROVISIONAL N=52 (`low_n`) |
| Fitted bank (canonical) | `eduLLM-Evals/data/TutorEval/rubrics_qmatrix_final_unidim_fitted.jsonl` | 1,186 criteria; unidim 2PL |
| Scenarios | `eduLLM-Evals/data/TutorEval/scenarios_final.jsonl` | 828 scenarios (single-turn: `conversation_context: []`) |
| Judge config | `eduLLM-Evals/judge_frozen.yaml` | Qwen3.5-9B `generic-binary`, frozen |
| Dimensionality evidence | `eduLLM-Evals/regenerated_figures/tutoreval_skillfit/` | ρ=0.981, ΔAIC/ΔBIC favor unidim |

Bank record (unidim — discrimination keyed by the modeled skill `ability`):

```json
{"criterion_id": "te_0000_c01", "scenario_id": "te_0000",
 "criterion": "this is not necessarily true", "primary_skill": "conceptual_understanding",
 "difficulty": -0.1155, "discrimination": {"ability": 0.68798}, "q_modeled": {"ability": 1}}
```

Scenario record:

```json
{"scenario_id": "te_0000", "prompt": "Anhedonia seems to be a common feature ...",
 "conversation_context": [], "reference_solution": "", "criterion_ids": ["te_0000_c01","te_0000_c02"]}
```

Judge config (`judge_frozen.yaml`):

```yaml
judge:
  judge_name: qwen
  model: Qwen/Qwen3.5-9B
  hf_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a
  adapter: generic-binary
  prompt_version: judge-validation-v3
  normalization_version: judge-normalization-v3
  evidence_policy_version: criterion-evidence-gate-v1
  temperature: 0.0
  max_tokens: 1024
  seed: 42
```

### The judge + checkpoint scripts to port (from `stash@{0}` + working tree)

These are the "tutor eval" scripts. `checkpoint.py`/`judge.py` sources live in
`git stash@{0}` ("On MockTraining: wip before checkpoint flows switch"); the
full-bank drivers are untracked in the working tree:

| Source | What to lift into the frozen `common/` |
|---|---|
| `stash@{0}:diagnostics/frq_cat/checkpoint.py` (`VLLMOpenAITutor.respond`, `build_tutor_messages`, `materialize_arm`, `merge_lora_to_hf`) | `common/respgen.py`: served tutor client + the HTTP-400 context-shrink retry + TIMING logs |
| `stash@{0}:diagnostics/frq_cat/judge.py` (`QwenJudgeClient`, `JudgeSpec`, `spec_from_config`, `build_messages` generic-binary v3, `parse_judgment`, `apply_evidence_gate`) | `common/judge.py`: the frozen judge (identity + prompt v3 + evidence gate) |
| `stash@{0}:diagnostics/frq_cat/config/judge_frozen.yaml` + `judge_generic_binary_prompt_v3.txt` + `judge_evidence_policy_v3.txt` | vendored judge assets the style's `config.yaml` selects |
| `diagnostics/frq_cat/full_bank_respgen.py` / `full_bank_judge.py` (working tree) | reference for CLI, S3 layout (`responses.jsonl`, `judge_results.jsonl`, `_READY`/`_READY_JUDGE`), and the frozen-judge sanity assertions (`adapter == generic-binary`, `revision == c202...`, `enable_thinking is False`) |

> Note the frozen scaffolding currently ships a *stub* `ServedJudge`/`ServedRespGen`.
> Finishing them to match the stashed generic-binary judge + tutor is scaffolding
> work that belongs on `CheckpointFlows` (shared, frozen once). This branch should
> assume that contract and only add the style; if a gap is found, fix it on
> `CheckpointFlows`, not here.

### On-node serving (from `.frq_respgen_run/` + `.frq_judge_run/`)

Two separate vLLM servers, both OpenAI-compatible on `:8000`:
- Tutor: `vllm serve <checkpoint> --served-model-name <name> --max-model-len 16384 ...`
- Judge: `vllm serve <Qwen3.5-9B snapshot> --served-model-name Qwen/Qwen3.5-9B --enforce-eager --limit-mm-per-prompt '{"image":0,"video":0}'`

The `run_checkpoint_diag.sh` seam already threads `TUTOR_ENDPOINT`, `JUDGE_ENDPOINT`,
and `JUDGE_CONFIG`; wiring both serves onto the node (one box, two ports, or two
boxes) is the FRQ integration point referenced in the wiring plan.

## What this branch adds (all under `diagnostics/frq_cat/styles/uni_frq/`)

```
diagnostics/frq_cat/styles/uni_frq/
  __init__.py        # @register("uni_frq")
  style.py           # the CAT script: unidim 2PL estimate_ability + selector + stopping + report + bank load
  config.yaml        # fit family (2pl), judge selection (judge_frozen.yaml), se_threshold, max_items
  bank/
    params.jsonl     # graduated fitted bank, ported to common/irt_params.py schema
    scenarios.jsonl  # graduated scenarios (prompt + conversation_context)
  judge_frozen.yaml  # copied frozen judge selection (identity only; judge code is NOT forked)
  FLOW_PACKAGE.md    # provenance: tag, n_persons=52, matrix sha256, low_n caveat, rerun/swap protocol
  evidence/          # tutoreval_skillfit figures + captions (why unidim)
  tests/             # schema-load + tiny CAT smoke with a stub respgen + stub judge
```

### Port the bank → `bank/params.jsonl`

Conform TutorEval's `rubrics_qmatrix_final_unidim_fitted.jsonl` to the frozen
`common/irt_params.py` schema (criterion-keyed; `discrimination` keyed by modeled
skill; scalar `difficulty`; `q_modeled`). The loader already accepts skill-keyed
mapping discrimination and `q_modeled`, so this is data-shaping, not a loader edit.

### The CAT script — `style.py`

- `download_bank(benchmark)` → load `bank/scenarios.jsonl` + `bank/params.jsonl` via
  `common/bank_loader` into an `FrqBank` (criteria are the items).
- `load_irt_params(source)` → `common/irt_params.load_irt_params` (unidim →
  `dimensions=1`).
- `estimate_ability(bank, responses, previous)` → unidimensional 2PL EAP over graded
  criterion outcomes (pass=1/fail=0); scalar θ + SE.
- `select_next_item(bank, state)` → max Fisher information over un-administered
  criteria at current θ (2PL). Respect that selecting a criterion may trigger a
  one-time respgen of its scenario (the frozen `cat_loop` caches per-scenario).
- `stopping_rule(state)` → SE < `se_threshold` (default 0.3) or `max_items`; MIN_ITEMS floor.
- `report(state)` → θ, SE, administered criteria + verdicts.

### `config.yaml`

```yaml
style: uni_frq
fit_family: 2pl
benchmark: tutoreval
judge_config: judge_frozen.yaml
se_threshold: 0.3
max_items: 40
min_items: 8
```

## How it runs (unchanged seam)

```bash
DIAGNOSTIC_MODALITY=frq_cat CAT_STYLE=uni_frq \
  CHECKPOINT=s3://.../step_1000 \
  TUTOR_ENDPOINT=http://localhost:8000/v1 \
  JUDGE_ENDPOINT=http://localhost:8001/v1 \
  JUDGE_CONFIG=diagnostics/frq_cat/styles/uni_frq/judge_frozen.yaml \
  tests/aws/run_checkpoint_diag.sh
# → uv run python -m diagnostics.frq_cat.runner --cat-style uni_frq --tutor-endpoint ... --judge-endpoint ... --judge-config ...
```

## Verification before PR

1. `ruff` + `ty` on `diagnostics/frq_cat/styles/uni_frq`.
2. `python -m diagnostics.frq_cat.runner --list-styles` shows `uni_frq`; `--dry-run` resolves it.
3. `tests/`: schema-load of the ported bank (assert `discrimination` skill-keyed,
   `q_modeled == {"ability":1}`, 1,186 criteria / 828 scenarios); tiny CAT smoke with
   a stub respgen + stub judge (SE decreases, terminates, report serializes).
4. Frozen-judge sanity: assert selected judge `adapter == generic-binary`,
   `revision == c202236235762e1c871ad0ccb60c8ee5ba337b9a`, `enable_thinking is False`.
5. `run_checkpoint_diag.sh --dry-run` for `CAT_STYLE=uni_frq` prints both endpoints + judge config.

## Merge-conflict discipline

- Touch **only** `diagnostics/frq_cat/styles/uni_frq/**` and this `Plan/flows/uni_frq/`.
- Do NOT fork the judge or respgen; select the shared frozen ones via `config.yaml`.
- Any shared gap (finishing `common/judge.py`/`common/respgen.py` to the stashed
  generic-binary contract, or a new `base.py` field) is fixed once on `CheckpointFlows`.
- Register exactly once, in `styles/uni_frq/__init__.py`.
