# Handoff: live two-machine biggen CAT smoke run

This is the operator-facing walkthrough for running the smoke and verifying it. Full
tooling reference is in [README.md](README.md).

## Topology

- **Tutor (candidate)** runs locally on the eval job's GPU (`vllm_server`).
- **Judge** (`Qwen/Qwen3.5-9B`, vLLM 0.26.0) runs on a **separate machine**, reached over
  `base_url`. It is validated as an external endpoint (`/version` == 0.26.0), never
  launched by the runner.

Two things the operator chooses; nothing else changes:
- `--judge-base-url` (where the judge server is hosted), and
- `--tutor-model` / `--tutor-revision` (the candidate checkpoint).

## What the smoke exercises

`olmo-eval run-modes` starts both providers once, then runs the `edullm_adaptive` live
CAT loop. For each scenario the CAT engine selects:

1. the local tutor generates a response;
2. the remote judge grades each rubric criterion (binary pass/fail via explicit P/F token
   probabilities; `failure_probability_threshold = 0.33`); blank/unparseable results are
   `no_decision` (missing data, never an automatic fail);
3. the graded outcomes update the ability estimate (theta); the engine then selects the
   next scenario.

Selection cannot advance until the current scenario is graded -- that sequential
dependency is the "grade after every scenario" requirement, with both machines live in
the loop. Stop rule (biggen locked): EAP-posterior SD <= 0.12, floor 8 scenarios, capped
at `max_scenarios` (12 for the smoke).

What the smoke proves: both machines connect, tutor generates, remote judge grades, theta
updates, selection is adaptive, the stop rule fires, artifacts are written. It does NOT
produce a meaningful theta -- that needs the real checkpoint and a higher `max_scenarios`.

## Run steps

1. **Judge up (judge machine, GPU):**
   ```bash
   JUDGE_PORT=8000 bash scripts/edullm/serve_frozen_judge.sh
   python scripts/edullm/verify_judge_endpoint.py --base-url http://JUDGE_HOST:8000/v1
   ```
   Expect `version ok` and `logprob_token_ids accepted`.

2. **Conform the bank + generate the config (laptop):** see README steps 1-2, passing the
   real `--judge-base-url` (and `--tutor-model`/`--tutor-revision` if not smoking with the
   default small model).

3. **Preflight against the live judge (eval job):**
   ```bash
   uv run olmo-eval run-modes --config run_two_machine_smoke.yaml --check
   ```
   This is the live version of `preflight_local.py`; it now actually probes the judge.

4. **Run the smoke (eval job):**
   ```bash
   uv run olmo-eval run-modes --config run_two_machine_smoke.yaml
   ```

## Verification checklist

- `verify_judge_endpoint.py` passes both checks.
- `--check` returns `preflight_passed` with `provider_names` = `candidate`, `judge` and a
  `judge_runtime.version` of `0.26.0`.
- During the run, `<output_dir>/modes/edullm_adaptive/cat_trace.jsonl` grows one row per
  scenario (the live loop).
- `cat_result.json`: `stop_reason` is the SE target or the `max_scenarios` cap;
  `criteria_no_decision` is small. A large `no_decision` count means the judge
  prompt/endpoint is misbehaving -- investigate before trusting any full run.
- `manifest.json` records `metadata.judge_policy_provenance`.

## Open assumption to watch

The integration judge (`local-judge-v4.3.1-curated` / `atomic-json-v4`) is treated as
equivalent to biggen's calibration-time grader (`judge-validation-v3` / `generic-binary`):
same Qwen weights and revision, different prompt/adapter lineage. No document asserts
equivalence. If the smoke shows a high `no_decision` rate or skewed pass rates, resolve
this before a full run.

## Cost and time

Authoritative platform cost/approval is NOT quoted here. Per [AGENTS.md](../../AGENTS.md),
prices, runtime bounds, and approval class live in reviewed platform config; read them
from the free, no-GPU command once the config is committed:

```bash
edullm check --json    # read `cost` and `approval_class` from the output
```

Engineering notes on the drivers (not billing):

- This is two separate GPU allocations (judge box + tutor box), likely billed
  independently. The judge is persistent and accrues time for its whole uptime, not just
  during the smoke -- bring it up, verify, run, tear it down.
- Smoke cost is dominated by judge weight download (~18 GB, first run only, cached after)
  and vLLM startup (~1-3 min), not inference. The loop is 8-12 tutor generations plus
  ~30-45 one-token judge calls, run sequentially (CAT is adaptive, cannot batch across
  scenarios), so raw inference is seconds to a few minutes.
- Judge sizing is the constraint: `Qwen3.5-9B` bf16 fits a 24 GB card (L4 / A10G / 4090)
  with tight KV headroom, comfortably on 40-48 GB (L40S / A6000 / A100-40G). A bigger
  judge GPU does not speed the sequential smoke loop; it only trims startup and adds KV
  headroom. The tutor box only needs to fit the chosen candidate checkpoint.
- End-to-end, once weights are cached, expect roughly 10-15 min wall clock, mostly
  startup/download rather than the eval itself.

## Full run (after smoke passes)

Regenerate the config with the real candidate checkpoint and a higher `--max-scenarios`
(biggen deploys ~19 scenarios median). The locked SE/floor/ridge stay as-is; the operating
point is per-benchmark and must not be changed or reused across banks.
