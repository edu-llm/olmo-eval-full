# olmo-eval architecture — how the eval suite runs

Scope: the outer suite in `src/olmo_eval/`. This is a fork of
[allenai/olmo-eval](https://github.com/allenai/olmo-eval). This doc is the
integration surface: to run ATLAS "as an eval" you either add a `Task`, add an
external eval, or post-process the results this pipeline already produces.

The `AdaptiveTesting/` inference harness (see doc 02) is **standalone and does not
use `olmo_eval` at all** today — that is the gap the integration closes.

---

## 1. Package map (`src/olmo_eval/`)

| Dir | Responsibility |
|---|---|
| `cli/` | Click CLI: `run`, `beaker`, `results`, `metrics`, `task`, `suite`, `run-external`. Entry `olmo-eval = olmo_eval.cli:main` (`pyproject.toml`). |
| `common/` | Core primitives: `types/base.py` (`Instance`, `LMRequest`, `LMOutput`, `Response`), `formatters.py`, `scorers/`, `metrics/`, `configs.py` (`expand_tasks`). |
| `data/` | `DataLoader` / `DataSource` — HF / local / S3 / GCS. |
| `evals/` | `tasks/` (benchmark defs + registry), `suites/` (composition), `external/` (standalone benchmarks). |
| `inference/` | Provider abstraction (`base.py`) + factory (`create_provider`): `mock`, `vllm`, `vllm_server`, `litellm`, `hf`, `olmo_core`. |
| `harness/` | Runtime around a provider: system prompt, tools, scaffolds, sandbox, metrics. |
| `runners/` | Orchestration: `asynq/runner.py` (`AsyncEvalRunner`) for tasks, `external/runner.py` for external evals. |
| `launch/` | Beaker/gantry launch (`launch/beaker/launcher.py`). |
| `storage/` | Postgres backends + repos; `alembic/` holds two migration trees (results + metrics). |

Registration is by **import side-effect**: `cli/__init__.py` imports
`olmo_eval.evals`, and `evals/tasks/__init__.py` auto-imports every sibling
`*.py` via `pkgutil`, so any `@register(...)`-decorated task in
`evals/tasks/<name>.py` is discovered with no manual wiring.

---

## 2. The Task abstraction

Defined in `evals/tasks/common/base.py` (`Task`, `TaskConfig`); registry in
`evals/tasks/common/registry.py`.

A task is responsible for four things:

| Member | Role |
|---|---|
| `instances` (property → `Iterator[Instance]`) | Load dataset rows into `Instance`s. |
| `format_request(instance) → LMRequest` | Turn an instance into a model request (typically `RequestType.LOGLIKELIHOOD` for MCQ). |
| `extract_answer` / `score_responses` | Extract + score model output (usually inferred from metrics). |
| `metrics = (...)` | Class-level tuple of `Metric`s; scorers are derived from them. |

`@register("arc_challenge")` on the class registers it; `register_variant(...)`
adds named presets (`arc_challenge:mc`, `:bpb`, `:olmo3base`, …).

### Why ARC matters for ATLAS: the id is preserved

`evals/tasks/arc.py` builds each instance with the **native ARC id** in metadata:

```41:53:src/olmo_eval/evals/tasks/arc.py
    return Instance(
        question=question,
        choices=tuple(choices),
        gold_answer=letter,
        metadata={
            "id": doc.get("id", f"{dataset}_{index}"),
            ...
            "gold_idx": gold_idx,
```

So an ARC-Challenge instance carries `metadata["id"] = "Mercury_7175875"` — the
**same `question_id`** used by (a) `AdaptiveTesting/Inputs/ATLAS/arc/atlas_idx_to_question_id.csv`
and (b) the MCQ response CSVs. This is the join key that makes the integration
tractable. MCQ scoring is log-likelihood over lettered/text continuations
(`LogprobMCAccuracyMetric` argmaxes the per-choice logprob sums) — a per-item
correct/incorrect signal, exactly what an IRT response vector needs.

---

## 3. Single-task run, end to end

CLI: `olmo-eval run -m <model> -t <task>`.

```
cli:main → cli/run.run()
  → RunConfigBuilder.build()            # model preset + task specs + overrides
  → RunnerFactory.create() → AsyncEvalRunner        (runners/asynq/runner.py)
  → runner.run() → asyncio.run(run_async()):
       expand_tasks(specs)              # suites → concrete task specs
       _prepare_tasks():
           get_task(spec)               # registry resolves name+variants
           list(task.instances)         # ALL instances materialized up front
           task.format_request(inst)    # → QueueItem per instance
       ProviderManager.start()          # N inference workers (subprocesses)
           worker: harness.provider.alogprobs(batch)   # LOGLIKELIHOOD path
       process_results():
           ResultItem → Response
           task.score_responses([resp]) # inline async scoring
       compute_metrics() → aggregate_results() → _finalize_and_save()
```

Two facts that shape the integration:

1. **The pipeline is static/batch.** All instances are enumerated before any
   inference; there is no built-in "score this, then pick the next item" hook.
   True online CAT (only running the ~40 selected items) does **not** fit the
   default runner without a custom runner/scaffold. See doc 03 for why the
   bare-bones path sidesteps this.
2. **Per-instance predictions are first-class.** Scoring produces a `Response`
   per instance with `scores`, and the runner can write instance-level
   predictions (JSONL locally, and Postgres `instance_predictions` with a
   `native_id` when `--store`). That per-item 0/1 signal is the ATLAS input.

Core types (`common/types/base.py`): `Instance` (question, choices, gold_answer,
metadata), `LMRequest` (request_type, prompt, continuations…), `LMOutput`
(text, logprobs…), `Response` (instance, request, outputs, scores).

---

## 4. Suites

`evals/suites/registry.py`: a `Suite` is a named tuple of task specs (and/or
nested suites) with an `AggregationStrategy` (`AVERAGE`, `AVERAGE_OF_AVERAGES`,
`DISPLAY_ONLY`, `NONE`). `expand_tasks()` flattens suites to task specs before
running. An "ATLAS suite" could group the ATLAS-backed tasks (arc, hellaswag,
winogrande, …) once each has an adaptive variant.

---

## 5. Providers

`inference/base.py` defines the provider interface; `create_provider(kind, model,
**kw)` is the factory. MCQ tasks use the **logprob** path
(`alogprobs`) — one `LMOutput` per continuation, scored by argmax. For a first
ATLAS integration the `mock` provider (`inference/providers/mock.py`) is enough
to exercise wiring end-to-end with no GPU.

---

## 6. Launching on Beaker (the "beaker thing")

`olmo-eval beaker launch` (`cli/beaker/launch.py` → `job_assembler.py` →
`launch/beaker/launcher.py`). It assembles a `BeakerJobConfig` and calls gantry's
`launch_experiment`. Gantry clones the repo into the job, runs an install step,
then runs the remote command — which for tasks is literally:

```
olmo-eval run -O /results -m <model> -t <task> [--store ...]
```

Minimal task launch:

```bash
olmo-eval beaker launch -m <model> -t arc_challenge -c h100 -w ai2/oe-data -B ai2/oe-base --dry-run
```

So "run ATLAS on Beaker whenever it runs" = make the launched `olmo-eval run`
command execute an ATLAS-backed task (or external eval). No Beaker-specific glue
is needed beyond a normal task/eval registration; `--dry-run` prints the spec.

### External-eval alternative

`evals/external/` supports standalone benchmarks (`ExternalEval` /
`SandboxedExternalEval`, `register_external_eval`, `ExternalEvalResult`), run via
`olmo-eval run-external -e <name>` or `beaker launch -E <name>`. The parent starts
a vLLM server; the eval talks to it over an OpenAI-compatible API and returns
`metrics`. This is the path if ATLAS needs its own environment / online item loop
rather than fitting the static task runner. Trade-off table in doc 03.

---

## 7. Results storage

`--store` persists to Postgres (`storage/backends/postgres/models.py`): tables
`experiments`, `task_results` (nested `{metric: {scorer: score}}` + `primary_metric`),
and `instance_predictions` (per-item metrics keyed by `native_id` + `task_hash`).
Query via `olmo-eval results query ... --instances`. This table is one viable
source of the per-item response vector ATLAS consumes (the other is reading the
local predictions JSONL, or computing θ inline in the task).

---

## 8. Tests + dev

- Task unit tests: `tests/evals/tasks/test_<name>.py` (use `get_task(...)`, no GPU).
- External eval tests: `tests/evals/external/benchmarks/test_<name>.py`.
- `mock` provider drives GPU-free dry runs / tests.
- Dev commands (see root `CLAUDE.md` / `DEVELOPMENT.md`): `uv sync --frozen`,
  `uv run ruff check src/ tests/`, `uv run ty check src/ alembic/`,
  `uv run pytest tests/ --ignore=tests/integration -v`.
