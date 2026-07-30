# ATLAS × olmo-eval — implementation status, how to test, Phase 3 handoff

This doc records **what is built (Phases 0–2)**, **exactly how to verify it**, and
**where a teammate should pick up Phase 3 (multi-benchmark + recalibration)**.

---

## 1. What is implemented

### Phase 0 — shared adaptive core (`src/olmo_eval/adaptive/`)
Pure, provider-agnostic numpy. Single source of truth for both entry points.

| File | Contents |
|---|---|
| `irt.py` | 3PL `prob`, `fisher_info`, grid EAP `eap_theta_se` (81-node grid, N(0,1) prior). |
| `bank.py` | `ItemBank` + `load_bank()` (ATLAS `a1,d,g → a,b,c`, bridges via `atlas_idx_to_question_id.csv`, records `bank_version`); `ItemBank.restrict()` to align a full run to the calibrated subset. Default dir = vendored ARC bank, overridable by arg or `$OLMO_EVAL_ATLAS_BANK_DIR`. |
| `cat.py` | `CatSession` (max-Fisher-info selection + EAP re-estimate + SE stop), `TableResponder`, `pirt_accuracy`, sync `run_cat`, async `run_cat_async`. |

### Phase 1 — offline CAT task (`src/olmo_eval/evals/tasks/atlas_arc.py`)
`@register("atlas_arc_challenge")` subclasses `ARCChallenge`; runs the full
ARC-Challenge eval unchanged, then `compute_metrics` derives per-item 0/1 via
`self.config.get_primary_metric().compute_instance`, joins to the bank on
`metadata["id"]`, runs CAT, and appends `atlas_theta / atlas_se /
atlas_n_items / atlas_pirt_accuracy` (+ `atlas_bank_version`). Full benchmark is
still run — this proves banks/ids/math end-to-end. Missing/empty bank ⇒ base
metrics only (graceful).

Correctness follows the task's **configured** metric rather than a hardcoded
`LogprobMCAccuracyMetric`, so scoring variants reach the CAT (needed for the
parity work in §3d). Two guards: an unresolvable primary metric, or a metric
producing non-binary per-item values (bpb, perplexity), skips the adaptive report
instead of feeding a truncated value into the IRT cell.

### Phase 2 — online CAT external eval (`src/olmo_eval/evals/external/benchmarks/atlas_arc/`)
`atlas_arc` (registered, **not** sandboxed). `execute()` runs the live loop: pick
max-Fisher-info unused item → build **one** ARC loglik request via the real
`arc_challenge` task (formatting parity) → `await provider.alogprobs([req])` →
argmax → re-estimate → stop at SE. **Only the selected items (~tens) are sent to
the model** — that is the inference saving. Returns `pirt_accuracy / theta / se /
n_items / bank_size` + metadata (`bank_version`, selected id sequence, `seed`).
Args: `se_stop` (0.3), `min_items` (8), `max_items` (200), `bank_path`, `task`, `seed`.

### Tests
`tests/adaptive/test_cat.py`, `tests/evals/tasks/test_atlas_arc.py`,
`tests/evals/external/benchmarks/test_atlas_arc.py`.

---

## 2. How to test / verify these changes

### 2a. Unit + integration tests (primary gate — no network needed)
```bash
uv run pytest tests/adaptive tests/evals/tasks/test_atlas_arc.py tests/evals/external/benchmarks/test_atlas_arc.py -v
```
Expected: **16 passed**. Notable cases:
- `test_matches_reference_atlas_numbers` — reproduces a recorded held-out row from
  `AdaptiveTesting/Experiments/atlas_transfer_published/results/atlas_arc_heldout_se0.3.csv`
  (`allenai/OLMo-2-1124-7B`: θ≈-1.3085, p-IRT≈0.5762, 8 items) **exactly**. Skips
  automatically if the vendored ARC bank/responses are absent.
- `test_online_matches_offline_and_queries_small_subset` — online == offline θ/order
  on identical responses, using only ~8 of ~842 items.
- Online eval tests assert it stops on the SE rule, never re-administers an item,
  and calls the provider exactly once per administered item.

### 2b. Lint / format / type gates (per root `CLAUDE.md`)
```bash
uv run ruff check src/olmo_eval/adaptive src/olmo_eval/evals/tasks/atlas_arc.py src/olmo_eval/evals/external/benchmarks/atlas_arc tests/adaptive tests/evals/tasks/test_atlas_arc.py tests/evals/external/benchmarks/test_atlas_arc.py
uv run ruff format --check src/olmo_eval/adaptive src/olmo_eval/evals/tasks/atlas_arc.py src/olmo_eval/evals/external/benchmarks/atlas_arc tests/adaptive tests/evals/tasks/test_atlas_arc.py tests/evals/external/benchmarks/test_atlas_arc.py
uv run ty check src/
```
On macOS/arm64 all three gates come back clean. (An earlier note here said `ty`
reports 4 pre-existing `signal.alarm/SIGKILL` platform warnings; those are
platform-dependent and do not appear on darwin. Either way, none are in ATLAS code.)

### 2c. Registration smoke tests (fast, no network)
```bash
uv run python -c "from olmo_eval.evals.tasks.common import get_task; print(type(get_task('atlas_arc_challenge')).__name__)"
uv run python -c "from olmo_eval.evals.external.registry import list_external_evals; print('atlas_arc' in list_external_evals())"
```

### 2d. End-to-end via the CLI (needs HF dataset download for ARC-Challenge)

**Verified working** — the offline task runs end to end under the `mock` provider,
which needs no GPU and exercises dataset load, the id join, bank restriction, the
CAT, and metric emission:
```bash
uv run olmo-eval run -m mock -t atlas_arc_challenge -O /tmp/atlas_mock
```
Emits all four metrics to `metrics.json` alongside the normal `accuracy:logprob`:
```
ATLAS CAT: theta=-1.9326 se=0.2632 n_items=8 pirt_acc=0.5203
           bank=arc:irt_item_parameters_combined.csv:n650:restrict650
```
Note what that run shows: mock scores 0.227 (chance), and p-IRT reconstructs 0.520.
That is the sub-chance floor from doc 02 §2 reproducing live, not a wiring bug.

**Stronger check against real model responses**, no GPU or network required — drive
the shipped `olmo_eval.adaptive` code over the 60 recorded held-out models and
compare to `atlas_transfer_published/results/`. All 60 rows reproduce bit-exactly
(max abs difference 0.0), which covers considerably more than the single OLMo-2 row
pinned in `test_matches_reference_atlas_numbers`.

**Still unverified**, with the reasons:
```bash
# Real local model: blocked on macOS. The async runner's GPU planner calls
# torch.cuda.device_count() (runners/asynq/runner.py:1031) and the hf provider is
# flagged requires_local_gpu, so MPS is not recognised. Needs a CUDA host.
uv run olmo-eval run -m <small-model> -t atlas_arc_challenge -o limit=50

# Online eval: needs a persistent vLLM server.
uv run olmo-eval run-external -m <small-model> -e atlas_arc -a max_items=40

# Beaker: -c/--cluster is required, and --dry-run still needs BEAKER_TOKEN because
# _handle_group_creation reads launcher.beaker.user_name before the dry-run guard.
uv run olmo-eval beaker launch -m <model> -t atlas_arc_challenge -c h100 \
  -w ai2/oe-data -B ai2/oe-base --dry-run
uv run olmo-eval beaker launch -m <model> -E atlas_arc -c h100 \
  -w ai2/oe-data -B ai2/oe-base --dry-run
```
Cross-check once a GPU host is available: online θ for a model should be close (not
identical) to the offline θ for the same model.

Caveat on `-o limit=N`: it truncates the instance set, so `ItemBank.restrict()`
aligns the bank to whatever survived. θ from a limited run is a wiring check, not a
measurement.

---

## 3. Phase 3 — where to pick up (multi-benchmark + recalibration)

Goal: extend beyond ARC-Challenge to more benchmarks, group them into an `atlas`
suite, and (separately) recalibrate a scoring-parity bank.

### 3a. Data reality (read first)
Only the **ARC** bank is vendored locally: `AdaptiveTesting/Inputs/ATLAS/arc/`
(and `arc_0p5_7b/`). Banks for hellaswag/winogrande/gsm8k/truthfulqa are in the
upstream ATLAS repo (`github.com/Peiyu-Georgia-Li/ATLAS`) but are **gitignored**
here (see root `.gitignore` lines for `AdaptiveTesting/Inputs/ATLAS/{hellaswag,winogrande,gsm8k,truthfulqa}/`).
So step one for each new benchmark is to **drop in that benchmark's
`irt_item_parameters_combined.csv` + `atlas_idx_to_question_id.csv`** under
`AdaptiveTesting/Inputs/ATLAS/<benchmark>/` (kept local / in S3, not committed).

### 3b. The id-bridge is the crux (per-benchmark)
The bank joins to olmo-eval on `Instance.metadata["id"]`. Confirmed id sources in
the base tasks:
- **ARC** (`arc.py`): native `question_id` (e.g. `Mercury_7175875`). ✅ aligns.
- **HellaSwag** (`hellaswag.py`): `metadata["id"] = doc.get("ind", index)` — native `ind`.
- **WinoGrande** (`winogrande.py`): `metadata["id"] = index` — **positional**, not a
  stable native id. The ATLAS bridge must use the *same ordering* or a real qID.
- **CommonsenseQA** (`csqa.py`): native `id`. **PIQA** (`piqa.py`): positional `index`.
Action per benchmark: verify the ATLAS `atlas_idx_to_question_id.csv` keys match
whatever the base task emits as `metadata["id"]`; if the task uses a positional
index, either regenerate the bridge against that ordering or patch the task to
carry the native id. Add a tiny fixture test like
`test_matches_reference_atlas_numbers` per benchmark once its bank + a recorded
result row exist.

### 3c. Suggested generalization (small refactor)
Turn the two ARC-specific entry points into config-driven ones:
1. Add `src/olmo_eval/adaptive/benchmarks.py`: a frozen `AtlasBenchmark(name,
   base_task, bank_subdir)` + a registry dict for arc_challenge / hellaswag /
   winogrande / csqa / piqa.
2. Offline: extract the `compute_metrics` logic in `atlas_arc.py` into a mixin and
   register one task per benchmark (`atlas_<name>`) by combining the mixin with the
   base task. It already uses the task's own primary metric rather than a hardcoded
   `LogprobMCAccuracyMetric`, so the logic is benchmark-agnostic as written.
3. Online: the `execute()` loop is already benchmark-agnostic — it takes `task` +
   `bank_path` args. Either register one `atlas_<name>` external eval per benchmark
   (defaults filled from the `AtlasBenchmark` config) or keep one eval and pass
   `-a task=... -a bank_path=...`.
4. Suite: add `src/olmo_eval/evals/suites/atlas.py` grouping the offline
   `atlas_<name>` tasks (see `suites/mmlu.py` / `suites/registry.py` for the
   pattern) so `olmo-eval run -s atlas` runs the whole set.

**Scoring caveat:** the online `_score_item` and the offline MC correctness assume
**log-likelihood MCQ** scoring (argmax over choice continuations vs `gold_idx`).
That covers arc/hellaswag/winogrande/csqa/piqa. **gsm8k is generative** and
**truthfulqa varies** — they need a different per-item correctness adapter
(generate + exact/GT match) before CAT applies. Treat those as a separate sub-task.

### 3d. Scoring-parity recalibration (independent of multi-benchmark)
The vendored ARC bank is the published 25-shot leaderboard calibration; olmo-eval
scores 0-shot loglik at eval time. For research-grade numbers, fit a fresh 3PL
bank on ARC responses scored the *same* way olmo-eval scores (reuse ATLAS R
`scripts/01_fit_irt*.r` + linking, or `eduLLM-Evals/tutor_cat/mcq_irt`), then swap
it in via `bank_path` / `$OLMO_EVAL_ATLAS_BANK_DIR`. No code change needed — just a
new bank directory.

**Read `02_atlas_and_adaptive_testing.md` §2 "Validation status" before starting.**
Three findings shape this work:

- **A scoring-parity recalibration has never actually been run.**
  `atlas_recalibrate_0p5_7b` re-fit on ATLAS's own 25-shot matrix filtered by model
  size, so it varied the calibration population rather than the scoring method.
- **That recalibration made transfer worse**, r 0.83 → 0.59 at SE 0.2. It is not a
  precedent for the approach working, and the published bank is currently the
  better one to run against.
- **p-IRT accuracy currently loses to a constant baseline** (MAE 0.147–0.172 vs
  0.084 for predicting the mean). Until a parity bank exists, prefer θ/SE for
  ranking and treat `atlas_pirt_accuracy` as diagnostic rather than reportable.

Decide the scoring target before fitting, and use the same one for the bank and the
eval task, or the mismatch simply reappears. Non-letter options already registered
in `evals/tasks/arc.py`:

| Option | Variant / metric | Notes |
|---|---|---|
| Raw loglik over choice text | `arc_challenge` default, `LogprobMCAccuracyMetric` | What ships today; most exposed to length/fluency bias, and the source of sub-chance scores. |
| PMI-normalized | `arc_challenge:rc`, `LogprobUncondMCAccuracyMetric` | Divides out the unconditional likelihood of each choice (see `arc.py` lines 173–187). Most likely to remove the sub-chance floor. |
| Length-normalized | `LogprobPerCharMCAccuracyMetric` | Blunter fix for the same bias. |
| Few-shot | `arc_challenge:olmes` (5-shot); up to 19 via `ARC_CHALLENGE_FIXED_FEWSHOT` | Moves the eval toward the bank's 25-shot regime instead of refitting the bank. |

These are testable now that per-item correctness follows the configured metric
(§1, Phase 1). Before the parity bank exists, running a normalized variant against
the current 25-shot bank tells you whether the sub-chance floor disappears, which
is the cheapest available signal on whether parity recalibration is worth the
effort.

### 3e. Optional — persistence
Persist per-item selections + θ to Postgres via the runner's `--store` path for
cross-model θ ranking (see `03_integration_assessment.md`).

---

## 4. Push hygiene (done)
Large/derived calibration artifacts under `eduLLM-Evals/` (supplement zips/tarballs,
`supplement_v2/`, `*.npy`, `run_data_supp*.jsonl`, `staging/` run outputs,
experimental q-matrix jsonl) are gitignored in `eduLLM-Evals/.gitignore`. The ATLAS
integration under `AdaptiveTesting/`, `src/olmo_eval/`, and `tests/` is source/docs
only (no large binaries) and is intended to be pushed. Non-ATLAS calibration banks
for other benchmarks stay **out** of git (root `.gitignore`).
