# Integration assessment: ATLAS adaptive testing → olmo-eval

How hard is it, where are the seams, and what's the smallest first cut. This feeds
the plan; it is not a plan itself.

---

## 1. Verdict

**Moderate, and de-risked by prior work.** The two hard parts of an IRT/CAT eval —
(a) a calibrated item bank and (b) a correct CAT + p-IRT implementation — **already
exist** in the repo (published + recalibrated 3PL banks; a working numpy CAT in
`atlas_diagnostic_validation.py`; a second CAT package in `tutor_cat/mcq_irt/`).
What's missing is a thin **adapter** that runs the CAT/θ computation *inside*
olmo-eval and reports it, plus the **id-alignment plumbing** so olmo-eval's
per-item predictions line up with a bank.

The single biggest determinant of scope is **online vs offline CAT** (§4).

---

## 2. The three things that must line up

1. **Item bank** `question_id → (a, b, c)` — have it for ARC (published +
   recalibrated). Missing for the other benchmarks until they're calibrated.
2. **Response signal** `question_id → 0/1` for the model under test — olmo-eval's
   MCQ tasks already produce this per instance (ARC keeps the native id in
   `Instance.metadata["id"]`; predictions land in `instance_predictions.native_id`
   or the local predictions JSONL).
3. **Calibration↔eval scoring parity** — the bank must be fit on responses scored
   the *same way* olmo-eval scores at eval time (0-shot loglik), or transfer
   degrades (the 25-shot published-bank caveat; recalibration exists for this).

If all three hold for ARC, the eval is: load bank, build the model's 0/1 vector
over the bank's `question_id`s, run CAT → (θ, SE, selected items) → p-IRT accuracy.

---

## 3. Recommended bare-bones first cut

Goal (matches the ask): *a model launched on Beaker runs through an ATLAS eval and
reports θ + p-IRT accuracy, end to end, on one benchmark (ARC-Challenge).*

**Offline-CAT-on-full-scores.** Run ARC-Challenge normally in olmo-eval (all items,
loglik) → get per-item correctness → run the existing Fisher-info CAT over that
vector → report θ, SE, n_items, p-IRT accuracy as metrics. The CAT still *selects*
a ~40-item subset and estimates θ from it (faithful ATLAS behavior); it just
doesn't *skip* running the other items. This deliberately avoids touching the
static batch runner.

Concrete shape (either works):

- **Task variant** `arc_challenge:atlas` whose `compute_metrics()` (or a custom
  scorer/metric) loads the 3PL bank + idx→qid map, aligns responses by
  `metadata["id"]`, calls the lifted `run_cat` / `pirt_accuracy`, and returns
  `{theta, se, n_items, pirt_accuracy}`. Fits the registry; no runner changes;
  `mock` provider covers a GPU-free test.
- **External eval** `atlas_arc` if you'd rather keep ATLAS self-contained and out
  of the task metric path.

Ship criteria: `olmo-eval run -m mock -t arc_challenge:atlas --dry-run` wires up;
a real small model reproduces the θ/p-IRT numbers from
`Experiments/atlas_transfer_published/`; `olmo-eval beaker launch ... -t arc_challenge:atlas --dry-run`
emits a valid spec. A unit test in `tests/evals/tasks/` pins the CAT on a fixed
response vector.

Effort: small — mostly packaging existing numpy code as an importable module
(e.g. `src/olmo_eval/adaptive/`) + one task variant + one test + vendoring the ARC
bank/map where the job can read it.

---

## 4. The fork in the road: online CAT

True ATLAS efficiency (only *run* the ~40 selected items) needs an **interactive
loop**: infer item → score → re-estimate θ → pick next item. olmo-eval's default
`AsyncEvalRunner` materializes all instances up front and has **no select-next
hook** (see doc 01 §3). Options, cheapest first:

| Approach | Runs only selected items? | Runner change | Notes |
|---|---|---|---|
| Offline CAT on full scores (**recommended first**) | No (selects, doesn't skip) | None | Faithful θ/p-IRT; simplest; good for Beaker smoke test. |
| External eval with its own item loop | Yes | None to core runner | ATLAS owns inference via vLLM server + OpenAI API; more moving parts, own env. |
| Custom scaffold / adaptive runner | Yes | Yes (new harness scaffold or runner) | Most faithful + efficient; largest lift; do after the bare-bones works. |

Recommendation: land offline first (proves banks, ids, math, Beaker path), then
decide if online efficiency is worth a scaffold.

---

## 5. Open risks / decisions for the plan

- **Bank scope.** Only ARC has an index→question_id map + validated bank.
  Extending to hellaswag/winogrande/gsm8k/truthfulqa needs (a) an id bridge for
  each and (b) recalibration on our scoring. Decide launch scope = ARC-only vs
  multi-benchmark suite.
- **Which bank.** Published (25-shot, transfer r≈0.59) vs recalibrated-on-our-
  responses. For a real eval, calibrate on olmo-eval-scored responses; keep the
  published bank only for the smoke test.
- **Response source.** Compute θ inline in the task (self-contained, no DB) vs
  read back `instance_predictions` after `--store` (decouples calibration/eval but
  adds a DB round-trip). Inline is simpler for v1.
- **Scoring parity.** Pin the olmo-eval ARC variant (few-shot count, mc vs rc,
  normalization) used for *both* calibration and eval so item params stay valid.
- **Harness convergence.** Long-term, generating response matrices via olmo-eval
  tasks (not the standalone `Test/Inference` sweep) removes duplication and
  guarantees scoring parity — but that's a follow-on, not v1.
- **R vs Python for calibration.** CAT/eval is Python already. New *calibration*
  can reuse ATLAS's R (`01_fit_irt*.r` + linking) or move to `tutor_cat/mcq_irt`'s
  Python fit. Pick one to avoid a two-language calibration path.

---

## 6. Component pointers (for the plan)

| Need | Use |
|---|---|
| CAT + p-IRT reference impl | `AdaptiveTesting/Experiments/atlas_transfer_published/atlas_diagnostic_validation.py` |
| General IRT/CAT package | `eduLLM-Evals/tutor_cat/mcq_irt/{matrix,calibrate,ability,cat,pipeline,report}.py` |
| ARC 3PL bank + id map | `AdaptiveTesting/Inputs/ATLAS/arc/{irt_item_parameters_combined.csv, atlas_idx_to_question_id.csv}` |
| olmo-eval task to extend | `src/olmo_eval/evals/tasks/arc.py` (`register_variant`) |
| Task/metric base | `src/olmo_eval/evals/tasks/common/base.py`, `common/metrics/base.py` |
| Beaker launch | `src/olmo_eval/cli/beaker/launch.py`, `launch/beaker/launcher.py` |
| External-eval path | `src/olmo_eval/evals/external/{base,registry,result}.py` |
| Per-item predictions store | `src/olmo_eval/storage/backends/postgres/models.py` (`instance_predictions`) |
| Calibration scripts (R) | `AdaptiveTesting/Inputs/ATLAS/scripts/01_fit_irt*.r`, `02_link_chunks_custom.r` |
