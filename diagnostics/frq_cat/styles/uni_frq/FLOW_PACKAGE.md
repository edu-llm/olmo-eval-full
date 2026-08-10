# uni_frq graduation package (TutorEval, unidimensional 2PL)

The `uni_frq` style ports TutorEval's canonical unidimensional calibration into the FRQ
CAT flow. The bank is config-independent: item parameters do not depend on the stopping
rule, so the CAT operating point lives in `config.yaml`, not here.

## Snapshot

- Source branch: `origin/frq/tutoreval`.
- Snapshot tag: `tutoreval-unidim-52models`.
- Instrument (per owner): unidimensional, single latent axis `ability`. Only the unidim
  bank graduates; the 2-skill variant does not, because at this sample size its loadings
  and the ~0.98 latent correlation are unstable.
- Status: PROVISIONAL, `n_persons = 52` (`low_n`), below the ~150 identifiability floor.
  Treat abilities as coarse.

## Payload (ported into this directory)

- `bank/params.jsonl` (1,186 criteria) from
  `eduLLM-Evals/data/TutorEval/rubrics_qmatrix_final_unidim_fitted.jsonl`. Each record is
  the IRT item: `criterion_id`, `scenario_id`, `criterion` text, `primary_skill`,
  `q_modeled = {"ability": 1}`, scalar `difficulty` (b), and `discrimination = {"ability": a}`.
  Read as-is by both `common/bank_loader.py` (criterion text) and `common/irt_params.py` (a/b).
- `bank/scenarios.jsonl` (828 scenarios) from `.../scenarios_final.jsonl`. All scenarios are
  single-turn (`conversation_context: []`), so the tutor prompt is the `prompt` field and no
  message history is needed. Extra provenance fields are ignored by the loader.
- `judge_frozen.yaml` selects the shared frozen judge (identity only). Reshaped to the
  `judge.spec_from_config` schema (top-level `model_id`/`revision`). `model_id` is a fill-in
  defaulting to `Qwen/Qwen3.5-9B`.
- `evidence/` dimensionality + recovery figures and metrics.

## Fit provenance

- Method: `unidimensional-2pl-mml-em` (re-run of `calibrate_tutoreval`'s EM; item params
  recomputed by `staging/tutoreval_calibration/export_tutoreval_fitted_bank.py`).
- `n_persons = 52`, ridge `0.01`, fit grid 7 nodes/dim, EAP grid 61 nodes.
- Response-matrix sha256: `f7830280a40f6bb774116bb6e7a44f9840073c88d7ce8fbd538b357cdcbd4781`
  (`runs/judge/TutorEval/response_matrix.csv`; a calibration input that lands in S3, not here).
- Negative loadings are preserved (not floored to 0) in this export.

## Held-out recovery (scenario-level 5-fold, from `evidence/metrics.json`)

- 52 models, mean 15.4 criteria / 9.96 scenarios administered, `max_se = 0.3`.
- OOS ability recovery (Pearson r): batch 0.935, MWLE 0.931, online 0.905.
- The estimator matches what this style ships (2PL EAP, grid 61, `max_se` 0.3). The
  administration policy does not (see below), so treat these as indicative of bank quality
  rather than a guarantee for a runtime CAT session.

## Operating point differs from the recovery study

The recovery study stopped on `min_evals_per_skill = 15` with `max_scenarios = 50`, a
scenario-oriented policy with an effective floor of 15 criterion evaluations. This style
ships `min_items: 8`, and the shared engine administers one criterion per step, capped by
the runner's `--max-items`. Two consequences:

- A floor of 8 is below the study's effective floor of 15. Raise `min_items` (and pass
  `--max-items 40`) to sit closer to the validated operating point.
- Criteria that share a scenario share one tutor response and are locally dependent.
  Administering them as independent items can understate SE. Testlet-style (whole-scenario)
  administration would be a change to the shared `common/cat_loop.py` on `CheckpointFlows`,
  not to this style.

## Deployment requirement: tutor context window

TutorEval prompts embed book passages and are long. Measured over the 828 shipped
scenarios (chars/4 estimate): median ~1.2k tokens, p90 ~4.5k, p99 ~8.4k, max ~9.7k.

- The tutor's own context window decides how much of the bank is scorable, and for some
  checkpoints it cannot be raised. OLMo-2-7B was trained at `max_position_embeddings=4096`,
  so vLLM will not serve it wider: with that tutor, 135 of 828 scenarios (16.3%) exceed
  ~90% of the window (118 exceed 4096 outright) and are simply unscorable. Scoring the full
  bank requires a long-context tutor, not a larger `--max-model-len`.
- Skipping is not neutral. The long-prompt scenarios are harder than average (mean
  difficulty 3.8 against 2.0), so a tutor limited to 4096 is measured on an easier subset.
  Expect `ungradable.alert` to fire, and read theta as covering that subset.
- Behaviour differs by entry point. The shared `frq_cat.runner` + `common/cat_loop.py` have
  no per-scenario error handling, so one over-long prompt aborts the session with no report.
  The style-local `run_uni_frq` + `session.py` skip that scenario (and its criteria),
  record it in `skipped.jsonl`, and carry on. Either way a small window silently shrinks
  the effective bank, so treat the context size as a prerequisite, not a tuning knob.

## Dimensionality (why unidimensional)

Evidence in `evidence/` (recovery figure + metrics). Unidimensional was chosen deliberately:
at N=52 the multi-dimensional loadings and latent correlation are not identifiable, so a
single `ability` axis is the shipped instrument.

## Runtime data flow (scoring a checkpoint)

1. Generate one tutor response per scenario in `bank/scenarios.jsonl` (respgen; the tutor is
   the checkpoint under test served at `--tutor-endpoint`).
2. Grade each administered criterion with the judge named by `--judge-config`. That file's
   `adapter` selects both the prompt and the parser that reads its replies, from
   `adapters.py`: `generic-binary` (the shared text contract, self-hosted Qwen in
   `judge_frozen.yaml`) or `generic-binary-strict` (the JSON contract the team froze for
   TutorEval, reproduced in `judge_frontier.yaml`). The pairing is not optional — the text
   parser reads a JSON reply as unscorable and abstains on every criterion.
3. Feed graded outcomes + the fitted bank (a/b) into the unidimensional 2PL CAT loop to
   estimate `ability` (theta) with SE; the runner writes `cat_report.json` to `--s3-out`.

## Caveats

- `low_n` (N=52): abilities are coarse; do not over-interpret small theta differences.
- Calibration faithfulness: this bank was fitted on a matrix graded under
  `judge-validation-v3` with an evidence gate, and neither shipped contract is that prompt,
  so no configuration here produces a calibration-faithful theta today. Every report says
  which contract graded it and marks the result `UNCALIBRATED`.
- Choosing `generic-binary-strict` narrows the gap on the evidence gate and widens it on
  strictness: that judge passes 13.9% of cells against difficulties fitted behind a more
  lenient one, so theta is biased low by an amount the SE does not carry. The fix is a
  refit on a matrix graded by whichever contract will run, not a flag. The 92,872-cell
  Gemini matrix that would support one exists but is not committed; see
  `eduLLM-Evals/api_judge_pilot/TUTOREVAL_RESULTS_2026-08-09.md` on `frq/tutorbench`.

## Rerun / swap protocol (larger cohort)

To replace this bank with a larger-N fit: rerun the TutorEval calibration to produce a new
`rubrics_qmatrix_final_unidim_fitted.jsonl` + response matrix, re-export the fitted bank,
copy it to `bank/params.jsonl`, refresh `evidence/`, and update the tag, `n_persons`, matrix
sha256, and recovery numbers above. Keep the judge identity fixed unless the whole bank is
recalibrated against a new judge.
