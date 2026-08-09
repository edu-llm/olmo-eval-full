# Frontier-API LLM-judge pilot: findings

Pivot rationale: the two-machine self-hosted Qwen judge is infeasible on the platform
(no cross-image network transfer; the pipeline isn't built for single-image multi-GPU
jobs). This pilot tests whether a **frontier API judge** behind the TrueFoundry
OpenAI-compatible gateway can replace it, measuring (1) agreement vs human labels and
(2) how parallelizable it is in terms of compute.

## Setup

- Dataset: 261 human-labeled judge cases (blinded scenario + conversation context +
  tutor response + one criterion), from the judge-validation set. Gold labels join on
  `case_id`.
- Prompt: reuses the existing evidence-gated binary judge prompt
  (`eduLLM-Evals/tutor_cat/judge.py`, judge-validation-v3 / generic-binary), so results
  are comparable to the frozen Qwen judge. Parsing: JSON `{verdict}` -> `[RESULT] <1-5>`
  (>=4 pass) -> keyword; unparseable -> fail (fail-closed, as in the judge).
- Transport: `POST {MODEL_API_BASE}/chat/completions`, temperature 0, max_tokens 512,
  async with a concurrency semaphore. Credentials from `eduLLM-Evals/.env`
  (`MODEL_API_KEY` / `MODEL_API_BASE`).

## Agreement vs human labels (261 cases, concurrency 12)

| model | accuracy | balanced | F1(pass) | MCC | false-pass | unscorable | cases/s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai-group/gpt-4.1 | 0.789 | 0.782 | 0.817 | 0.571 | 28.4% | 0 | 8.04 | 1.92 |
| claude-group/claude-sonnet-4-6 | 0.770 | 0.775 | 0.779 | 0.547 | 18.1% | 2 | 2.15 | 8.66 |
| claude-group/claude-opus-4-6 | 0.766 | 0.759 | 0.796 | 0.524 | 30.2% | 2 | 1.78 | 10.10 |
| gemini-group/gemini-2.5-flash | 0.682 | 0.704 | 0.638 | 0.435 | 9.5% | 148 | 4.46 | 3.48 |
| gemini-group/gemini-3-flash-preview | 0.678 | 0.703 | 0.622 | 0.443 | 6.9% | 160 | 4.46 | 3.15 |

Reading it:
- **gpt-4.1** has the best agreement (accuracy 0.789, MCC 0.571, F1 0.817) and is the
  fastest, but its **false-pass rate is high (28.4%)** -- it over-grants passes, which is
  the dangerous error mode for a judge.
- **claude-sonnet-4-6** is close on agreement (0.770 / MCC 0.547) with the **lowest
  false-pass rate among the clean parsers (18.1%)** -- attractive if false-pass is the
  priority for a grading judge.
- **claude-opus-4-6** is not better than sonnet here and is the slowest.
- The **Gemini flash models are confounded**: 148-160 of 261 outputs were **unscorable**
  (they emit long feedback and never reach the `[RESULT]`/JSON within 512 tokens, so they
  fail-close). Their low agreement and low false-pass are largely a format/`max_tokens`
  artifact, not judgment quality. Do not conclude on Gemini until re-run fairly (below).

For reference, the frozen-judge validation study targeted macro-F1 >= 0.80; gpt-4.1
(F1 0.817) clears it and sonnet-4-6 (0.779) is just under, on this label set.

> NOTE: the table above mixes prompts unfairly (all used the ported Prometheus 1-5
> prompt, but Gemini couldn't comply within 512 tokens). The **unified apples-to-apples
> table below supersedes it** for cross-model comparison.

## Unified apples-to-apples (all 5 models, canonical Qwen prompt + JSON, max_tokens 2048)

Every model on the exact frozen-Qwen `generic-binary` prompt with
`response_format=json_object`, concurrency 12, 261 cases (`results_unified/`):

| model | accuracy | balanced | F1(pass) | MCC | false-pass | unscorable | cases/s | p95 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai-group/gpt-4.1 | 0.751 | 0.746 | 0.780 | 0.494 | 30.2% | 0 | 9.45 | 1.82 |
| claude-group/claude-sonnet-4-6 | 0.751 | 0.741 | 0.787 | 0.492 | 34.5% | 2 | 3.35 | 5.61 |
| claude-group/claude-opus-4-6 | 0.747 | 0.729 | 0.796 | 0.491 | 43.1% | 4 | 2.49 | 8.02 |
| gemini-group/gemini-2.5-flash | 0.747 | 0.742 | 0.775 | 0.486 | 30.2% | 24 | 2.60 | 10.60 |
| gemini-group/gemini-3-flash-preview | 0.736 | 0.741 | 0.745 | 0.478 | 21.6% | 45 | 2.26 | 10.01 |

Headline findings (these are the ones to trust):
- **On an identical prompt, agreement is remarkably model-insensitive: all five converge
  to ~0.74-0.75 accuracy / MCC ~0.48-0.49.** The apparent gpt-4.1 lead in the first table
  (MCC 0.571) was a prompt effect, not a model effect.
- **Prompt choice dominates model choice.** Switching Claude/GPT from the Prometheus 1-5
  prompt to the Qwen generic-binary prompt *lowered* their MCC (gpt-4.1 0.571 -> 0.494,
  sonnet 0.547 -> 0.492) and *raised* false-pass (to 30-43%). The generic-binary prompt is
  the more lenient grader here. If deploying an API judge, the prompt is the primary lever
  to tune, more than the model.
- **gpt-4.1 wins decisively on compute at equal agreement**: 9.5 cases/s and p95 1.8s
  (3-4x the others), with zero unscorable. Gemini flash is competitive on agreement under
  the fair prompt but pays a heavy verbosity tax (mean 655 / 988 completion tokens, 24 / 45
  residual JSON truncations, ~2.3-2.6 cases/s).
- False-pass is high across the board on generic-binary (30-43%, worst on opus). For a
  judge, that leniency is the main risk to address (via prompt, not model).

## Parallelizability (compute): gpt-4.1, 60-case subset

| concurrency | cases/s | wall s | p95 latency s | errors |
| --- | --- | --- | --- | --- |
| 1 | 0.84 | 71.3 | 1.64 | 0 |
| 4 | 3.00 | 20.0 | 2.00 | 0 |
| 8 | 5.79 | 10.4 | 1.66 | 0 |
| 16 | 12.13 | 4.95 | 1.47 | 0 |
| 32 | 19.92 | 3.01 | 1.70 | 0 |
| 48 | 18.32 | 3.28 | 1.83 | 0 |

- Throughput scales **near-linearly up to ~32 in-flight requests** (~24x over serial),
  then **plateaus around ~20 cases/s** (gateway/provider concurrency ceiling for this
  model). Per-request p95 latency stays flat (~1.5-2s), so the gateway absorbs the
  concurrency without queueing degradation until the plateau. **Zero errors** throughout.
- This is the key contrast with the self-hosted Qwen judge: the API judge is
  embarrassingly parallel (bounded by provider RPM/TPM, not one GPU's sequential
  throughput), so grading throughput is a tunable dial, not a hardware constraint.

## Strict binary prompt (reduce false-pass; keep binary)

Kept the binary verdict + JSON contract but folded the Prometheus rubric's strictness
into a binary gate (`--adapter generic-binary-strict`): PASS only when clear, direct
evidence fully satisfies every required part; partial/vague/implied/indirect -> FAIL;
default to fail, no benefit of the doubt; evidence emitted before the verdict. All 5
models, JSON mode, max_tokens 2048, 261 cases (`results_strict/`), vs the lenient
generic-binary baseline (`results_unified/`):

| model | accuracy | MCC | false-pass (lenient -> strict) |
| --- | --- | --- | --- |
| claude-sonnet-4-6 | 0.751 -> 0.770 | 0.492 -> 0.536 | 34.5% -> 24.1% |
| claude-opus-4-6 | 0.747 -> 0.778 | 0.491 -> 0.548 | 43.1% -> 31.0% |
| gemini-2.5-flash | 0.747 -> 0.766 | 0.486 -> 0.526 | 30.2% -> 26.7% |
| gemini-3-flash-preview | 0.736 -> 0.713 | 0.478 -> 0.448 | 21.6% -> 18.1% |
| openai-group/gpt-4.1 | 0.751 -> 0.751 | 0.494 -> 0.492 | 30.2% -> 34.5% |

Findings:
- **The strict binary prompt is a clear win for the Claude models and gemini-2.5-flash**:
  false-pass drops while accuracy AND MCC rise. claude-opus-4-6 reaches the best agreement
  (acc 0.778 / MCC 0.548, FP 31%); **claude-sonnet-4-6 gives the best false-pass/agreement
  balance (FP 24.1%, acc 0.770, MCC 0.536)**.
- **Strictness steerability is model-dependent.** gpt-4.1 barely moved and its false-pass
  even rose (30.2% -> 34.5%) -- it does not respond to the "default to fail" framing the
  way Claude does. gemini-3-flash-preview over-shot (false-pass down to 18.1% but accuracy
  and MCC fell -- too many false negatives).
- Net recommendation for a low-false-pass judge: **claude-sonnet-4-6 with the strict binary
  prompt** (best FP among the strong performers) if grading conservatism matters most;
  gpt-4.1 remains the throughput king (~8-9 cases/s) but its false-pass is stubborn.
- 24-31% false-pass is better but still non-trivial; pushing lower risks the
  gemini-3-preview failure mode (tanking recall). Further gains likely need targeted
  prompt work (e.g., an explicit "would a strict grader reject this?" second check) or
  criterion-type-specific handling, with recall watched.

## Gemini re-run: canonical Qwen prompt + normalized JSON response

The initial Gemini numbers were confounded by output-format non-compliance. Re-ran both
flash models with the exact frozen-Qwen prompt (`--adapter generic-binary`, verbatim from
`run_judge_validation.py`) and OpenAI-style JSON mode (`--json-mode`,
`response_format={"type":"json_object"}`), sweeping `max_tokens`:

| model | max_tokens | accuracy | balanced | F1 | MCC | false-pass | unscorable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-2.5-flash | 512 (prometheus prompt) | 0.682 | 0.704 | 0.638 | 0.435 | 9.5% | 148 |
| gemini-2.5-flash | 1024 (generic-binary+json) | 0.720 | 0.722 | 0.737 | 0.442 | 25.9% | 62 |
| gemini-2.5-flash | 2048 (generic-binary+json) | 0.728 | 0.722 | 0.761 | 0.447 | 33.6% | 25 |
| gemini-3-flash-preview | 512 (prometheus prompt) | 0.678 | 0.703 | 0.622 | 0.443 | 6.9% | 160 |
| gemini-3-flash-preview | 1024 (generic-binary+json) | 0.732 | 0.747 | 0.715 | 0.505 | 11.2% | 105 |
| gemini-3-flash-preview | 2048 (generic-binary+json) | 0.724 | 0.729 | 0.733 | 0.456 | 22.4% | 46 |

Interpretation:
- Normalizing the format (JSON mode) + raising `max_tokens` recovers most of the
  unscorable outputs: gemini-2.5-flash dropped 148 -> 25 unscorable, gemini-3-flash-preview
  160 -> 46. Both are extremely verbose in JSON mode (mean 675 / 993 completion tokens;
  gemini-3-preview appears to do long internal reasoning), so residual unscorable are JSON
  truncation at the token cap, and throughput drops (2.3-3.5 cases/s).
- Even fully corrected, **Gemini flash lands at ~0.72-0.73 accuracy / MCC ~0.45 -- below
  the Claude/GPT tier (0.77-0.79 acc, MCC 0.52-0.57)**. So the flash models are a genuinely
  weaker judge for this rubric, not merely a parsing artifact; the earlier catastrophic
  numbers were format truncation, but the corrected numbers still trail.
- Apples-to-apples caveat: the 5-model table above used the ported Prometheus 1-5 prompt
  (`prometheus15`); this Gemini re-run uses `generic-binary`. For a strict cross-model
  comparison, run all models on the same `--adapter generic-binary --json-mode` (see next
  steps). Artifacts: `results_gemini_json/` (1024), `results_gemini_json_2048/` (2048).

## Chosen config parallelizability: claude-sonnet-4-6 (strict binary + JSON)

Concurrency sweep on the selected judge config (60-case subset, `results_sonnet_sweep/`):

| concurrency | cases/s | p95 latency s | errors |
| --- | --- | --- | --- |
| 1 | 0.25 | 6.48 | 0 |
| 4 | 0.96 | 7.09 | 0 |
| 8 | 1.86 | 5.95 | 0 |
| 16 | 3.57 | 6.28 | 0 |
| 32 | 4.09 | 9.95 | 0 |
| 48 | 4.63 | 6.26 | 0 |

- NOTE: the 60-case subset above under-measures throughput -- too few cases to keep the
  concurrency filled, so per-request latency dominates. On the FULL 261-case set at higher
  concurrency the real curve is much better (`results_sonnet_ceiling/`):

| concurrency | cases/s | p95 latency s | errors |
| --- | --- | --- | --- |
| 48 | 7.4-9.7 | 6.5 | 0 |
| 96 | 12.2 | 6.6 | 0 |
| 128 | 13.1 | 8.3 | 0 |
| 192 | 18.8 | 7.2 | 0 |

- **Real sustained ceiling is >= ~19 cases/s and still rising at c=192**, with zero 429s
  and flat latency -- headroom remains (test c=256/384 to find the wall). Per-call latency
  ~5-7s, but that is hidden by concurrency on a full queue. The earlier ~4.6 c/s figure was
  a small-subset artifact, not the ceiling.

## Full-response regrade throughput (biggen, 1-model slice, c=256)

Grading a real benchmark cell set (all 2,678 biggen criteria for one model's responses,
`regrade_benchmark.py`, sonnet-4-6 strict + JSON, `results_biggen_slice/`):

- **~48 judge calls/s** at concurrency 256 (2,678 cells in 55.6s), 0 errors, 0 auto-fails,
  13 unscorable. This is the real sustained throughput -- the earlier ~19 c/s was a
  small-subset artifact; a full 2,678-cell queue actually saturates the concurrency.
- **Sanity agreement vs the Qwen matrix: 87.4%** (2,665 comparable cells). Pass rate 60.6%.
- Mean 1,592 prompt tokens/call (long tutor responses) + 161 completion tokens.
- Full-job extrapolation (100 models x [biggen 2678 + tutoreval 1793] = 447,100 calls):
  **~2.6 hours** wall at this rate; ~712M prompt + ~72M completion tokens. Time is cheap;
  token volume is the cost driver (truncating very long responses would cut cost most).
- Concurrency sweet spot is ~256: c=256 -> 48 calls/s, but **c=512 -> 27 calls/s** (98.6s
  vs 55.6s, 0 errors) -- the gateway soft-throttles past ~256, so higher concurrency is
  counterproductive. Use c=256. The grader is resumable (incremental verdicts.jsonl) so a
  multi-hour run survives interruption.

## Cost (dashboard-implied sonnet rate: ~$3/M input, ~$15/M output)

Per call ~1,592 input + 161 output tokens => ~$0.0072/call.

| scope | judge calls | input tokens | output tokens | est. cost |
| --- | --- | --- | --- | --- |
| biggen, 100 models | 267,800 | 426M | 43M | ~$1,930 |
| tutoreval, 100 models | 179,300 | 285M | 29M | ~$1,290 (biggen token profile; measure) |
| both, 100 models | 447,100 | 712M | 72M | ~$3,200 |

Input tokens are ~2/3 of cost and are driven by long tutor responses (mean 1,592/call);
capping response length fed to the judge is the biggest cost lever.

## Tier-0 acceptance-gate audit (free; where the false-pass lives)

Joined each model's strict verdicts to human_labels.csv (criticality + primary_skill),
zero API calls (`audit_verdicts.py` / `audit_compare.py`). Strict prompt, 261 cases:

| model | acc | FP% | crit-sensitivity | crit-FP% | content-FP% |
| --- | --- | --- | --- | --- | --- |
| claude-sonnet-4-6 | 0.770 | 24.1 | 0.775 | 22.5 | 35.7 |
| claude-opus-4-6 | 0.778 | 31.0 | 0.706 | 29.4 | 32.1 |
| gemini-2.5-flash | 0.766 | 26.7 | 0.745 | 25.5 | 25.0 |
| gemini-3-flash-preview | 0.713 | 18.1 | 0.814 | 18.6 | 14.3 |
| gpt-4.1 | 0.751 | 34.5 | 0.686 | 31.4 | 39.3 |

- The Qwen v3 gate targeted critical-failure sensitivity >= 0.90; none of these clear it,
  but the **Gemini flash models are best on false-pass / critical-failure** -- the metric
  that matters for a judge. sonnet/opus/gpt-4.1 look strong on accuracy but are the worst
  at catching critical failures.
- sonnet's weakness is **content criteria** (35.7% false-pass); gemini-2.5-flash halves
  that to 25% and gemini-3-flash-preview to 14.3%.
- Caveat: gemini-3-flash-preview's strictness is partly inflated by unscorable->fail
  truncations at max_tokens 2048; re-validate at 4096 before trusting it.

## Cost by model (full 100-model x 2-benchmark job, ~447,100 calls)

Confirmed sonnet rate from the billing dashboard (~$3/M in, ~$15/M out); others are public
rates -- CONFIRM on the gateway dashboard. Input ~711.8M tokens shared across models
(prompt+response); output per model varies.

| model | est. cost | note |
| --- | --- | --- |
| claude-sonnet-4-6 | ~$3,200 | over budget; worst content FP |
| gpt-4.1 | ~$1,820 | worst critical-failure sensitivity |
| gemini-2.5-flash | ~$950 | better content FP than sonnet; recommended |
| gemini-3-flash-preview | ~$1,000-1,500 | safest false-pass; re-validate max_tokens |
| claude-haiku-4-5 | ~$1,150 | untested; Claude family |

Input tokens are ~2/3 of cost (mean 1,592/call, long tutor responses); truncating the
response fed to the judge is the biggest orthogonal cost lever for any model.

## Gemini unscorable: diagnosis + fix (keeps Gemini viable)

Captured raw outputs + finish_reason + token usage on the 261 cases (`diagnose_gemini.py`).
Root causes of the Gemini unscorable rate (NOT model quality):

1. Truncation with verdict last. gemini-3-flash-preview dumped entire worked solutions
   into the `evidence` field and hit the 2048 token cap (`finish=length`,
   completion_tokens 2027-2034) before emitting the verdict, which was the last JSON field.
   100% of gemini-3's 46 unscorable were this.
2. LaTeX breaks strict JSON. Several "unscorable" outputs were actually correct, e.g.
   `{"evidence":"\\[ K_a = \\frac{...} \\]", ..., "verdict":"pass"}` -- LaTeX single
   backslashes are invalid JSON escapes, so `json.loads` threw and the good verdict was
   discarded (the gemini-2.5-flash "other" bucket).

Fixes (both general, no extra token cost):
- Prompt v3 emits `"verdict"` FIRST, so truncation cannot eat the decision.
- The parser extracts the verdict by regex (`"verdict"\s*:\s*"(pass|fail)"`) instead of
  requiring strict `json.loads`, recovering truncated-after-verdict and LaTeX-broken JSON;
  v3 also instructs "no LaTeX backslashes in JSON".

Result at the same max_tokens 2048 (unscorable count / 261):

| model | v1 strict | v2 brevity | v3 verdict-first + robust parser |
| --- | --- | --- | --- |
| gemini-3-flash-preview | 48 (18.4%) | 24 | 0 |
| gemini-2.5-flash | 21 (8.0%) | 10 | 0 (+1 transient network error) |

Implication: the regex parser is a pure win (keep it always). Gemini is fully viable again.

### v3 recompute of the acceptance gate (261 cases) -- and a caution

Recomputed all 5 models under v3 (`results_strict_v3/`). v1 evidence-first -> v3
verdict-first:

| model | FP% v1->v3 | crit-sens v1->v3 | content-FP% v1->v3 |
| --- | --- | --- | --- |
| claude-sonnet-4-6 | 24.1 -> 35.3 | 0.775 -> 0.667 | 35.7 -> 46.4 |
| claude-opus-4-6 | 31.0 -> 44.0 | 0.706 -> 0.598 | 32.1 -> 50.0 |
| gemini-2.5-flash | 26.7 -> 35.3 | 0.745 -> 0.647 | 25.0 -> 32.1 |
| gemini-3-flash-preview | 18.1 -> 28.4 | 0.814 -> 0.725 | 14.3 -> 28.6 |
| gpt-4.1 | 34.5 -> 27.6 | 0.686 -> 0.745 | 39.3 -> 25.0 |

- gemini-3's earlier lead was truncation-inflated (unscorable->fail); under fair v3 its
  critical-sensitivity is 0.725, not 0.814.
- CAUTION: verdict-first (v3) traded judgment quality for robustness. sonnet and gpt-4.1
  had 0 unscorable in BOTH versions, yet their false-pass shifted materially (sonnet worse:
  FP 24->35, crit-sens 0.775->0.667). Parsing was not their variable, so the cause is the
  prompt change -- emitting the verdict before reasoning (and terse rationale) makes the
  judge more lenient. Evidence-first reasoning graded more strictly/accurately.
- Recommended design going into the gold sets: evidence-first ordering + the robust regex
  parser + brevity + enough max_tokens to avoid truncation (robust AND accurate), rather
  than verdict-first. Confirm on the biggen/tutoreval/tutorbench gold sets, since prompt
  ordering and model clearly interact.

## v4 (analysis -> verdict -> evidence): best config so far on tutorbench

v4 puts a brief analysis first (reason before deciding -> quality), the verdict second
(before the verbose evidence -> truncation-robust), evidence last. Acceptance gate,
261 cases (`results_strict_v4/`), with v4 unscorable counts:

| config | critSens | FP% | contentFP% | unscorable | cost | speed |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-4.1 + v4 | 0.784 | 25.0 | 25.0 | 0 | ~$2,000 | 33 c/s |
| sonnet-4-6 + v1 | 0.775 | 24.1 | 35.7 | 0 | ~$3,200 | 2 c/s |
| sonnet-4-6 + v4 | 0.725 | 28.4 | 35.7 | 0 | ~$3,200 | 15 c/s |
| gemini-2.5-flash + v4 | 0.686 | 31.9 | 28.6 | 9 (3.4%) | ~$950 | 4 c/s |
| gemini-3-flash-preview + v4 | 0.814* | 19.8 | 17.9 | 34 (13%) | ~$1,200 | 8 c/s |

- Prompt x model interaction is the headline: **gpt-4.1 went from the worst judge under v1
  (critSens 0.686) to the best under v4 (0.784)** -- it responds strongly to analysis-first
  structure. Best clean config on tutorbench = **gpt-4.1 + v4**: highest critical-failure
  sensitivity, lowest false-pass and content-FP, 0 unscorable, fastest, ~at budget.
- *gemini-3-flash-preview's 0.814 is STILL truncation-inflated (13% unscorable->fail); it
  ignores brevity and only reaches 0 unscorable under v3 (verdict-first), which hurts
  quality. It is hard to use reliably -- deprioritize.
- gemini-2.5-flash is cheapest but the weakest judge (critSens 0.686, lenient).
- All on 261 tutorbench cases with known label noise -> directional. The winning config
  may differ per benchmark; confirm on the biggen/tutoreval gold sets. Keep the robust
  regex parser in all cases.

## Caveats

- Gemini numbers are not a fair read (format non-compliance under `max_tokens 512`).
  Re-run with higher `max_tokens` (1024-2048) and/or a JSON/structured-output mode.
- Label noise: the judge-validation notes flagged a couple of human labels as
  adjudication candidates, so ~1.0 agreement is not the ceiling.
- False-pass vs accuracy is a real trade-off for judge selection; pick based on whether
  the downstream CAT is more harmed by lenient or strict grading.
- Cost not quoted here (each case = one short call, ~110-450 completion tokens); measure
  per-provider before a full-scale run.

## Recommended next steps

1. Done: Gemini re-run with the canonical Qwen prompt + JSON mode (see section above).
   For a strict apples-to-apples table, re-run ALL models with
   `--adapter generic-binary --json-mode --max-tokens 2048` so every model uses the exact
   frozen-Qwen prompt and normalized output.
2. If false-pass is the priority, shortlist claude-sonnet-4-6; if raw agreement, gpt-4.1.
   Gemini flash trails on this rubric even after format normalization.
3. Integration note: the `edullm_adaptive` CAT mode currently hardcodes and validates the
   frozen Qwen judge (and its explicit-token-logprob contract). Using an API judge in the
   live CAT requires relaxing that judge contract to accept a gateway/API judge -- a
   separate code change from this standalone pilot.

## How to reproduce

```bash
# materialize gold labels once (they live on the frq/tutorbench ref):
git show frq/tutorbench:eduLLM-Evals/graderValidationStuff/_recovery/human_labels.csv > eduLLM-Evals/api_judge_pilot/human_labels.csv

# agreement across models (261 cases):
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/run_api_judge_pilot.py \
  --labels eduLLM-Evals/api_judge_pilot/human_labels.csv --concurrency 12

# parallelizability sweep on one model:
uv run --no-project --with httpx python eduLLM-Evals/api_judge_pilot/run_api_judge_pilot.py \
  --labels eduLLM-Evals/api_judge_pilot/human_labels.csv \
  --models openai-group/gpt-4.1 --limit 60 --sweep 1 4 8 16 32 48
```

Artifacts: `results/summary.json`, `results/<model>.verdicts.jsonl`,
`results_sweep/summary.json`.

Note: this pilot lives under `eduLLM-Evals/` (untracked local, matching the repo's
convention for this data). It is not committed; a branch switch that checks out a tree
without these paths will leave untracked files in place, but do not `git clean` them.
