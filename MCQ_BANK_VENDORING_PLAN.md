# Vendoring the pedagogy / piqa / socialiqa MCQ banks

Bring three self-calibrated MCQ item banks into `calibrated_datasets/` so the CAT runner can
select items from them, following the layout the nine existing banks already use.

**Status 2026-08-09.** Row order is proven. Fit families are settled by measurement. The
substantive remaining work is that *no per-item IRT parameters exist yet for any of these
banks* — they must be fit and persisted — plus three pipeline capabilities that turn out not
to exist, and one join boundary still unverified.

---

## 1. The banks

| Bank | Items | Models | Kept by `filter_items` | Item id | Positional? | Fit family |
| --- | --- | --- | --- | --- | --- | --- |
| `pedagogy` | 920 | 78 | 318 | `pedagogy_<question_id>`, `0..919` | No — id travels with the row | **1PL** |
| `piqa` | 1838 | 52 | 831 | `piqa_<i:05d>` | **Yes** | **2PL** |
| `socialiqa` | 1954 | 52 | 969 | `socialiqa_<i:05d>` | **Yes** | **1PL** |

Both halves of the source data live under
`s3://edullm-adaptive-inference-056956104102/full200/`:

- `mcq_cache/<bank>.n2000.s1234.jsonl` — items **exactly as the sweep saw them**, written
  2026-08-01T15:23:51Z. Every bank is under the 2000 cap, so `_cap` never fired.
- `results/Outputs/mcq/<bank>/<model>.csv` — one file per model; columns
  `question_id, model, benchmark, predicted, gold, result, scoring_method`.

Aug-1 loaders: `load_pedagogy` / `load_piqa` / `load_socialiqa` in
`AdaptiveTesting/Test/Inference/datasets_registry.py` on `Research` (worktree `_research_wt`,
HEAD `6998270f`).

## 2. Execution order

| # | Step | Blocking? |
| --- | --- | --- |
| 0 | Row order proven | **done** |
| 0b | Verify task enumeration matches the cache | do first — cheap, offline |
| 1 | Pull the three response matrices from S3 | gates step 2 |
| 2 | Fit and persist per-item parameters | the real work |
| 3 | Per-token mean normalization (protocol change) | independent of 2 |
| 4 | Write the pedagogy `olmo_eval` task | gates step 5 |
| 5 | Vendor items + manifests | needs 0b, 2, 4 |
| 6 | Ingress + register | needs 5 |

---

## 3. Phase 0 — row order is proven (done)

Position enters at **exactly one boundary**: a loader minting `f"piqa_{i:05d}"` from the
enumeration index. Everything downstream is label-keyed — `load_benchmark` (`matrix.py:47`)
builds the matrix as `pd.Series(correct, index=df["question_id"])` and pandas aligns by
label — so verifying that one boundary verifies the chain.

1. **Content diff (primary).** Re-fetched all three datasets and compared against the
   `mcq_cache` snapshot: 920 / 1954 / 1838 items, **100% strict-exact on stem, choices and
   gold**, zero orphans, whitespace-normalized rate identical to strict. Every qid's numeric
   suffix still equals today's row index. Fingerprints: pedagogy `5cc94c54…`, socialiqa
   `1177045b…`, piqa `79b636e7…`.
2. **Provenance (corroborating).** Every upstream ref predates the sweep. Weaker alone — a
   force-push or backdated commit would defeat it — so the content diff leads.

Pins to adopt in place of the mutable refs the loaders used:

| Bank | Loader used | Pin to | Tip |
| --- | --- | --- | --- |
| `piqa` | `refs/convert/parquet` | `142c51238b3ca2bc61e9a075913871b8b600e8e1` | 2024-03-07 |
| `socialiqa` | `refs/convert/parquet` | `537a2ec8ec565adc0b70b70752893e59e024df26` | 2024-05-23 |
| `pedagogy` | `main` (**unpinned**) | `55133a11b8f3c05adc186c11d8e4e100219dd36d` | 2025-06-24 |

## 4. Phase 0b — task enumeration is not yet verified (do first)

Phase 0 proved *cache ≡ HF rows*. It did **not** prove *cache ≡ what the `olmo_eval` task
enumerates*, which is the boundary vendoring actually joins on. Two facts make this a real
risk:

- **Tasks render text differently from the Aug-1 loaders.** `socialiqa.py:68` builds
  `f"{context} {question_text}"`; the loader built
  `f"{context.strip()}\n\n{question.strip()}"`. piqa's task keeps raw `sol1`/`sol2` where the
  loader stored `t.strip()` choices.
- **Tasks filter rows.** `socialiqa.py:65` skips a doc with empty context or question. The
  Aug-1 loaders skipped **zero** rows in all three banks (ids are dense), so if a task skips
  even one row, every later position shifts and a positional bridge silently misaligns.

This does not mean the join is broken. It means the bridge must be built **positionally
against the task's own enumeration** — bank index *i* ↔ the task's *i*-th instance, with the
content hash computed from **the task's** rendering, never from our cached text. Hashing our
cache text against task hashes would miss on every item.

**RESULT: PASS for both banks.** piqa enumerates 1838 instances and socialiqa 1954, matching
the raw counts, with **every** position agreeing on a whitespace-normalized key and gold
answers agreeing throughout. Zero rows filtered — socialiqa's empty-context guard never
fires. Both tasks resolve the same commits the Aug-1 loader used (`142c5123…`, `537a2ec8…`).
Normalized-key fingerprints: piqa `e1c81887ba66399f`, socialiqa `69b75b61c2035e37`. A
positional bridge is sound for both.

The whitespace normalization was load-bearing rather than cosmetic: strict-exact stem
agreement on socialiqa is **0 / 1954**, because the task space-joins where the loader used
`\n\n`. Choices are 1954/1954 strict. A one-position shift control breaks every position, so
the key is sharp enough to detect a single filtered row.

**Two landmines this surfaced, both for Phase 5:**

- **The bridge is valid only for the bare task spec.** `socialiqa:xlarge` and
  `socialiqa:mc_olmo3base` set `limit=10000`, which switches the loader to validation+train
  and then `random.Random(1234).sample()`s it — 10,000 instances in an unrelated order, which
  would misalign *silently*. The `DatasetSpec` must name `socialiqa` or `socialiqa:full`.
  `piqa`, `piqa:full`, `piqa:olmes` and `socialiqa:olmes` all enumerate correctly.
- **socialiqa has 19 duplicate content keys** (piqa has none). `drop_ambiguous_ids` drops
  *every* claimant of a colliding hash, so expect to lose up to 38 items there.

**Environment, corrected.** Two blockers reported during this check turned out not to be
real: bare repo ids resolve fine under `huggingface_hub` 1.26.1 once truststore is injected
(`piqa` → `ybisk/piqa`, `social_i_qa` → `allenai/social_i_qa`), and the earlier `HfUriError`
was a TLS failure wearing a different hat. `olmo_eval` does resolve, but **only when running
from the `olmo-eval-full` directory** — run vendoring from there.

## 5. Phase 1 — per-token mean normalization

**Verified.** Both Aug-1 backends average rather than sum: the HF path ends `cont_lp.mean()`
(`engine.py:255`), vLLM computes `total / max(1, n)` (`:236`), and `mcq_scoring.py:56` takes a
plain argmax with no further normalization.

`MCQ_SCORE_NORMALIZATIONS` (`common/inference.py:608`) offers only
`unnormalized_sum_of_continuation_logprobs` and `continuation_logprob_per_character`.
Neither matches, and **this is not addable as a third dict entry**: the `ScoreNormalization`
protocol is `score(total_logprob: float, continuation: str)` (`inference.py:547`) and is
deliberately tokenizer-blind, while `_continuation_logprob` computes `cont_len` and discards
it (`:806`, `:1194`). Real scope: the protocol signature, both backends'
`_continuation_logprob`, both `score_items` call sites (`:853`, `:1235`), and the
positional-call tests (`test_score_normalization.py:118`).

Load-bearing — length normalization changes which option wins, so these banks must not fall
through to `DEFAULT_SCORE_NORMALIZATION`.

**Backend divergence to settle:** vLLM clamps `min(clen, len(f_ids)-1)` with a non-prefix
fallback (`engine.py:216-217`); HF slices raw `clen-1` with none. Identical in the normal
case, divergent when tokenization is not a clean prefix. `configs/inference.full200.yaml`
says `backend: vllm`; confirm no Aug-1 model took the documented `hf_fallback`.

## 6. Phase 2 — fit and persist the parameters

### 6.1 No per-item parameters exist, in any fit family

**This is the actual work.** Every prior study fit in memory, ran CAT, recorded metrics, and
discarded the parameters. An exhaustive search of the Research tree finds no per-item `a`/`b`
for any of the three banks under 1PL, 2PL or 3PL. What was persisted is only: the per-model
response CSVs (in S3); per-model CAT predictions (`per_model_predictions.csv`); **aggregate**
fit diagnostics (`fit_diagnostics.csv` — 9 rows, one per benchmark × family, holding item
counts and the min/max of `a` and `c`, not per-item values); and accuracy-scale linear
calibration coefficients. `run_pl_comparison.py` never writes `a`/`b` to disk.

The recipe is fully pinned, so this is mechanical: girth `rasch_mml` (1PL) and `twopl_mml`
(2PL), `filter_items` on the train slice, the 40/12 split from
`numpy.default_rng(7).permutation`. Recorded cost: pedagogy 1PL 0.9 s, piqa 1PL 0.2 s, piqa
2PL 338 s — the whole refit is minutes.

**Correctness gates:** the refit must reproduce 318 / 831 / 969 kept items, and any 1PL fit
must return `a_min = a_max = 1.0`.

For an export format to copy, `Experiments/openlm_atlas_3pl/run_3pl_diagnostic.py` and
`Inputs/ATLAS/scripts/build_atlas_idx_bridge.py` already write `irt_item_parameters` CSVs for
other benchmarks.

### 6.2 Fit family — decided by measurement

Two studies looked contradictory; they are not, because they measure different calibration
pool sizes. `pl_1_2_3_comparison/README.md:83-91` ran calibrate → CAT → predict on all three
banks off 52 models (40 calibration, 12 held out). Pearson r at SE ≤ 0.30:

| Bank | 1PL | 2PL | 3PL |
| --- | --- | --- | --- |
| pedagogy | **0.890** | 0.716 | 0.793 |
| piqa | 0.916 | **0.937** | 0.790 |
| socialiqa | **0.760** | 0.271 | 0.292 |

`kfold_calib_sweep/README.md:37-54` then swept pedagogy's pool size and found 2PL overtakes
1PL near N = 56, reaching 2PL 0.837 vs 1PL 0.767 at N = 70.

**Decision: pedagogy 1PL, piqa 2PL, socialiqa 1PL.** Pedagogy takes 1PL *deliberately*,
against that N = 70 result, for three reasons — recorded so nobody later reads it as an
oversight:

- The single best pedagogy number anyone measured is 1PL at N = 40, r = 0.890, higher than
  2PL reaches at any pool size.
- Pedagogy's calibration population sits at chance (§8), so per-item discriminations
  estimated from it are mostly noise — exactly what 2PL spends the sample on.
- Fit-then-filter on the same data biases surviving discriminations upward (§8), shrinking
  CAT standard errors below their true value. Rasch has no per-item discrimination to
  inflate, so it cannot fail this way.

The crossover was measured on pedagogy only and does not override piqa's own direct
measurement, so piqa stays 2PL.

### 6.3 `filter_items` — what "reproduce exactly" means

`matrix.py:77` first `dropna(axis=0)`, dropping whole **model rows**, then screens items on
`p >= 1.0`, `p <= 0.0`, or point-biserial below **0.05** (default). It removes 50–65% of every
bank: 318/920, 831/1838, 969/1954. This also supplies socialiqa's item screen under 1PL,
where the non-positive-discrimination rule has no fitted `a` to use.

## 7. Phase 3 — vendor the items

Write `calibrated_datasets/<bank>/{items.jsonl, params.json, manifest.json}` matching
`hellaswag/`. Record the pinned SHAs, the fingerprints, `positional_ids: true` for piqa and
socialiqa, and — improving on the existing manifests, which omit it — the upstream license.

Three prerequisites that do not currently exist:

- **Pedagogy has no `olmo_eval` task.** `vendor_bank.py:962` calls `get_task(spec.task)` and
  `DatasetSpec.__post_init__` (`datasets.py:366`) requires one; `piqa.py` and `socialiqa.py`
  exist, `pedagogy.py` does not. It must render the `answer_a..answer_g` layout and read the
  pinned gated revision.
- **`FIT_FAMILIES` has no `1pl`.** `datasets.py:66` lists only `("2pl", "3pl")`, and
  `check_parameter_family` (`vendor_bank.py:306`) classifies purely on `g != 0`, so a Rasch
  fit is silently stamped `2pl`. Rasch *is* storable today as 2PL with `a ≡ 1` — identical
  mathematically — but with two of three banks Rasch, that would leave two manifests
  misreporting their own family. Add the label.
- **Dedup.** `drop_ambiguous_ids` (`vendor_bank.py:202`) drops *every* claimant of a colliding
  content hash; hellaswag lost 204 items this way. Count duplicate question+choices pairs per
  bank before vendoring and state the expected loss.

Also: `check_alignment` (`vendor_bank.py:500-507`) requires the bank's max `X` index to equal
the bridge row count, so survivors must keep **original 1-based positions with gaps** rather
than being renumbered `X1..Xn`.

## 8. Validity — what is established, what is not

**Target population.** Evaluation targets are **1B–7B models**; sub-1B checkpoints exist only
to exercise the pipeline. The calibration population is 0.5B–7B, so targets and calibration
population coincide, and the target-mismatch failure that afflicts the MATH bank in
`RESULT_CAVEATS.md` does not apply here.

**Per-model behaviour of the chosen fits.** `pl_1_2_3_comparison/per_model_predictions.csv`
(216 rows) holds per-model rows for all three families. Spearman rank agreement between
predicted and actual accuracy at SE ≤ 0.30, with counts of structurally impossible
sub-chance predictions:

| Bank | 1PL, all 12 | 1PL, 1B–7B | 2PL, all 12 | 2PL, 1B–7B |
| --- | --- | --- | --- | --- |
| pedagogy | **+0.923** (2 sub) | **+0.900** (1 sub) | +0.692 (1 sub) | +0.600 (0 sub) |
| piqa | +0.783 (0 sub) | +0.718 (0 sub) | +0.699 (1 sub) | **+0.609** (0 sub) |
| socialiqa | **+0.720** (3 sub) | **+0.636** (2 sub) | +0.315 (4 sub) | +0.555 (4 sub) |

This supports the 1PL choices, and it retires an earlier alarm: socialiqa's apparent rank
inversion (the weakest model ranked 3rd best, 4 of 11 sub-chance predictions) was a **2PL**
artifact. **Do not cite `diag_*_se0.3.csv` against the 1PL banks** — those files are the 2PL
fit, as `mcq_diagnostic_summary.csv` confirms (0.7163 / 0.9373 / 0.2713).

One tension noted but not acted on: piqa scores better under 1PL on rank agreement
(+0.718 vs +0.609) with zero sub-chance predictions, though Pearson r favours 2PL (0.937 vs
0.916). Pearson is the study's headline and Spearman on eleven points is noisy, so piqa stays
2PL — but this is the first thing to revisit if piqa looks miscalibrated at the low end.

**Open validity gaps.** These are properties of the data or procedure and survive every
reframe above:

- **Pedagogy's calibration population sits at chance** — 78 models spanning 0.225–0.312
  against a 0.250 floor, 10 of them below chance. A bank that 1B–7B models can barely score
  on carries limited information whatever the fit family.
- **Pedagogy has ~191 all-fail items** — zero of 78 models correct on a ≥4-choice item, which
  guessing should make near-impossible. That suggests mis-keying or items adversarial to
  per-token-mean ranking, and it is ~21% of the bank. Report it rather than letting the
  filter absorb it silently.
- **Predicted accuracy is over-dispersed** — `expanded_linear_calibration.csv` slopes of
  0.15–0.27 make the predicted-accuracy axis four to six times wider than reality.
- **Fit-then-filter inflates confidence.** At N = 78 a null point-biserial has SD ≈ 0.114, so a
  0.05 threshold keeps roughly two thirds of pure noise and biases survivors upward; CAT
  standard errors come out too small. `kfold_calib_sweep/` refits the filter per fold but
  reports only Pearson r, never SE coverage.
- **Unidimensionality is unmeasured, not merely unstated.** `m2_goodness_of_fit/` covers
  ifeval/math/gpqa/musr and `mirt_within_benchmark/` is bbh-only — none of these three banks
  has any dimensionality analysis, and `datasets_registry.py` stores no subdomain field to
  group pedagogy on.

**Prompt parity — every dimension, not just normalization.** The parameters describe the task
*as calibrated*: the `"Question: {stem}\nAnswer:"` template (`question_answer` in
`MCQ_PROMPT_STYLES`), the single-leading-space option prefix, `num_fewshot = 0`,
`add_special_tokens=True` on both halves, socialiqa's `context\n\nquestion` stem assembly, and
pedagogy's variable 2–7 option count. Note a per-token mean makes the divisor depend on *our*
checkpoint's tokenizer, so the metric is not tokenizer-invariant across checkpoints the way a
per-character mean would be.

## 9. Phase 4 — ingress (the assumed path does not exist)

`load_bank` reads parameters only via `git_show(source_ref, spec.params_csv)`
(`vendor_bank.py:455`), and `--bank-dir` (`:1618`) relocates *within a ref* — there is no
local-path flag. "Emit a CSV and let `vendor_bank` ingest it unchanged" is therefore not
possible today. Either commit the refit to a ref and point `source_ref` at it, or add the
local-path code path. Decide explicitly.

## 10. Phase 5 — register

Add three `DatasetSpec` entries in `styles/uni_mcq/datasets.py` with `fit_family` per bank
(pedagogy `1pl`, piqa `2pl`, socialiqa `1pl`), update the allowlist test, and record each
bank's caveats in its notes: pedagogy's chance-level calibration population and all-fail
items, socialiqa's low resolving power, and the cross-bank scale mismatch.

---

## Decisions

1. **License — RESOLVED, vendor all three.** piqa and socialiqa are unspecified upstream (HF
   `cardData` `unknown` / `None`, empty README licensing sections); pedagogy is MIT. Approved
   on the precedent of the nine banks already committed this way. Record `"unspecified"` in
   those two manifests rather than blank or guessed.
2. **Fit family — RESOLVED.** pedagogy 1PL, piqa 2PL, socialiqa 1PL. See §6.2.
3. **socialiqa ships** — no longer a blocker. Under 1PL its in-range rank agreement is +0.636
   with 2 of 11 sub-chance predictions. It remains the weakest of the three (r = 0.760 against
   0.890 and 0.937), so ship it with that stated in its notes.
4. **Cross-bank scale — OPEN.** piqa is 2PL while pedagogy and socialiqa are 1PL, so piqa's
   theta is on a different scale. Anything pooling, averaging, or leaderboarding across banks
   must state how it reconciles them.
5. **Supported ability range — OPEN.** A guard rather than a live problem, since inside 1B–7B
   piqa and pedagogy produce no sub-chance predictions. Set the range to the calibration
   population's span and refuse — do not clamp — outside it, so a sub-1B pipeline-test
   checkpoint cannot silently produce a reportable number.
6. **Ingress mechanism — OPEN.** Commit the refit to a ref, or add a local-path flag (§9).

## Standing caveats

- **Pedagogy's content anchoring is weaker than it sounds.** Its `question_id` values are
  exactly the row indices, so the anchor gives no protection if upstream ever renumbers during
  a reorder. The pin makes this moot by construction.
- **Pedagogy is gated** (401 unauthenticated). Vendoring removes the *runtime* token
  requirement — confirmed: `style.py:310-325` loads items via `resolve()` plus
  `load_items_from_jsonl`, a pure local filesystem path, and `get_task` never appears in the
  runtime path. But *vendoring itself* still needs a token, since `vendor_bank` enumerates the
  task.
- **Environment.** A corporate MITM cert breaks Python TLS here, so `datasets` and `requests`
  cannot reach HF; `truststore.inject_into_ssl()` fixes it and the `olmo-eval-full` venv
  carries truststore. `curl.exe` works unconditionally. There is no repo-root `.venv`.
- Two parallel shell calls in a stateful PowerShell session can race on the working directory
  and silently return empty listings. Use absolute paths.
