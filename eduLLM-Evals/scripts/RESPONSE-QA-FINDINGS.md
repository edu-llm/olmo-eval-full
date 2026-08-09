# Tutor response run 8-9F — QA findings on the first 7 models

**Scope audited:** `eduLLM-Evals/TutorModelRun8-9F/FirstFew/small_lt2b/{BiGGen,TutorEval}/*.jsonl`
— 7 models × 2 benchmarks = 14 shards, 10,647 rows.

**Verdict:** the harness plumbing is healthy; the **generation config and prompt
fitting are not**. Four defects (P0–P1 below) make a large share of rows unusable,
and one of them is currently *invisible* on small-context models and will surface
on most of the remaining roster. All are fixable without touching the runner's
architecture. **Recommend fixing P0+P1 and re-running these 7 before continuing.**

---

## 0. Usable-row estimate

A row is counted unusable if the output is empty, the prompt was truncated, the
output is a degenerate loop (4-gram repetition ≥ 0.7 over ≥60 words), or the text
is under 15 characters.

| shard | rows | usable | | shard | rows | usable |
|---|---:|---:|---|---|---:|---:|
| TutorEval / gpt2-large | 828 | **3%** | | BiGGen / bloom-1b1 | 693 | 46% |
| TutorEval / pythia-410m | 828 | **14%** | | TutorEval / bloom-1b1 | 828 | 55% |
| BiGGen / gpt2-large | 693 | **22%** | | BiGGen / bloomz-1b7 | 693 | 63% |
| BiGGen / pythia-410m | 693 | **27%** | | BiGGen / SmolLM2-135M | 693 | 73% |
| TutorEval / bloomz-1b7 | 828 | 29% | | TutorEval / Qwen2.5-1.5B | 828 | 74% |
| TutorEval / SmolLM2-135M | 828 | 48% | | TutorEval / SmolLM2-1.7B | 828 | 77% |
| | | | | BiGGen / SmolLM2-1.7B | 693 | 87% |
| | | | | BiGGen / Qwen2.5-1.5B | 693 | 100% |

## 0b. What is verified healthy — do not spend time here

- **Coverage is exact.** Every shard has all 693 BiGGen / 828 TutorEval scenarios.
  Zero missing, zero duplicate, zero malformed JSON lines.
- **No hard failures.** Zero `Finish Reason == "error"`, zero `Issue == 1` rows.
- **`Chat Template Applied == 0` is correct** for all 7 (six base models; BLOOMZ
  ships no chat template).
- **Determinism config is uniform**: `temperature 0.0`, `top_p 1.0`, `seed 0`.

---

## P0-1 — Prompt truncation deletes the system instruction, keeping only the tail

**Symptom.** When a prompt overflows the window, the model receives a fragment
that starts mid-document and has no task instruction. **100% of truncated rows
lost the `System:` header** — 1,055 rows in this sample:

| shard | rows truncated | lost `System:` header |
|---|---:|---:|
| TutorEval / gpt2-large | 460 (56%) | 460 |
| TutorEval / pythia-410m | 343 (41%) | 343 |
| TutorEval / bloomz-1b7 | 136 (16%) | 136 |
| TutorEval / SmolLM2-1.7B | 39 (5%) | 39 |
| TutorEval / SmolLM2-135M | 39 (5%) | 39 |
| BiGGen / gpt2-large | 32 (5%) | 32 |
| BiGGen / pythia-410m | 6 (1%) | 6 |

**Evidence.** Diffing the same scenario between a 32k model and gpt2-large gives
`common prefix kept: 0 chars | common suffix kept: 3920 chars` — the head is
discarded wholesale. Scenario `te_0002`, 10,374 chars → 3,920 chars:

*Intended prompt:*
```
System: You are an expert science tutor helping a student. Answer the student's
question accurately and clearly. If reference material is provided, ground your
answer in it; otherwise rely on your own knowledge.

Student: Here is a passage from a textbook I am trying to understand:
"""
Learning Objectives
• Distinguish between the disorders of schizophrenia and depression
...
```

*What gpt2-large actually received:*
```
 diseases affecting the mind onset by brain damage or genetics
• schizophrenia: a psychiatric diagnosis denoting a persistent, often chronic,
mental illness variously affecting behavior, thinking, and emotion
...
```

Its answer was `" Dr. Robert J. Schatzberg, MD, FRCP, FACP, FRCS, FRCP, FRCP, FRCP, ..."`
— it had no idea it was supposed to tutor.

**Root cause.** `eduLLM-Evals/tutor_cat/respgen/runner.py:116-121`:

```116:121:eduLLM-Evals/tutor_cat/respgen/runner.py
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    prompt_cap = max(1, max_model_len - MIN_GEN)
    truncated = len(ids) > prompt_cap
    if truncated:
        ids = ids[-prompt_cap:]  # keep the most recent turn (the student's latest prompt)
        text = tokenizer.decode(ids, skip_special_tokens=False)
```

`ids[-prompt_cap:]` keeps the tail. The intent is sound and documented in
`backends.py:152-157` — the generation cue (`Tutor:` / chat gen token) sits at the
tail and must survive, so `truncation_side = "left"`. **The intent is right; the
implementation is too blunt** — it protects the tail by sacrificing the head,
when the material that should be dropped is the *middle* of the reference text.

**Fix — middle-out truncation.** Keep both ends, drop from the middle:

1. Render the prompt in three parts: `head` (system instruction + the student's
   opening line), `body` (reference passage), `tail` (the actual question +
   `\n\nTutor:`).
2. If the total exceeds `prompt_cap`, trim **`body`** only, from its middle, and
   splice in an explicit marker (e.g. `\n[...reference material truncated...]\n`)
   so the model and the judge can both see that material is missing.
3. Only if `head + tail` alone still overflows should anything else be cut — and
   at that point the item should be **skipped**, not mangled (see §Design below).

Add a post-fit assertion: a fitted prompt must still start with the rendered
system prefix and end with the generation cue. Fail loudly otherwise.

---

## P0-2 — Empty responses from immediate EOS, silently logged as success

**Symptom.** The model emits exactly **1 token**, which is EOS, so `Output` is
`""`. It is recorded as `Finish Reason: "stop"`, `Truncated: 0`, `Issue: 0` — i.e.
indistinguishable from a successful generation.

Qwen2.5-1.5B on TutorEval: **209 / 828 rows (25%) empty**, and **204 / 443 (46%)
of rows with prompts ≥ 1000 tokens**. The rate tracks prompt length:

| prompt tokens | <200 | 200–500 | 500–1k | 1k–2k | 2k–4k | 4k–8k | >8k |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 367 | 1 | 17 | 136 | 148 | 134 | 25 |
| empty | 1 | 0 | 4 | 40 | 78 | 68 | 18 |
| **rate** | 0% | 0% | 24% | **29%** | **53%** | **51%** | **72%** |

Example — `te_0790`, prompt 11,753 tokens inside a 32,768 window, nothing
truncated, prompt correctly ends with `...up to $20$ matches all be zero?\n\nTutor:`
→ `Output = ''`, `Output Tokens = 1`, `Finish Reason = 'stop'`.

**Why this is the most dangerous finding.** Qwen is the *only* model here that
exhibits it, because it is the only one whose window is large enough to receive
the long prompts intact — every other model had its prompt truncated below the
threshold first, so the bug was masked. Per-model rate on prompts ≥1000 tokens:

```
Qwen_Qwen2.5-1.5B             n=443  empty=204 (46%)   otok==1 = 204
bigscience_bloomz-1b7         n=443  empty= 12 ( 3%)   otok==1 =  12
EleutherAI_pythia-410m        n=450  empty=  0 ( 0%)
HuggingFaceTB_SmolLM2-1.7B    n=452  empty=  0 ( 0%)
HuggingFaceTB_SmolLM2-135M    n=452  empty=  0 ( 0%)
bigscience_bloom-1b1          n=443  empty=  0 ( 0%)
openai-community_gpt2-large   (no rows >=1000 ptok — all truncated below it)
```

**Every remaining long-context model is exposed**: Qwen2.5 / Qwen3, Llama-3.2,
Falcon3, Granite 3.1/3.3, Phi-3/3.5, Mistral, SmolLM3, OLMo-2 all have ≥32k
windows. Expect ~25–50% empty TutorEval rows across the rest of the run.

**Root cause.** `SamplingParams` is constructed without a floor on generated
tokens, so at `temperature=0.0` a base model that judges EOS most likely after a
long flat-rendered context terminates immediately:

```82:89:eduLLM-Evals/tutor_cat/respgen/backends.py
        def _sp(max_tokens: int) -> "SamplingParams":
            return SamplingParams(
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=max_tokens,
                repetition_penalty=params.repetition_penalty,
                seed=params.seed,
            )
```

**Fix.**
1. Pass `min_tokens=8` (vLLM `SamplingParams`) so EOS cannot win at step 0. For
   the HF path, the equivalent is `min_new_tokens=8` in `model.generate(...)`
   (`backends.py:208`).
2. Treat `Output Tokens <= 1` or blank `Output` as `Issue = 1` with a description,
   so these stop being logged as successes and get retried on resume. Note
   `runner.py:356` already treats `Prompt Tokens > 1` as a validity condition —
   extend the same logic to the output side.

---

## P1-1 — No stop sequences: base models write both sides of the dialogue

**Symptom.** Outputs continue the fake transcript, inventing student turns.
Across the 14 shards the outputs contain:

| leaked header | `Tutor:` | `Student:` | `System:` | `User:` | `Assistant:` | `Human:` |
|---|---:|---:|---:|---:|---:|---:|
| occurrences | 93,349 | 83,579 | 364 | 239 | 122 | 1 |

Per-shard leak rate runs 3%–75%; SmolLM2-1.7B on TutorEval is 621/828 (75%).
Example (`te_0000`, SmolLM2-1.7B) — the tutor answer is fine, then:

```
 No, not at all. Schizophrenia is a different disorder from depression. ...

Student: I'm confused about this. How can you say that one condition has
another as a symptom when they're so different?

Tutor: Well, there are many similarities between them. ...
```

The judge scores this whole blob as the tutor's response.

**Fix.** Add stop sequences to `GenParams` / `SamplingParams` and the HF path:
`["\nStudent:", "\n\nStudent:", "\nSystem:", "\n\nSystem:", "\nUser:", "\n\nUser:"]`.
Keep the flat renderer's role labels and the stop list in one place so they cannot
drift apart. This also reclaims a lot of wasted compute.

---

## P1-2 — BLOOM/BLOOMZ context window over-declared, two different ways

| shard | `Max Model Len` recorded | real window | note |
|---|---:|---:|---|
| bloom-1b1 (both benchmarks) | **32768** | 2048 | equals `max_model_len_cap` — no clamp applied |
| bloomz-1b7 (both benchmarks) | **4096** | 2048 | equals `DEFAULT_MAX_MODEL_LEN` — config read failed |

Two different failure modes inside one family, which means the resolution path is
not just wrong but non-deterministic. `models_registry.py:135` already has
`("bloom", 2048)` in `KNOWN_CONTEXT_WINDOWS`, so the static table was not
consulted or was overridden.

**Consequence.** bloom-1b1 was fed prompts up to **9,612 tokens on a
2,048-position model** with `Truncated: 0`. BLOOM uses ALiBi so it degrades
instead of erroring — which is exactly why it shows 45–54% severe repetition
rather than a crash. Example `te_0790`: 9,612 prompt tokens, output is
`"It is not easy to solve it."` repeated for the full 4,096-token budget.

**Fix.** After resolving `max_model_len`, assert it against the static
`KNOWN_CONTEXT_WINDOWS` entry and take the **minimum**; if `config.json` could not
be read, fail the model rather than silently using the cap or the 4096 default.
The SSL/`truststore` bootstrap noted in `AdaptiveTesting/Test/Inference/README.md`
§12b is the usual reason a config read fails — check that first.

---

## P1-3 — `repetition_penalty: 1.1` is too weak for sub-1B base models

gpt2-large hit the output cap on **686/693 (99%)** of BiGGen and **823/828 (99%)**
of TutorEval rows; pythia-410m 84% / 89%. Severe-loop rates are 59–75%.

Verbatim, gpt2-large `te_0067` (4-gram repetition 0.98, 256 tokens):
`"\n\nMicrosoft Access\n\nMicrosoft Access\n\nMicrosoft Access\n\n..."` for the entire budget.
pythia-410m `te_0760` (1,980 tokens): `"Student: Thank you very much.\n\nTutor: Good luck!"` on a loop.
bloom-1b1 `bgb_0457` (4,096 tokens): `"The answer is that the answer is that..."`.

**Fix.** Raise `repetition_penalty` (≈1.15–1.2) for models under ~1B, or add
`no_repeat_ngram_size`. Independently, **drop `max_new_tokens` from 4096 to
~768** — no tutor turn needs 4k tokens, and right now the budget is mostly
financing loops. Combined with P1-1's stop sequences this is a large wall-clock
saving on the remaining run.

---

## P2 — Lower severity, fix opportunistically

**`Latency (s)` is a per-model constant, not a measurement.** All 1,521 Qwen rows
read `0.0751`; all 1,521 bloom rows read `1.8858`. Source:
`_effective_latency(g.latency_s, elapsed, len(todo))` at `runner.py:466` divides
the batch wall-clock by the row count. Harmless but useless for spotting hangs —
either record real per-row latency or rename the column to
`Mean Batch Latency (s)` so nobody trusts it.

**`max_new_tokens` silently collapses to a 256 floor.** The manifest says 4096;
the recorded per-row value is `min(4096, max(256, window − prompt_tokens))` from
`MIN_GEN = 256` (`runner.py:30`, `runner.py:123`). gpt2-large on TutorEval sits at
768/768 prompt tokens and 256 output tokens on essentially every row. This is
working as designed, but it means small-window models get a mangled prompt *and* a
clipped answer — it is the multiplier on P0-1, not an independent bug.

**Some source items have the question clipped mid-word.** Confirmed by inspection
on TutorEval `te_0011`, `te_0042`, `te_0043` and others — the question text ends
inside a word, **identically across all 7 models** regardless of window size and
with `Truncated: 0`, so this is upstream of the harness in the dataset prep:

```
te_0011: ..."I'm not sure how to define that the behaviour of the network doesn't chang"
te_0042: ..."compare with traditional methods in red-black tr"
te_0043: ..."what should we do during an insertion operation to maintain the tree's balanc"
```
then the renderer appends `\n\nTutor:`.

The audit script's `prompt_end_advisory` flag surfaces candidates but **over-reports
and should not be read as a count** — many prompts legitimately end on a bare word
(BiGGen's field style, `Phenomenon: Increased intensity of hurricanes`, or a list
ending `...Notre Dame Cathedral`), and separating those from real clips needs a
dictionary. Treat the flagged rows as a list to eyeball. Because it hits every
model equally it does not bias model-vs-model comparison, but the affected items
are corrupted and their IRT difficulty estimates will be meaningless, so it is
worth a pass at the data-prep step to quantify and fix properly.

**BLOOMZ-1b7 is arguably not a usable tutor model.** Median output is **2 tokens**
on TutorEval, with 578/828 (70%) responses under 15 characters:
`' No'`, `' Yes'`, `'...'`, `' We need to use more nitrogen'`. This is genuine
BLOOMZ behaviour — it is prompt-tuned for terse task completion, not dialogue —
not a harness bug. Decide whether it earns a roster slot; as-is it contributes
almost no rubric signal.

---

## Design question: how much truncation is actually necessary?

Measured on the 32k model, where nothing is truncated:

| benchmark | n | median prompt | p90 | max |
|---|---:|---:|---:|---:|
| TutorEval | 828 | 1,215 tok | 5,822 | 11,753 |
| BiGGen | 693 | 153 tok | 518 | 5,017 |

Items that do **not** fit a given window (before the 256-token answer reservation,
which tightens the effective cap to `window − 256`):

| window | TutorEval | BiGGen |
|---|---:|---:|
| 1,024 | 440 (53%) | 16 (2%) |
| 2,048 | 302 (36%) | 4 (1%) |
| 4,096 | 154 (19%) | 1 (0%) |
| 8,192 | 25 (3%) | 0 |
| 32,768 | 0 | 0 |

So truncation is **genuinely unavoidable** for ≤2k-window models on TutorEval —
you cannot put 11,753 tokens into gpt2-large's 1,024 positions — and **almost
never needed on BiGGen**. But note what that implies for calibration:

> A truncated item is **not the same item**. If gpt2-large answers a mutilated
> `te_0002` while Qwen answers the full one, they have not taken the same test,
> which breaks the common-item assumption the IRT fit depends on. Item difficulty
> for long-context TutorEval items is currently estimated from a mixture of
> respondents who saw the question and respondents who saw a fragment.

**Recommendation:** prefer **skip-and-record-missing** over truncate. Write the row
with `Issue = 1`, `Issue Description = "prompt exceeds context window"`, and a
null output. The calibration path already treats missing cells as NaN and
marginalises them (that is the documented handling for judge `no_decision`), so a
missing cell is statistically clean while a truncated cell is silent contamination.
Middle-out truncation (P0-1) is still worth implementing for the mild overflows
(the 3% on 8k models, where only a little reference text is lost), but for a
model that can only see 8% of the prompt, skipping is the honest answer.

Concretely, a reasonable policy: truncate middle-out while `body` retains ≥50% of
its tokens; otherwise skip the item as missing.

---

## Fix checklist

- [ ] **P0-1** Middle-out truncation in `_fit_prompt_and_budget`
      (`runner.py:103-125`); assert the fitted prompt still starts with the system
      prefix and ends with the generation cue.
- [ ] **P0-2** `min_tokens=8` in `SamplingParams` (`backends.py:82-89`) and
      `min_new_tokens=8` on the HF path (`backends.py:208`).
- [ ] **P0-2** Flag `Output Tokens <= 1` / blank output as `Issue = 1` so it is
      retried rather than counted as a success (`records.py`, `shard.scan_shard`).
- [ ] **P1-1** Stop sequences on both backends, shared with the flat renderer's
      role labels.
- [ ] **P1-2** Clamp resolved `max_model_len` to `KNOWN_CONTEXT_WINDOWS`; fail
      loudly when `config.json` cannot be read.
- [ ] **P1-3** Raise `repetition_penalty` for <1B models; drop `max_new_tokens`
      4096 → ~768.
- [ ] **Policy** Decide truncate-middle vs skip-as-missing for the ≥50%-loss case.
- [ ] **Data** Fix the mid-word question clipping upstream (~5% of items).
- [ ] Re-run the 7 models in `FirstFew` and re-audit before launching the rest.

## How to reproduce this audit

`audit_responses.py` in this folder reproduces every number above:

```bash
uv run python eduLLM-Evals/TutorModelRun8-9F/audit_responses.py \
    eduLLM-Evals/TutorModelRun8-9F/FirstFew
```

It prints coverage, a per-shard flag matrix, config/token stats, a usable-row
estimate, and examples of each pathology. It **exits non-zero** while any blocking
defect remains, so it can go straight into the run script as a gate.

Blocking flags are `finish_error`, `output_null`, `output_empty`,
`single_token_output`, `lost_system_header`, `prompt_collapsed`, `transcript_leak`.
Current baseline on `FirstFew`:

```
RESULT: FAIL — blocking defects present
  transcript_leak            3904 rows
  lost_system_header         1055 rows
  output_empty                223 rows
  single_token_output         222 rows
```

Everything else (`finish_length`, `loop_severe`, `output_tiny`,
`prompt_end_advisory`) is reported but not gated, because some of it is genuine
small-model behaviour rather than a defect. Target after the fixes: all four
blocking counts at zero, and `loop_severe` down to low single digits for the tiny
base models.

Add the roster's remaining models to `REAL_WINDOW` in the script as you go — that
table is what catches the P1-2 class of bug, and a model absent from it is not
checked.
