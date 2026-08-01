# BUG — Head-Truncated Prompts in the full200 FRQ Run

> **The code to fix lives in the SIBLING repo, not this one:**
> `olmo-eval-full/AdaptiveTesting/Test/Inference/` — specifically `frq_generate.py`,
> `frq_shard.py`, `engine.py`, and `models_registry.py`.
> **Do not spend time in `eduLLM-Evals/tutor_cat/respgen/`.** That package is a dormant
> ancestor of the same pipeline. It carries an identical copy of the defect
> (`tutor_cat/respgen/runner.py`, `_fit_prompt_and_budget`), but it did **not** produce the
> `full200-results/` data and is not on the path any future sweep runs through. This memo
> lives here only because this is the repo you can commit and push.

**Purpose:** everything needed to fix the prompt-fitting logic in the 200-model FRQ
(free-response) generation pipeline and repair the rows it has already corrupted.
Self-contained — no prior context required.

**Status:** diagnosed from the 6-model smoke test under
`eduLLM-Evals/full200-results/_full200/`. Every number below was measured from those files
or read from live `config.json` on the Hub; nothing is estimated except the roster-wide
projection in §1 and §4.4, which is explicitly labelled as a projection and shows its
arithmetic.

---

## 1. TL;DR / impact

When a rendered FRQ prompt does not fit a model's context window, the pipeline keeps the
**tail** of the token sequence and throws away the **head**. Because every prompt is built
with the system turn first, the part that gets deleted is the task instruction, and the part
that survives is the trailing generation cue. The model is therefore asked to answer a
question it was never given, and the row is recorded as a normal success (`Issue = 0`,
`Finish Reason` of `stop` or `length`), so the judge scores it and IRT calibrates on it.
The single line responsible is `ids = ids[-prompt_cap:]` in `_fit_prompt_and_budget`
(`AdaptiveTesting/Test/Inference/frq_generate.py`, line 97 at time of writing).

Projected across the full 187-model roster (`AdaptiveTesting/Inputs/Models/models_200.yaml`),
using live context windows for the 153 ids whose `config.json` could be read anonymously and
the measured prompt-length distributions from this smoke test: **about 29,939 of 662,184 FRQ
rows (4.5%) will arrive with no instruction. 106 of those 153 models are affected at all, and
10 of them exceed 10% of their own rows.** In the 6-model smoke test itself, 1,255 of 25,968
rows are already corrupted this way.

**TutorEval is the epicenter and it is not close.** It contributes 24,939 of the 29,939
projected rows, which is **83.3%** — higher than the "roughly two thirds" figure that a
back-of-envelope pass suggests, because TutorEval carries long reference material and its
truncation rate stays material even at a 4096-token window (19.7%). WildBench adds 3,642
rows (12.2%), TutorBench 1,009 (3.4%), and BiGGen 349 (1.2%). **Bridge and InFoBench are
completely clean at every window down to 1024 tokens** — their prompts are short enough that
they never overflow. If you have to trade scope for speed, fixing TutorEval and WildBench
recovers 95% of the damage.

---

## 2. The defect

### 2.1 The prompt fitter drops the head

```76:102:AdaptiveTesting/Test/Inference/frq_generate.py
def _fit_prompt_and_budget(
    text: str, tokenizer, max_model_len: int, max_new_tokens: int
) -> tuple[str, int, bool, int]:
    """Fit the prompt into the context window and size this item's generation
    budget. Returns (text, prompt_tokens, truncated, gen_budget).

    Without a tokenizer (mock backend) tokens are estimated at ~4 chars each and
    the same window math is applied, so a smoke test exercises fitting and
    budgeting rather than silently bypassing them.
    """
    prompt_cap = max(1, max_model_len - MIN_GEN)
    if tokenizer is None:
        prompt_tokens = max(1, len(text) // 4)
        truncated = prompt_tokens > prompt_cap
        if truncated:
            text = text[-prompt_cap * 4 :]
            prompt_tokens = max(1, len(text) // 4)
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        truncated = len(ids) > prompt_cap
        if truncated:
            ids = ids[-prompt_cap:]  # keep the most recent turn (the student's latest prompt)
            text = tokenizer.decode(ids, skip_special_tokens=False)
        prompt_tokens = len(ids)
    gen_budget = min(max_new_tokens, max(MIN_GEN, max_model_len - prompt_tokens))
    gen_budget = max(1, min(gen_budget, max_model_len - 1))
    return text, prompt_tokens, truncated, gen_budget
```

**Line 97 is the bug:** `ids = ids[-prompt_cap:]`. The negative slice keeps the last
`prompt_cap` tokens and discards everything before them. The inline comment ("keep the most
recent turn") states the intent correctly — for a plain chat transcript, keeping the latest
turn is a reasonable policy — but it is wrong for these prompts, because the instruction is
not in the latest turn. It is at position zero.

Note also that `prompt_cap` is `max_model_len - MIN_GEN`, where `MIN_GEN = 256` is defined at
`frq_generate.py` line 42. It is deliberately **not** `max_model_len - max_new_tokens`; that
older formula collapsed to 1 on any model whose window was smaller than the answer budget and
was fixed previously. The current arithmetic is sound. Only the truncation direction is wrong.

### 2.2 Why the head is the instruction

```191:202:AdaptiveTesting/Test/Inference/frq_prompts.py
def build_chat_messages(scenario) -> list[dict[str, str]]:
    """[system?, ...role-mapped context, user(prompt)], coalesced to alternate."""
    use_case = getattr(scenario, "use_case", "") or _DEFAULT_USE_CASE
    system = system_prompt_for_scenario(scenario)
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for turn in getattr(scenario, "conversation_context", None) or []:
        role = ROLE_MAP.get(turn.get("role", "user"), "user")
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": scenario.prompt})
    return normalize(messages, separator=lambda role: _separator(use_case, role))
```

The system turn is appended first, then conversation history, then the scenario prompt.
Whatever template or flat rendering is applied afterwards preserves that order, so
head-truncation deletes the system turn before it deletes anything else, then eats forward
through the conversation history, then into the body of the question itself. What always
survives is the very end of the string — which for the flat base rendering is the literal
`Tutor:` cue, and for a chat template is the assistant generation prompt.

The per-benchmark system prompts that get destroyed are the verbatim ones in
`SYSTEM_PROMPTS_BY_BENCHMARK` (`frq_prompts.py`, lines 129-145) and the per-use-case
TutorBench personas in `SYSTEM_PROMPTS` (lines 101-118). BiGGen instead uses each instance's
own native `system_prompt`, and InFoBench deliberately has no system turn at all
(`_NO_SYSTEM_BENCHMARKS`, line 127) — that matters for the detection heuristic in §4.2, not
for the bug.

### 2.3 Concrete evidence: scenario `WB_0000`

`WB_0000` is a WildBench item containing a large game-configuration dump, roughly 15,000
characters. Both prompts the reviewer originally flagged turn out to be this same scenario
seen through two different windows.

What the models actually received:

```
naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B   Max Model Len 8192   Prompt Tokens 7936   Truncated 1
  'OLUTIONARY: false,\n\t\t\t\n\t\t\tAI_TYPE: "DEFAULT",\n\t\t\t\n\t\t\tR: 0,\n\t\t\tG: 255,\n\t\t\tB: 50\n\t\t},\n\t\t{\n\t\t'

malteos/gpt2-xl-wechsel-german                          Max Model Len 1024   Prompt Tokens  768   Truncated 1
  'MINISTRATION_COST_CAPITAL: 0.5,\n\t\t\t\n\t\t\tCOST_OF_MOVE: 5,\n\t\t\tCOST_OF_MOVE_TO_THE_SAME_PROV: '
```

Both begin mid-word — `OLUTIONARY` is the tail of `REVOLUTIONARY` and `MINISTRATION_COST_CAPITAL`
the tail of `ADMINISTRATION_COST_CAPITAL` — which is the signature of a cut on a token
boundary in the middle of the text rather than at a message boundary.

What the same scenario looks like when it fits (`OpenBMB/MiniCPM4-0.5B`, 32768-token window,
identical flat base rendering to gpt2-xl):

```
'System: You are a helpful assistant. Respond to the user's request as helpfully, accurately,
 and thoroughly as you can.\n\nStudent: add 10 more balanced governments[aoc2]\n{\n\tGovernment: [\n\t\t{\n\t\t\tName: "'
```

And an untruncated row from gpt2-xl itself, confirming that the same model does render the
preamble correctly when the prompt is short enough:

```
'System: You are a helpful assistant. Respond to the user's request as helpfully, accurately,
 and thoroughly as you can.\n\nStudent: Make this introduction two pages, make it detailed and more contents: '
```

Every one of the 1,255 truncated rows in the smoke test still ends on its generation cue: for
gpt2-xl, 369/369 WildBench, 460/460 TutorEval, 312/312 TutorBench and 80/80 BiGGen truncated
rows end with the literal string `Tutor:`. The model was politely asked to continue a tutor
turn with no idea what it was tutoring.

---

## 3. This is NOT a context-window detection bug

This is the natural first hypothesis, and it is wrong, so please rule it out before touching
`models_registry.py`. I fetched `config.json` live from the Hub for all six smoke-test models
and compared against the `Max Model Len` recorded in the response rows:

| Model | config.json key | Declared value | Recorded `Max Model Len` | Verdict |
|---|---|---|---|---|
| `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B` | `max_position_embeddings` | **8192** | 8192 | correct |
| `malteos/gpt2-xl-wechsel-german` | `n_positions` | **1024** | 1024 | correct |
| `OpenBMB/MiniCPM4-0.5B` | `max_position_embeddings` | 32768 | 32768 | correct |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `max_position_embeddings` | 131072 | 32768 | correctly clamped by `max_model_len_cap` |
| `NousResearch/Hermes-3-Llama-3.2-3B` | `max_position_embeddings` | 131072 | 32768 | correctly clamped by `max_model_len_cap` |
| `Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct` | `max_position_embeddings` | 32768 | 32768 | correct |

**HyperCLOVAX-SEED-Text-Instruct-0.5B genuinely has an 8192-token window.** It is not being
under-detected. Its truncation is a legitimate consequence of a real limit — the problem is
purely what the pipeline *does* about that limit.

Two further confirmations that resolution worked as designed during the run:

- **The Hub was reachable.** `KNOWN_CONTEXT_WINDOWS` in `models_registry.py` (lines 117-194)
  is the offline fallback table, and its entries for these families are `("minicpm", 4096)`
  and `("deepseek", 4096)`. Both models recorded 32768 instead, which is only possible if
  `declared_context_window` successfully read a live `config.json`. The static table was not
  in play, so the `DEFAULT_MAX_MODEL_LEN = 4096` fallback (`models_registry.py` line 97) did
  not fire either.
- **No model with adequate context was truncated.** Across all 36 (benchmark, model) pairs,
  every pair with a 32768-token window has a truncation rate of exactly 0.0%. Truncation
  occurs only on the two genuinely small-window models. There is no case of the budget
  arithmetic firing spuriously.

The `min(request, declared, cap)` logic in `resolve_max_model_len` (`models_registry.py`,
lines 305-329) is behaving correctly. Leave it alone except for the optional cap change in
P5 below.

---

## 4. Measurements

All figures below were read directly from
`eduLLM-Evals/full200-results/_full200/open/<bench>/<slug>.responses.jsonl`.

### 4.1 Provenance of the data

The `full200` artifacts were written by `AdaptiveTesting/Test/Inference/`, not by
`tutor_cat/respgen/`. The directory shapes are produced verbatim by `common.py`
(`mcq_output_path`, `open_responses_path`, `done_marker_path`, lines 147-160), and the row
schema is `frq_records.build_record`. The run used one of the vLLM configs
(`configs/inference.yaml`, `inference.parallel.yaml` or `inference.node.yaml`), all of which
declare `overrides: {}`; the proof is that every full-window model records
`Generation Params.max_new_tokens = 4096`, the manifest default, rather than the 512-1024
per-benchmark values that `configs/inference.cpu.yaml` (lines 18-31) would have injected via
`_frq_overrides` in `run_benchmark.py` (lines 114-127).

Item counts per benchmark: WildBench 1001, TutorEval 828, BiGGen 695, TutorBench 662,
Bridge 642, InFoBench 500 — 4,328 per model, 25,968 rows across the six models.
`Issue = 1` appears on **zero** of those rows.

### 4.2 Per-(benchmark, model) results

`Trunc%` is the share of rows with `Truncated = 1`. `NoSysHead%` is the share of rows whose
`Rendered Prompt` no longer contains the benchmark's verbatim system-prompt text from
`frq_prompts.py` — that is, rows that provably lost their instruction. It is reported as
`n/a` for BiGGen (per-instance native system prompts, no fixed string to search for) and
InFoBench (no system turn by design). `PT` is `Prompt Tokens`.

| Benchmark | Model | `Max Model Len` | True window | Trunc% | PT median / max | NoSysHead% | Chat template |
|---|---|---|---|---|---|---|---|
| wildbench | DeepSeek-R1-Distill-Qwen-1.5B | 32768 | 131072 | 0.0% | 329 / 7767 | 0.0% | 1 |
| wildbench | **gpt2-xl-wechsel-german** | 1024 | 1024 | **36.9%** (369/1001) | 480 / 768 | **36.9%** | 0 |
| wildbench | **HyperCLOVAX-…-0.5B** | 8192 | 8192 | **0.1%** (1/1001) | 351 / 7936 | 0.1% | 1 |
| wildbench | Hermes-3-Llama-3.2-3B | 32768 | 131072 | 0.0% | 330 / 7561 | 0.0% | 1 |
| wildbench | MiniCPM4-0.5B | 32768 | 32768 | 0.0% | 371 / 11903 | 0.0% | **0** |
| wildbench | Vikhr-Qwen-2.5-1.5B-Instruct | 32768 | 32768 | 0.0% | 338 / 7775 | 0.0% | 1 |
| tutoreval | DeepSeek-R1-Distill-Qwen-1.5B | 32768 | 131072 | 0.0% | 1183 / 11750 | 0.0% | 1 |
| tutoreval | **gpt2-xl-wechsel-german** | 1024 | 1024 | **55.6%** (460/828) | 768 / 768 | **55.6%** | 0 |
| tutoreval | **HyperCLOVAX-…-0.5B** | 8192 | 8192 | **3.9%** (32/828) | 1219 / 7936 | 3.9% | 1 |
| tutoreval | Hermes-3-Llama-3.2-3B | 32768 | 131072 | 0.0% | 1163 / 10892 | 0.0% | 1 |
| tutoreval | MiniCPM4-0.5B | 32768 | 32768 | 0.0% | 1313 / 12840 | 0.0% | **0** |
| tutoreval | Vikhr-Qwen-2.5-1.5B-Instruct | 32768 | 32768 | 0.0% | 1191 / 11758 | 0.0% | 1 |
| tutorbench | DeepSeek-R1-Distill-Qwen-1.5B | 32768 | 131072 | 0.0% | 553 / 8430 | 0.0% | 1 |
| tutorbench | **gpt2-xl-wechsel-german** | 1024 | 1024 | **47.1%** (312/662) | 745 / 768 | 46.2% | 0 |
| tutorbench | **HyperCLOVAX-…-0.5B** | 8192 | 8192 | **0.2%** (1/662) | 560 / 7936 | 0.2% | 1 |
| tutorbench | Hermes-3-Llama-3.2-3B | 32768 | 131072 | 0.0% | 543 / 8155 | 0.0% | 1 |
| tutorbench | MiniCPM4-0.5B | 32768 | 32768 | 0.0% | 601 / 8823 | 0.0% | **0** |
| tutorbench | Vikhr-Qwen-2.5-1.5B-Instruct | 32768 | 32768 | 0.0% | 562 / 8438 | 0.0% | 1 |
| biggen | DeepSeek-R1-Distill-Qwen-1.5B | 32768 | 131072 | 0.0% | 151 / 5015 | n/a | 1 |
| biggen | **gpt2-xl-wechsel-german** | 1024 | 1024 | **11.5%** (80/695) | 240 / 768 | n/a | 0 |
| biggen | HyperCLOVAX-…-0.5B | 8192 | 8192 | 0.0% | 163 / 4168 | n/a | 1 |
| biggen | Hermes-3-Llama-3.2-3B | 32768 | 131072 | 0.0% | 159 / 3824 | n/a | 1 |
| biggen | MiniCPM4-0.5B | 32768 | 32768 | 0.0% | 169 / 5535 | n/a | **0** |
| biggen | Vikhr-Qwen-2.5-1.5B-Instruct | 32768 | 32768 | 0.0% | 159 / 5023 | n/a | 1 |
| bridge | all six models | 1024 / 8192 / 32768 | — | **0.0%** | max PT 904 | 0.0% | 1, except gpt2-xl and MiniCPM4 = 0 |
| infobench | all six models | 1024 / 8192 / 32768 | — | **0.0%** | max PT 567 | n/a | 1, except gpt2-xl and MiniCPM4 = 0 |

Two things to read off this table:

1. **The `Truncated` flag is accurate, and truncation is exactly equivalent to instruction
   loss.** For the three benchmarks with a single fixed system prompt (WildBench, TutorEval,
   Bridge), `NoSysHead%` matches `Trunc%` to the row. Every truncated row lost its entire
   instruction, and no untruncated row did. The TutorBench figure differs slightly (46.2% vs
   47.1%) only because TutorBench has three different use-case personas and a handful of
   truncated rows retain a fragment of one.
2. **The two failures are confined to the two small-window models**: 1,221 rows on
   `malteos/gpt2-xl-wechsel-german` and 34 on
   `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B`, 1,255 in total.

### 4.3 How much of the prompt is actually lost

`malteos/gpt2-xl-wechsel-german` and `OpenBMB/MiniCPM4-0.5B` both render through the flat
base path (`Chat Template Applied = 0`), so MiniCPM4's 32768-token rows are a byte-exact
reference for what gpt2-xl *should* have received for the same scenario. Comparing them
confirms the direction of the cut and quantifies the loss:

| Benchmark | Truncated rows | Is an exact tail-suffix of the full prompt | Median characters kept | Worst case kept | Median head characters dropped |
|---|---|---|---|---|---|
| tutoreval | 460 | 460 / 460 | 19.6% | 4.2% | 7,770 |
| wildbench | 369 | 369 / 369 | 58.2% | 6.7% | 1,685 |
| tutorbench | 312 | 311 / 312 | 76.8% | 9.0% | 557 |
| biggen | 80 | 80 / 80 | 81.3% | 16.9% | 558 |
| bridge | 0 | — | — | — | — |
| infobench | 0 | — | — | — | — |

The "exact tail-suffix" column is the proof of direction: 1,220 of 1,221 truncated gpt2-xl
prompts are literally a suffix of the correct prompt, meaning the loss is entirely at the
front. The single TutorBench exception differs only by a boundary character from the
token-level cut. On TutorEval the median row keeps under a fifth of its prompt and the worst
keeps 4%.

### 4.4 Projected blast radius across the roster

Method: read the 187 ids in `AdaptiveTesting/Inputs/Models/models_200.yaml`; fetch each
`config.json` from the Hub and resolve the window the pipeline would use, `min(declared,
max_model_len_cap = 32768)`; then apply the measured `Prompt Tokens` distribution from
`NousResearch/Hermes-3-Llama-3.2-3B` (32768-token window, zero truncation, so its
distribution is the true untruncated one) and count items exceeding `window - MIN_GEN`.

153 of 187 configs resolved anonymously; the other 34 are gated or private without a token
and would resolve in a real run with `HF_TOKEN` set, so the projection is a mild
undercount. Resolved windows: 8 models at 1024, 2 at 1536, 39 at 2048, 39 at 4096, 14 at
8192, 6 at 16384, 41 at 32768, and 4 whose config exposes no positional key (these fall to
`DEFAULT_MAX_MODEL_LEN = 4096`, so they are counted in the 4096 bucket for a total of 43).

Share of each benchmark's items that would be head-truncated at each window:

| Benchmark | 1024 | 1536 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|
| tutoreval | 55.4% | 48.6% | 40.6% | 19.7% | 3.0% | 0.0% | 0.0% |
| wildbench | 24.0% | 11.6% | 3.6% | 0.2% | 0.0% | 0.0% | 0.0% |
| tutorbench | 16.0% | 2.0% | 0.3% | 0.2% | 0.2% | 0.0% | 0.0% |
| biggen | 3.7% | 1.7% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| bridge | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| infobench | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Weighting those rates by the bucket populations gives the roster-wide projection:

| Benchmark | Projected head-truncated rows | Share of all corrupted rows |
|---|---|---|
| tutoreval | 24,939 | 83.3% |
| wildbench | 3,642 | 12.2% |
| tutorbench | 1,009 | 3.4% |
| biggen | 349 | 1.2% |
| bridge | 0 | 0.0% |
| infobench | 0 | 0.0% |
| **Total** | **29,939 of 662,184 (4.5%)** | |

106 of the 153 models have at least one affected row; 10 exceed 10% of their own rows. The
worst are the eight 1024-token checkpoints — `stanford-crfm/BioMedLM`,
`rinna/japanese-gpt-1b`, `openai-community/gpt2-xl`, `openai-community/gpt2-large`,
`malteos/gpt2-xl-wechsel-german`, `benjamin/gerpt2-large`, `MBZUAI/LaMini-GPT-774M` and
`MBZUAI/LaMini-GPT-1.5B`, each at 19.2% — followed by the two 1536-token
`sdadas/polish-gpt2-*` models at 12.5%.

> The 6-model smoke test badly understates the problem: it happens to contain only one
> small-window model out of six, whereas the real roster is roughly half small-window.

---

## 5. Three independent failure modes

The single line at `frq_generate.py:97` produces three distinct problems. Fixing the slice
addresses only the first; the other two need their own changes.

### 5.1 It drops the wrong end

Covered in §2. The instruction is deleted first and the generation cue is preserved last,
which is the worst possible ordering for a prompt whose whole purpose is to state a task.
A defensible truncation policy for these prompts must protect the system turn and the final
user turn, and give up interior conversation history instead.

### 5.2 It fails silently, and that is worse than failing loudly

Every one of the 1,255 corrupted rows carries `Issue = 0`, a normal `Finish Reason` of
`stop` or `length`, and a plausible-looking `Output`. Nothing downstream can tell them apart
from a genuine attempt without re-deriving the truncation itself. The consequence is not
extra noise, it is a systematic bias:

- The judge reads `Rendered Prompt` and `Output` and scores the answer against the rubric.
  A model that was never told the task scores near zero, so the row becomes evidence of
  incapability.
- The IRT layer then cannot distinguish "this model's context window is too small for this
  item" from "this model cannot do this task." Because truncation is strongly correlated with
  small models and with one benchmark, the error is not random. Small-window models get
  systematically depressed ability estimates, and the TutorEval items that overflow get
  systematically inflated difficulty. Both parameters are corrupted, not merely noisy.
- The correct semantics for a prompt that cannot fit is **missing data**, which IRT handles
  natively, rather than a scored failure.

`Truncated = 1` is recorded faithfully and is a usable filter today (see §7), but nothing in
the pipeline or the judge currently consults it.

### 5.3 Resume will not repair the existing rows

```50:59:AdaptiveTesting/Test/Inference/frq_shard.py
def _is_valid_row(obj: dict[str, Any]) -> bool:
    """A row is a real, reusable result when generation succeeded (Issue==0) AND a
    genuine prompt reached the model (Prompt Tokens > 1). Prompt Tokens <= 1 is the
    fingerprint of a truncation bug that discarded the whole prompt; no real
    scenario prompt (system prompt + student turn) is ever that short. A row
    missing the field (legacy/minimal) is assumed complete."""
    if obj.get("Issue", 0) == 1:
        return False
    pt = obj.get("Prompt Tokens")
    return pt is None or (isinstance(pt, int) and pt > 1)
```

The validity test was written for an *older* truncation bug that collapsed prompts to a
single token, so it only rejects `Prompt Tokens <= 1`. The current corrupted rows have
`Prompt Tokens` of 768 or 7936 and sail through as valid. `scan_shard` therefore adds them to
`done`, `generate_frq` filters them out of `todo`, and a post-fix re-run with resume enabled
**skips every single affected row.** On top of that, `is_pair_done` in `common.py` (line 167)
short-circuits the whole (benchmark, model) pair before generation is even considered if a
`.done` marker exists in `Outputs/_manifests/`, which it does for all 36 pairs. See §7 for
the operational consequence.

---

## 6. Prioritized fixes

### P0 — Truncate the middle, not the head

Change `_fit_prompt_and_budget` (`AdaptiveTesting/Test/Inference/frq_generate.py`, lines
76-102) to fit at the **message level, before templating**, rather than on the flat token
array afterwards. The policy that preserves construct validity is:

1. Always keep the system turn and the final user turn verbatim.
2. Drop or elide interior `conversation_context` turns oldest-first until the rendered prompt
   fits, inserting a visible marker such as `[…earlier conversation omitted…]` so the model
   and any human auditor can see that history was removed.
3. If the system turn plus the final user turn still do not fit, truncate the **interior** of
   the final user turn — keep its opening span and its closing span — so both the instruction
   and the question framing survive. This is the case that matters for TutorEval, where a
   single reference passage is what overflows.

This requires passing the message list into the fitter, so `_render_prompt` (`frq_generate.py`,
lines 65-73) and the render-and-fit loop in `generate_frq` (lines 172-181) change alongside it:
the fitter needs to re-render candidate message lists through `engine.render_chat` to measure
them, instead of receiving a finished string. Keep `MIN_GEN` and the `gen_budget` arithmetic
exactly as they are — that part is correct.

Mirror the same change in `tutor_cat/respgen/runner.py` (`_fit_prompt_and_budget`, the
`ids[-prompt_cap:]` line) **only if** that dormant path might be revived. It did not produce
this data.

### P1 — Fail loud instead of mangling

Add a hard floor to the fitter: if the system turn plus the final user turn cannot fit within
`max_model_len - MIN_GEN` even after P0's interior elision, do not generate at all. Write an
`R.error_record(...)` (`AdaptiveTesting/Test/Inference/frq_records.py`, line 82) with
`Issue = 1`, `Finish Reason = "error"`, and an `Issue Description` of the form
`prompt_too_long: needs N tokens, window W`, and store the **full untruncated**
`Rendered Prompt` for audit rather than the fragment. The judge and IRT then see the cell as
missing, which is the correct semantics.

For the softer middle tier — prompts that fit only after interior elision — keep
`Truncated = 1` and add an explicit field such as `Truncation Strategy` with values like
`none` / `history_elided` / `body_elided`, so downstream stages can weight or exclude those
rows deliberately instead of guessing.

### P2 — Make the corrupted rows regenerable

Extend `_is_valid_row` (`frq_shard.py`, lines 50-59) to reject rows produced by the broken
fitter. The simplest correct predicate is to treat `Truncated == 1` as invalid, which forces
regeneration of exactly the affected rows and nothing else. If you would rather not
invalidate legitimately-elided rows after P0 ships, gate on a schema or pipeline version
stamp written into each record instead. Either way this must land before the re-run, or pair
it with the manual procedure in §7.

### P3 — Fix chat-template derivation (independent bug, see §8)

`_derive_chat` (`AdaptiveTesting/Test/Inference/models_registry.py`, lines 77-79) matches the
lowercased model id against the `CHAT_MARKERS` substring tuple (lines 26-32). Invert the
default: attempt `apply_chat_template` whenever the tokenizer actually ships a
`chat_template`, and use the id heuristic only to *suppress* it for known base checkpoints.
`Engine.render_chat` (`engine.py`, lines 170-194) already degrades safely to `applied=False`
when no template exists, so the permissive direction is the safe one. Alternatively, set
`apply_chat_template` explicitly per entry in `models_200.yaml`, whose `defaults` block
currently leaves it `null`.

### P4 — Close the budget-arithmetic and instrumentation gaps

These are all small, and all in `engine.py`:

- **Restore input truncation on the HF generate path.** `_generate_hf` (lines 310-331) calls
  `self._tok(p, return_tensors="pt")` at line 319 with no `truncation` and no `max_length`,
  so an over-length prompt reaches the model unguarded. This also means the
  `self._tok.truncation_side = "left"` set in `_init_hf` (line 122) is dead code — the
  comment above it describes a protection that does not exist. The ancestor implementation
  did pass `truncation=True, max_length=input_cap` (`tutor_cat/respgen/backends.py`, in
  `HFBackend.generate`); that guard was lost in the port. Restore it with
  `max_length = max_model_len - mt`, and set `truncation_side` to match whatever direction
  P0 settles on.
- **Reconcile `add_special_tokens`.** The fitter counts with `add_special_tokens=False`
  (`frq_generate.py`, line 94) while `_generate_hf` re-encodes with the library default of
  `True`. On a BOS-adding tokenizer that makes `prompt_tokens + gen_budget` one token larger
  than `max_model_len`. Either pass `add_special_tokens=False` at generation time or reserve
  one token in `prompt_cap`.
- **Use the backend's own finish reason.** `frq_generate.py` line 239 computes
  `"length" if n_out >= budgets[i] else "stop"` from a re-tokenized output count, which
  misreports whenever decode-then-encode is not a round trip. vLLM already supplies the real
  reason, but `_generate_vllm` discards it — it returns only `o.outputs[0].text` at
  `engine.py` line 308. Plumb it through.

### P5 — Optionally raise the cap

`max_model_len_cap` defaults to 32768 in `ModelSpec` (`models_registry.py`, line 58), which
clamps `DeepSeek-R1-Distill-Qwen-1.5B` and `Hermes-3-Llama-3.2-3B` down from their declared
131072. This costs nothing today, since the longest prompt observed anywhere in the smoke
test is 12,840 tokens, but raising it where VRAM allows removes a class of future truncation
for free. Lower priority than everything above.

---

## 7. Re-run recipe for the already-affected rows

Fixing the code does not fix the data, and the resume machinery will actively prevent a
naive re-run from touching it. Do all of the following.

**Step 1 — decide the scope.** The clean way is to regenerate every (benchmark, model) pair
that has any row with `Truncated = 1`. For the current smoke-test output that is seven pairs:
`wildbench`, `tutoreval`, `tutorbench` and `biggen` for
`malteos__gpt2-xl-wechsel-german`, and `wildbench`, `tutoreval` and `tutorbench` for
`naver-hyperclovax__HyperCLOVAX-SEED-Text-Instruct-0.5B`. Bridge and InFoBench need nothing.
For a full 200-model sweep, regenerate any pair whose model window is below 16384, since no
benchmark truncates at or above that.

**Step 2 — invalidate the bad rows.** Either land P2 (make `_is_valid_row` in `frq_shard.py`
reject `Truncated == 1`), which lets a normal resumed run repair exactly the affected rows
and leave everything else untouched, **or** run the affected pairs with `--no-resume`, which
calls `rewrite_shard(out_path, [])` in `generate_frq` and regenerates the whole pair from
scratch. P2 is strongly preferred: it is cheaper, it is idempotent, and it does not throw
away correct rows.

**Step 3 — delete the `.done` markers.** This is the step that is easy to forget and silently
makes the whole re-run a no-op. `is_pair_done` (`AdaptiveTesting/Test/Inference/common.py`,
line 167) returns true whenever
`AdaptiveTesting/Outputs/_manifests/<benchmark>__<slug>.done` exists, and the runner skips
the pair before `generate_frq` is ever called. Delete the marker for each pair you intend to
regenerate — for example `_manifests/tutoreval__malteos__gpt2-xl-wechsel-german.done` — or the
new code will never run.

**Step 4 — validate.** After the re-run, over the regenerated shards confirm that:

1. Row counts per shard are unchanged (WildBench 1001, TutorEval 828, BiGGen 695,
   TutorBench 662, Bridge 642, InFoBench 500) with no duplicate `Scenario` ids.
2. No row has a `Rendered Prompt` that fails to contain its benchmark's system-prompt text,
   for the benchmarks that have a fixed one. On WildBench the string to grep for is
   `You are a helpful assistant. Respond to the user's request`; on TutorEval,
   `You are an expert science tutor helping a student.`; on Bridge,
   `You are an AI math tutor. The student has just made a mistake`. Before the fix this check
   fails on 36.9% / 55.6% / 0.0% of gpt2-xl's rows respectively; after it should be 0.0%
   everywhere.
3. No `Rendered Prompt` begins mid-word. A cheap proxy that catches the current failures: the
   prompt must start with the model's own template preamble or with the literal `System: `.
4. Any row that is still marked `Truncated = 1` carries an explicit `Truncation Strategy`
   (P1), and any prompt that genuinely cannot fit is `Issue = 1` with a `prompt_too_long`
   description rather than a scored row.
5. Rows that were already clean are byte-identical to before, confirming the fix did not
   perturb the 34 unaffected pairs.

---

## 8. Separate, unrelated bug — the chat-template fallback

This came up during the same investigation and must not be conflated with the truncation
bug. **It is independent of truncation and uncorrelated with it.**

`OpenBMB/MiniCPM4-0.5B` has `Chat Template Applied = 0` on **all six** benchmarks — all
4,328 of its rows — so every prompt it received used the flat `Student:` / `Tutor:` base
rendering from `render_base_prompt` rather than its own chat template. Its
`tokenizer_config.json` on the Hub **does** contain a `chat_template`; I verified this
directly. The cause is purely `_derive_chat` (`models_registry.py`, lines 77-79): none of the
`CHAT_MARKERS` substrings (lines 26-32) appear in `openbmb/minicpm4-0.5b`, so
`apply_chat_template` resolves to `False`, and `models_200.yaml` leaves the field `null` so
nothing overrides it. `Engine.render_chat` then returns `applied=False` and the caller falls
back to base rendering.

Why the two bugs are orthogonal, stated numerically so nobody has to re-derive it:

- MiniCPM4-0.5B has `Chat Template Applied = 0` on all six benchmarks and a truncation rate
  of **0.0%** on all six.
- HyperCLOVAX has `Chat Template Applied = 1` on all six benchmarks and is truncated on
  three of them.
- `malteos/gpt2-xl-wechsel-german` also shows `Chat Template Applied = 0` everywhere, but for
  a legitimate reason: its `tokenizer_config.json` genuinely ships no `chat_template`, so the
  base rendering is correct for it. Only MiniCPM4 is misrouted.

The fix is P3 above. The follow-up work is to audit the whole 187-model roster for other
instruct-tuned checkpoints being silently rendered as base models; substring matching on
model ids will not scale, and MiniCPM4 is unlikely to be the only miss.

---

## 9. Slack version (copy-paste)

```
Found the cause of the cut-off prompts in the full200 FRQ run. Fixes belong in the sibling
repo olmo-eval-full/AdaptiveTesting/Test/Inference/, not in eduLLM-Evals/tutor_cat/respgen/,
which is a dormant copy that did not produce this data.

Root cause: in frq_generate.py, _fit_prompt_and_budget does `ids = ids[-prompt_cap:]`
(line 97). When a prompt overflows the window it keeps the tail and discards the head.
Prompts are built system-turn-first by frq_prompts.build_chat_messages, so what gets
deleted is the task instruction. Both flagged examples are the same WildBench item,
WB_0000, and start mid-word because the cut lands inside a token.

This is not a context-window detection bug. Live config.json for all six smoke-test models
confirms every recorded Max Model Len is correct: HyperCLOVAX really is 8192, gpt2-xl
really is 1024. No model with adequate context was truncated, so leave
resolve_max_model_len alone.

Blast radius, projected over the 187-model roster from measured prompt lengths: roughly
29,939 of 662,184 rows (4.5%), 106 of the 153 resolvable models, 10 above 10%. TutorEval
is the epicenter at 83% of the damage, WildBench 12%, Bridge and InFoBench clean. In the
smoke test 1,255 of 25,968 rows are already corrupted and the worst TutorEval rows kept
4% of their prompt.

It also fails silently: Issue=0 and a normal finish reason, so the judge scores the answer
and IRT reads "can't do the task" rather than "no task given," which biases ability and
difficulty rather than merely adding noise. And resume won't repair it -
frq_shard._is_valid_row only rejects Prompt Tokens <= 1, so these 768- and 7936-token rows
count as done and a post-fix re-run skips them.

Top three fixes: truncate the middle, keeping the system turn and final user turn; write
Issue=1 prompt_too_long instead of generating when even those don't fit; and make
_is_valid_row reject Truncated==1, deleting the affected Outputs/_manifests/*.done markers
or is_pair_done skips the pair.

Full memo: eduLLM-Evals/plans+prds/BUG - Head-Truncated Prompts in full200 FRQ Run.md
```
