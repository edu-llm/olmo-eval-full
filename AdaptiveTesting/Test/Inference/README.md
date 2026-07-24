# AdaptiveTesting — Inference Job Plan

**Status:** Plan / design doc (no scripts written yet)
**Location of eventual code:** `AdaptiveTesting/Test/Inference/` (scripts) + reuse of the existing `AdaptiveTesting/Inputs/` tree for datasets, models, and outputs.
**Target runtime:** AWS GPU job (Batch / SageMaker / EC2 spot), one benchmark at a time, sweeping ~100 open-source models in the 0–7B range.

> This README is the build spec for a set of *temporary* inference scripts. It describes what to build, the directory layout, the exact output formats, the model + benchmark catalog, the batching/optimization strategy, and the AWS execution plan. Nothing here is quantization-based.

---

## 1. Goal

For every benchmark below, download it, then run inference for ~100 open-source models (0–7B params) and record per-question results.

Two families of benchmark, two scoring paths:

| Family | What we store | How it is graded |
| --- | --- | --- |
| **MCQ** | For each question: `correct` / `wrong` only | Deterministic (model prediction vs gold key) |
| **Open-ended** | The full model response text | **Prometheus** LLM-judge → `pass`/`fail` + `reasoning` |

**Iteration order (hard requirement) + efficiency reconciliation:** the fleet must make progress **benchmark-first** — all 100 models finish benchmark *k* before the fleet reports benchmark *k+1* as complete. This is enforced at the **fleet/aggregation level**, NOT by reloading each model per benchmark. Each worker loads a model **once** (resident) and runs all its assigned benchmarks; a completion barrier + the `_manifests/*.done` markers determine when a benchmark is "fully done across all models." This satisfies the "more models per benchmark before moving on" requirement while avoiding ~1,700 redundant model loads (see §7 and risk **A1** in §12).

**No quantization.** Optimize throughput with batching, vLLM continuous batching, **model co-location (multiple models per B200 GPU)**, prefix caching, dtype (bf16/fp16), and paged attention — never bit-width reduction. Tensor parallelism is unused for ≤7B (kept at `tp=1`); the 180 GB B200 is instead filled with **many concurrent models** (see §6.4 / risk **B1**).

---

## 2. Benchmark catalog

The user grouped these into "MCQ" and "Open-Ended." Several items originally listed under *Open-Ended* are natively multiple-choice / fixed-answer (SciQ, HellaSwag, PIQA, BoolQ, WinoGrande, MathQA). **Decision (confirmed):** classify by the *native answer format* — anything with a gold option/key is scored deterministically as MCQ (cheaper + exact); anything free-form goes through the Prometheus judge.

**Sample caps (bounded runs):** each benchmark has a configurable `max_samples` in `configs/inference.yaml` (default **2000**, seeded sampling). This bounds otherwise-huge splits (SQuAD 2.0 dev ~11.9k, HellaSwag val ~10k, EduBench large) so total time/cost is predictable. Set `max_samples: null` for a full-set run (see risk **A2**). The chosen split (train/val/test) per benchmark is fixed in the registry, preferring **labeled** splits (e.g., validation where test labels are hidden).

### 2.1 MCQ (deterministic scoring)

| Benchmark | Source | HF / download id | Answer format | Notes |
| --- | --- | --- | --- | --- |
| ARC-Easy | allenai/ai2_arc | `allenai/ai2_arc` (`ARC-Easy`) | 4-way MCQ | grade-school science |
| ARC-Challenge | allenai/ai2_arc | `allenai/ai2_arc` (`ARC-Challenge`) | 4-way MCQ | hard subset |
| OpenBookQA | allenai/openbookqa | `allenai/openbookqa` (`main`) | 4-way MCQ | open-book facts |
| Pedagogy benchmark | HF | `AI-for-Education/pedagogy-benchmark` | MCQ | teaching/pedagogy MCQ |
| SciQ | allenai/sciq | `allenai/sciq` | 4-way MCQ | keep support passage optional |
| HellaSwag | rowanzellers | `Rowan/hellaswag` | 4-way MCQ | commonsense NLI |
| PIQA | ybisk | `ybisk/piqa` | 2-way MCQ | physical commonsense |
| BoolQ | google-research-datasets | `google/boolq` | yes/no (2-way) | treat as 2-class MCQ |
| WinoGrande | allenai | `allenai/winogrande` (`winogrande_xl`) | 2-way MCQ | coref |
| EducationQ (MMLU-Pro-Stratified) | derived from MMLU-Pro | `TIGER-Lab/MMLU-Pro` + stratified sampler | up to 10-way MCQ | build stratified subset script |
| MathQA | allenai/math_qa | `allenai/math_qa` | 5-way MCQ | keep rationale for logging only |

### 2.2 Open-ended (Prometheus LLM-judge scoring)

| Benchmark | Source | HF / download id | Response type | Judge rubric focus |
| --- | --- | --- | --- | --- |
| TutorBench | HF | `tutorbench/tutorbench` | tutoring turn | pedagogical quality / correctness |
| TutorEval | princeton-nlp | `princeton-nlp/TutorEval` | free-form answer | factuality vs reference |
| EduBench | DirectionAI | `DirectionAI/EduBench` | free-form | task-specific rubric |
| SQuAD 2.0 | Kaggle mirror / HF | `rajpurkar/squad_v2` | span / "no answer" | answerability + EM/F1 assisted judge |
| SVAMP | arkilpatel/SVAMP (GitHub) | git repo JSON | numeric answer + reasoning | final-answer correctness (judge or exact) |
| MathTutorBench (MathDial) | HF | `eth-nlped/mathdial` | tutoring dialogue | tutoring move quality |

**Hybrid note:** SVAMP and SQuAD 2.0 have exact gold answers. We still store the full generation, but the judge prompt is given the gold answer so the `pass/fail` decision is grounded (and we can add a cheap exact-match/F1 column alongside the judge). This avoids the judge hallucinating correctness.

**Datasets needing credentials / manual handling:**
- **SQuAD 2.0** — load from the HF mirror `rajpurkar/squad_v2` (no Kaggle).
- **SVAMP** — plain GitHub repo, `git clone` or raw JSON fetch.
- **EEDI — dropped** (required Kaggle; excluded per decision).

### 2.3 Per-dataset format handling (loaders must normalize each of these)

Every dataset has a **different raw schema**; each `PreProcess/` loader converts it to one common record:

```json
// MCQ normalized record
{"question_id": "...", "prompt": "...", "options": ["...","..."], "gold_index": 0, "meta": {...}}
// Open-ended normalized record
{"question_id": "...", "prompt": "...", "reference": "...", "rubric_key": "...", "meta": {...}}
```

Raw formats and the specific quirks each loader MUST handle:

| Benchmark | Raw fields | Quirk the loader must handle |
| --- | --- | --- |
| ARC-E / ARC-C | `question`, `choices={text[],label[]}`, `answerKey` | **Variable # of choices (3–5)**; labels are sometimes `A/B/C/D`, sometimes `1/2/3/4` → map `answerKey` through `choices.label` to a `gold_index`, don't assume position. |
| OpenBookQA | `question_stem`, `choices={text[],label[]}`, `answerKey` | Prompt uses `question_stem`; same label-mapping as ARC. |
| Pedagogy | (confirm columns: question / options / correct) | Confirm exact column names + subject field; verify whether answer is a letter or the option text. |
| SciQ | `question`, `correct_answer`, `distractor1..3`, `support` | **Gold is text, not an index** → assemble `[correct_answer, distractor1..3]`, shuffle deterministically (seeded), record `gold_index`; `support` passage optional (off by default). |
| HellaSwag | `ctx_a`, `ctx_b`, `ctx`, `endings[4]`, `label` | Prompt = `ctx` (or `ctx_a`+`ctx_b`); `label` is a **string** index → int; **test split has empty labels** → use validation for scoring. |
| PIQA | `goal`, `sol1`, `sol2`, `label` | 2 options from `sol1/sol2`; `label` 0/1; test split unlabeled → score on validation. |
| BoolQ | `question`, `passage`, `answer(bool)` | 2-class yes/no; build options `["no","yes"]`, `gold_index = int(answer)`; include `passage` in prompt. |
| WinoGrande | `sentence` (has `_`), `option1`, `option2`, `answer("1"/"2")` | Fill the `_` blank with each option (cloze), `gold_index = int(answer)-1`; pick a config (`winogrande_xl`). |
| EducationQ (MMLU-Pro-Strat.) | `question`, `options[]` (up to 10), `answer`(letter), `answer_index`, `category` | **Up to 10 options**; build the **stratified subset by `category`** (separate sampler script, seeded); use `answer_index` directly. |
| MathQA | `Problem`, `options` (single string), `correct`(letter), `Rationale` | **`options` is ONE string** like `"a ) 10 , b ) 12 , ..."` → parse into a list by letter markers; map `correct` letter → index; keep `Rationale` for logs only. |
| TutorBench | (confirm columns: prompt/dialog + reference + rubric) | Confirm whether it ships its own rubric per item; may be multi-turn → serialize dialog into the prompt. |
| TutorEval | `question`, textbook/chapter context, `key_points` | Grading is **key-point based** → feed `key_points` into the judge rubric as the reference; include chapter context in prompt. |
| EduBench | multiple scenarios/task types | **Heterogeneous: several sub-tasks in one dataset** → loader must branch per scenario/task-type and pick the right rubric each. |
| SQuAD 2.0 | `context`, `question`, `answers={text[],answer_start[]}` | **Unanswerable questions have empty `answers.text`** → treat "no answer" as valid gold; judge/EM must credit a correct abstention. |
| SVAMP | `Body`, `Question`, `Equation`, `Answer`, `Type` | Prompt = `Body` + `Question`; gold is **numeric** (`Answer`) → normalize number formatting for final-answer match; `Equation` for logs. |
| MathTutorBench (MathDial) | `question`, `ground_truth`, `student_incorrect_solution`, `conversation` | **Multi-turn dialogue** → serialize `conversation` (Teacher/Student turns) into the prompt; reference = `ground_truth`; the student's wrong solution is context, not the answer. |

Loaders marked "confirm" (Pedagogy, TutorBench) need their exact column names verified against the live dataset card before implementation; the loader interface is fixed regardless.

---

## 3. Directory layout

Build on the existing `AdaptiveTesting/Inputs/` tree, and add scripts under `Test/Inference/`:

```
AdaptiveTesting/
├── Inputs/
│   ├── Models/
│   │   └── models.yaml                 # curated ~100-model manifest (see §5)
│   ├── MCQ/
│   │   ├── Benchmarks/                 # downloaded/normalized MCQ datasets (jsonl)
│   │   │   ├── arc_easy.jsonl
│   │   │   ├── arc_challenge.jsonl
│   │   │   └── ...
│   │   ├── PreProcess/                 # loaders that normalize each MCQ dataset
│   │   └── InferenceScripts/           # (existing) MCQ inference entrypoints
│   └── Open/
│       ├── Benchmarks/                 # downloaded/normalized open-ended datasets
│       ├── PreProcess/                 # loaders that normalize each open dataset
│       └── LLM-Judge/                  # Prometheus judge prompts + rubrics
└── Test/
    └── Inference/
        ├── README.md                   # this file
        ├── run_benchmark.py            # core driver (1 benchmark × N models)
        ├── engine.py                   # vLLM wrapper (batching, TP, prefix cache)
        ├── mcq_scoring.py              # loglikelihood / letter-extraction scoring
        ├── open_generate.py            # free-form generation + storage
        ├── judge_prometheus.py         # Prometheus judge client + rubric runner
        ├── datasets_registry.py        # benchmark id -> loader + type + rubric
        ├── models_registry.py          # loads models.yaml, sizing hints
        ├── orchestrate.sh              # loops benchmarks (outer) × models (inner)
        ├── aws/
        │   ├── Dockerfile              # CUDA + vLLM + prometheus-eval image
        │   ├── batch_job_def.json      # AWS Batch job definition template
        │   ├── submit_batch.py         # array-job submitter (1 model = 1 task, or shard)
        │   └── entrypoint.sh           # in-container: pull data/model, run, push to S3
        ├── aggregate.py                # roll per-question files -> _summary/summary.csv
        └── configs/
            ├── inference.yaml          # batch sizes, max_len, max_samples/bench, gen params, co-location packs, fsync N/T
            └── judge.yaml              # Prometheus model, rubric mapping, pass threshold (raw 1-5 kept)
```

**Outputs** (written locally, then synced to S3):

```
Outputs/
├── mcq/
│   └── <benchmark>/
│       └── <model_slug>.csv            # one file per (benchmark,model); appended per question, fsync'd
├── open/
│   └── <benchmark>/
│       ├── <model_slug>.responses.jsonl   # raw generations, appended per question
│       └── <model_slug>.judged.csv        # per question: pass/fail + reasoning, appended per question
└── _manifests/
    └── <benchmark>__<model_slug>.done     # written only after the LAST question is persisted
```

`<model_slug>` = HF id with `/`→`__`. Every (benchmark, model) pair owns exactly one result file, written incrementally (see §4.4).

---

## 4. Output schemas (exact)

### 4.1 MCQ — `Outputs/mcq/<benchmark>/<model_slug>.csv`

```
question_id,model,benchmark,predicted,gold,result
arc_e_0001,Qwen2.5-1.5B,arc_easy,C,C,correct
arc_e_0002,Qwen2.5-1.5B,arc_easy,A,D,wrong
```

- `result` ∈ {`correct`,`wrong`}. This is the only column the user strictly requires; the rest are for traceability.
- One file per (benchmark, model). Every question listed, right or wrong.

### 4.2 Open-ended — raw responses `Outputs/open/<benchmark>/<model_slug>.responses.jsonl`

```json
{"question_id": "tutoreval_0007", "model": "Mistral-7B-v0.3", "benchmark": "tutoreval", "prompt": "...", "response": "...full model text..."}
```

### 4.3 Open-ended — judged `Outputs/open/<benchmark>/<model_slug>.judged.csv`

```
question_id,model,benchmark,result,reasoning
tutoreval_0007,Mistral-7B-v0.3,tutoreval,pass,"Answer is factually correct and pedagogically sound because..."
tutoreval_0008,Mistral-7B-v0.3,tutoreval,fail,"Missed the key misconception; explanation is incorrect because..."
```

- `result` ∈ {`pass`,`fail`} (one column).
- `reasoning` = Prometheus feedback (one column).
- Optional extra columns for hybrid benchmarks: `exact_match`, `f1` (SQuAD), `final_answer_match` (SVAMP).

### 4.4 One file per (benchmark, model) — durable, incremental writes

**Hard requirements:**

1. **Exactly one result file per (benchmark, model) pair.** Naming is deterministic: `Outputs/mcq/<benchmark>/<model_slug>.csv` (MCQ) and `Outputs/open/<benchmark>/<model_slug>.{responses.jsonl,judged.csv}` (open-ended). `<model_slug>` is the HF id with `/`→`__` (e.g., `Qwen__Qwen2.5-1.5B`) so the filename is globally unique across families. No shared/combined result files — a single benchmark×model's answers never mix with another's.
2. **Results are saved constantly (streaming), not at the end.** Each writer appends **one row per question the moment it is scored/generated** and `flush()`es it to the OS immediately (`line_buffering=True`). To avoid throttling high-QPS small models (tens of thousands of tiny rows), the expensive `os.fsync()` is **batched** — called every `N` rows (default 50) or every `T` seconds (default 5), whichever comes first, and always once at close (see risk **B2**). Net effect: at most a few seconds / N rows of writes are at risk on a hard crash, never a whole benchmark. Never buffer a whole benchmark in memory and write once.
3. **Header written once, atomically.** On first open, write the CSV header; subsequent appends are append-only (`mode="a"`). JSONL is naturally append-only (one JSON object per line).
4. **Partial-file resume.** On startup for a (benchmark, model) pair, if its file already exists, read the `question_id`s already present and **skip** them; resume from the first missing question. This makes every pair independently restartable without recomputation. The `.done` marker is written only after the *last* question is persisted.
5. **S3 sync cadence.** Locally the file is the source of truth (fsync'd per row). Periodically (every N rows or T seconds) sync the in-progress file to S3 so partial progress survives instance loss; final sync + `.done` marker on completion.
6. **Judge stage is decoupled & also incremental.** `responses.jsonl` is fully persisted per question *before* judging. The judge reads `responses.jsonl`, appends to `judged.csv` per question with the same flush/resume rules, so a judge crash never loses generations and resumes mid-file.

### 4.5 Summary metrics (derived, non-authoritative)

A cheap post-pass (`aggregate.py`) rolls the per-question files up into `Outputs/_summary/summary.csv` with one row per (benchmark, model): `n`, `n_correct`/`n_pass`, `accuracy`/`pass_rate`, `scoring_method`, `run_timestamp`. This is purely derived from the per-question files (the source of truth) and can be regenerated at any time. It does not replace the per-question outputs the user requested (see risk **A8**).

---

## 5. Model catalog (curated ~100, 0–7B)

Stored as `AdaptiveTesting/Inputs/Models/models.yaml`. Each entry: HF id, param size, family, dtype, and vLLM hints (TP degree, `trust_remote_code`, max model len). Below is the initial curated list (~100 checkpoints ≤ ~7B). Prune/adjust before launch.

> Sizing rule of thumb for a single 80GB GPU (A100/H100), bf16, no quant: ≤7B fits with room for KV cache; ≤3B allows large batch / high concurrency. Models >7B are intentionally excluded per the 0–7B constraint (e.g., Llama-3.1-8B, Gemma-2-9B are dropped).

```yaml
# AdaptiveTesting/Inputs/Models/models.yaml  (initial curated manifest)
defaults:
  dtype: bfloat16
  trust_remote_code: true
  max_model_len: 4096
  tp: 1                       # tensor parallel; 1 for all <=7B (fill the GPU via co-location instead)
  apply_chat_template: false  # base models -> raw prompt; instruct entries override to true (A4)
  scoring_method: loglikelihood  # SAME method for ALL models for comparability (A3)

models:
  # --- Qwen2.5 family ---
  - {id: Qwen/Qwen2.5-0.5B,             params_b: 0.5}
  - {id: Qwen/Qwen2.5-0.5B-Instruct,    params_b: 0.5}
  - {id: Qwen/Qwen2.5-1.5B,             params_b: 1.5}
  - {id: Qwen/Qwen2.5-1.5B-Instruct,    params_b: 1.5}
  - {id: Qwen/Qwen2.5-3B,               params_b: 3}
  - {id: Qwen/Qwen2.5-3B-Instruct,      params_b: 3}
  - {id: Qwen/Qwen2.5-7B,               params_b: 7}
  - {id: Qwen/Qwen2.5-7B-Instruct,      params_b: 7}
  - {id: Qwen/Qwen2.5-Math-7B,          params_b: 7}
  # --- Qwen3 family (<8B) ---
  - {id: Qwen/Qwen3-0.6B,               params_b: 0.6}
  - {id: Qwen/Qwen3-1.7B,               params_b: 1.7}
  - {id: Qwen/Qwen3-4B,                 params_b: 4}
  # --- Qwen2 family ---
  - {id: Qwen/Qwen2-0.5B,               params_b: 0.5}
  - {id: Qwen/Qwen2-1.5B,               params_b: 1.5}
  - {id: Qwen/Qwen2-7B,                 params_b: 7}
  # --- Llama 3.2 (small) ---
  - {id: meta-llama/Llama-3.2-1B,           params_b: 1}
  - {id: meta-llama/Llama-3.2-1B-Instruct,  params_b: 1}
  - {id: meta-llama/Llama-3.2-3B,           params_b: 3}
  - {id: meta-llama/Llama-3.2-3B-Instruct,  params_b: 3}
  # --- Gemma 2 / 3 (<8B) ---
  - {id: google/gemma-2-2b,             params_b: 2}
  - {id: google/gemma-2-2b-it,          params_b: 2}
  - {id: google/gemma-3-1b-it,          params_b: 1}
  - {id: google/gemma-3-4b-it,          params_b: 4}
  # --- Phi ---
  - {id: microsoft/phi-2,               params_b: 2.7}
  - {id: microsoft/Phi-3-mini-4k-instruct,   params_b: 3.8}
  - {id: microsoft/Phi-3.5-mini-instruct,    params_b: 3.8}
  # --- Mistral 7B ---
  - {id: mistralai/Mistral-7B-v0.1,          params_b: 7}
  - {id: mistralai/Mistral-7B-v0.3,          params_b: 7}
  - {id: mistralai/Mistral-7B-Instruct-v0.3, params_b: 7}
  # --- OLMo ---
  - {id: allenai/OLMo-2-1124-7B,        params_b: 7}
  - {id: allenai/OLMo-2-0425-1B,        params_b: 1}
  # --- SmolLM2 ---
  - {id: HuggingFaceTB/SmolLM2-135M,           params_b: 0.135}
  - {id: HuggingFaceTB/SmolLM2-360M,           params_b: 0.36}
  - {id: HuggingFaceTB/SmolLM2-1.7B,           params_b: 1.7}
  - {id: HuggingFaceTB/SmolLM2-1.7B-Instruct,  params_b: 1.7}
  # --- TinyLlama ---
  - {id: TinyLlama/TinyLlama-1.1B-Chat-v1.0,   params_b: 1.1}
  # --- Pythia suite ---
  - {id: EleutherAI/pythia-70m,   params_b: 0.07}
  - {id: EleutherAI/pythia-160m,  params_b: 0.16}
  - {id: EleutherAI/pythia-410m,  params_b: 0.41}
  - {id: EleutherAI/pythia-1b,    params_b: 1}
  - {id: EleutherAI/pythia-1.4b,  params_b: 1.4}
  - {id: EleutherAI/pythia-2.8b,  params_b: 2.8}
  - {id: EleutherAI/pythia-6.9b,  params_b: 6.9}
  # --- StableLM ---
  - {id: stabilityai/stablelm-2-1_6b,          params_b: 1.6}
  - {id: stabilityai/stablelm-2-zephyr-1_6b,   params_b: 1.6}
  - {id: stabilityai/stablelm-zephyr-3b,       params_b: 3}
  # --- Falcon ---
  - {id: tiiuae/falcon-7b,             params_b: 7}
  - {id: tiiuae/falcon-7b-instruct,    params_b: 7}
  - {id: tiiuae/Falcon3-1B-Base,       params_b: 1}
  - {id: tiiuae/Falcon3-3B-Base,       params_b: 3}
  - {id: tiiuae/Falcon3-7B-Base,       params_b: 7}
  # --- MPT ---
  - {id: mosaicml/mpt-7b,              params_b: 7}
  - {id: mosaicml/mpt-7b-instruct,     params_b: 7}
  # --- GPT-Neo / GPT-J-ish small ---
  - {id: EleutherAI/gpt-neo-1.3B,     params_b: 1.3}
  - {id: EleutherAI/gpt-neo-2.7B,     params_b: 2.7}
  # --- OPT ---
  - {id: facebook/opt-1.3b,           params_b: 1.3}
  - {id: facebook/opt-2.7b,           params_b: 2.7}
  - {id: facebook/opt-6.7b,           params_b: 6.7}
  # --- BLOOM(z) ---
  - {id: bigscience/bloom-560m,       params_b: 0.56}
  - {id: bigscience/bloom-1b1,        params_b: 1.1}
  - {id: bigscience/bloom-1b7,        params_b: 1.7}
  - {id: bigscience/bloom-3b,         params_b: 3}
  - {id: bigscience/bloom-7b1,        params_b: 7.1}
  - {id: bigscience/bloomz-1b7,       params_b: 1.7}
  - {id: bigscience/bloomz-3b,        params_b: 3}
  - {id: bigscience/bloomz-7b1,       params_b: 7.1}
  # --- IBM Granite (<8B) ---
  - {id: ibm-granite/granite-3.1-2b-base,        params_b: 2}
  - {id: ibm-granite/granite-3.1-2b-instruct,    params_b: 2}
  - {id: ibm-granite/granite-3.0-2b-base,        params_b: 2}
  # --- H2O Danube ---
  - {id: h2oai/h2o-danube3-4b-base,       params_b: 4}
  - {id: h2oai/h2o-danube2-1.8b-base,     params_b: 1.8}
  # --- Yi ---
  - {id: 01-ai/Yi-6B,                 params_b: 6}
  - {id: 01-ai/Yi-6B-Chat,            params_b: 6}
  # --- InternLM ---
  - {id: internlm/internlm2-1_8b,     params_b: 1.8}
  - {id: internlm/internlm2-7b,       params_b: 7}
  - {id: internlm/internlm2_5-7b,     params_b: 7}
  # --- MiniCPM ---
  - {id: openbmb/MiniCPM-2B-sft-bf16, params_b: 2.7}
  # --- RedPajama INCITE ---
  - {id: togethercomputer/RedPajama-INCITE-Base-3B-v1,  params_b: 3}
  - {id: togethercomputer/RedPajama-INCITE-7B-Base,     params_b: 7}
  # --- Cerebras-GPT ---
  - {id: cerebras/Cerebras-GPT-1.3B,  params_b: 1.3}
  - {id: cerebras/Cerebras-GPT-2.7B,  params_b: 2.7}
  - {id: cerebras/Cerebras-GPT-6.7B,  params_b: 6.7}
  # --- DeepSeek (<8B) ---
  - {id: deepseek-ai/deepseek-llm-7b-base,     params_b: 7}
  - {id: deepseek-ai/deepseek-math-7b-base,    params_b: 7}
  - {id: deepseek-ai/deepseek-coder-6.7b-base, params_b: 6.7}
  # --- Zephyr / Vicuna (Mistral/Llama derived, 7B) ---
  - {id: HuggingFaceH4/zephyr-7b-beta,   params_b: 7}
  - {id: lmsys/vicuna-7b-v1.5,           params_b: 7}
  # --- Nemotron mini ---
  - {id: nvidia/Nemotron-Mini-4B-Instruct, params_b: 4}
  # --- MobileLLM / OpenELM small ---
  - {id: apple/OpenELM-1_1B,          params_b: 1.1}
  - {id: apple/OpenELM-3B,            params_b: 3}
  # --- Amber / OpenLM ---
  - {id: LLM360/Amber,                params_b: 7}
  # --- Mamba / RWKV alt-arch (optional, verify vLLM support) ---
  - {id: state-spaces/mamba-2.8b-hf,  params_b: 2.8}
  # --- GPT-2 baselines (very small sanity floor) ---
  - {id: openai-community/gpt2,        params_b: 0.124}
  - {id: openai-community/gpt2-medium, params_b: 0.355}
  - {id: openai-community/gpt2-large,  params_b: 0.774}
  - {id: openai-community/gpt2-xl,     params_b: 1.5}
  # --- Top-up to exactly 100 ---
  - {id: Qwen/Qwen2.5-Coder-7B,           params_b: 7}
  - {id: microsoft/phi-1_5,               params_b: 1.3}
  - {id: stabilityai/stablelm-3b-4e1t,    params_b: 3}
  - {id: HuggingFaceTB/SmolLM2-360M-Instruct, params_b: 0.36}
```

**That is exactly 100 concrete entries** (base + instruct variants both kept, per decision). `models_registry.py` should validate each id resolves on HF and warn on gated ids (Llama, Gemma, Mistral need HF token / license acceptance).

**Per-entry overrides the registry applies:**
- `apply_chat_template: true` for every `*-Instruct` / `*-it` / `*-chat` / `zephyr` / `vicuna` entry (base entries stay `false`). MCQ is still scored by log-likelihood under the correct format (**A3/A4**).
- `gated: true` for Llama, Gemma, Mistral (require accepted licenses + `HF_TOKEN`); the AWS entrypoint injects the token secret, and the registry **skips + logs** any gated id that 401s instead of failing the whole run.
- `backend: hf_fallback` for models vLLM may not support at the pinned version (e.g., `state-spaces/mamba-2.8b-hf`, `apple/OpenELM-*`, possibly `google/gemma-3-*`). A startup **capability probe** test-loads all 100 and marks unsupported ones to skip or route to the HF-transformers fallback (**A5**).

---

## 6. Inference engine design

`engine.py` wraps **vLLM** (the repo already declares a `vllm` extra in `pyproject.toml`). Key choices:

- **Model loaded once per worker (resident)** and reused across *all* of that worker's benchmarks — never reloaded per benchmark (**A1**).
- **Batching:** submit all prompts for a (model, benchmark) at once and let continuous batching schedule them. Cap with `max_num_seqs` and `max_num_batched_tokens` from `configs/inference.yaml`.
- **Prefix caching** (`enable_prefix_caching=true`) — big win for MCQ where the shared question stem prefixes each option continuation.
- **Tensor parallelism** stays `tp=1` for all ≤7B models; instead of one big model per GPU we **co-locate many models per GPU** (see §6.4).
- **dtype:** bf16 default, fp16 fallback for older archs. **No quantization.**
- **Determinism:** `temperature=0` / greedy for MCQ and for reproducible open-ended runs (configurable per benchmark).
- **Backend fallback:** models unsupported by the pinned vLLM version route to an HF-`transformers` generation/scoring path or are skipped-with-log (**A5**); a boot-time capability probe test-loads all 100.
- **Prompt formatting:** per-model `apply_chat_template` decides raw vs chat-templated prompts; the same setting is used for both MCQ log-likelihood scoring and open-ended generation so results are internally consistent (**A4**).

### 6.1 MCQ scoring (`mcq_scoring.py`)

**One scoring method for all 100 models — log-likelihood — for cross-model comparability (A3):**

- **Log-likelihood ranking:** for each answer option, score `logP(option | question)` (length-normalized), pick argmax. Robust for both base and instruct models (instruct models just get the option scored under their chat template). This mirrors the existing `olmo_eval` MCQ tasks (see `src/olmo_eval/evals/tasks/constants/{arc,piqa,hellaswag,winogrande,...}.py`).
- **Letter generation** is available only as a *secondary, logged* signal for instruct models (extra column `predicted_letter`), never as the primary `result`, so accuracy is comparable across the whole sweep.

Output: append one row per question with `result = correct|wrong` (plus `scoring_method` for traceability).

### 6.2 Open-ended generation (`open_generate.py`)

- Greedy or low-temp generation with a per-benchmark `max_tokens` and stop sequences.
- Persist full `response` to `.responses.jsonl` **before** judging (so a judge failure never loses generations).

### 6.3 Prometheus judge (`judge_prometheus.py`)

- Use **Prometheus** — default judge model **`prometheus-eval/prometheus-7b-v2.0`** via the `prometheus-eval` library, self-hosted with vLLM and kept **resident on dedicated GPU(s)** so judging runs in parallel with downstream generation instead of as a serial phase (**B3**). Judge calls are batched across models per benchmark. (`prometheus-8x7b-v2.0` remains a config override.)
- Absolute-grading mode with a rubric per benchmark (`Inputs/Open/LLM-Judge/<benchmark>.rubric.txt`), each on a **1–5 scale** with required written feedback.
- Store the **raw 1–5 `score`** in addition to mapping it to `pass`/`fail` via a configurable threshold in `configs/judge.yaml` (default: score ≥ 4 = pass). Keeping the raw score lets thresholds be re-tuned without re-running the judge (**A6**).
- **Calibration check:** on a small sampled slice, spot-check Prometheus verdicts against a stronger judge; record agreement in the run log. Rubrics for reference-less benchmarks (e.g., TutorBench) are authored to lean on explicit criteria, not gold answers.
- Store: `result` (pass/fail), `reasoning` (feedback), `score` (1–5). For hybrid benchmarks pass the gold answer into the rubric.
- **The repo already depends on `prometheus_client`** (that's the metrics client, unrelated). The judge needs the *`prometheus-eval`* package + the Prometheus model weights — add to the AWS image, not necessarily to the main `pyproject.toml`.

### 6.4 GPU co-location (make P6/B200 actually busy) — **B1**

A single ≤7B model uses ~14 GB of the B200's **180 GB** and is memory-bandwidth-bound during decode → one-model-per-GPU wastes the card (effective MFU ~10–30%, single-digit for sub-2B). Instead:

- **Pack multiple models per GPU.** Run several independent vLLM engine processes on one B200 (or use CUDA MPS), each pinned via `CUDA_VISIBLE_DEVICES` + a `gpu_memory_utilization` fraction, so ~4–8 small/mid models (or ~6–8×7B) share a card concurrently.
- **Size the pack by model memory** from `models.yaml` (`params_b` → weights + KV budget) with a bin-packing scheduler in `run_benchmark.py`; leave headroom for KV cache and prefix cache.
- **Target aggregate utilization ~70–90%**, cutting wall-clock and cost ~3–5× vs one-model-per-GPU.
- Judge GPU(s) are excluded from the inference pack and run their own resident Prometheus engine(s).

---

## 7. Orchestration & iteration order

**Requirement restated:** the *fleet* must finish a benchmark across all 100 models before advancing — but each **worker keeps its model resident** and does not reload per benchmark (**A1**). We get benchmark-first progress by making **benchmark the outer coordination phase** while models are distributed across workers, each worker holding its model(s) in memory for the whole phase.

Per-worker loop (a worker owns a shard of models, co-located per GPU per §6.4):

```
# datasets pre-normalized once and cached on NVMe/S3 (not reloaded per model)
load_resident(models_shard)                      # each model loaded ONCE, packed on GPU(s)
for benchmark in BENCHMARKS:                      # outer PHASE (fleet-synchronized)
    dataset = load_cached(benchmark)[:max_samples]
    for model in models_shard:                    # inner: iterate resident models (no reload)
        outfile = path(benchmark, model)
        if done_marker(benchmark, model): continue            # pair-level resume
        done_ids = read_existing_question_ids(outfile)        # question-level resume
        open outfile APPEND (header if new)
        for q in dataset if q.id not in done_ids:
            if benchmark.type == MCQ:
                row = loglik_score(q, model)                  # correct/wrong (single method)
            else:
                resp = generate(q, model); persist(resp)      # responses.jsonl, flush
                row = judge(resp, q)                          # Prometheus (resident, batched)
            append(outfile, row); flush()                     # fsync batched every N/T (§4.4)
        write done_marker(benchmark, model)
    barrier()                                     # wait until ALL workers finish this benchmark
    mark_benchmark_complete(benchmark)            # fleet-level "benchmark done" signal
```

- **Iteration order is preserved at the fleet level:** the `barrier()` ensures benchmark *k* is complete across all models before *k+1* is reported done, satisfying "more models per benchmark before moving on" — without per-benchmark reloads.
- **Checkpointing (two levels):**
  - *Pair-level:* `_manifests/<benchmark>__<model>.done` lets a restart skip fully-finished pairs.
  - *Question-level:* for an unfinished pair, the writer reads the `question_id`s already in the (single) result file and resumes from the first missing one — so a spot/Capacity-Block interruption loses at most the in-flight question, never completed rows (see §4.4).
- **Judge:** Prometheus stays **resident on dedicated GPU(s)** and is called in batches per (benchmark, model) so judging overlaps generation rather than blocking it (**B3**).
- **Sharding for scale:** `aws/submit_batch.py` shards the 100 models across workers/GPUs (bin-packed by size, §6.4). Every worker walks benchmarks in the same order and hits the shared `barrier()`, so the fleet advances benchmark-by-benchmark while each model is loaded exactly once.

---

## 8. AWS execution plan

**Recommended:** AWS Batch (managed GPU compute environment) + S3 for data/artifacts + ECR for the image. SageMaker Training Jobs are an acceptable alternative.

1. **Image (`aws/Dockerfile`):** CUDA base + `uv` + project deps + `vllm` extra + `prometheus-eval`. Bake nothing model-specific; pull weights at runtime.
2. **Storage:**
   - Input datasets → normalize locally, upload to `s3://<bucket>/adaptivetesting/inputs/...` (or download-on-start via HF).
   - Model weights → stream from HF at task start (cache to a shared FSx/EBS if repeated). No weights in the image.
   - Outputs → sync `Outputs/` to `s3://<bucket>/adaptivetesting/outputs/...` after each (benchmark, model) pair.
3. **Compute (target: `p6-b200.48xlarge`, 8× B200 180 GB):**
   - **Provisioning:** P6 is delivered through **EC2 Capacity Blocks for ML / reservations**, *not* ordinary Spot. Reserve the block and launch into it; if using AWS Batch, attach a compute environment tied to the Capacity Block (managed on-demand scaling into P6 is not guaranteed) — or launch the array via plain EC2 in the reservation (**C1**). Checkpoint-based resume (§4.4) still protects against any interruption/expiry.
   - **Co-location is mandatory here (§6.4 / B1):** pack several models per B200 so the 180 GB and compute are actually used; one-model-per-GPU leaves P6 ~15–25% utilized and is not cost-justified (~$110/instance-hr, **C2**). If P6 capacity is unavailable, `g6e`/`p5` are cheaper-per-token fallbacks for ≤7B.
   - Dedicate 1–2 of the 8 B200s to the **resident Prometheus judge**; the rest run the co-located inference packs.
4. **Secrets:** `HF_TOKEN` (gated models), `OPENAI_API_KEY` only if a fallback judge is wanted (Prometheus is primary). Inject via Batch job definition `secrets` from SSM/Secrets Manager. Note existing `.apienv` holds `OPENAI_API_KEY` locally — do **not** bake it into the image.
5. **Job definition (`aws/batch_job_def.json`):** vcpus/memory/GPU count, the ECR image, env, and the array size = number of model shards.
6. **Entrypoint (`aws/entrypoint.sh`):** read `AWS_BATCH_JOB_ARRAY_INDEX` → pick model shard → run `run_benchmark.py --benchmarks all --models <shard>` → push outputs → exit.

---

## 9. Dependencies

- Core: `vllm` (existing extra), `transformers`, `datasets`, `torch`, `huggingface_hub`.
- Judge: `prometheus-eval` + Prometheus weights.
- Download helpers: `gitpython`/`requests` (SVAMP); everything else via `datasets`/`huggingface_hub`.
- AWS: `boto3`, AWS CLI (in image).
- Keep these in the AWS image / an `inference` optional-extra; avoid disturbing the main `olmo-eval` dependency set unless we decide to upstream.

---

## 10. Resolved decisions

1. **Benchmark classification** — SciQ/HellaSwag/PIQA/BoolQ/WinoGrande/MathQA scored as **deterministic MCQ**. ✅
2. **Sources** — SQuAD 2.0 from HF `rajpurkar/squad_v2`; **EEDI dropped** (needed Kaggle). ✅
3. **Judge** — default **`prometheus-7b-v2.0`** (8x7b available as override). ✅
4. **Model variants** — keep **both base + instruct** (each counts toward 100). ✅
5. **Model count** — manifest is **exactly 100**. ✅

Still to confirm against live dataset cards during implementation (interface unaffected): exact column names for **Pedagogy** and **TutorBench**.

---

## 11. Build checklist (implementation order)

- [ ] `datasets_registry.py` + `Inputs/*/PreProcess/*` loaders → normalize every benchmark to a common schema (`question_id`, `prompt`, `options?`, `gold?`); fixed split + `max_samples` per benchmark (**A2**).
- [ ] `models_registry.py` + finalize `Inputs/Models/models.yaml` (validate ids, gated flags, `apply_chat_template`, `backend` fallback) (**A4/A5**).
- [ ] `engine.py` (vLLM wrapper, batching, prefix cache, no-quant, HF fallback backend, capability probe).
- [ ] **Co-location scheduler** in `run_benchmark.py` — bin-pack models per B200, per-process `gpu_memory_utilization` (**B1**).
- [ ] `results_writer.py` (shared) — append-per-question + flush, **batched fsync (N/T)**, header-once, existing-id resume, `.done` markers, one file per (benchmark, model) (**B2**).
- [ ] `mcq_scoring.py` (**log-likelihood as the single primary method**; letter-gen as logged secondary) → MCQ CSV via `results_writer` (**A3**).
- [ ] `open_generate.py` → responses.jsonl (append per question, flush).
- [ ] `judge_prometheus.py` (**resident, batched**) + rubrics under `Inputs/Open/LLM-Judge/` → judged.csv (pass/fail + reasoning + raw 1–5 score) (**A6/B3**).
- [ ] `run_benchmark.py` (**resident model, benchmark-phase loop + `barrier()`** for fleet-level ordering + checkpointing) (**A1**).
- [ ] `aggregate.py` → `_summary/summary.csv` (**A8**).
- [ ] `orchestrate.sh` (local dry-run of full sweep).
- [ ] `aws/` (Dockerfile, **Capacity Block launch path**, submitter, entrypoint) + S3 wiring (**C1**).
- [ ] Smoke test: capability-probe all 100 models; then 1 MCQ + 1 open-ended benchmark × 2 small models (e.g., `Qwen2.5-0.5B`, `pythia-410m`) end-to-end locally, then a 1-task run in the Capacity Block.

---

## 12a. Implemented code & usage

The plan above is now implemented. Modules (all under `Test/Inference/`):

| File | Role |
| --- | --- |
| `common.py` | unified `Question` record, path helpers, slugify, done markers |
| `config.py` | loads `configs/inference.yaml` + `configs/judge.yaml` |
| `models_registry.py` | loads `Inputs/Models/models.yaml`; derives `apply_chat_template`, `gated`, `backend`; sharding |
| `datasets_registry.py` | all benchmark loaders → normalized JSONL cache (+ offline `synth_*`) |
| `results_writer.py` | append-per-item CSV/JSONL, batched fsync, header-once, resume |
| `engine.py` | `vllm` / `hf` / `mock` backends; log-likelihood + generation; chat templates; capability routing |
| `mcq_scoring.py` | log-likelihood MCQ → `correct`/`wrong` CSV |
| `open_generate.py` | free-form generation → `responses.jsonl` |
| `judge_prometheus.py` | Prometheus absolute grading → `pass`/`fail` + reasoning + raw score |
| `run_benchmark.py` | driver: resident model, benchmark ordering, sharding, resume |
| `aggregate.py` | derived `_summary/summary.csv` |
| `orchestrate.sh` | local `smoke` / `mcq` / `open` / `all` driver |
| `aws/` | `Dockerfile`, `entrypoint.sh` (array-shard + S3 sync), `submit_batch.py`, `batch_job_def.json` |

**Install:** `uv pip install -r requirements.txt` (vLLM + Prometheus are Linux/CUDA-only and skipped on macOS).

**Local offline smoke test (no GPU, no network):**

```bash
./orchestrate.sh smoke      # mock backend + synthetic data, exercises the full pipeline
```

**Real run (GPU):**

```bash
# one worker = one model shard; benchmarks in fixed order (fleet advances benchmark-first)
python run_benchmark.py --benchmarks all --shard-index 0 --num-shards 100
python aggregate.py
```

Key flags: `--benchmarks all|mcq|open|<list>`, `--models <ids>`, `--shard-index/--num-shards`,
`--backend vllm|hf|mock`, `--max-samples N`, `--resident-all` (co-located benchmark-outer order),
`--no-judge`. Backends route automatically (`hf_fallback` models → HF). Everything resumes from the
per-(benchmark, model) files + `.done` markers.

---

## 12. Review, risks & compute estimate (P6 / B200)

**Target hardware:** `p6-b200.48xlarge` = 8× NVIDIA B200 (180 GB HBM3e each, 1,440 GB/instance), 192 vCPU, 2 TiB RAM, 30 TB NVMe, up to 3.2 Tbps EFA.

### 12.1 Key risks (all addressed in the plan)

| # | Sev | Issue | Resolution (where) |
| --- | --- | --- | --- |
| A1 | High | **Benchmark-outer/model-inner loop reloads each model per benchmark** (~100×17 ≈ 1,700 loads → tens of GPU-h wasted). | ✅ Model **resident** per worker; benchmark-first ordering via fleet `barrier()` (§1, §7). |
| A2 | High | **No per-benchmark sample cap** (SQuAD dev ~11.9k, HellaSwag ~10k, EduBench large) → unbounded time/cost. | ✅ `max_samples` (default 2k, seeded) per benchmark (§2, `configs/inference.yaml`). |
| A3 | Med | **Mixed MCQ scoring** (LL for base, letter-gen for instruct) → non-comparable across the 100 models. | ✅ **Log-likelihood for all**; letter-gen logged as secondary only (§6.1). |
| A4 | Med | **Chat templates** — instruct vs base formatting changes LL results. | ✅ Per-model `apply_chat_template` in `models.yaml` (§5, §6). |
| A5 | Med | **Alt-arch models may not load in vLLM** (Mamba, OpenELM, Gemma-3, some `trust_remote_code`). | ✅ Boot capability probe + HF fallback backend + skip-with-log (§5, §6). |
| A6 | Med | **Single 7B judge = noisy binary verdicts**; reference-less rubrics (TutorBench) hard. | ✅ Store raw 1–5 score, configurable threshold, calibration spot-check (§6.3). |
| A8 | Low | No summary metrics (only per-question rows). | ✅ Derived `aggregate.py` → `_summary/summary.csv` (§4.5). |
| B1 | **Crit** | **B200 oversized for ≤7B** → effective MFU ~10–30% (single-digit for sub-2B). Paying B200 for A10-class work. | ✅ **Co-location scheduler** packs many models per B200 → 70–90% util (§6.4, §8). |
| B2 | Med | **Per-question `fsync`** throttles high-QPS small models. | ✅ Per-row `flush()`, **batched `fsync` every N/T** (§4.4). |
| B3 | Med | Judge as a serial phase doubles latency. | ✅ **Resident, batched** Prometheus on dedicated GPUs, overlaps generation (§6.3, §7, §8). |
| C1 | High | **P6 is via Capacity Blocks/reservation, not plain spot**; Batch managed scaling may not provision it. | ✅ Capacity Block launch path documented; checkpoint-resume still applies (§8). |
| C2 | Med | ~**$110/instance-hr** — overkill for ≤7B. | ✅ Co-location mandatory on P6; `g6e`/`p5` fallback noted (§8). |

### 12.2 Assumptions
17 benchmark splits (ARC×2) × 100 models (avg ~3B) + Prometheus-7B judge. B200 conservative batched throughput: **prefill ~120k tok/s/GPU**, **decode ~25k tok/s/GPU**. Scenario A = ~2k samples/benchmark; Scenario B = full eval sets.

### 12.3 Time estimate (post-fix design)

Assumes A1 fixed (models resident, one load each) and B1 applied (co-location keeping the 8 B200s busy).

| Scenario | Compute GPU-hours | 1× p6-b200 (8 GPU) | 2× (16 GPU) |
| --- | --- | --- | --- |
| A — capped (~2k/bench) | ~15–18 | **~2–3 h** | ~1.5–2 h |
| B — full sets | ~30–40 | **~4–6 h** | ~2.5–4 h |

Notes:
- Without co-location (**B1**) the same GPU-hours stretch to ~2–3× the wall-clock because the B200s sit ~15–25% utilized.
- Without the resident-model fix (**A1**) add ~40–80 GPU-h of pure load overhead — now avoided by design.
- Diminishing returns past ~2 instances (100 models bin-packed; 7B models dominate imbalance). The real lever is **co-location**, not more instances.

### 12.4 GPU utilization
- SM/occupancy (busy) during active inference: **~60–85%**.
- Effective utilization (**MFU**): **~10–30% overall**, single-digit for sub-2B models (B200 oversized).
- **With co-location (B1): ~70–90%** — the only way to make P6 cost-sensible here.
