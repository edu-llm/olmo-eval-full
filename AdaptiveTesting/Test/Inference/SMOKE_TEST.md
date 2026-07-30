# Pre-calibration Smoke Test

`smoke_test.py` runs a handful of models through the **entire pre-calibration
pipeline** — model download, engine load, MCQ log-likelihood scoring, FRQ
free-response generation, durable output writing — and prints a report with the
model outputs, any errors found in the responses, and a full timing breakdown.

It exists to answer one question on a new machine: *does this pipeline actually
work here, and how fast?* It is not an evaluation; the scores it prints are
meaningless (5 questions).

---

## 1. What it does

For each of N models (default 5, chosen at random from the roster):

| Step | What runs | Timed as |
| --- | --- | --- |
| Fetch weights | `huggingface_hub.snapshot_download` | `download` |
| Load model | `run_benchmark.make_engine` → capability routing, up to 3 load attempts, a smoke generation that must produce output, vLLM→transformers fallback | `load` |
| Score MCQ | `mcq_scoring.score_mcq` — log-likelihood ranking over option continuations | `mcq:<bank>` |
| Generate FRQ | `frq_generate.generate_frq` — per-benchmark system prompt + conversation history, prompt fit to the context window, per-item token budget, Model Output schema | `frq:<bank>` |
| Free memory | `Engine.close` (CUDA graph / KV-cache reclamation) | `close` |

By default it samples **5 MCQ questions and 5 FRQ scenarios spread round-robin
across the banks**, so several benchmarks are exercised rather than five items
from one. The MCQ and FRQ item counts are totals, not per-bank.

It calls the driver's own `make_engine` and `_frq_overrides` rather than
reimplementing them, so if the driver breaks, this test breaks.

**Not covered:** the judging stage (`judge_all.py`) and aggregation
(`aggregate.py`). Judging needs a 7B Prometheus model; test it separately.

### Isolation guarantee

Everything is written under `AdaptiveTesting/Outputs/_smoke/<timestamp>/`. The
real `Outputs/mcq`, `Outputs/open` and `Outputs/_manifests` trees are never
touched, and **no `.done` markers are created** — so running this cannot cause a
later real sweep to skip pairs it should have run. Delete
`Outputs/_smoke/` whenever you like.

---

## 2. Prerequisites

### 2.1 Repository layout

FRQ items load from the sibling **eduLLM-Evals** repo, not from HuggingFace, so
the two repos must sit side by side:

```
<anywhere>/
├── AdaptiveTesting/          # this repo
└── eduLLM-Evals/
    └── data/
        ├── TutorBench/scenarios.jsonl
        ├── TutorEval/scenarios.jsonl
        ├── Bridge/scenarios.jsonl
        ├── BiGGen/scenarios.jsonl
        ├── InFoBench/scenarios.jsonl
        └── WildBench/scenarios.jsonl
```

If your layout differs, point an env var at the eduLLM-Evals root instead:

```bash
export EDULLM_EVALS_ROOT=/path/to/eduLLM-Evals      # Windows: $env:EDULLM_EVALS_ROOT="..."
```

You do **not** need the whole eduLLM-Evals repo. Those six `scenarios.jsonl`
files total ~16 MB; copying just them (keeping the `data/<Benchmark>/` structure)
is enough.

### 2.2 Software

- **Python 3.11+**
- Packages: `pip install -r requirements.txt`
- **PyTorch** is not in `requirements.txt` — install the build that matches your
  CUDA version from <https://pytorch.org/get-started/locally/>.
- **vLLM is Linux + CUDA only.** On Windows or macOS it will not install; use
  `--backend hf` (transformers) or `--backend mock`. The `requirements.txt` line
  is already guarded for macOS.

### 2.3 Hardware

A CUDA GPU for `--backend vllm`. `--backend hf` works on CPU but is *very* slow
(minutes per generation). `--backend mock` needs no GPU and no weights.

`configs/inference.yaml` sets `gpu_memory_utilization: 0.30`, sized for
co-locating several models on one large GPU. This test loads one model at a time,
so on a single smaller GPU raise that value (0.85 is reasonable) — otherwise a
model that would fit will fail to allocate its KV cache.

### 2.4 HuggingFace token

Some roster models are gated (`meta-llama`, `google`, `mistralai`). Put the token
in a `.env` file — the pipeline loads it automatically via
`common.bootstrap_env()`:

```
# AdaptiveTesting/.env   (or AdaptiveTesting/Test/Inference/.env)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

`bootstrap_env()` also installs the OS trust store (`truststore`), which is what
makes Hub reads work on networks that TLS-intercept with a corporate root CA.
Keep `truststore` and `python-dotenv` installed.

### 2.5 Disk

Weights land in the HuggingFace cache (`~/.cache/huggingface`, or `HF_HOME` if
set). The roster is 187 models, median 1.8B, max 7B, so **five random models is
~24 GB** (12–22 GB across the first few seeds). Cap the size to stay small:

| Command | Model pool | Download for 5 |
| --- | --- | --- |
| `python smoke_test.py` | 187 | ~24 GB |
| `python smoke_test.py --max-params-b 1.5` | 74 | ~12 GB |
| `python smoke_test.py --max-params-b 1.0` | 34 | ~9 GB |

The plan block prints the estimate for the models it actually picked before
downloading anything, and preflight warns below 20 GB free. The estimate assumes
2 bytes/param — repos that ship fp32, or both `.bin` and `.safetensors`, download
more.

---

## 3. Running it

From `AdaptiveTesting/Test/Inference/`:

```bash
# 0. Check the environment and see what would run — no downloads, no models.
python smoke_test.py --dry-run

# 1. Fully offline sanity check: no weights, no network, no GPU.
#    Proves the plumbing (prompt building, budgeting, writers, reports).
python smoke_test.py --backend mock --synthetic

# 2. The real thing: 5 random models x 5 MCQ x 5 FRQ.
python smoke_test.py --backend vllm

# 3. Same, but keep the download small.
python smoke_test.py --backend vllm --max-params-b 1.5

# 4. No GPU / no vLLM (Windows, macOS, CPU box).
python smoke_test.py --backend hf --max-params-b 1.0 --max-new-tokens 64

# 5. Pin specific models instead of sampling.
python smoke_test.py --models Qwen/Qwen2.5-1.5B-Instruct,microsoft/phi-1_5
```

Ids passed to `--models` must appear in the roster. List what's available with
`python models_registry.py`.

Capture the console output — it is part of the result:

```bash
python smoke_test.py --backend vllm 2>&1 | tee smoke.log
```

### Options

| Flag | Default | Notes |
| --- | --- | --- |
| `--num-models` | 5 | how many models to sample |
| `--num-mcq` / `--num-frq` | 5 / 5 | item totals, spread across banks; `0` disables that half |
| `--models` | — | comma list of ids or trailing names, instead of sampling |
| `--max-params-b` | — | only sample models at or below this size |
| `--seed` | 0 | same seed picks the same models and items — use it to reproduce a failure |
| `--backend` | from `configs/inference.yaml` (`vllm`) | `vllm` \| `hf` \| `mock` |
| `--max-new-tokens` | 256 | FRQ generation cap so the test finishes quickly; `0` uses the real 4096 |
| `--synthetic` | off | use the offline `synth_mcq`/`synth_open` banks (no dataset download) |
| `--models-yaml` | `Inputs/Models/models_200.yaml` | roster file |
| `--item-cap` | 200 | items loaded per bank before sampling |
| `--no-cache` | off | re-normalize MCQ items instead of reusing the local cache |
| `--skip-download-timing` | off | let the engine fetch weights (merges download into `load`) |
| `--dry-run` | off | preflight + plan, then exit |

Exit code is `0` only if every model completed with zero problems.

### Runtime expectations

| Configuration | Wall clock |
| --- | --- |
| `--backend mock --synthetic` | ~5 s |
| 5 small models, weights cached, GPU | 3–10 min |
| 5 random models, cold cache, GPU | 20–60 min, mostly downloading |
| `--backend hf` on CPU | 10+ min per model; use `--max-new-tokens 64` |

---

## 4. Reading the report

### Console

```
Salesforce/codegen-2B-mono   2.0B   [OK]
  backend  declared=vllm  effective=vllm
  context  resolved=2048  static_table=2048
  timings
     download                 41.30s   downloaded 4.21 GB
     load                     18.72s   backend=vllm, max_model_len=2048
     mcq:pedagogy              2.10s   2 question(s) scored
     frq:tutoreval             9.44s   1 generated, complete=True, status=ok
     close                     1.02s
  model total 72.58s   |   cumulative at finish 72.58s
  results  mcq 1/2 correct   |   frq 5 rows, 0 issue, 0 empty, 1 prompt-truncated, 0 hit-budget
    [mcq  pedagogy/pedagogy_00031] predicted=C gold=A -> wrong
    [frq  TutorEval/te_0554] chat_template=1 prompt_tok=1792/2048 out_tok=256/256 finish=length
      prompt> System: You are an AI tutor...
      output> To find the derivative you first...
```

Field by field:

- **`backend declared` vs `effective`** — `declared` is the roster's routing
  intent, `effective` is where it actually ran. A mismatch is flagged
  `*** DEGRADED ***`: vLLM could not serve the model and it fell back to
  transformers. Responses are still valid, just produced on a slower path.
  Models the roster marks `hf_fallback` run on `hf` by design and are not
  degradations.
- **`context resolved` vs `static_table`** — `resolved` is the window actually
  used, read from the checkpoint's `config.json` on the Hub and clamped by the
  manifest cap. `static_table` is the offline fallback value. If `resolved`
  equals the table for every model *and* you saw a
  `warning: cannot reach the HuggingFace Hub` line, the Hub was unreachable and
  windows came from the static table — see Troubleshooting.
- **`prompt_tok=1792/2048`** — prompt tokens vs the context window.
- **`out_tok=256/256 finish=length`** — the answer used its whole budget and was
  cut off. Expected with `--max-new-tokens 256`; in a real sweep (4096) it means
  a genuinely long answer.
- **`PROMPT-TRUNCATED`** — the prompt did not fit the window, so its head was
  dropped (the tail, i.e. the student's latest turn, is always kept). Normal for
  long TutorEval prompts on small-context models; a signal only if it happens on
  a model with a large window.
- **`chat_template=1/0`** — whether the tokenizer's chat template was applied. `0`
  on a base model is correct. `0` on an `-Instruct`/`-it` model means the roster
  flag or the tokenizer's template is wrong, and is worth reporting.

### Alerts and problems

A model is counted as a problem when any of these occur, and each is listed with
its cause:

| Symptom | Meaning |
| --- | --- |
| `load failed` | the model never came up (after 3 retries + hf fallback) |
| `degraded` | ran on transformers instead of vLLM |
| `N issue` | rows with `Issue=1` — generation raised; the error text is printed |
| `N empty` | generation "succeeded" but produced no text — a real failure |
| `download failed` | usually a 401 on a gated repo, or TLS |

### JSON

`Outputs/_smoke/<run-id>/smoke_report.json` holds everything the console
truncates: full outputs, full rendered prompts (first 2000 chars), per-stage
timings, the preflight results, and host/platform info. **Send this file back
along with the console log.**

Raw pipeline artifacts are alongside it, in the exact format the real sweep
produces: `mcq/<bank>/<model>.csv` and
`open/<bank>/<model>.responses.jsonl`.

### Expected oddities (not bugs)

- **MCQ accuracy is meaningless** — 5 questions, and under `--backend mock`
  scores are a hash, so `0/5` is normal there.
- **`--backend mock` always reports `chat_template=0`** — there is no tokenizer.
  Verifying chat templates requires `hf` or `vllm`.
- **The first `frq:` stage of each model is ~2 s slower** than the rest. FRQ
  generation resolves the model's commit SHA from the Hub for provenance; the
  first call pays connection setup. It inflates `frq:` timings by one Hub
  round-trip per bank — subtract that when judging throughput.
- **`Inputs/MCQ/Benchmarks/*.n200.s0.jsonl`** appears: the normalized MCQ item
  cache. Harmless, and gitignored.
- **`benchmark item load` is reported separately** from the model timings. It
  happens once, before any model, and on a cold machine it includes downloading
  the three MCQ datasets from HuggingFace. It is excluded from the wall clock
  attributed to models.

---

## 5. Troubleshooting

**`preflight FAILED: package X`** — `pip install -r requirements.txt`. If it is
`torch`, install it separately (§2.2).

**`preflight FAILED: FRQ scenario banks — MISSING: ...`** — the eduLLM-Evals
banks aren't where the code looks. The preflight line above it prints the
resolved root; fix the layout or set `EDULLM_EVALS_ROOT` (§2.1).

**`preflight FAILED: CUDA availability — no CUDA device`** — with
`--backend vllm`. Use `--backend hf` (CPU, slow) or `--backend mock`.

**`SSLError` / `CERTIFICATE_VERIFY_FAILED` / `Cannot send a request, as the
client has been closed`** — TLS interception. Confirm `truststore` is installed
(preflight lists it); it is injected before the first HTTPS connection. If it
persists, export your corporate root CA:
`export SSL_CERT_FILE=/path/to/ca-bundle.pem` (and
`REQUESTS_CA_BUNDLE` to the same path).

**`401 Unauthorized` / `GatedRepoError`** — the model is gated and needs a token
with the license accepted. Put `HF_TOKEN` in `.env` (§2.4), or skip gated models
by pinning ids with `--models`.

**`CUDA out of memory` at load** — lower `gpu_memory_utilization` in
`configs/inference.yaml`, or use `--max-params-b` to pick smaller models. If it
only happens on the *second* model onward, `Engine.close()` is failing to reclaim
memory — report that, it is a real bug.

**vLLM load fails, then everything runs on `hf`** — that is the fallback working
as designed. The vLLM stderr tail is printed above the fallback message; include
it in the report.

**Everything hangs on download** — the roster has 7B models. Ctrl-C and use
`--max-params-b 1.5`.

---

## 6. What to send back

1. The console log (`smoke.log`).
2. `Outputs/_smoke/<run-id>/smoke_report.json`.
3. Machine details: OS, GPU, CUDA version, and whether vLLM installed.

The single most useful line is the `SUMMARY` block: how many models were clean,
the download/load/inference split, and any `ALERT` lines.
