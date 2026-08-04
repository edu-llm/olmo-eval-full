# RUN_EDUBENCH.md

How to run the **response-generation** stage of `tutor-cat` over the manually-built,
unmerged EduBench scenarios. This produces the model-response matrix (one generated
answer per model x scenario). It does **not** grade anything.

> Scope: generation only. EduBench ships `difficulty`/`discrimination` as `null`
> (uncalibrated), so the judge / MIRT stage (`tutor-cat run`) cannot score this bank
> yet and is out of scope here.

Run every command below from the repo root `eduLLM-Evals/` (the directory holding
`models.yaml`, `benchmarks.yaml`, and `pyproject.toml`), because benchmark paths are
resolved relative to it.

---

## TL;DR

```bash
# 0. one-time, on a Linux CUDA GPU box, from eduLLM-Evals/
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gen]"
export HF_TOKEN=<your_hf_token>

# 1. place the provided scenarios file (it is gitignored, see step 3)
#    -> data/EduBench/augmented_qmat/unmerged/scenarios.jsonl

# 2. add the EduBench entry to benchmarks.yaml (see step 5)

# 3. preview (no GPU), then smoke test, then full run
tutor-cat generate --benchmarks benchmarks.yaml --only EduBench --dry-run --limit 3
tutor-cat generate --benchmarks benchmarks.yaml --only EduBench --model Qwen/Qwen2.5-0.5B-Instruct --limit 2 --gpu-ids 0
tmux new -s edubench
tutor-cat generate --benchmarks benchmarks.yaml --only EduBench --out-dir runs/responses
```

Output lands at `runs/responses/EduBench/<model>.jsonl`.

---

## 1. Prerequisites

- A **Linux host with an NVIDIA CUDA GPU**. The `[gen]` runtime deps (`vllm`, `torch`)
  have no Windows wheels; generation runs on the GPU box only. (The pure logic is
  importable anywhere, which is what makes the offline `--dry-run` in step 6 work.)
- **Python 3.10+** and this `eduLLM-Evals` repo cloned onto the box.
- A **Hugging Face token** in `HF_TOKEN` for gated repos in `models.yaml`
  (`meta-llama/*`, `google/gemma*`, `mistralai/*`). Without it those models fail to
  load and are recorded as `Issue=1` cells (the run still finishes).

## 2. Install

```bash
cd eduLLM-Evals
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gen]"
export HF_TOKEN=<your_hf_token>
```

## 3. Place the EduBench scenarios file (important)

Every EduBench bank file is **gitignored**, so a fresh clone will NOT contain one. Two
ways to get the sampled bank this doc assumes:

```bash
# (a) build it from the cleaned bank, if you have unmerged/scenarios.json
python scripts/stratified_sample_bank.py \
    --scenarios data/EduBench/augmented_qmat/unmerged/scenarios.json \
    --stratify-by use_case --per-group 252 \
    --out-scenarios data/EduBench/augmented_qmat/unmerged/scenarios_252.jsonl

# (b) or copy the file you were given to exactly that path
```

Notes:
- The loader reads `.jsonl` only; the pretty `.json` twin is the source the sampler
  converts from, not something generation can read.
- The sampled bank is 2,268 text-modality scenarios: 252 for each of the 9 task types,
  drawn from the 7,040 cleaned scenarios. All are generated; there is no non-text
  content to skip. Use `--per-group` to change the quota, or omit it to convert the
  whole 7,040-scenario bank.
- Generation reads only each scenario's `prompt`, `use_case`, and
  `conversation_context`. It does not read the rubrics, so you do not need
  `rubrics.jsonl` for this stage.

## 4. Verify the layout

```bash
ls models.yaml benchmarks.yaml                                       # you are in the repo root
wc -l data/EduBench/augmented_qmat/unmerged/scenarios_252.jsonl      # expect 2268
```

## 5. Register EduBench as a benchmark

EduBench is already registered in `benchmarks.yaml`; confirm the entry matches the file
you placed in step 3 (paths are relative to the repo root):

```yaml
  - name: EduBench
    scenarios: data/EduBench/augmented_qmat/unmerged/scenarios_252.jsonl
    enabled: true
```

Then always pass `--only EduBench` on the command line so **only** EduBench runs; the
other benchmarks already registered in the file are left untouched (`--only` overrides
each entry's `enabled` flag and runs exactly the named subset).

Do **not** use the `--scenarios <path>` shortcut for this: that shortcut hardcodes the
benchmark label as `TutorBench`, so rows would be tagged `TutorBench` and written to
`runs/responses/TutorBench/`, mislabeling the data and colliding with a real TutorBench
run. The registry entry above tags rows `EduBench` and shards them under
`runs/responses/EduBench/`.

## 6. Run

**Offline preview (no GPU, no network).** Prints the resolved per-model config and the
exact rendered messages for a few scenarios, including the system turn and the ES
multi-turn `conversation_context`. Use this first to eyeball prompts.

```bash
tutor-cat generate --benchmarks benchmarks.yaml --only EduBench --dry-run --limit 3
```

**Smoke test (one small model, 2 scenarios, one GPU).** Confirm how this node numbers
its GPUs first — CUDA indices are 0-based and per-node.

```bash
nvidia-smi -L                                   # e.g. "GPU 0 ... GPU 7"
tutor-cat generate --benchmarks benchmarks.yaml --only EduBench \
    --model Qwen/Qwen2.5-0.5B-Instruct --limit 2 --gpu-ids 0
```

**Full run (all 100 models x 9,163 scenarios).** Run inside `tmux` so it survives a
disconnect. This is large — roughly 100 x 9,163 ~= 916k generations — so expect a long
run and use as many GPUs as the box has.

```bash
tmux new -s edubench
tutor-cat generate --benchmarks benchmarks.yaml --only EduBench --out-dir runs/responses
#   detach: Ctrl-b then d   |   reattach: tmux attach -t edubench
```

Useful flags:
- `--gpus N` — use N worker GPUs (default: all detected). The fleet runs one model per
  GPU (data-parallel), pulling the 100 models off a shared queue.
- `--gpu-ids 2,3` — pin the fleet to specific physical device indices (each worker gets
  `CUDA_VISIBLE_DEVICES=<id>`). `--gpu-ids 0` runs the whole roster on GPU 0 only. An
  out-of-range index fails fast with the valid range.
- `--model <id>` — run only one model id from `models.yaml`.
- `--limit N` — cap scenarios per model (smoke tests).
- `--s3-uri s3://<bucket>/prefix` — upload each finished shard (uses instance IAM);
  optional, results are also on local disk.
- `--no-resume` — ignore existing shards and regenerate from scratch.

**Resuming:** re-running the exact same command continues where it stopped — shards are
keyed by scenario id, and completed cells are skipped. An interrupted or preempted run
just picks up on the next invocation.

## 7. Output

One JSONL shard per model:

```
runs/responses/EduBench/<sanitized_model_id>.jsonl
```

Each line is one `(model, scenario)` row with the PRD "Model Output" keys:

| Key | Meaning |
| --- | --- |
| `Benchmark` | `EduBench` |
| `Scenario` | scenario id (e.g. `eb_4335`) |
| `Model`, `Model Revision` | model id + pinned commit SHA |
| `Rendered Prompt`, `Chat Template Applied` | exact prompt fed to the model |
| `Generation Params`, `Max Model Len`, `Prompt Tokens`, `Output Tokens`, `Finish Reason`, `Truncated`, `Latency (s)` | decoding + telemetry |
| **`Output`** | **the model's generated response (the payload)** |
| `Issue`, `Issue Description` | `1` + reason on a failed cell, else `0` / `N/A` |

Failed cells (load error, generation error) are written as `Issue=1` with an empty
`Output` rather than dropped, so every `(model, scenario)` has exactly one row. The
command also prints a per-model JSON summary line (status, rows written, throughput) at
the end.

## 8. EduBench-specific behavior to know

- **ES (`mental_health`) scenarios are multi-turn.** Their Agent/Student dialogue lives
  in `conversation_context` and the `prompt` holds only the instruction. The runner turns
  those turns into alternating `assistant`/`user` chat messages before the final user
  prompt. Confirm the rendering in the `--dry-run` output.
- **No system prompt.** EduBench is in `respgen/prompts.py::_NO_SYSTEM_BENCHMARKS`
  (alongside IFEval and InFoBench), so the system turn is **omitted entirely** — the
  message list starts at the first user turn. This is deliberate: an EduBench `prompt`
  already states the subject, education level, task and required output sections, so a
  tutor persona would compete with it. The `--dry-run` output shows the roles per
  scenario; expect `roles=['user']` for single-turn tasks and no `system` entry for ES.
  Getting this right depends on the benchmark label being exactly `EduBench` — that
  string is what selects the rule, which is the other reason not to use the
  `--scenarios` shortcut below.
- **Generation only.** Grading (LLM judge) and MIRT calibration are a separate,
  per-benchmark stage and do not apply to this uncalibrated bank. This doc stops at
  producing the response matrix.
