# OLMoE native benchmark smoke test

This handoff answers one narrow question: **can the OLMoE checkpoint load through
OLMo Eval and complete a real benchmark using OLMo's native runner?** It runs three
deterministically selected GSM8K cases. There are no handwritten education prompts and
no EduLLM `Scenario` or tutor adapter in this path.

## Frozen test contract

- Native OLMo task: `gsm8k`
- Cases: 3, selected by the task's seed (`42`)
- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- Hugging Face model revision: `7f1c97f440f06ce36705e4f2b843edb5925f4498`
- Architecture expected by vLLM: `OlmoeForCausalLM`
- Precision: `bfloat16`
- Context limit: 4096 tokens
- GPUs / tensor parallelism: one GPU, TP=1
- Generation: greedy, one sample, maximum 128 new tokens

GSM8K is public and non-gated. OLMo Eval downloads it through the normal Hugging Face
dataset loader and uses the benchmark's built-in eight-shot completion format, answer
extractor, and exact-match scorer. The source resolved to `openai/gsm8k` at revision
`740312add88f781978c0658806c59bc2815b9866` when this handoff was prepared. The current
OLMo task does not pin that dataset revision, so the observed revision is provenance,
not an enforced download constraint.

The checkpoint has about 6.9B total parameters even though roughly 1.3B are active per
token. Use a **BF16-capable GPU** (typically NVIDIA Ampere or newer) with approximately
24 GB of VRAM to leave room for vLLM and its KV cache. The script checks both the chosen
visible GPU's BF16 support and the pinned model's `config.json` architecture before
starting vLLM; it does not silently fall back to another dtype.

## Run on an already-provisioned Linux GPU worker

From the repository root:

```bash
uv sync --frozen

uv run python scripts/edullm/run_olmoe_benchmark_smoke.py \
  --output-dir runs/olmoe_gsm8k_smoke \
  --gpu-id 0 \
  --dtype bfloat16
```

Use a new or empty output directory. Do not replace a scheduler-provided
`CUDA_VISIBLE_DEVICES`; `--gpu-id 0` means the first GPU already visible to the job.
The first run needs network access to Hugging Face for the model and GSM8K unless both
are already cached.

This script contains no AWS calls or credentials. It is a handoff for an
already-provisioned Linux GPU worker; it does not add or change platform submission
configuration.

## What the script actually exercises

The script selects only `standard_olmo` and delegates to OLMo Eval's native
`AsyncEvalRunner`. OLMo owns benchmark loading, few-shot formatting, the completion
requests, managed vLLM startup, generation, answer extraction, exact-match scoring,
metrics, request/prediction files, worker cleanup, and server shutdown.

GSM8K accuracy is recorded as diagnostic evidence but is **not** a smoke-test gate.
This is a three-case compatibility check, not a meaningful benchmark estimate. The
command passes only if the native run succeeds, exactly three requests and predictions
are stored, all three raw/final outputs are nonblank, request settings and IDs agree,
the exact-match score is present, the pinned provider contract is recorded, and a
managed vLLM log is captured.

If GPU/model preflight, model download, vLLM startup, inference, or cleanup fails, the
wrapper exits nonzero and makes a best effort to preserve the exact command and a failed
`olmoe_benchmark_validation.json` with the stage and error. Use a new output directory
for the next attempt.

## Output artifacts

The output directory contains:

| Artifact | Meaning |
| --- | --- |
| `manifest.json` | Shared native OLMo mode-run status and pinned model provenance. |
| `report_index.json` | Index of the native mode and its artifacts. |
| `olmoe_preflight.json` | BF16 GPU evidence and pinned HF architecture/config evidence. |
| `modes/standard_olmo/metrics.json` | GSM8K exact-match score, task count, and provider config. |
| `modes/standard_olmo/predictions/...jsonl` | Three raw OLMoE outputs, extracted answers, and per-case scores. |
| `modes/standard_olmo/requests/...jsonl` | Exact native GSM8K few-shot prompts and generation settings. |
| `modes/standard_olmo/mode_result.json` | Native mode status and artifact index. |
| `logs/` | Managed vLLM startup/server logs. |
| `reproduction_command.txt` | Exact command recorded by the wrapper. |
| `olmoe_benchmark_validation.json` | Final strict compatibility gates; check `status == "passed"`. |

## GPU-free checks

The contract tests do not download GSM8K or load the model:

```bash
uv run --no-group vllm pytest -q \
  tests/edullm/test_edullm_olmoe_benchmark_smoke.py
```

They validate the pinned provider/task configuration and exercise the artifact gates
with local fixtures. Only the groupmate's live GPU run can prove that the real OLMoE
checkpoint loads and generates.
