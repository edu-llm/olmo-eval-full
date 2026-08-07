# Frontier judge validation

This is a separate API-based validation path for three frontier judges:

| Judge | Excluded tutor family | Eligible cases per wave |
|---|---|---:|
| GPT-5.5 | OpenAI | 174 |
| Opus 4.8 | Anthropic | 174 |
| Gemini 3.6 Flash | Google/Gemini | 174 |

Every one of the 261 criterion cases is judged by exactly two models. The
human labels are not included in any judge prompt. They are loaded only while
preparing the family-routing sidecar and later when calculating metrics.

## 1. Install and configure

Use Python 3.10 or newer:

```bash
python3 -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` if `.env` does not already exist, then fill in
only your TrueFoundry key locally. The confirmed endpoint and catalog slugs are
already supplied:

```env
TFY_API_KEY=...
TFY_BASE_URL=https://tfy.promptlens.trilogy.com/api/llm/api/inference/openai
TFY_TOKEN_PARAM=max_completion_tokens
TFY_TEMPERATURE_MODE=zero
FRONTIER_OPENAI_MODEL=openai-group/gpt-5.5
FRONTIER_ANTHROPIC_MODEL=claude-group/claude-opus-4-8
FRONTIER_GOOGLE_MODEL=gemini-group/gemini-3.6-flash
```

`.env` is gitignored. Do not put keys in a command, result archive, Slack, or
source file. Keep the endpoint, request profile, and selected catalog slugs
fixed for the entire study. TrueFoundry catalog slugs are gateway aliases, so
the runner also records the model identifier returned by every response.

The default backend sends all three models through TrueFoundry's
OpenAI-compatible Chat Completions endpoint. It does not load these models onto
an AWS GPU; a normal CPU machine or CPU-based AWS instance with outbound
network access is sufficient. The old direct-provider route remains available
only through an explicit `--backend direct`. That optional route uses its
provider key and built-in direct model ID unless a separate
`FRONTIER_DIRECT_*_MODEL` variable or CLI model override is supplied; the
TrueFoundry catalog slugs above are never sent to a native provider API.

## 2. Prepare and verify routing without API calls

```bash
python3 scripts/run_frontier_judge_validation.py prepare \
  --cases runs/judge_validation_v2/judge_cases.blinded.jsonl \
  --human-labels runs/judge_validation_v2/human_labels.csv \
  --out-dir runs/frontier_judge_validation/prepared

python3 scripts/run_frontier_judge_validation.py plan \
  --cases runs/frontier_judge_validation/prepared/judge_cases.blinded.jsonl \
  --routing runs/frontier_judge_validation/prepared/case_routes.blinded.jsonl \
  --json-out runs/frontier_judge_validation/plan.json
```

The plan must show 87 cases in each tutor family, 174 eligible cases per judge
per wave, no same-family cases, and 3,132 calls for the complete six-wave run.

## 3. Run the study

First run one case for each judge into a separate smoke directory. This makes
three paid calls and verifies authentication, model access, parsing, and the
frozen request profile before the full study:

```bash
python3 scripts/run_frontier_judge_validation.py run \
  --cases runs/frontier_judge_validation/prepared/judge_cases.blinded.jsonl \
  --routing runs/frontier_judge_validation/prepared/case_routes.blinded.jsonl \
  --judge gpt-5.5 --limit 1 --concurrency 1 --batch-size 1 --max-retries 0 \
  --output runs/frontier_judge_validation/smoke/gpt.jsonl
```

Repeat that command with `--judge opus-4.8` and
`--judge gemini-3.6-flash`, using different output filenames. Do not mix smoke
artifacts with the full result directory. If the gateway explicitly rejects
the token field or temperature, rerun all smoke and full commands with the
same frozen override: `--tfy-token-param max_tokens` or
`--tfy-temperature-mode omit`.

The suite runs three identical-prompt repeats and three small prompt variants
for each judge. Raw gateway payloads and normalized pass/fail decisions are
both retained:

```bash
python3 scripts/run_frontier_judge_validation.py suite \
  --cases runs/frontier_judge_validation/prepared/judge_cases.blinded.jsonl \
  --routing runs/frontier_judge_validation/prepared/case_routes.blinded.jsonl \
  --output-dir runs/frontier_judge_validation/results
```

If the full run is interrupted, repeat the `suite` command with `--resume`.
After a completed run containing API or parse failures, repeat it with both
`--resume --retry-errors`; replaced error rows are preserved in retry-history
files. A successful wave manifest must have `status: "complete"`, exactly one
nonblank `resolved_provider_models` value, and
`model_provenance_consistent: true`. The runner marks within-wave model drift
as a failure, and the comparator rejects drift across waves.

## 4. Compare with the human labels

Run this only after all 18 wave files exist:

```bash
python3 scripts/compare_frontier_judges.py \
  runs/judge_validation_v2/human_labels.csv \
  --results-root runs/frontier_judge_validation/results \
  --json-out runs/frontier_judge_validation/comparison.json \
  --csv-out runs/frontier_judge_validation/comparison.csv \
  --review-out runs/frontier_judge_validation/human_review.csv
```

The report includes human-label accuracy, macro-F1, critical-failure
sensitivity, coverage, test-retest agreement, prompt flip rate, skill and tutor
family slices, pairwise agreement on common cases, and two-judge consensus and
human-review workload. It also applies the existing acceptance thresholds.

Do not rank the judges from pooled 174-case accuracy alone. Each judge omits a
different tutor family, and each pair shares only 87 cases from one family. Use
the family-specific and pairwise common-case sections for fair comparisons.
