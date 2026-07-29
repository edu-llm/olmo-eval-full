# Cluster handoff: presentation-criteria grading (targeted add-on)

This is a **small add-on run** to the full Qwen calibration study. It grades the
**662 per-scenario presentation (`style_surface`) criteria** that were `optional`
(non-gating) in the main run and therefore never scored. We flip them to gating
so `prepare` includes them, then grade them with the **identical frozen judge**
so the verdicts merge cleanly with the existing `run_data.jsonl`.

- **Scope:** 82 tutor models × 662 presentation criteria = **54,284 cells**
  (~11% of the 505,615-case main run). Expect a small number of missing cells
  from the same blank / over-length response units as the 32K run (≈ up to 121,
  one per affected model×scenario); `prepare` prints the exact number.
- **Judge:** unchanged — `Qwen/Qwen3.5-9B` zero-shot, 32,768-token context,
  fail when `p_fail >= 0.33`. **Do not change the judge config.**
- **Reversible:** this only adds verdicts for presentation criteria. It does not
  alter the main 6,180-criterion results or the official curated bank.

## Prerequisite: reuse the 32K packet you already ran

You already have `qwen_calibration_handoff_32k_20260728/` (the full-run packet)
with its `scripts/`, `tutor_cat/`, `tutorbench-responses-v2.zip`,
`calibration_cohort_policy.json`, `calibration_judge_config.qwen_zero_shot.json`,
validated env (`vllm==0.26.0`), and — ideally — the `calibration_id_key.txt` you
generated. **Reuse all of it.** The only change here is the rubric bank.

> The `id_key` does not need to match the main run: our ingestion reads the
> **de-blinded** `run_data.jsonl` (the one with a `tutor_model` column), so a
> fresh key is fine. If you still have the original key, reusing it is harmless.

## 1. Drop in the presentation bank

Copy `rubrics_presentation_gating.jsonl` (in this folder) next to the 32K packet,
e.g. `qwen_calibration_handoff_32k_20260728/rubrics_presentation_gating.jsonl`.

## 2. Prepare (CPU-only) with a FRESH work dir

Run from inside the 32K packet directory. Use a **new** `CAL_WORK_ROOT` so you do
not touch the main run's prepared/results dirs.

```bash
cd qwen_calibration_handoff_32k_20260728

export CAL_WORK_ROOT=/path/to/edullm-presentation-work   # NEW, empty path
test ! -e "$CAL_WORK_ROOT" || { echo "choose a fresh CAL_WORK_ROOT" >&2; exit 1; }
mkdir -p "$CAL_WORK_ROOT/source"
unzip -q tutorbench-responses-v2.zip -d "$CAL_WORK_ROOT/source"

umask 077
export CAL_ID_KEY_FILE="$CAL_WORK_ROOT/calibration_id_key.txt"
# reuse the main run's key if you have it, else generate one (both fine):
test -s "$CAL_ID_KEY_FILE" || openssl rand -hex 32 > "$CAL_ID_KEY_FILE"
export EDULLM_CALIBRATION_ID_KEY="$(tr -d '\n' < "$CAL_ID_KEY_FILE")"

python scripts/run_calibration_judging.py prepare \
  --responses "$CAL_WORK_ROOT/source/responses" \
  --scenarios data/TutorBench/curated/scenarios_curated.jsonl \
  --rubrics   rubrics_presentation_gating.jsonl \
  --cohort-policy calibration_cohort_policy.json \
  --blank-response-policy missing \
  --id-key-env EDULLM_CALIBRATION_ID_KEY \
  --output-dir "$CAL_WORK_ROOT/presentation_prepared"
```

**Expected:** a line reporting **`82 tutors x 662 criteria = 54,284 cells`** and a
handful of shards (far fewer than the 26 of the full run). Note the exact
missing-cell count it prints. Keep `presentation_prepared/private/` private.

## 3. Generate + submit the GPU shard commands (same judge config)

```bash
python scripts/run_calibration_judging.py commands \
  --prepared-dir "$CAL_WORK_ROOT/presentation_prepared/public" \
  --config calibration_judge_config.qwen_zero_shot.json \
  --results-dir "$CAL_WORK_ROOT/presentation_results" \
  --output "$CAL_WORK_ROOT/presentation_shard_commands.txt"
```

Submit each line through the scheduler exactly as in the full run (same GPU /
partition / wall-time conventions, 40 GB+ GPU). This is a fraction of the compute
of the main run.

## 4. Status + merge

```bash
python scripts/run_calibration_judging.py status \
  --prepared-dir "$CAL_WORK_ROOT/presentation_prepared/public" \
  --config calibration_judge_config.qwen_zero_shot.json \
  --results-dir "$CAL_WORK_ROOT/presentation_results" \
  --json-out "$CAL_WORK_ROOT/presentation_status.json"

# once all shards report complete:
python scripts/run_calibration_judging.py merge \
  --prepared-dir "$CAL_WORK_ROOT/presentation_prepared" \
  --config calibration_judge_config.qwen_zero_shot.json \
  --results-dir "$CAL_WORK_ROOT/presentation_results" \
  --output-dir "$CAL_WORK_ROOT/presentation_merged"
```

## 5. What to return to the eval team

Please return, same as the full run:

1. **The de-blinded `run_data.jsonl`** for this run — the flat verdict table that
   includes a **`tutor_model`** column (identical schema to the `run_data.jsonl`
   you already sent). This is what we ingest.
2. `presentation_merged/` and the preparation manifests.
3. If your merge does not emit a de-blinded `run_data.jsonl` directly, also send
   `presentation_prepared/private/case_index.jsonl` and we will de-blind on our
   side.

That's it — everything else is identical to the 32K run.
