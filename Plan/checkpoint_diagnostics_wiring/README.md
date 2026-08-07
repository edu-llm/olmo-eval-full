# Checkpoint diagnostics wiring (the shared invocation seam)

How the CAT diagnostics actually run on a checkpoint on AWS, in a way that is **shared by
all four style branches** (`flow/uni-mcq`, `flow/mirt-mcq`, `flow/uni-frq`,
`flow/mirt-frq`) and never becomes a merge-conflict point. This is the cross-cutting
companion to [`Plan/mcq_cat_diagnostics/README.md`](../mcq_cat_diagnostics/README.md) and
[`Plan/frq_cat_diagnostics/README.md`](../frq_cat_diagnostics/README.md).

## The problem the seam solves

Today the AWS checkpoint flow (`tests/aws/run_checkpoint_eval.sh` = the per-checkpoint
atom, and `tests/aws/run_checkpoint_batch.sh` = the batch driver) runs `olmo-eval run`. It
has no notion of diagnostics. If each style branch wired its own diagnostic into those
shell scripts, all four branches would edit the same lines of the same file and conflict
on the second merge.

The fix is a single, **style-agnostic, parameterized** entry point, committed and frozen
once. Adding a style never touches it, because the style name is just an argument that the
Python runners resolve through their auto-discovering registries.

## The one generic command

Both modalities expose the same runner shape:

```
python -m diagnostics.mcq_cat.runner  --cat-style <STYLE> --checkpoint <URI> --s3-out <URI> [...]
python -m diagnostics.frq_cat.runner  --cat-style <STYLE> --checkpoint <URI> \
        --tutor-endpoint <URL> --judge-endpoint <URL> --s3-out <URI> [...]
```

`<STYLE>` is resolved by the registry via directory scan, so `runner.py` is never edited
when a style is added. The MCQ runner already exists and works this way; the FRQ runner is
built to match (see the FRQ plan).

## Deliverable: `tests/aws/run_checkpoint_diag.sh`

A new orchestrator that **reuses the atom's proven mechanics** (pre-flight caps, launch,
wait-for-SSM, stage, install, detached run + 24 KB-safe sentinel poll, verify, triple
self-terminate teardown) and swaps only the on-node run command from `olmo-eval run` to
the diagnostics runner. It mirrors the atom's command strings rather than refactoring the
atom, exactly as `run_checkpoint_batch.sh` does, so the validated atom stays untouched.

Selection is entirely by environment variable — no per-style code path:

| Env var | Meaning | Example |
|---|---|---|
| `DIAGNOSTIC_MODALITY` | which runner package | `mcq_cat` \| `frq_cat` |
| `CAT_STYLE` | registered style name (auto-discovered) | `uni_mcq`, `mirt_frq` |
| `CHECKPOINT` | checkpoint under test | `s3://.../step_1000` |
| `CHECKPOINT_KIND` | `hf` (default) \| `olmo_core` | `hf` |
| `BENCHMARK` / `IRT_PARAMS` | optional overrides (else style defaults) | |
| `SE_THRESHOLD` / `MAX_ITEMS` | CAT stop knobs | `0.3` / `40` |
| `JUDGE_ENDPOINT` | FRQ only: served frozen-judge `/v1` URL | |
| `S3_BUCKET` / `S3_PREFIX` / `S3_GROUP` | result namespace (same story as the atom) | |

On-node command (schematic — the modality/style are just interpolated values):

```bash
uv run python -m diagnostics.${DIAGNOSTIC_MODALITY}.runner \
  --cat-style "${CAT_STYLE}" \
  --checkpoint "${CHECKPOINT}" --checkpoint-kind "${CHECKPOINT_KIND}" \
  --s3-out "s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/${CAT_STYLE}/step_${N}/" \
  ${FRQ_ONLY_ENDPOINT_FLAGS}
```

Because every distinguishing value is an argument, the four style branches all invoke the
exact same script; none of them edits it.

## FRQ two-endpoint shape (served)

FRQ needs two served vLLM endpoints on the box:

- **tutor** = the checkpoint under test (respgen; greedy/deterministic).
- **judge** = the frozen Qwen `generic-binary` judge (grading), defined once in
  `frq_cat/common/judge.py` and selected per style via `config.yaml`.

`run_checkpoint_diag.sh` serves both (or accepts an external `JUDGE_ENDPOINT`) before
launching the FRQ runner. MCQ scores in-process and needs neither endpoint. This branch in
the orchestrator keys off `DIAGNOSTIC_MODALITY` only — still no per-style logic.

## Idempotency + results namespace

Reuse the atom/batch conventions so a re-run is safe and a sweep can resume:

- Skip-if-done against the per-style result prefix (`_SUCCESS` / the runner's report marker).
- Results land under `.../<S3_GROUP>/<CAT_STYLE>/step_<N>/` so styles never collide.
- The batch driver can drive many checkpoints for one style by setting `CAT_STYLE` +
  `DIAGNOSTIC_MODALITY` and pointing at a `RUN_URI`, exactly as it does for `olmo-eval run`.

## Freeze order (do this before cutting the 4 branches)

1. Confirm the MCQ scaffolding is frozen (it is) and build + freeze the FRQ scaffolding.
2. Commit `tests/aws/run_checkpoint_diag.sh` (the seam) — style-agnostic, env-driven.
3. Add CODEOWNERS covering the shared surfaces:
   - `diagnostics/mcq_cat/{base,registry,runner}.py`, `diagnostics/mcq_cat/common/**`
   - `diagnostics/frq_cat/{base,registry,runner}.py`, `diagnostics/frq_cat/common/**`
   - `tests/aws/run_checkpoint_eval.sh`, `run_checkpoint_batch.sh`, `run_checkpoint_diag.sh`
4. Smoke each runner with `--dry-run` (and `run_checkpoint_diag.sh --dry-run`) on a tiny
   checkpoint to prove the seam end to end.
5. Cut the four branches from that frozen commit.

## Why this cannot conflict

- The seam script is committed once and frozen; branches pass values, never edit it.
- Style names resolve by registry directory scan, so no shared list or dispatch exists.
- Per-style results, config, and IRT params live under disjoint paths.
- CODEOWNERS forces any shared-surface change to be a separate, serialized PR.

## Out of scope

- Building the style implementations themselves (that is each branch's job).
- Calibration/fitting of IRT parameters and the judge (done on `frq-lab` / `frq/*`).
- Changing the validated atom (`run_checkpoint_eval.sh`) or batch driver internals; the
  seam mirrors their mechanics instead of modifying them.
