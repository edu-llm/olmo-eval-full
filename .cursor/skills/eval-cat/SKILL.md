---
name: eval-cat
description: Evaluate a model checkpoint on one benchmark using adaptive testing (CAT), reporting an ability score and predicted accuracy from a few dozen items instead of the full benchmark. Use when the user asks to evaluate, score, or diagnose a checkpoint on a named benchmark, mentions CAT or adaptive testing, or asks how good a checkpoint is on ARC, HellaSwag, WinoGrande, GSM8K, GPQA, IFEval, MATH, MuSR or BBH.
---

# Evaluate a checkpoint with adaptive testing

Estimates a checkpoint's ability on one benchmark by administering items adaptively —
selecting each next item to be maximally informative at the current ability estimate, and
stopping once the standard error is small enough. A run typically takes 13–40 items instead
of the benchmark's full 1,000–5,000, and reports both an ability score (theta) and a
predicted accuracy.

**One benchmark per run.** If the user names several, run them one at a time and report each
separately; do not try to combine them into a single invocation.

---

## What you need from the user

| Input | Required | Notes |
|---|---|---|
| Checkpoint | yes | Local path or `s3://` URI. Must be HuggingFace format. |
| Benchmark | yes | One name, from the ready list you query in step 1. |
| Output location | yes | `s3://` URI or local directory for `cat_report.json`. |

If any is missing, ask for it before doing anything else. Do not guess a benchmark, and do
not default the output location.

---

## Step 1: preflight, before touching the checkpoint

Run both checks first. They are cheap, and they catch the three failure modes that would
otherwise surface only after staging gigabytes.

Write this to a temporary file and run it from the repo root, rather than passing it inline —
the quoting does not survive most shells.

```python
from diagnostics.mcq_cat.styles.uni_mcq import resolve
from diagnostics.mcq_cat.styles.uni_mcq.datasets import ready_names

name = "BENCHMARK"
print("ready:", ready_names())
try:
    r = resolve.resolve(name)
    print("OK", name, "items=", r.manifest.get("items"),
          "fit=", r.fit_family, "modality=", r.spec.modality)
except resolve.DatasetNotAvailable as exc:
    print("UNAVAILABLE", name, "->", exc)
```

**`PYTHONPATH` must contain both the repo root and `src`.** `diagnostics` lives at the root
and `olmo_eval` under `src`, so `PYTHONPATH=src` alone fails with
`ModuleNotFoundError: No module named 'diagnostics'`.

```bash
# Unix
cd <repo-root> && PYTHONPATH=.:src .venv/bin/python /tmp/preflight.py
```

```powershell
# Windows
cd <repo-root>; $env:PYTHONPATH=".;src"; & .venv\Scripts\python.exe "$env:TEMP\preflight.py"
```

**Do not hardcode the benchmark list.** It changes as banks are vendored. `ready_names()` is
the source of truth; anything absent from it cannot be run today.

If the benchmark is unavailable, the exception message says specifically why — no calibrated
bank exists, or the bank exists but is blocked, or the name is unknown. Relay that reason to
the user verbatim rather than paraphrasing it as "not supported", and offer the ready list as
alternatives.

Then confirm the checkpoint exists. For `s3://`, list the prefix; for a local path, check the
directory and that it contains a `config.json` and weights.

---

## Step 2: run it

```bash
# Unix
cd <repo-root>
PYTHONPATH=.:src .venv/bin/python -m diagnostics.mcq_cat.runner \
  --cat-style uni_mcq \
  --benchmark BENCHMARK \
  --checkpoint s3://BUCKET/PATH/TO/CHECKPOINT \
  --s3-out s3://BUCKET/PATH/FOR/RESULTS
```

```powershell
# Windows
cd <repo-root>; $env:PYTHONPATH=".;src"
& .venv\Scripts\python.exe -m diagnostics.mcq_cat.runner `
  --cat-style uni_mcq --benchmark BENCHMARK `
  --checkpoint s3://BUCKET/PATH/TO/CHECKPOINT `
  --s3-out s3://BUCKET/PATH/FOR/RESULTS
```

Running `-m` from the repo root puts the root on `sys.path` implicitly, but set `PYTHONPATH`
anyway — the task registry that vendoring and some graders reach for needs `src` on it.

`uni_mcq` is currently the only style; confirm with `--list-styles` if unsure.

Leave `--se-threshold` (0.3), `--max-items` (40) and `--batch-size` (16) at their defaults
unless the user asks otherwise. **The first two are pinned to match the reference
implementation, and changing either makes the resulting theta non-comparable to other runs** —
the report records when they depart from the pin, but a user comparing checkpoints will not
notice.

This step requires `torch` and `transformers`, so it runs on a GPU box, not a laptop. Runs
take minutes for a multiple-choice benchmark and considerably longer for a generative one,
since those sample up to 1,024 tokens per item rather than scoring four short continuations.

---

## Step 3: report the result

Read `cat_report.json` from the output location. The fields that matter:

- `ability.theta` — the ability estimate, on a logit scale centred near 0. **This is only
  comparable across runs of the same benchmark**, never across benchmarks, because each
  bank's scale is anchored to its own calibration population.
- `ability.standard_error` — precision of that estimate. At the default threshold a completed
  run lands at or below 0.3.
- `metadata.pirt_accuracy` — predicted accuracy, which is the number to quote to a user who
  wants "how good is it". Report it alongside
  `metadata.pirt_accuracy_denominator`, because it is over the **calibrated subset**, not the
  full benchmark split.
- `metadata.observed_accuracy` — raw fraction correct on the items actually administered. This
  will differ from the predicted accuracy and that is expected, since adaptive testing
  deliberately concentrates on items near the model's ability, where it is near 50/50.
- `metadata.n_items_administered` and `metadata.stop_reason`.
- `metadata.ungradable` — if present with a nonzero `rate`, say so. An `alert` field there
  means the harness could not grade a large share of items and the theta is a floor rather
  than a measurement; that is a broken dependency, not a weak checkpoint.
- `metadata.bank_provenance` — which bank was used and how it was calibrated.

Always surface `metadata.scoring_note` and any caveat in `bank_provenance`. Several banks
were calibrated under a different prompting convention than we administer, which shifts
theta's absolute value while leaving checkpoint-to-checkpoint comparison intact. A user told
only the number will over-read it.

---

## Handling failures

The runner catches everything and exits 1 with a single `mcq_cat.runner failed:` line. Read
that line; it names the cause. Map it as follows.

**Benchmark not in the pipeline** — `DatasetNotAvailable`. Preflight should have caught this.
The message states the specific reason. Give the user that reason and the ready list.

**Checkpoint not found** — `No objects found under s3://...` or `Checkpoint path does not
exist:`. Distinguish the two: the first is an empty or wrong S3 prefix (or missing
credentials), the second a bad local path. Ask the user to confirm the URI, and check whether
they pointed at a parent directory rather than the checkpoint itself.

**Checkpoint present but unloadable** — the failure comes from `transformers`, typically a
missing `config.json`, an unrecognised architecture, or truncated weight shards. Report what
the checkpoint directory actually contains, since a partially-uploaded checkpoint is the
common case and looks identical to a complete one until it is loaded.

**Wrong checkpoint format** — `olmo_core scoring is a training-env integration point`. Only
HuggingFace-format checkpoints work. A raw OLMo-core checkpoint must be converted first;
`--checkpoint-kind olmo_core` is a declared but unimplemented path, so do not suggest it as a
workaround.

**Modality mismatch** — `ModalityMismatch`, naming the dataset, its declared modality and up
to five offending item ids. This means the bank and its grader disagree, which is a repo
inconsistency rather than anything the user did. Do not attempt to work around it by changing
flags; report it as a bug.

**Scoring convention mismatch** — a `DatasetNotAvailable` naming a field, the manifest's value
and the configured one. Same category: the harness is configured differently from how the bank
was vendored. Report it rather than overriding, since the whole point of that check is that
overriding it silently corrupts the ability estimate.

**IFEval specifically** — grading needs two things on the box, not one. The `ifbench` package,
without which items come back ungradable rather than failing outright, so watch
`metadata.ungradable` and treat a high rate as a missing dependency. And NLTK's `punkt_tab`
and `averaged_perceptron_tagger_eng` corpora, which several verifiers download on first use:

```python
import nltk
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
```

That download failing is the nastier of the two, because it surfaces as a `LookupError`
raised mid-grade on whichever item happens to select such a verifier — so it lands *after*
the checkpoint is staged, and only on some runs. Pre-fetch both before a real IFEval run. On
a machine behind a TLS-inspecting proxy the download needs the OS trust store, via
`import truststore; truststore.inject_into_ssl()` before calling `nltk.download`.

In every case, prefer relaying the harness's own message to the user over restating it in your
own words. These messages were written to name the specific cause and the fix; paraphrasing
loses that.

---

## What not to do

- Do not lower `--se-threshold` or raise `--max-items` to "get a better number". A wider cap
  buys precision, not accuracy, and breaks comparability with existing runs.
- Do not run a benchmark absent from `ready_names()` by passing a path to its bank directly.
  The datasets excluded are excluded for recorded reasons, several of which are that the
  resulting number would be meaningless.
- Do not compare theta across benchmarks. Predicted accuracy is the cross-benchmark quantity.
- Do not present a theta from a run whose report carries an `ungradable` alert as a measurement
  of the model.
