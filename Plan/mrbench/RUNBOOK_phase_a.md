# MRBench Phase-A runbook (judge validation)

Push-button operator guide for validating the LLM judge against MRBench human
gold. **Nothing here picks a judge model** — you supply one via
`MRBENCH_JUDGE_MODEL` at run time. Steps run offline until you explicitly pass
`--live`. Costs quoted are from `diagnostics/mrbench/cost.py` at the assumed
Claude Haiku 4.5 rates ($1/MTok in, $5/MTok out — verify before spending).

All commands run from the repo root. Flags/env vars below were verified against
`gold_damr_check.py`, `judge_run.py`, `judge_client.py`, `cost.py`, and
`metrics.py`.

---

## 1. Prerequisites / environment

The judge client (`diagnostics/mrbench/judge_client.py`) reads three knobs from
the environment. The first present value in each group wins.

```bash
# REQUIRED: the judge model id (TrueFoundry provider_account/model_name).
# No usable default — the live run refuses until this is a real id.
export MRBENCH_JUDGE_MODEL='<MRBENCH_JUDGE_MODEL>'

# REQUIRED: gateway API key. Any ONE of these (aliases, first present wins):
export OPENAI_API_KEY='<api-key>'          # or:
# export TFY_API_KEY='<api-key>'
# export TRUEFOUNDRY_API_KEY='<api-key>'

# OPTIONAL: gateway base URL. Any ONE (default: https://gateway.truefoundry.ai):
# export OPENAI_BASE_URL='<gateway-base-url>'
# export TFY_GATEWAY_BASE_URL='<gateway-base-url>'
# export TRUEFOUNDRY_BASE_URL='<gateway-base-url>'
```

**Security:** the API key is **runtime-only**. Never commit it, never bake it
into an image or a checked-in file, and **rotate it after the run**. Export it in
the shell that launches the run; nothing in the repo persists it.

**Data / network:** the dataset is vendored at
`diagnostics/mrbench/data/MRBench_V1.json` (attribution in
`diagnostics/mrbench/data/NOTICE.md`). Steps (a) and (b) below make **zero**
network calls. Only `--live` (steps c/d) contacts the gateway. Judge results are
cached at `diagnostics/mrbench/runs/judge_cache.jsonl` (override with `--cache`);
the cache is idempotent, so re-running skips completed (conversation, tutor,
dimension) calls.

---

## 2. Phase-A flow (offline → cheap pilot → full), gated

### (a) Offline gold-DAMR sanity check — no API

```bash
uv run python -m diagnostics.mrbench.gold_damr_check
```

Confirms parsing + arithmetic against the paper's Table 3. **Expect a `FAIL`
verdict at the default tolerance** — this is the *accepted* data-vs-paper
divergence (see `README.md` → "Known data-vs-paper divergence"): 6/9 tutors
reproduce within ~4 pp; Expert, Gemini, and Novice are caveated. What you are
checking: the command runs, prints the computed table, and the 6 reproducing
tutors are close. Loosen with `--tolerance 4.0` to see the clean split.

### (b) Offline dry-run: prompt fidelity + cost estimate — no API

```bash
# Print the exact Figure-6 system+user prompt for a few responses, plus the
# full-run cost estimate and live-run readiness. No network.
uv run python -m diagnostics.mrbench.judge_run --samples 3

# Cost estimate only (no prompt dump); add --phase-b for the Phase-B estimate:
uv run python -m diagnostics.mrbench.judge_run --samples 0 --phase-b
```

Look for: the printed prompt matches the paper's Figure 6; the readiness block
says whether `MRBENCH_JUDGE_MODEL` + a key are set. **Full-run estimate:
~$10.19** (12,712 judge calls = 1,589 responses × 8 dimensions; band
$9.34–$11.29).

### (c) Cheap live pilot — eyeball fidelity before spending

`--limit N` caps the number of judge **calls**, where one call = one
(response × dimension). So 8 responses = 64 calls.

```bash
# ~$0.05 pilot: 8 responses across all 8 dimensions (64 calls).
uv run python -m diagnostics.mrbench.judge_run --live --limit 64
# (absolute-minimum smoke: --limit 8 = one response's 8 dims, ~$0.01)
```

Then inspect the cache to confirm Figure-6 fidelity and clean parsing:

```bash
# Each line has: raw (the judge's text), score (1-3), label, ok, used_fallback.
uv run python - <<'PY'
import json, pathlib
p = pathlib.Path("diagnostics/mrbench/runs/judge_cache.jsonl")
for line in p.read_text().splitlines()[:8]:
    r = json.loads(line)
    print(r["dimension"], "| score", r["score"], "| ok", r["ok"],
          "| fallback", r["used_fallback"], "|", repr(r["raw"][:120]))
PY
```

Look for: `ok=true`, `used_fallback=false`, a trailing `[RESULT] N`, and labels
that make sense. If parsing is failing, stop and fix before the full spend.

### (d) Full live run — AC-vs-gold + metrics tables

```bash
# Runs all remaining (cached calls are skipped), then prints metrics.
uv run python -m diagnostics.mrbench.judge_run --live --metrics --bootstrap 1000
```

Useful knobs (verified in `judge_run.py`): `--concurrency N` (default 8),
`--cache PATH` (default `diagnostics/mrbench/runs/judge_cache.jsonl`),
`--limit N`. `--metrics` computes per-dimension Pearson **AC** (judge vs gold),
judge-derived DAMR, per-tutor×dimension tables, and diffs vs the paper's
Prometheus2 / Llama-3.1-8B baselines; `--bootstrap 1000` adds 95% CIs. Self-bias
is **N/A** for the Claude Haiku 4.5 judge — it is not one of the MRBench tutors,
so there is no self-authored subset to surface separately.

You can also recompute metrics from an existing cache without any network:

```bash
uv run python -m diagnostics.mrbench.judge_run --metrics --bootstrap 1000
```

### (e) Apply the DECIDED acceptance gate

From `README.md` (§Phase A acceptance gate — DECIDED):

- **Per-dimension pass:** AC ≥ **0.30** **AND** the 95% CI lower bound > **0**.
- **Judge validated if:** passes on **≥ 6/8** dimensions **AND** beats **both**
  paper baselines (Prometheus2 and Llama-3.1-8B) on **≥ 7/8** dimensions.
- **Weak dimensions** are **caveated, not fatal** (don't fail the whole judge).
- **Human-likeness** is **reported but excluded** from the gate.
- **Self-bias:** N/A for the Claude Haiku 4.5 judge (not one of the MRBench
  tutors). The report-both-and-exclude-self rule only applies when the judge model
  is itself one of the tutors.
- Thresholds may be relaxed later if warranted.

---

## 3. Optional secondary variants (opt-in, OFF by default)

These are **secondary columns** for analysis. They do **not** replace the
byte-faithful Figure-6 zero-shot judge, which stays the headline/comparability
number. Verified flags/env:

- **Macro-F1 (BEA-comparable secondary):** add `--macro-f1` to a `--metrics` run.

  ```bash
  uv run python -m diagnostics.mrbench.judge_run --metrics --macro-f1
  ```

- **Reference-guided judging** (injects `Ground_Truth_Solution` for the four
  correctness-linked dimensions only): flag `--reference-guided`, or env
  `MRBENCH_JUDGE_REFERENCE_GUIDED=1`.

  ```bash
  uv run python -m diagnostics.mrbench.judge_run --live --reference-guided --limit 64
  ```

- **Self-consistency (k-sample majority vote):** flag `--k N` (standalone) or env
  `MRBENCH_JUDGE_SAMPLES=N` (both paths). Default `k=1` = paper single-sample.
  Cost scales ~`k×` on the judge side (the estimate reflects it).

  ```bash
  uv run python -m diagnostics.mrbench.judge_run --live --k 3 --limit 64
  ```

Combine as needed; none is on by default anywhere.

---

## 4. Phase B (tutor under test) — quick reference

Phase B scores a **model under test** as a tutor, then judges its generations.
It runs through the native olmo-eval task and needs the **same judge env** as
Phase A (`MRBENCH_JUDGE_MODEL` + a key).

```bash
export MRBENCH_JUDGE_MODEL='<MRBENCH_JUDGE_MODEL>'
export OPENAI_API_KEY='<api-key>'
uv run olmo-eval run -m <model-under-test> -t mrbench -O <output-dir>
# optional opt-ins: MRBENCH_JUDGE_SAMPLES=k, MRBENCH_JUDGE_REFERENCE_GUIDED=1
```

- **Cost ≈ $1.55 / model** (192 generation calls + 192×8 = 1,536 judge calls at
  k=1; ~$0.30 generation at a generic $3/$15 model + ~$1.24 judge at Claude Haiku
  4.5 $1/$5 rates).
- **Reports DAMR only** — the primary `damr` aggregate plus 8 `damr_<Dim>`. There
  is **no** AC or macro-F1 in Phase B, because there is no human gold for the
  model under test (those are judge-vs-gold metrics, Phase A only).

---

## 5. Cost-gate + go/no-go checklist (tick before the full spend)

- [ ] Judge model chosen and exported in `MRBENCH_JUDGE_MODEL` (a real id, not the
      placeholder).
- [ ] API key exported (`OPENAI_API_KEY` / `TFY_API_KEY` / `TRUEFOUNDRY_API_KEY`);
      it is runtime-only and will be rotated after.
- [ ] Base URL correct (default gateway, or override exported).
- [ ] (a) Gold-DAMR sanity check ran; divergence understood (FAIL vs Table 3 is
      expected/accepted).
- [ ] (b) Dry-run prompt matches Figure 6; cost estimate reviewed (**~$10.19**
      full Phase A).
- [ ] (c) ~$0.15 pilot (`--limit 64`) ran; cache shows `ok=true`,
      `used_fallback=false`, clean `[RESULT] N`.
- [ ] Assumed pricing re-checked against current provider rates.
- [ ] Budget approved for the full run (and per-model **~$1.55** if Phase B follows).
- [ ] `--concurrency` set appropriately for gateway rate limits.
- [ ] GO / NO-GO: _______

---

*Judge stays unpicked in the repo; every model reference above is a runtime
placeholder. See `README.md` for the full design, divergence note, and the
elicitation/judging strategy backlog.*
