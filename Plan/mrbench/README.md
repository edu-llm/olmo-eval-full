# MRBench Evaluation — Plan / Spec

Status: **Phase A: VALIDATED (judge = Claude Haiku 4.5)** — full live run over all
12,712 judge calls completed 2026-08-09; the judge passes the decided acceptance
gate. See **[`PHASE_A_VALIDATION.md`](./PHASE_A_VALIDATION.md)** for the verdict,
tables, and caveats. (M0 accepted with caveats; M1 harness + M2 metrics
implemented and now exercised end-to-end.) Branch: `frq/mrbench`.

> **Validation record:** **[`PHASE_A_VALIDATION.md`](./PHASE_A_VALIDATION.md)** —
> the committable Phase-A summary (VALIDATED verdict, per-dimension AC + CIs +
> gate, baseline comparison, macro-F1, caveats, cost). Full machine-readable
> metrics + provenance live under the (gitignored) `diagnostics/mrbench/runs/`.
>
> **Operator runbook:** to re-run or run for another judge, follow
> **[`RUNBOOK_phase_a.md`](./RUNBOOK_phase_a.md)** — a copy-pasteable,
> offline→pilot→full gated sequence (with the Phase-B quick reference and the
> go/no-go cost checklist).
>
> **Phase B on eduLLM:** **[`PHASE_B_EDULLM_PLAN.md`](./PHASE_B_EDULLM_PLAN.md)** —
> the AGENTS.md-compliant plan to run a model-under-test (e.g.
> `allenai/OLMoE-1B-7B-0125-Instruct`) on the `mrbench` task via `edullm`
> (generate-on-cluster / judge-locally split; `edullm` blockers documented).

## Purpose

Evaluate **MRBench** directly — **not** through CAT / adaptive testing. MRBench is a
fixed, fully-annotated benchmark, so there is no item selection and no ability
model; the work is data parsing, a per-response LLM judge, and correlation
arithmetic.

Two phases, run in order:

- **Phase A — validate our judge.** Score MRBench tutor responses with our own
  judge (Claude Haiku 4.5, served via the **TrueFoundry AI Gateway**) and measure how
  well the judge agrees with the human gold labels. This tells us whether the
  judge is trustworthy before we rely on it for anything new.
- **Phase B — score a tutor (later).** Only once Phase A shows the judge is
  reliable, use that validated judge to score a *tutor model's* generations on
  the same protocol.

**Phase A gates Phase B.** If the judge does not correlate acceptably with gold,
we do not proceed to Phase B.

### What MRBench contains (Phase A vs Phase B) — factual finding

**MRBench already contains the tutor responses that Phase A scores.** Every
`anno_llm_responses[tutor]` entry carries both a `response` string and the human
`annotation` (verified: 0 of 1,589 tutor entries lack `response`). The nine
tutors (GPT-4, Gemini, Sonnet, Mistral, Llama-3.1-8B, Llama-3.1-405B, Phi3,
Expert, Novice) are baked in. So **Phase A generates nothing** — it re-scores
these pre-contained responses with our judge and correlates against gold.

**Phase B does require generation.** MRBench does not contain responses for an
arbitrary *model under test*, so scoring a new tutor means generating its
response for each conversation, then judging it. Generation uses the paper's
Figure-2 tutor prompt (below); judging reuses the Phase-A path unchanged.

## Dataset

- **MRBench V1**: 192 conversations, 1,596 responses (per paper), 8 dimensions.
- Source: `github.com/kaushal0494/UnifyingAITutorEvaluation`, file
  `MRBench/MRBench_V1.json`.
- License: **CC BY-SA 4.0** — requires **attribution** and **share-alike**
  (derivatives distributed under the same license). Any redistributed artifact
  built from MRBench must carry this notice. Full attribution, license link,
  required-credit text, citation, and the upstream MathDial/Bridge lineage are
  recorded in **`diagnostics/mrbench/data/NOTICE.md`** (license verified against
  the official repo's README License section).
- Paper: Maurya et al., "Unifying AI Tutor Evaluation: An Evaluation Taxonomy for
  Pedagogical Ability Assessment of LLM-Powered AI Tutors", **NAACL 2025**
  (`aclanthology.org/2025.naacl-long.57/`).

### Schema

Each conversation carries `conversation_history`, `Data` (`MathDial` | `Bridge`),
`Split`, `Topic`, `Ground_Truth_Solution`, and `anno_llm_responses`. Each entry of
`anno_llm_responses[model]` has a `response` string and an `annotation` object
with the eight dimension labels.

### The 8 dimensions, label sets, and desired labels

DAMR (below) counts only the **exact desired label**; "To some extent" never
counts.

| Dimension | Label set | Desired |
|---|---|---|
| Mistake_Identification | Yes / To some extent / No | **Yes** |
| Mistake_Location | Yes / To some extent / No | **Yes** |
| Revealing_of_the_Answer | Yes (correct) / Yes (incorrect) / No | **No** |
| Providing_Guidance | Yes / To some extent / No | **Yes** |
| Actionability | Yes / To some extent / No | **Yes** |
| Coherence | Yes / To some extent / No | **Yes** |
| Tutor_Tone | Encouraging / Neutral / Offensive | **Encouraging** |
| Humanlikeness | Yes / To some extent / No | **Yes** |

Note: the V1 file stores the eighth dimension under the lowercase key
`humanlikeness` (schema/Table 3 spell it `Humanlikeness`); the parser resolves
this case-insensitively.

### Tutors

Nine tutors: seven LLM-based (GPT-4, Gemini, Sonnet, Mistral, Llama-3.1-8B,
Llama-3.1-405B, Phi3) plus two human tutors (Expert, Novice). The paper's tutor
names differ from the JSON keys; the faithful 1:1 mapping is:

| Paper name | JSON key |
|---|---|
| GPT-4 | `GPT4` |
| Gemini | `Gemini` |
| Sonnet | `Sonnet` |
| Mistral | `Mistral` |
| Llama-3.1-8B | `Llama318B` |
| Llama-3.1-405B | `Llama31405B` |
| Phi3 | `Phi3` |
| Expert | `Expert` |
| Novice | `Novice` |

**Novice** is scored only on the Bridge dialogues; all other tutors on all 192.

## Metrics

- **DAMR (Desired Annotation Match Rate)** — *tutor quality.* Per (tutor,
  dimension): the percentage of that tutor's responses whose **gold** label
  equals the desired label. This is a property of the human annotations and needs
  no judge (used by Milestone 0).
- **AC (Agreement / Correlation)** — *judge reliability.* Per dimension, the
  **Pearson correlation** between the judge's scores and the gold labels across
  responses. This is the Phase-A validation number.

## Judge protocol (Phase A / M1)

Follow the paper's **Figure 6** (Appendix E) byte-faithfully: per-dimension
**absolute** scoring with a system + user prompt, and the judge emits
`Feedback: ... [RESULT] N` where `N ∈ {1, 2, 3}` (mapped to the dimension's
three-way label via the Figure-6 rubric).

- Judge = **Claude Haiku 4.5**, **temperature 0**, one call per (response, dimension)
  → **8 calls per response**. Our V1 file yields 1,589 responses → **12,712**
  calls (the paper's 1,596 would be 12,768).
- **Prompt fidelity.** The exact Figure-6 system text, user template,
  per-dimension definitions, and scoring rubrics live in `judge_prompt.py`,
  copied verbatim from the paper. One flagged decision and one known quirk:
  - The paper's `### Assistant:` generation cue (`# Generate Assessment Score:`)
    is folded onto the end of the **user** message, since OpenAI-style chat has
    no separate assistant-priming slot. System→`system`, User→`user`.
  - The `Revealing_of_the_Answer` "Score 1" line is missing its closing paren in
    the paper PDF; it is reproduced verbatim (see `KNOWN_PAPER_QUIRKS`). Curly
    apostrophes are normalised to ASCII. **Confirm** whether to keep the paper's
    typo or correct it.

### Inference routing — TrueFoundry AI Gateway

All inference is routed through the **TrueFoundry AI Gateway**, which is
**OpenAI-compatible**, so we use the `openai` SDK pointed at the gateway (not any
provider-native SDK). Three config knobs, all from env (never hardcoded):

| Knob | Env vars (first match wins) | Default |
|---|---|---|
| Gateway base URL | `OPENAI_BASE_URL`, `TFY_GATEWAY_BASE_URL`, `TRUEFOUNDRY_BASE_URL` | `https://gateway.truefoundry.ai` |
| API key (TFY PAT/VAT) | `OPENAI_API_KEY`, `TFY_API_KEY`, `TRUEFOUNDRY_API_KEY` | — (required for live) |
| Model id | `MRBENCH_JUDGE_MODEL` | placeholder — **user must supply** |

The model id is TrueFoundry's `provider_account/model_name` form (e.g.
`claude-group/claude-haiku-4-5`); the exact string depends on the user's
account and must be copied from their **Playground → "Code Snippet"**. There is
no working hardcoded default — the live path refuses to run on the placeholder.

- **Self-bias caveat: N/A.** Claude Haiku 4.5 (the judge) is **not** one of the
  nine MRBench tutors, so there is no self-authored subset to exclude. (The
  `Sonnet` tutor in the data is unrelated to the judge model.)

## Milestones

- **M0 — parse-first gold-DAMR sanity check (no judge).** *Implemented.* Parse
  MRBench V1, compute per-(tutor × dimension) gold DAMR, and diff it against the
  paper's Table 3 with a PASS/FAIL verdict. Pure data + arithmetic; no API, no
  network to any LLM. Entry point:

  ```bash
  uv run python -m diagnostics.mrbench.gold_damr_check
  ```

- **M1 — Claude Haiku 4.5 judge run.** *Built; not yet run live.* The Figure-6 judge over
  all responses, persisting per-response, per-dimension judge labels to an
  idempotent JSONL cache (skip-if-done), with concurrency + rate-limit backoff.
  Safe by default: no flags → dry-run (prints sample prompts + cost estimate,
  zero network). Live run is gated on `--live` **and** a TFY key in env **and** a
  real model id.

  ```bash
  uv run python -m diagnostics.mrbench.judge_run              # dry-run + cost
  uv run python -m diagnostics.mrbench.judge_run --validate   # offline self-test
  uv run python -m diagnostics.mrbench.judge_run --live --limit 8   # gated pilot
  ```

- **M2 — AC + judge-DAMR + baseline comparison.** *Implemented; pending M1 data.*
  Per-dimension AC (Pearson judge-vs-gold) overall and per tutor, a judge-derived
  DAMR (reusing the M0 desired-label logic), and per-cell comparison against the
  paper's Prometheus2 (Table 5) and Llama-3.1-8B (Table 6) baselines. (Self-bias
  is N/A: the Claude Haiku 4.5 judge is not one of the MRBench tutors.)

  ```bash
  uv run python -m diagnostics.mrbench.judge_run --metrics    # reads the cache
  ```

## Known data-vs-paper divergence (M0 result) — ACCEPTED

Running M0 against the official `MRBench_V1.json` **does NOT fully reproduce
Table 3** at ±1 pp (15/72 cells within tolerance, max diff 75.00 pp). The split
is clean at ±4 pp — 6 tutors reproduce, 3 do not:

- **Reproduce within ~4 pp** (parsing/arithmetic validated): GPT-4, Sonnet,
  Mistral, Llama-3.1-8B, Llama-3.1-405B, Phi3. Residual ~1–4 pp is consistent
  with a slightly different underlying conversation set.
- **Expert**: matches Table 3 on 7/8 dimensions (~+6 pp) but **Tutor_Tone is
  inverted** — computed 17.19% Encouraging vs paper 92.19%.
- **Gemini**: uniformly **~+20–27 pp** high on every dimension — no match.
- **Novice**: scored on **53** Bridge dialogues, not the paper's **60**; tone off
  by ~−35 pp.
- Published V1 split is **139 MathDial / 53 Bridge (1,589 responses)**; the paper
  reports **132 / 60 (1,596)**.

No faithful tutor-key remapping fixes Expert/Gemini (verified by exhaustive
pairwise matching; the same pattern holds in `MRBench_V2.json`, and V1 has a
single upstream commit, so it was not silently revised). Conclusion: the
published JSON differs from the paper's frozen Table-3 snapshot.

**Decision (user, accepted):** use `MRBench_V1.json` **as-is**; treat **Expert,
Gemini, and Novice** as **caveated** for strict paper-comparability. The judge
validation (Phase A) still runs on all tutors; interpret those three rows with
this caveat. (Machine-readable copy: `reference.TABLE3_DIVERGENCE`.)

## Milestone 1 — dry-run + cost (no live calls yet)

- **Prompt fidelity:** dry-run prints the exact Figure-6 `system` + `user`
  messages per (response, dimension); verified against the paper.
- **Offline validation:** parser (`[RESULT] N`, fallback, parse-failure) and
  metrics (Pearson +1/−1/nan, judge-DAMR, parse-failure exclusion) — 12/12 PASS.
- **Cost estimate (full run):** 12,712 calls; ~7.65M input tokens (measured from
  real prompts, ~2,408 chars/call, 4 chars/token) + ~0.51M output tokens
  (assumed 40/call). At assumed Claude Haiku 4.5 rates **$1 / MTok input, $5 / MTok
  output** (TrueFoundry passes provider pricing through): **≈ $10.2**, range
  **$9–$11**. Verify rates against current pricing before spending. (The actual
  validated live run came in at **≈ $12.10**; see `PHASE_A_VALIDATION.md`.)

To run live we need: the gateway base URL (default is fine for SaaS), a
**TrueFoundry API key** in `OPENAI_API_KEY`/`TFY_API_KEY`/`TRUEFOUNDRY_API_KEY`,
and the **Claude Haiku 4.5 model id** (`claude-group/claude-haiku-4-5`) in
`MRBENCH_JUDGE_MODEL` (from the Playground Code Snippet), plus explicit go-ahead.

## Phase A — acceptance criteria (judge validation)

Phase A asks: *is our judge reliable enough to trust in Phase B?* The metric is
per-dimension Pearson **AC** (judge score 1–3 vs gold score 1–3), computed over
all responses, with 95% bootstrap CIs (`--bootstrap`).

Context: the paper's own critics (Prometheus2, Llama-3.1-8B; `baselines.py`) are
**mostly negatively correlated** with gold — every dimension except
Human-likeness is ≤ 0 for both. So "beat the baseline" is a near-trivial bar; we
require *positive correlation with gold in absolute terms*, not merely beating a
weak baseline.

**Gate — DECIDED (user, accepted):**

- A **dimension passes** if AC ≥ **0.30** *and* its 95% bootstrap-CI lower bound
  > **0** (significantly positive, not just a point estimate).
- The **judge is "validated"** if it passes on **≥ 6 of 8** dimensions *and*
  strictly exceeds *both* paper baselines on **≥ 7 of 8** dimensions (a sanity
  floor, easily cleared given the baselines are negative).
- **Weak dimensions** (fail the 0.30/CI bar) are **not fatal** (DECIDED): the
  judge is still usable, but those dimensions are marked **judge-unreliable** and
  reported with a caveat in Phase B rather than driving decisions.
- **Human-likeness** is special-cased (DECIDED): gold is near-constant (mostly
  "Yes"), so Pearson is unstable/NaN. Report it, but do not let it fail the gate.
- **Self-bias — DECIDED (report-both):** if the judge model is itself one of the
  MRBench tutors, compute AC both **including and excluding** the judge's
  own-authored tutor subset; if any per-dimension delta > **0.10**, use the
  **excluding-self** number for the gate. **N/A for the current judge:** Claude
  Haiku 4.5 is not one of the nine tutors, so there is no self-authored subset.

*Note:* these thresholds (0.30, 6/8, 7/8, 0.10) are the accepted defaults and
**may be relaxed later if warranted.** They live nowhere in code yet — applied by
reading the `--metrics --bootstrap` output.

## Phase B — tutor-generation spec

Goal: score a **model under test** as a tutor on MRBench and report its DAMR,
using the *validated* judge from Phase A. Judge-agnostic throughout: the tutor
model and the judge model are independent and both configured via the same
env-driven client (`judge_client.py`); nothing hardcodes either.

**Step 1 — generate.** For each of the 192 conversations, prompt the model under
test with the paper's **Figure-2** template (Appendix B), byte-faithful:

- *Bridge* (elementary): system = "You are an experienced elementary school math
  teacher, and you are going to respond to a student's mistake in a useful and
  caring way"; user = `The problem your student is solving is on the topic:
  {topic}\nConversation History: {history}` + the one-sentence response cue.
- *MathDial* (middle school): system = "...experienced middle school math
  teacher..."; user = `The conversation history of the problem your student is
  solving is: {history}` + the response cue. (No `{topic}`; MathDial has none.)
- The `### Assistant:` cue ("Tutor response (maximum one sentence ...)") is folded
  onto the user turn, same convention as the judge. The MathDial cue's "hisotry"
  is a verbatim paper typo (flagged). These templates currently live as constants
  in `cost.py` for sizing; when Phase B is implemented they move to a dedicated
  `tutor_prompt.py` mirroring `judge_prompt.py`.

**Step 2 — judge.** Feed each generated response through the exact Phase-A path:
`judge_prompt.build_messages` → `judge_client.score_messages` → `parse.parse_result`,
one call per (response, dimension), cached in the same JSONL cache
(`judge_cache.py`) keyed by `(conversation_id, "<model-under-test>", dimension)`.

**Step 3 — DAMR on the tutor.** Reuse the existing arithmetic, no new formula:

- Map each judged label to desired via `reference.DESIRED_LABELS` — identical to
  `gold_damr.py`'s rule (exact desired label only; "To some extent" never counts).
- Per-dimension DAMR and aggregate come from `metrics.compute_judge_damr`
  (already used for the pre-contained tutors); the model under test is just
  another `tutor` key. 95% CIs via `bootstrap_ci_rate`.
- Present the tutor-under-test as an extra row alongside the paper's Table-3
  tutors for context (with the judge-reliability caveat per dimension).

**Cost (measured, judge-agnostic).** `estimate_phase_b_cost` (in `cost.py`,
surfaced via `judge_run.py --phase-b`) counts `192` generation calls + `192 × 8 =
1,536` judge calls, sizes generation prompts from the real Figure-2 templates and
judge prompts from an assumed response length (default = mean V1 response ≈ 174
chars), and prices generation and judge **independently** (`--gen-rate-*`,
`--judge-rate-*`). At assumed Claude Haiku 4.5 judge rates ($1/$5 per MTok) and a
generic $3/$15 generation model it is **≈ $1.55 per model under test** (gen ≈
$0.30, judge ≈ $1.24). No live calls to produce this.

## Reporting & artifacts

Tables to emit (all already computed by the code, or trivially derived):

| Table | Source | Mirrors |
|---|---|---|
| Gold DAMR (tutor × dim) + Table-3 diff | `gold_damr_check.py` | paper Table 3 |
| AC judge-vs-gold, per dim (overall) + 95% CI | `judge_run --metrics --bootstrap` | — |
| AC judge-vs-gold, tutor × dim | `judge_run --metrics` | paper Tables 5/6 |
| AC diff vs Prometheus2 / Llama-3.1-8B | `compare_ac_to_baselines` | Tables 5/6 |
| Judge-derived DAMR, tutor × dim (+ CI) | `compute_judge_damr` | paper Table 3 |
| Phase B: model-under-test DAMR per dim + aggregate | Phase B (above) | Table 3 row |

**Where results land — DECIDED (default for now):** use the per-run layout
below, consistent with the `eval-checkpoints` skill's `<root>/<run-id>/...` +
provenance convention and the existing cache path:

```
diagnostics/mrbench/runs/<run_id>/
  judge_cache.jsonl      # raw per-call judge results (already written here)
  metrics.json           # AC + judge-DAMR + baseline diffs + CIs
  tables/*.csv|*.md       # the tables above
  run_provenance.json    # judge model id, gateway, temp, git sha, data hash
```

with `run_id = {phase}_{judge_model_slug}_{YYYYMMDD}`.

*Note:* the referenced `tests/aws/P3_results_namespace.md` lives on the
**AWS-checkpoint branch** and is **absent on this `frq/mrbench` branch**, so it is
not used here. If MRBench later merges toward the checkpoint driver, the two
schemes should be reconciled (likely adding an S3 mirror like the eval-checkpoints
`s3://.../evals/<exp>/`).

## Statistical rigor — DECIDED

- **Single sample, temperature 0** (DECIDED). Matches the paper, deterministic,
  cache-friendly, and avoids multiplying cost by *k* for negligible gain at temp
  0. Revisit only if we later move to a nonzero-temperature judge.
- **Bootstrap CIs: on, N = 1000** (DECIDED) — cheap and non-invasive, already
  implemented (`bootstrap_ci_pearson`, `bootstrap_ci_rate`; `--bootstrap N`,
  default off, run with `--bootstrap 1000`). Used to (a) decide "AC significantly
  > 0" for the Phase-A gate and (b) put error bars on DAMR.
- **Correlation choice: Pearson only** (DECIDED) on the 1–3 ordinal, for direct
  comparability with the paper's Tables 5/6. Spearman intentionally omitted.

## Elicitation & judging strategies (from the literature)

Research-only synthesis of how people get the *best* results with MRBench, split
into **generation** (eliciting a good tutor response) and **judging** (getting
judge scores that match human gold). Tags: **[V]** = verified from a source we
read; **[U]** = unverified/inferred.

### Angle 1 — Generation (tutor response elicitation)

- **[V] The paper's Figure-2 prompt is zero-shot with a role/persona.** System =
  "You are an experienced elementary/middle school math teacher … respond to a
  student's mistake in a useful and caring way"; user = topic (Bridge only) +
  conversation history; assistant cue = "Tutor response (maximum one sentence
  …)". **No exemplars, no chain-of-thought, no self-refine.** The one hard
  constraint is **max one sentence**. (paper Fig 2 / App B; template adapted from
  Wang et al. 2024a, Bridge.)
- **[V] The paper specifies no temperature/sampling** for generation (or for
  judging) anywhere in the text — the word never appears. So "match the paper"
  means only: this exact prompt, one sentence, single response. Decoding is
  underspecified; temp 0 is a reasonable deterministic default.
- **[V] The BEA-2025 shared task deliberately did *not* re-open generation.** It
  reused the *same* organizer-generated responses and focused entirely on
  *assessment*, i.e. all the community effort below is on the judging side, not on
  better tutor prompts. (BEA-2025 Findings §2.)
- **[U] No follow-up we found reports a "better" MRBench tutor-generation prompt**
  (few-shot / CoT / persona-tuning) benchmarked against the Figure-2 zero-shot on
  MRBench's own DAMR. Prompt-engineering for *tutoring* generation exists broadly
  (e.g. Mollick & Mollick persona prompts; NAISTeacher's prompt+rerank in BEA-2023)
  but is not validated on MRBench. Treat any deviation from Fig-2 as off-protocol.

### Angle 2 — Judging (matching human gold)

- **[V] The paper's judge (Figure 6) is Prometheus2-style absolute, per-dimension,
  reference-free, zero-shot, single-sample.** One call per (response, dimension)
  with a dimension-specific definition + 3-point rubric, emitting
  `Feedback … [RESULT] N`. It is **reference-free** (no gold answer in the prompt;
  the paper notes human-likeness in Prometheus2 wanted gold references "which …
  were unavailable in our case"). No few-shot exemplars, no self-consistency.
- **[V] Off-the-shelf critics correlate poorly.** Prometheus2 and Llama-3.1-8B are
  mostly **negative** Pearson AC on all dimensions except human-likeness (paper
  Tables 5/6). The paper proposes **no fix** — it explicitly leaves "extensive
  prompt engineering and experimentation with other LLMs as critics" to future
  work (§6 Reliability + Limitations).
- **[V] That future work is the BEA-2025 shared task** (Kochmar et al. 2025;
  arXiv 2507.10579), which turns 4 of the 8 dimensions (Mistake Identification,
  Mistake Location, Providing Guidance, Actionability) + a tutor-ID track into a
  3-class prediction problem scored by **macro-F1** (not Pearson AC), on a larger
  labeled dev set (~2,476 responses). 50+ teams. Best exact macro-F1:
  **Mistake Id 0.7181, Mistake Loc 0.5983, Guidance 0.5834, Actionability 0.7085**
  vs a majority-class baseline of **~0.22–0.29** — i.e. huge, *learnable* gains
  over the paper's negative-AC critics. What worked, roughly in order of impact:
  - **[V] Fine-tuning a small open critic (LoRA/SFT) on the labeled data was the
    dominant winning strategy.** Examples: MSA (Mathstral-7B + LoRA, won Guidance),
    bea-jh (GLM-4-9B + GRPO with explicit rationales, won Actionability), BLCU-ICALL
    (Qwen2.5-32B SFT + ICL + RLHF, won Mistake Location), TBA (per-dimension
    FLAN-T5-XL merged via DARE-TIES). Fine-tuned ~3–32B models routinely beat
    zero-shot frontier LLMs.
  - **[V] Strong zero-shot prompting can still win a track**: BJTU won Mistake
    Identification (0.7181) with **zero-shot** GPT-class prompting + task-aware
    prompt refinement + class-imbalance handling — *no exemplars*. So few-shot is
    not required for the "easy" dimensions.
  - **[V] Few-shot / in-context learning helps but was rarely the single best
    lever.** NeuralNexus's retrieval-augmented few-shot (top-k similar dev
    examples + GPT-4o + schema-constrained output) beat their own non-few-shot
    baselines and hit F1 0.584 on Mistake Id, but ranked mid-pack (37th); BLCU-ICALL
    combined few-shot ICL *with* SFT+RLHF to win Mistake Location. Retrieved
    (semantically similar) exemplars > static exemplars.
  - **[V] Reasoning/CoT and rubric decomposition help the harder dimensions.**
    K-NLPers (GPT-4.1, "multi-perspective reflective" CoT; and decomposing Guidance
    into sub-questions feeding a classifier); SG (Gemma-3-27B two-step: first
    derive the correct solution, then rate the response against it); bea-jh's
    reasoning-with-rationales. These target Mistake Location / Guidance, the
    lowest-scoring dimensions.
  - **[V] Class-imbalance handling is essential**: "To some extent" is heavily
    under-represented, so synthetic augmentation, over/under-sampling, and
    class-weighted loss recurred across top teams.
  - **[V] Ensembling / self-consistency helps**: majority voting, stacking, and
    "disagreement-aware" inference (MSA) improved minority-label coverage.
- **[U] Exact AC lift vs the paper's protocol is not directly reported.** BEA uses
  macro-F1 3-class, the paper uses Pearson AC; the two are not a clean 1:1. The
  qualitative conclusion is solid (a fine-tuned or well-prompted critic goes from
  negative AC / ~0.28 F1 to ~0.58–0.72 F1), but a same-metric AC delta would have
  to be measured on our own run.

### Recommendations for OUR setup

Two buckets: **cheap/safe & paper-faithful** vs **higher-accuracy but off-protocol**
(deviating from Fig-2/Fig-6 hurts comparability to the paper's Tables 3/5/6).

**Phase A (maximize judge↔gold AC):**
- **[safe, on-protocol]** Keep the byte-faithful Figure-6 zero-shot, per-dimension,
  reference-free, temp-0 single-sample judge as the *primary/comparability* run —
  this is what our harness already does and what maps to Tables 5/6.
- **[safe, cheap, mostly on-protocol]** Use a **strong modern judge** (Claude
  Haiku 4.5): the paper itself invites "more powerful LLMs as critics", so swapping
  the *model* (not the prompt) is the single most defensible accuracy lever and
  stays close to protocol. This is our current plan.
- **[cheap add-on, mildly off-protocol]** Add a **reference-guided** variant that
  puts `Ground_Truth_Solution` (present in the V1 schema) into the judge prompt for
  the correctness-linked dimensions (Mistake Id/Location, Guidance, Revealing).
  Report it as a *secondary* column, not the comparability number.
- **[medium, off-protocol]** **Few-shot / retrieval-augmented judging**: prepend a
  few gold-labeled exemplars (ideally retrieved to be similar) per dimension.
  Community evidence says this helps, especially for the ambiguous middle class.
  Off-protocol → secondary column.
- **[medium, off-protocol]** **Self-consistency**: sample k judgments at temp>0 and
  majority-vote per (response, dimension). Our stats note currently says single
  temp-0 sample; revisit only if a nonzero-temp judge is adopted.
- **[expensive, most off-protocol, highest ceiling]** **Fine-tune / GRPO a small
  open critic** on the BEA labeled data — the demonstrated top strategy. Biggest
  accuracy ceiling but the largest departure from the paper and a real build cost;
  only if judge accuracy becomes the bottleneck.
- Prioritization: strong-model zero-shot (already) → reference-guided secondary →
  few-shot/retrieval secondary → (only if needed) fine-tuned critic.

**Phase B (elicit the tutor under test):**
- **[safe, on-protocol]** Use the **exact Figure-2 zero-shot persona prompt** with
  the one-sentence constraint, temp 0. This is required for comparability to Table 3
  and is what `cost.py` already encodes.
- **[caution]** Any "better tutor" prompt engineering (few-shot exemplars of good
  remediations, CoT-then-summarize, self-refine, persona tuning) may raise DAMR but
  **breaks comparability** with the paper's Table-3 tutors, who were all elicited
  zero-shot. If explored, run it as a clearly-labeled *extra* condition alongside
  the faithful Figure-2 baseline — never as the headline number.
- **[note]** The one-sentence cap materially shapes several dimensions (guidance,
  actionability); relaxing it is the most tempting and most comparability-breaking
  change. Keep it for the primary run.

**Sources read:** paper PDF (`data/_paper_refs/paper.pdf`; Figs 2 & 6, Tables
3/5/6, §5–6, Limitations); BEA-2025 Findings (arXiv 2507.10579); NeuralNexus
system paper (2025.bea-1.100); Archaeology system paper (2025.bea-1.98);
GitHub `kaushal0494/UnifyingAITutorEvaluation`.

### Backlog — status

Every strategy above, tagged `[IMPLEMENTED-NOW]` (built, opt-in, OFF by default),
`[DEFERRED]` (understood, not built yet), or `[BLOCKED-on-judge]` (waiting on the
judge-model decision). The byte-faithful Figure-6 zero-shot judge and the
Figure-2 zero-shot tutor remain the untouched headline/comparability path — none
of the implemented items is on by default.

- **Strong modern judge model (model swap, not prompt)** — `[BLOCKED-on-judge]`.
  Already the plan; the harness is judge-agnostic and no model is picked here.
- **Reference-guided judging (inject `Ground_Truth_Solution`)** —
  `[IMPLEMENTED-NOW]`. Opt-in via `MRBENCH_JUDGE_REFERENCE_GUIDED` (or the
  `reference_guided=` arg to `judge_prompt.build_messages`); default OFF, default
  output byte-identical to Figure 6. Injects the solution for the four
  correctness-linked dimensions only — Mistake Identification, Mistake Location,
  Providing Guidance, Revealing of the Answer (`REFERENCE_GUIDED_DIMENSIONS`);
  tone/coherence/human-likeness are style judgements the solution does not inform,
  and Actionability is deliberately excluded to match the secondary set documented
  above. Reported as a secondary column, never the comparability number.
- **Self-consistency / k-sample majority vote** — `[IMPLEMENTED-NOW]`. Opt-in
  `k` (`--k` in `judge_run.py`, `MRBENCH_JUDGE_SAMPLES` for the olmo-eval task);
  default `k=1` reproduces the paper's single-sample behaviour exactly. Majority
  vote over parsed `[RESULT] N` with a deterministic tie-break (smallest score);
  per-sample raw outputs are recorded. `k×` cost is reflected in `cost.py`.
- **Macro-F1 secondary metric (BEA-comparable)** — `[IMPLEMENTED-NOW]`.
  Per-dimension 3-class macro-F1 of judge-vs-gold in `metrics.py`
  (`compute_macro_f1`), surfaced by `judge_run.py --metrics --macro-f1` as
  SECONDARY reporting only. It does **not** touch the primary DAMR metric or the
  Pearson-AC path. In the olmo-eval Phase-B task there is no human gold for the
  model-under-test, so macro-F1 is not applicable there; it would only surface as
  a secondary metric on a gold-labelled Phase-A cache, never displacing
  `primary=damr`.
- **Retrieval-augmented few-shot judging** — `[DEFERRED]`. Needs an
  embedding/similarity index over the labelled dev set; larger build, not started.
- **Static few-shot judging exemplars** — `[DEFERRED]`. Cheap to add later
  (prepend a few fixed gold-labelled exemplars per dimension); skipped now to
  avoid scope creep and because retrieved exemplars beat static ones.
- **CoT / rubric-decomposition judging for hard dimensions** — `[DEFERRED]`.
  Targets Mistake Location / Guidance; a prompt/protocol change best evaluated
  once a judge model is fixed.
- **Class-imbalance handling + judge ensembling** — `[DEFERRED]`. Over/under-
  sampling, class-weighting, stacking/disagreement-aware inference — most relevant
  to a fine-tuned critic.
- **Fine-tuned / GRPO small critic on BEA labels** — `[DEFERRED]`. The biggest
  build and the demonstrated top strategy; pursue only if judge accuracy becomes
  the bottleneck.
- **Phase B alternate tutor conditions (few-shot / CoT / relax one-sentence
  cap)** — `[DEFERRED]`. These break comparability with the paper's Table-3
  tutors (all elicited zero-shot). If ever explored they must be run as clearly
  labelled *extra* conditions alongside the faithful Figure-2 baseline, never as
  the headline number.

## Open items (still undecided)

Everything else above is DECIDED. These two remain open by choice:

1. **Judge model choice** — on hold pending the user's teammates. The harness is
   judge-agnostic; no model is picked or hardcoded (placeholder
   `MRBENCH_JUDGE_MODEL` refuses the live path until supplied).
2. **Phase B response-length cost assumption** (was #10) — the judge-input sizing
   uses the mean V1 response length (≈ 174 chars) as a proxy for the
   model-under-test's output. Revisit once a specific model under test is chosen.

## olmo-eval integration (investigation + proposal)

*Investigation only — nothing built here, no live calls, judge-agnostic.* All
citations are to `src/olmo_eval` on this branch.

### 1. Does a reusable pipeline exist? **Yes.**

olmo-eval has a mature task registry + generation + LLM-as-judge pipeline that
MRBench Phase B slots into with **no core changes** — only a new task module.

**Registry & discovery.** Tasks self-register via a decorator and are
auto-imported at startup:

- `@register("name")` and `register_variant("name", "variant", **overrides)` —
  `evals/tasks/common/registry.py:42` and `:73`. Lookup is `get_task(spec)`
  (`:248`), spec form `task[:variant[:variant...]]`.
- Auto-discovery imports every module under `evals/tasks/` to trigger the
  decorators — `evals/tasks/__init__.py:8` (`_discover_and_load_tasks`). Dropping
  a new file in that dir is all it takes to register a task.

**Task contract.** `Task` ABC + `TaskConfig` in `evals/tasks/common/base.py`:
`process_doc(doc)->Instance` (`:376`), `format_request(instance)->LMRequest`
(`:360`), `request_type` derived from the formatter (`:334`), `score_responses`
(`:508`), `compute_metrics` (`:875`). `TaskConfig` (`:112`) carries `metrics`,
`sampling_params`, `primary_metric`, and — importantly for a gateway judge —
`required_secrets` (`:166`), the env-var names the beaker launcher mounts as
user-scoped secrets.

**End-to-end run.** `olmo-eval run` is the click entry in
`cli/run/__init__.py:30` (`-m model`, `-t task` (repeatable), `-o override`,
`--harness`, `--store`, `--s3-bucket/--s3-prefix`, `--db-*`). Flow:
`RunConfigBuilder.build()` (`cli/run/config.py`) resolves task specs via the
registry → `StorageSetup.setup()` (`cli/run/storage.py`) opens Postgres + S3
backends → `RunnerFactory.create()` (`cli/run/factory.py`) →
`runner.run()` (`runners/asynq/runner.py`). The `Harness`
(`harness/harness.py:22`) wraps the primary inference provider plus
`auxiliary_providers` (`harness/config.py:37`) and metrics collection. Results
are written through `storage/` (backends: `storage/backends/postgres`, S3 via
`cli/results/s3.py`, artifacts `storage/artifacts.py`) as the standard
`metrics.json` (`tasks[].metrics`), plus `predictions/` and `requests/`.

### 2. Generation vs. likelihood: **both supported.**

`request_type` is decided by the formatter (`base.py:334`); the harness routes
generation vs. loglikelihood accordingly.

- **Generation (free text).** `simpleqa.py` (`ChatFormatter` → `RequestType.CHAT`,
  temp 0, `simpleqa.py:42`), `harmbench.py:29`, `researchqa.py:264`.
- **Multiple-choice / likelihood.** `arc.py`, `piqa.py`, `hellaswag.py` (PPL /
  MC formatters, scored by loglikelihood, no free-text judge).

Phase B is a **generation** task — exactly the `simpleqa`/`researchqa` shape.

### 3. Existing LLM-as-judge / rubric scaffolding: **yes, reusable.**

`common/scorers/llm_judge.py` is a full judge framework:

- `LLMJudgeScorer` ABC (`:220`) with `format_judge_prompt` + `parse_judge_response`
  and async `ascore_with_context` (`:322`). Two routing paths:
  1. **`judge_fn`** — OpenAI SDK via `build_openai_judge_fn` (`:93`); default
     model resolved from `$OLMO_EVAL_JUDGE` (`build_default_judge_fn`, `:197`).
  2. **`provider_name`** — an **auxiliary provider** pulled from the inference
     pool (`_score_with_provider`, `:268`), i.e. a judge served alongside the
     model under test.
- Concrete scorers: `RubricJudgeScorer` (`:398`, regex `Score:\s*(\d+...)`,
  `max_score`) — structurally identical to MRBench's `[RESULT] N`;
  `SimpleQAJudgeScorer` (`:340`); `SafetyScorer` (`:470`).
- The harness already runs judge scorers **concurrently with a semaphore**
  (`base.py:689`, `ctx_semaphore = Semaphore(context.scoring_concurrency)`) and
  records per-output scoring errors — our cache/backoff concerns are partly
  handled upstream.
- Task wirings to copy: `simpleqa:judge` (`simpleqa.py:118`,
  `metrics=(AccuracyMetric(scorer=SimpleQAJudgeScorer),)`); `harmbench:wg_judge`
  (`harmbench.py:134`, auxiliary `provider_name="wg_judge"`); and the strongest
  analog **`researchqa.py`** — a temp-0 generation task that declares
  `required_secrets = ("OPENAI_API_KEY",)` (`:279`), overrides `score_responses`
  to run its **own multi-call judge loop** with retries + kept-zero parse
  failures (`:345`–`:435`), uses a placeholder scorer (`:159`) plus **custom
  per-type `Metric` subclasses** that read precomputed `response.scores`
  (`:182`–`:223`). MRBench's "8 dimensions → 8 metrics, DAMR = mean desired
  indicator" is the same pattern with 8 dims in place of ResearchQA's coverage
  types.

**Gateway routing.** `build_openai_judge_fn` currently constructs
`AsyncOpenAI(api_key=...)` with **no `base_url`** (`:143`), so it can't reach
TrueFoundry as-is. Two clean options, both judge-agnostic: (a) reuse our
existing `diagnostics/mrbench/judge_client.py` (already sets the gateway
`base_url`) inside an overridden `score_responses`, ResearchQA-style — **preferred,
least-invasive**; or (b) register the judge as an **auxiliary LiteLLM provider**
— `inference/providers/litellm.py` already supports `base_url` for
OpenAI-compatible endpoints (`:42`–`:107`) — and use the `provider_name` path.

**TutorBench / TutorEval / MRBench state: not integrated.** `rg -i
"mrbench|tutorbench|tutoreval|edubench"` over `src/` returns **nothing** — there
is no partial tutor scaffolding in the validated path. MRBench today lives
entirely in `diagnostics/mrbench/`.

### 4. Fit of the two phases

**Phase A (re-score 1,589 baked-in responses → correlate to gold): keep
standalone.** It has **no model-under-test inference** — the thing olmo-eval
exists to drive. Forcing it in would mean a "task" whose model output is ignored
and whose real metric is a per-(tutor×dimension) **Pearson AC against gold**,
which the per-instance `Metric.compute` contract (`base.py:875`) doesn't express
naturally (it needs paired judge/gold vectors and bootstrap CIs across the whole
tutor set). The current `diagnostics/mrbench` offline analysis is the right home;
olmo-eval adds nothing. **Recommendation: leave Phase A in `diagnostics/mrbench`.**

**Phase B (generate with the model under test → judge 8 dims → DAMR): native
olmo-eval task.** This *is* "run a model on a task": the model under test is
`-m`, the 192 Figure-2 prompts are the instances, the judge is a post-hoc metric.
It maps directly onto the `researchqa` pattern.

### Recommended path — **hybrid (a)+(c): native Phase B task, standalone Phase A**

Register Phase B as a native generation task whose scoring **reuses the existing
`diagnostics/mrbench` judge/DAMR code** unchanged; keep Phase A as the standalone
offline analysis it already is. This is strictly additive (one new task module,
no edits to olmo-eval core) and keeps the judge logic in one place.

**Concrete extension points (Phase B):**

1. New module `src/olmo_eval/evals/tasks/mrbench.py` (auto-discovered):
   - `@register("mrbench")` `class MRBench(Task)` with
     `data_source` = the V1 JSON (local/S3/HF via `DataSource`),
     `sampling_params = SamplingParams(temperature=0.0, max_tokens=...)`,
     `required_secrets = (<gateway key env>,)`.
   - `process_doc` → `Instance(question=<Figure-2 prompt>,
     metadata={conversation_id, data_source(Bridge|MathDial), topic, ...})`,
     mirroring the tutor-prompt spec above (move the Figure-2 templates out of
     `cost.py` into a `tutor_prompt.py`).
   - `format_request` → `LMRequest(RequestType.CHAT, messages=...)` (see
     `researchqa.py:319`).
2. Scoring: override `score_responses` (ResearchQA-style, `researchqa.py:345`) to
   call `judge_prompt.build_messages` → `judge_client` → `parse.parse_result`
   for each of the 8 dimensions, then map to desired via
   `reference.DESIRED_LABELS` and store 8 per-dimension DAMR indicators in
   `response.scores`. Reuse `judge_cache.py` keyed by `(conversation_id,
   "<model-under-test>", dimension)`.
3. Metrics: 8 per-dimension `Metric` subclasses (+ an aggregate) reading
   `response.scores`, plus a placeholder `Scorer` — copy
   `researchqa.py:159`–`:223`. `primary_metric` = aggregate DAMR.
4. Judge config stays env-driven (`judge_client.py`); **no model hardcoded**, so
   the task is judge-agnostic and satisfies `required_secrets`.

Invocation would then be the standard:

```bash
olmo-eval run -m <model-under-test> -t mrbench   # + --store/--s3-* to persist
```

**Checkpoint batch driver.** Because Phase B becomes an ordinary task name, it
plugs into the existing `eval-checkpoints` sweep for free: add `"mrbench"` to
`.cursor/skills/eval-checkpoints/scripts/benchmarks.json` and the driver runs it
per checkpoint (one `olmo-eval run` per checkpoint, shared vLLM boot), writing
`metrics.json` + `accuracy_wide.csv` per step — a DAMR-vs-training-step curve. The
one caveat: the **judge** is an external API call per response, so the sweep's
per-task cost model must include judge calls (192×8 per checkpoint), and the
judge key must be present in the sweep environment.

### Open decisions for the user

1. **Judge routing for the native task:** reuse `diagnostics/mrbench/judge_client.py`
   inside `score_responses` (preferred, one judge code path) **vs.** register the
   judge as an auxiliary LiteLLM provider and extend `build_openai_judge_fn` to
   pass `base_url`. (Both judge-agnostic; the former is less invasive.)
2. **Where the V1 data lives for the task loader** — package it as a `DataSource`
   (local path already in-repo, or push to S3/HF). Note the CC BY-SA 4.0
   share-alike obligation on any redistributed copy.
3. **Phase A stays standalone — confirm.** (Recommendation above.) If a single
   `olmo-eval`-native surface is later desired for reporting parity, Phase A would
   need a bespoke non-per-instance metric path — larger change, not recommended now.
4. **Register `mrbench` in the checkpoint `benchmarks.json`?** — **DONE on this
   (`frq/mrbench`) branch** for the `eval-checkpoints` *skill* registry
   (`.cursor/skills/eval-checkpoints/scripts/benchmarks.json` + `BENCHMARKS.md`);
   see the cross-branch TODO below for the separate AWS-checkpoint-branch driver.
   Running it is still only *meaningful* once a judge model is chosen and its
   per-checkpoint judge cost is acceptable.
5. Still gated on the outstanding **judge-model choice** (§Open items above); no
   integration code should pick one.

### Deferred: packaging for non-checkout (wheel) installs

The native task lives at `src/olmo_eval/evals/tasks/mrbench.py` but reuses the
judge/DAMR/reference code from the repo-root `diagnostics/mrbench/` tree. It
bridges to that tree by inserting the repo root (computed as
`Path(__file__).resolve().parents[4]`) onto `sys.path`.

- **Editable / checkout install works — verified.** An end-to-end smoke via the
  real `olmo-eval` console-script entrypoint (`olmo-eval run -m mock -t mrbench
  -o limit=4`) loaded `diagnostics.mrbench.*` successfully and produced all nine
  DAMR metrics; the only failure without a judge was the expected missing-API-key
  error raised *inside* `judge_client`, i.e. well past the import. So under
  `uv run` / `pip install -e .` (where `olmo_eval.__file__` points at
  `<repo>/src/...` and `diagnostics/` sits beside `src/`), the bridge resolves.

- **`[DEFERRED]` — a shipped wheel WITHOUT the `diagnostics/` tree beside `src/`
  would break scoring.** If `olmo_eval` is `pip install`ed from a wheel into
  site-packages with no repo checkout, `parents[4]` will not contain
  `diagnostics/`, the import fails, and the task (by design) registers but refuses
  to *score* with a clear error (task discovery never breaks). **Recommended fix
  (only if a non-checkout `pip install` becomes a real requirement):** bundle the
  reusable judge/DAMR/reference modules — and `data/MRBench_V1.json` — as package
  data under the installed `olmo_eval` package (or move them under
  `src/olmo_eval/…/mrbench/` and import them normally), then drop the `sys.path`
  bridge. Until that requirement is real, the checkout/editable layout is
  sufficient and this is intentionally not built (no package restructure now).

### Checkpoint-sweep integration — cross-branch TODO

**Done on this branch (`frq/mrbench`).** The `eval-checkpoints` *skill* registry
already carries `mrbench`:

- `.cursor/skills/eval-checkpoints/scripts/benchmarks.json` — added the `mrbench`
  entry (`instances: 192`, `choices: 1`, `split: "all"`, the 9 `damr*` metric
  keys), matching the existing schema. The skill's `run_eval_sweep.sh` inherits
  the caller's exported environment into `uv run`, so exporting the judge secrets
  before launching is sufficient there (no script edit needed).
- `.cursor/skills/eval-checkpoints/BENCHMARKS.md` — documented mrbench as a judged
  generative task: 192 generations + `192 × 8 × k` external judge calls per
  checkpoint (k default 1 → 1,536), the judge-cost caveat (invisible to the vLLM
  prompt count; estimate via `diagnostics/mrbench/cost.py`), and the required
  `OPENAI_API_KEY` + `MRBENCH_JUDGE_MODEL` env.

**Not done here — must be executed on the AWS-checkpoint branch** (where the
fleet driver `tests/aws/run_checkpoint_batch.sh` and `tests/aws/P3_results_namespace.md`
live; both are ABSENT on `frq/mrbench`, so they are intentionally NOT recreated
here). Checklist for a future worker on that branch:

- [ ] **(a) Judge secrets in the worker env.** Make `MRBENCH_JUDGE_MODEL` and the
  gateway key (`OPENAI_API_KEY`, or `TFY_API_KEY` / `TRUEFOUNDRY_API_KEY`)
  available to each sweep worker that runs `-t mrbench`. Tie this to the
  execution-identity / user-scoped-secrets concern already documented for beaker
  (`required_secrets = ("OPENAI_API_KEY",)` on the task): the driver must mount
  the judge key as a secret the worker process can read, not bake it into an
  image or a committed file. Optionally thread `MRBENCH_JUDGE_SAMPLES` (k) and
  `MRBENCH_JUDGE_REFERENCE_GUIDED` through as well.
- [ ] **(b) Judge cost in the driver's cost model.** Add mrbench's per-checkpoint
  judge cost — `#responses × 8 dims × k` judge calls (192 × 8 × 1 = 1,536 at the
  default k), on top of the 192 generation prompts — to whatever pre-flight cost
  estimate the AWS driver prints. The skill-side vLLM-prompt estimate does **not**
  include judge calls; mirror the `diagnostics/mrbench/cost.py`
  `estimate_phase_b_cost` breakdown so the dollar figure is visible before spend.
- [ ] **(c) Results namespace.** Ensure mrbench `metrics.json` (+ predictions/
  requests) land under the **P3 results namespace** used by the AWS driver
  (`tests/aws/P3_results_namespace.md`) — the same per-run layout the other
  benchmarks use, so the DAMR-vs-step curve aggregates alongside them. Reconcile
  with the skill-branch default (`diagnostics/mrbench/runs/<run_id>/…`) if a
  single reporting surface is wanted.
- [ ] **(d) Worker capabilities.** The judge is an **external API call per
  response**, so an mrbench sweep worker needs **network egress to the judge
  gateway + secret access**, not just a GPU. Flag mrbench in the driver so it is
  only scheduled on workers that have both; a GPU-only, network-egress-blocked
  fleet node will generate fine and then fail every judge call.

Everything above stays **judge-agnostic**: no model id is chosen or hardcoded on
either branch; the worker supplies `MRBENCH_JUDGE_MODEL` at run time.

## Implementation layout

```
diagnostics/mrbench/
  reference.py         # dimensions, desired labels, tutor mapping, Table 3, divergence note
  gold_damr.py         # M0: parse V1 + compute gold DAMR (pure data/arithmetic)
  gold_damr_check.py   # M0 CLI: table + per-cell diff vs Table 3 + PASS/FAIL
  judge_prompt.py      # Figure-6 system/user templates, definitions, rubrics, score<->label
  parse.py             # parse "Feedback: ... [RESULT] N" -> score + label
  judge_client.py      # TrueFoundry (OpenAI-compatible) config + client + retry/backoff
  judge_cache.py       # idempotent JSONL cache keyed by (conv_id, tutor, dimension)
  metrics.py           # M2: AC (Pearson) + judge-DAMR + baseline comparison
  baselines.py         # paper Table 5 (Prometheus2) & Table 6 (Llama-3.1-8B) AC scores
  cost.py              # measure real prompt sizes -> dollar estimate
  judge_run.py         # M1/M2 CLI: dry-run | --validate | --metrics | --live (gated)
  data/MRBench_V1.json # downloaded from the official repo (CC BY-SA 4.0)
  data/NOTICE.md       # CC BY-SA 4.0 attribution/notice for the vendored data
  data/_paper_refs/    # paper + annotation-guideline PDFs (git-ignored, local only)
```

Dependency: `openai` SDK (added via `uv add openai`) — used only as the
OpenAI-compatible client for the TrueFoundry gateway; imported lazily so
dry-run / metrics / validation need no live SDK or network.
