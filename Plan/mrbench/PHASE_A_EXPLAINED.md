# MRBench Phase A, explained for a newcomer

This is a self-contained walkthrough of the **MRBench Phase A judge-validation**
work. It assumes no prior context. Read it top to bottom and you should
understand what we built, what the numbers mean, and why we concluded the judge
is trustworthy. The authoritative results record is
[`PHASE_A_VALIDATION.md`](./PHASE_A_VALIDATION.md); the design spec is
[`README.md`](./README.md); the operator steps are
[`RUNBOOK_phase_a.md`](./RUNBOOK_phase_a.md). This doc explains them rather than
replacing them.

---

## 1. What MRBench is, and why we care

**MRBench** ("Mistake Remediation Bench") is an evaluation benchmark for **AI
math tutors**. The setting: a student is working a math problem, makes a mistake
in a dialogue, and a tutor writes a short response to help them. MRBench asks how
*good* that tutor response is — not "is the final answer right", but "is this
good teaching".

It scores each tutor response along **8 pedagogical dimensions** (see §4), each
against **gold human annotations** — labels that human experts assigned. The
dataset (MRBench V1, from the NAACL-2025 paper by Maurya et al.) contains **192
math-tutoring conversations** and **1,589 tutor responses** already written by
nine different tutors (seven LLMs plus a human "Expert" and a human "Novice"),
each with its human gold labels baked in.

We care because we want to measure how well a *model* can act as a math tutor.
But to measure that automatically at scale, we need a cheap, reliable **judge** —
an LLM that reads a tutor response and scores it the way a human would. That is
what Phase A is about.

## 2. The two phases, and why validation comes first

The work has two phases, run in order:

- **Phase A — judge validation (this document).** Does our LLM-as-judge agree
  with the human gold labels well enough to trust it? We run the judge over the
  1,589 responses that *already have* human labels, and check how closely the
  judge's scores track the humans'.
- **Phase B — tutor evaluation (what's next).** Once the judge is trusted, use
  it to score a brand-new model acting as a tutor (the model generates responses,
  the judge scores them). Phase B is not covered here.

**Why this order matters.** In Phase B there is *no human gold* for the new
model's responses — the judge's score is the only score we get. If the judge is
unreliable, every Phase-B number is garbage and we would not know it. So we must
first prove the judge on data where we *do* have human labels to compare against.
**Phase A gates Phase B**: if the judge does not agree with humans acceptably, we
do not proceed.

## 3. The one-paragraph result

Phase A is **✅ VALIDATED**. On 2026-08-09 we ran the judge — **Claude Haiku 4.5**,
served through the **TrueFoundry AI Gateway** — over all 1,589 responses × 8
dimensions = **12,712 judge calls**, at **temperature 0** (deterministic), for
**≈ $12.10**, with **0 failures, 0 parse errors, 0 fallbacks**. The judge passed
the acceptance gate: it agrees positively and significantly with human gold on
**6 of the 7 gate-eligible dimensions** (only *Coherence* falls short), and it
beats both of the paper's baseline critics on **7 of 8** dimensions. Details and
the exact table are in §7–§8.

## 4. The 8 dimensions, in plain English

Each dimension is a yes/no-ish question a human asked about the tutor response.
The "desired" label is the answer a *good* tutor should produce (used later for
tutor-quality scoring, not for judge validation).

| Dimension | The question it asks | Desired answer |
|---|---|---|
| **Mistake_Identification** | Did the tutor notice the student made a mistake at all? | Yes |
| **Mistake_Location** | Did the tutor correctly point to *where/what* the mistake is? | Yes |
| **Revealing_of_the_Answer** | Did the tutor just give away the final answer? | **No** (giving it away is bad teaching) |
| **Providing_Guidance** | Did the tutor offer a correct, relevant hint/explanation? | Yes |
| **Actionability** | Is it clear what the student should do next? | Yes |
| **Coherence** | Is the response logically consistent with the conversation? | Yes |
| **Tutor_Tone** | Is the tone encouraging (vs neutral/offensive)? | Encouraging |
| **Humanlikeness** | Does it sound natural rather than robotic? | Yes |

Each dimension is scored on a **3-point scale** (e.g. Yes / To some extent / No),
which we treat as an ordinal number **1–3**. Note the desired answer for
*Revealing_of_the_Answer* is "No" — a good tutor guides rather than hands over
the solution.

## 5. The metrics, and how to read them

Four numbers show up throughout. Here is what each means and what "good" looks
like.

- **AC — Annotation Correlation (the Phase-A star metric).** The **Pearson
  correlation** between the judge's 1–3 score and the human gold 1–3 score,
  across all responses in a dimension. It answers *"when humans score a response
  higher, does the judge also score it higher?"* Range −1 to +1. **Higher is
  better**; ~0 means the judge is uncorrelated with humans (useless); negative
  means it disagrees. Our acceptance bar is **AC ≥ 0.30**. This is the number
  that decides whether the judge is trustworthy.

- **The 95% bootstrap CI (confidence interval).** A range around the AC point
  estimate, produced by resampling the data 1,000 times and seeing how much AC
  wobbles. It tells us whether an AC is *reliably* positive or could just be
  noise. **Read it as: if the lower bound is above 0, the positive correlation is
  statistically real.** Our gate requires the CI lower bound **> 0** in addition
  to AC ≥ 0.30.

- **DAMR — Desired Annotation Match Rate (a tutor-quality metric, used in Phase
  B, not Phase A).** The percentage of a tutor's responses whose label equals the
  *desired* label for a dimension. It measures how *good the tutor is*, not how
  good the judge is. We mention it because it is the headline Phase-B number; it
  is **not** what validates the judge.

- **macro-F1 (a secondary, comparison-only metric).** A classification score that
  treats the 3 labels as three classes and averages the per-class F1, so the rare
  middle class ("To some extent") counts as much as the common ones. It is
  reported because the **BEA-2025 shared task** uses it, so it lets us compare to
  that community. It is **not** our primary metric — AC is. Higher is better;
  ~0.22–0.29 is roughly a majority-class guesser, and the BEA winners land around
  0.58–0.72 on the easier dimensions.

**One-line summary:** *AC (+ its CI) validates the judge; DAMR measures tutors;
macro-F1 is a side comparison.*

## 6. The judge protocol (how the judge actually scores)

The judge follows the paper's **Figure 6** protocol byte-for-byte:

- **Per-dimension, absolute scoring.** One judge call per (response, dimension) —
  so 8 calls per response. Each call gets a system prompt ("you are a critic…"),
  the conversation history, that dimension's definition and 3-point rubric, and
  the tutor response.
- **Fixed output format.** The judge must reply `Feedback: … [RESULT] N` where
  `N` is 1, 2, or 3. A small parser extracts `N` and maps it to the dimension's
  label. In our run every call parsed cleanly (0 fallbacks).
- **Deterministic.** Temperature 0, a single sample per call — matches the paper
  and makes the run cache-friendly and reproducible.
- **Reference-free.** The judge is not shown the ground-truth solution (there is
  an opt-in "reference-guided" variant, off by default; it did not run here).

**Model and routing.** The judge model is **Claude Haiku 4.5**, reached through
the **TrueFoundry AI Gateway**, which is OpenAI-compatible (we use the `openai`
SDK pointed at the gateway). The model id and gateway key are supplied at run
time via environment variables; nothing is hardcoded in the repo. The key lives
only in a **gitignored `.apienv`** (runtime-only, never committed).

> Note: the older design text in `README.md` was written before the judge was
> chosen and still says "Sonnet 4.6" as a placeholder. The judge is deliberately
> *unpicked* in code; the **actual validated run used Claude Haiku 4.5**, as
> recorded in `PHASE_A_VALIDATION.md`. Trust the validation record.

## 7. The acceptance gate, spelled out

Before running, we fixed the pass/fail rules so the verdict could not be
massaged after seeing the numbers:

- **A dimension passes** if **AC ≥ 0.30** *and* its **95% CI lower bound > 0**
  (a real, significant positive correlation — not just a lucky point estimate).
- **The judge is "validated"** if it (a) **passes ≥ 6 of 8 dimensions** *and*
  (b) **beats both paper baselines** (Prometheus2 and Llama-3.1-8B) on **≥ 7 of
  8** dimensions.
- **Human-likeness is excluded** from the gate — its gold labels are nearly all
  "Yes", which makes a correlation unstable and near-meaningless. So there are
  really **7 gate-eligible dimensions**.
- **Weak dimensions are caveated, not fatal.** A dimension that fails the bar
  doesn't sink the whole judge; it is flagged as low-confidence.
- **Self-bias check: N/A here.** The rule says to re-check AC excluding the
  judge's *own*-authored responses (in case a judge favours itself). Claude Haiku
  is **not** one of the nine tutors baked into MRBench, so there is no
  self-authored subset to exclude.

### The actual per-dimension result

Judge vs human gold, over **n = 1,553** (judge, gold) pairs per dimension
(a handful of responses lack a parseable gold label, so this is slightly below
the 1,589 total):

| Dimension | AC | 95% CI | macro-F1 | Gate |
|---|---|---|---|---|
| Mistake_Identification | +0.49 | [+0.46, +0.54] | 0.47 | **PASS** |
| Mistake_Location | +0.37 | [+0.32, +0.41] | 0.44 | **PASS** |
| Revealing_of_the_Answer | +0.72 | [+0.68, +0.77] | 0.60 | **PASS** |
| Providing_Guidance | +0.48 | [+0.43, +0.52] | 0.54 | **PASS** |
| Actionability | +0.42 | [+0.37, +0.46] | 0.53 | **PASS** |
| Coherence | +0.25 | [+0.20, +0.30] | 0.36 | **FAIL** (AC < 0.30) |
| Tutor_Tone | +0.44 | [+0.40, +0.47] | 0.45 | **PASS** |
| Humanlikeness | +0.38 | [+0.31, +0.46] | 0.48 | *excluded* |

**6 of the 7 gate-eligible dimensions pass** (only Coherence fails). Macro-F1
averages **0.48** across dimensions — reported for comparison only.

## 8. The baseline comparison, and what it tells you

The paper published two off-the-shelf "critic" judges — **Prometheus2** and
**Llama-3.1-8B** — and both correlate *poorly* with human gold: near-zero or
**negative** AC on almost every dimension. That is the whole reason the paper
invited "more powerful LLMs as critics" as future work. Our Haiku judge is the
answer to that invitation.

Per-dimension AC, averaged across the nine tutors (comparable to the paper's
Tables 5/6):

| Dimension | Ours | Prometheus2 | Llama-3.1-8B | Beats both |
|---|---|---|---|---|
| Mistake_Identification | +0.38 | −0.16 | −0.16 | ✓ |
| Mistake_Location | +0.30 | −0.15 | −0.18 | ✓ |
| Revealing_of_the_Answer | +0.71 | −0.22 | −0.28 | ✓ |
| Providing_Guidance | +0.31 | −0.24 | −0.31 | ✓ |
| Actionability | +0.28 | −0.10 | −0.16 | ✓ |
| Coherence | +0.15 | −0.12 | −0.16 | ✓ |
| Tutor_Tone | +0.42 | −0.32 | −0.36 | ✓ |
| Humanlikeness | +0.04 | +0.08 | +0.08 | ✗ |

**Beats both baselines on 7/8** — the only miss is Human-likeness, which is
excluded from the gate anyway. For a newcomer, the takeaway is: the previously
published automatic judges basically *anti-correlated* with humans, and our judge
flips that to a strong positive on nearly every dimension. "Beating the baseline"
was an easy bar; the real bar was *positive correlation in absolute terms*, which
we cleared.

## 9. Caveats a newcomer should keep in mind

- **Coherence is weak.** AC 0.25 (below the 0.30 bar; the CI upper bound only
  reaches 0.30) and the lowest macro-F1 (0.36). Treat Coherence judge-scores as
  **low-confidence**. It does not sink the judge, per the decided policy.
- **Mistake_Location is moderate.** It passes, but AC is only 0.37 — solid, not
  stellar.
- **Human-likeness is excluded.** Its pooled AC (0.38) looks okay but is inflated
  by differences *between* tutors; within a single tutor it is ~0, and it is the
  one dimension that does **not** beat the baselines. Don't lean on it.
- **Self-bias is N/A.** Because Haiku isn't one of the judged tutors, there is no
  own-response favouritism to worry about here (the "Sonnet is one of the tutors"
  note in the tooling is about *Sonnet-the-tutor* in the data, not the judge).

## 10. The known data-vs-paper divergence (accepted, not a bug)

There is one wrinkle worth understanding so it doesn't surprise you. There is an
offline "sanity check" (`gold_damr_check.py`) that recomputes DAMR **from the
human gold labels alone** — no judge, no API — and compares it to the paper's
published **Table 3**. This is a check on our *parsing and arithmetic*.

When you run it, it prints **FAIL** at the default tolerance. That FAIL is
**expected and accepted** — it is not a bug in our code:

- **6 of the 9 tutors reproduce** the paper's Table 3 within ~4 percentage points
  (GPT-4, Sonnet, Mistral, Llama-3.1-8B, Llama-3.1-405B, Phi3) — this validates
  our parser and math.
- **3 tutors do not**: *Expert* (its Tutor_Tone is inverted vs the paper),
  *Gemini* (uniformly ~20–27 pp high), and *Novice* (scored on 53 Bridge
  dialogues in our file, not the paper's 60).
- The published file's split is **139 MathDial / 53 Bridge (1,589 responses)**;
  the paper reports **132 / 60 (1,596)**.

The conclusion, after checking that no tutor-name remapping fixes it (and the
same pattern appears in MRBench V2): **the officially published `MRBench_V1.json`
simply differs from the paper's frozen Table-3 snapshot.** The **decision was to
use the official file as-is** and caveat those three tutors for strict
paper-comparability. It does not affect the judge validation, which runs over all
tutors regardless. In short: *the data drifted from the paper, our code is
faithful to the data.*

## 11. How to reproduce, and where the data lives

The full run is cached, so you can regenerate every metric **with no API calls
and no network**:

```bash
uv run python -m diagnostics.mrbench.judge_run --metrics --bootstrap 1000 --macro-f1
```

- **Raw judge outputs** (one line per call: the judge's text, parsed score,
  label, ok flag) live in a cache under `diagnostics/mrbench/runs/`. That whole
  `runs/` directory is **gitignored** — it is a local record, not committed.
- **The committed summary** is `PHASE_A_VALIDATION.md` (the verdict + tables).
- **The dataset** is vendored at `diagnostics/mrbench/data/MRBench_V1.json`
  (CC BY-SA 4.0; attribution in `data/NOTICE.md`).
- **To re-run live or for a different judge model**, follow
  `RUNBOOK_phase_a.md`, which walks the gated offline → cheap pilot → full
  sequence, including the go/no-go cost checklist. (The offline steps make zero
  network calls; only `--live` contacts the gateway.)

> Cost note: the runbook's pre-run estimate (~$30.58) assumed the more expensive
> Claude *Sonnet* pricing ($3/$15 per MTok). The actual run used *Haiku* rates
> ($1/$5 per MTok), so it came in far cheaper at **≈ $12.10**. Same call count,
> cheaper model.

## 12. What's next: Phase B

With the judge validated, **Phase B** uses it to score a *model under test* as a
tutor: the model generates a response for each of the 192 conversations, and the
same Phase-A judge scores those generations across the 8 dimensions, reporting
**DAMR** (there is no AC or macro-F1 in Phase B because there is no human gold for
a brand-new model). Phase B runs as a native `olmo-eval` task (~$4 per model at
current settings). See `README.md` and `PHASE_B_EDULLM_PLAN.md`.

## 13. Glossary

- **MRBench** — Mistake Remediation Bench; the AI-math-tutor evaluation benchmark
  (192 conversations, 1,589 responses, 8 dimensions).
- **Dimension** — one of the 8 pedagogical questions scored per response
  (e.g. Mistake_Identification).
- **Gold / gold label** — the human expert annotation for a response, treated as
  ground truth.
- **Judge / LLM-as-judge** — the model (here Claude Haiku 4.5) that scores tutor
  responses automatically.
- **AC (Annotation Correlation)** — Pearson correlation between judge and gold
  scores; **the** Phase-A judge-reliability metric. Bar: ≥ 0.30 with CI > 0.
- **DAMR (Desired Annotation Match Rate)** — % of a tutor's responses with the
  desired label; a **tutor-quality** metric, used in Phase B.
- **macro-F1** — secondary 3-class classification metric, reported for BEA-2025
  comparability, not primary.
- **Bootstrap CI** — a resampled confidence interval showing whether an AC is
  reliably above 0.
- **Figure 6** — the paper's judge prompt/rubric protocol, reproduced
  byte-faithfully.
- **TrueFoundry AI Gateway** — the OpenAI-compatible gateway the judge calls are
  routed through.
- **Prometheus2 / Llama-3.1-8B** — the paper's baseline critic judges, both
  mostly negatively correlated with gold.
- **Phase A / Phase B** — judge validation (this doc) / tutor evaluation (next).
