# Bridge

Bridge reformatted into the Scenario + Rubric schema. Source:
https://huggingface.co/datasets/rose-e-wang/bridge (splits: `train` / `validation` /
`test`, 700 tutor–student math remediation conversations). Paper: *Bridging the
Novice-Expert Gap via Models of Decision-Making: A Case Study on Remediating Math
Mistakes* (Wang et al.).

Built by [`scripts/ingest_bridge.py`](../../scripts/ingest_bridge.py); the synthetic IRT
parameters are appended by [`scripts/assign_irt_params.py`](../../scripts/assign_irt_params.py).

> **Status: candidate benchmark — not confirmed for the bank.** Bridge is a *potential*
> option under evaluation, not a committed part of the calibration set. Its presence in
> `data/` does not mean it is wired into any run — the default bank in `config.yaml`
> remains TutorBench, and Bridge only loads when you explicitly flip `SKILLS` to its
> 5-skill axis (below) and point `config.yaml` at these files.

> **⚠️ Deliberate hand-authoring exception.** The standing rule (see
> [`data/README.md`](../README.md)) is that a benchmark "must ship its own rubrics — no
> hand-authoring." Bridge ships **no** per-response criterion checklist — its native rubric
> is a 4-dimension conversation-level scale, not binary per-response criteria. So the
> 15-criterion rubric here (and its criteria→skill q-matrix) is **hand-authored**, applied
> uniformly to every scenario. This is a conscious, user-directed departure made because
> Bridge has no native per-response rubric to derive from. Only the **applicability** of
> those criteria is data-driven (from the expert error-type label `e`).

## The 5 skills (q-matrix axis)

Bridge uses its own 5-skill axis, **not** the engine default
(`content` / `diagnosis` / `scaffolding`). Like InFoBench's 5-skill artifact, this is an
on-disk artifact only and is **not** engine-loadable under the default 3-skill
`tutor_cat.SKILLS`. The list order is the fixed q-matrix column order.

| Skill | Meaning |
|-------|---------|
| `diagnosis` | correctly identifying the student's specific error and its cause |
| `strategy` | pedagogical decision-making: choosing an effective remediation move |
| `math` | mathematical correctness of the tutor's own statements |
| `communication` | clarity, structure, and audience-appropriateness of the explanation |
| `affective` | supportive stance: tone, encouragement, preserving student agency |

## The 15 criteria and their q-matrix

Each scenario gets the same criteria (a fixed checklist), with stable id suffixes
(`D1`=`c01` … `A3`=`c15`). Five criteria are cross-loaded onto a second skill; the rest
are single-skill.

| Code | Primary | Also loads | Criterion (abbrev.) |
|------|---------|-----------|----------------------|
| D1 | diagnosis | — | identifies the specific error |
| D2 | diagnosis | — | addresses the real mistake, invents none |
| D3 | diagnosis | strategy | addresses the underlying misconception |
| P1 | strategy | — | guides rather than gives the answer away |
| P2 | strategy | — | remediation move fits the situation |
| P3 | strategy | affective | keeps the student involved / doesn't take over |
| M1 | math | — | all math statements correct |
| M2 | math | — | worked steps correct & complete |
| M3 | math | diagnosis | correction resolves the student's specific error |
| C1 | communication | — | explanation is clear and organized |
| C2 | communication | — | language is audience-appropriate |
| C3 | communication | strategy | focused; not overwhelming |
| A1 | affective | — | tone is supportive |
| A2 | affective | communication | frames the mistake constructively |
| A3 | affective | — | avoids discouraging / judgmental language |

## Applicability (deterministic, no API)

The expert error-type label `e` drives two rules:

1. **Exclusion.** Rows whose `e` is not one of the six clean error types — the free-text
   "the student did not make a mistake" / "end session" / "unresponsive" annotations — are
   dropped (not mistake-remediation items) and logged to `dropped.jsonl`.
2. **D3 (misconception)** is attached only when `e` is a conceptual error (`misinterpret`
   or `diagnose`); it is skipped for the non-conceptual slips (`guess`, `right-idea`,
   `careless`, `imprecise`). Every other criterion is attached to every kept scenario.

Response-dependent skips (e.g. M2 when the tutor shows no worked steps) are **not** applied
at ingest — they are a judge-time concern (`JudgeVerdict.unscorable_reason`).

As of the current dataset (700 rows): **642 kept** (58 dropped); 172 conceptual scenarios
carry D3 (15 criteria each), 470 carry 14 → **9,160 criteria** total. The exact per-skill
q-matrix loads are printed by the ingester.

## Repeated conversations (`source_id` is non-unique by design)

Bridge annotates many conversations more than once: the same conversation point is
reviewed by different experts, who may assign a different error type (`e`), strategy
(`z_what`), and revised reply (`c_r_`). Across the 700 rows there are only 430 unique
conversation ids, and 116 span more than one HF split. All such rows are **kept** as
separate scenarios — the differing expert responses are genuine signal (there is rarely a
single correct remediation), and dropping them is irreversible.

`source_id` (the Bridge `c_id`) is the **conversation key** and is therefore intentionally
non-unique across scenarios; `scenario_id` remains unique. Because the same conversation
point can carry different `e` labels, its repeats may even get different criteria (one gets
D3, another does not).

The resulting local dependence and cross-split overlap are best handled at **calibration**,
not here: group by `source_id` to add a testlet random effect, hold out whole conversation
groups to avoid leakage, or de-duplicate post-hoc — all keyed on `source_id`. The ingester
prints how many scenarios share a conversation so this stays visible.

## How the source maps into the schema

| Bridge field | Schema field |
|--------------|--------------|
| `c_id` | `source_id` (join key back to HuggingFace) |
| `c_h[-1].text` | `prompt` (the student's final turn = the mistake) |
| `c_h[:-1]` | `conversation_context` (`{role, content}`, role `student`/`tutor`) |
| `c_r_` (joined) | `reference_solution` (expert revised reply = the gold key) |
| `c_r` (joined) | `novice_response` (original tutor reply; provenance) |
| `e` | `error_type` (drives exclusion + D3 applicability) |
| `z_what` / `z_why` | `expert_strategy` / `expert_intention` (provenance) |
| `lesson_topic` | `lesson_topic` (curriculum code; provenance) |
| HF split | `native_split` (provenance; `split` is the pipeline role `calibration`) |

`use_case` is `mistake_remediation`; the response generator has no dedicated system prompt
for it, so it falls back to the `adaptive_explanation` prompt, which preserves the
multi-turn context.

## Rebuild

```
python scripts/ingest_bridge.py
python scripts/assign_irt_params.py \
    --input data/Bridge/rubrics.jsonl \
    --skills diagnosis,strategy,math,communication,affective \
    --log-dir data/Bridge/irt_logs --no-backup
```

Re-running the ingester strips the synthetic IRT params, so always re-run the assign step
after a rebuild.
