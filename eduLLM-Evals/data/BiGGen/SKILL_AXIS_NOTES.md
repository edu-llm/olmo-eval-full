# BiGGen skill axis — prep notes

**Status: exploratory. Nothing here is decided.** These are findings from the BiGGen
authors' released judgments, gathered before our own responses are graded so that the
q-matrix pass has a grounded starting point. Every number below should be recomputed on
our own atom-level verdicts once the larger run finishes; our criteria are binary and
atomic, while everything here is holistic 1–5 scoring at the instance level.

`q_mapping`, `primary_skill`, `q_rationale`, `criticality`, `objectivity`, and
`explicitness` are all null in `rubrics.jsonl` pending that decision.

## Data source

`prometheus-eval/BiGGen-Bench-Results`, split `llm_as_a_judge`: 68,805 judgments forming
a **99-model × 695-instance** score matrix, each instance scored by five evaluator LMs.
We use GPT-4-Turbo (highest human agreement in the paper, 0.623) and Claude as two
independent replicates. Multilingual is a separate split and is excluded, matching our
ingest.

Scripts: `scripts/biggen_factor_probe.py` (collinearity by model band, task purity),
`scripts/biggen_residual_structure.py` (residual clustering with cross-evaluator
replication).

## The paper's own grounding

BiGGen is built top-down as capability → task → instance criteria, and each capability
definition in Appendix A states what its criteria measure — "correctness of the final
prediction and the logical flow" (reasoning), "concreteness and feasibility" (planning),
"acts sensitively to the given input components" (grounding), "how effectively the
response incorporates the provided feedback" (refinement), and so on. That made a
capability-derived axis the natural first candidate.

The paper also excludes its own subjective tasks from average performance: the safety
moral-dilemma task (our `safety/moral_belief`) and two grounding conflict tasks. We
independently flagged `moral_belief` during atomization, so that judgment is corroborated
— those probably should not contribute to θ.

## Finding 1 — capability scores are dominated by general ability

Correlation between capability mean scores across models, by band:

| band | models | score sd | capability mean r | min pair | 4-skill mean r |
|---|---|---|---|---|---|
| all 99 | 99 | 0.664 | 0.929 | 0.808 | 0.933 |
| ≤7B (base + chat) | 37 | 0.596 | 0.919 | 0.750 | 0.919 |
| ≤7B, post-trained only | 25 | 0.425 | 0.844 | 0.498 | 0.817 |

Replicates under Claude (all-99 mean r = 0.937). At the item level the first factor
carries 41% of variance.

Our tutor population spans roughly 220M–6B and mixes base with instruct models, so the
**mixed ≤7B row is the relevant one** and r ≈ 0.92 is the expected regime.

Banding by *score* rather than parameter count is invalid — it selects on the outcome,
removes ability variance, and deflates correlations. An earlier pass that did this
produced a misleadingly encouraging r = 0.74.

Safety is the one capability that separates, and it does so in every band (0.50 vs tool
usage among post-trained ≤7B models). The six most separable capability pairs all involve
safety.

**Implication:** a 4-skill capability-derived axis is not supported. Input fidelity,
derivational correctness, and social inference sit at 0.91–0.98 with each other. What the
data supports is closer to two dimensions: general competence and normative restraint.

## Finding 2 — capability is not what organizes the variance

For each of the 70 tasks, taking its three most-correlated tasks across the ≤7B models:
**24% share its capability, against a 12% chance baseline.** Structure exists but is
weak, and 33 of 70 tasks have no same-capability task in their top three.
`grounding/role_playing` is closest to `theory_of_mind/multistep_tom` (0.94);
`instruction_following/lexical_constraint` is closest to
`theory_of_mind/interplanetary_diplomacy` (0.90).

So a capability-aligned axis is doubly unsupported: the dimensions are collinear, and
capability is not the grouping the scores follow.

## Finding 3 — real cross-cutting structure under the general factor

Regressing each task on model general ability removes 74–75% of task variance. The
residual correlation matrices from the two evaluators agree at **r = 0.681** (raw
task correlations agree at 0.821), so the residuals carry real signal rather than noise.

Clustering the evaluator-averaged residuals, the partition that best replicates across
evaluators is **k = 3** (adjusted Rand index 0.447; k=2 0.313, k=4 0.298, k=5 0.375).
The three groups cut straight across capability:

**Cluster A — verifiable / structured production (21 tasks, 7 capabilities).**
`reasoning/{competition_mwp, first_order_logic, high_school_mwp, math_proof, table_reason}`,
`tool_usage/{api_documentation, coding_for_math, item_recommendation, multi_step, search_engine, tool_making}`,
`planning/{executable_planning, reward_modeling, world_modeling}`,
`refinement/{code_revision, rationale_revision}`,
`theory_of_mind/{checklist_generation, knowledge_graph}`,
`grounding/{false_context, json_csv_xml}`, `instruction_following/lexical_constraint`.
Common thread: a checkable answer or a strict output format — math, formal logic, code,
tool invocation, structured data.

**Cluster B — refusal and normative compliance (6 tasks).**
`safety/{honesty, if_else_statements, knowledge_unlearning, safety_alignment}`,
`instruction_following/{alignment, instruction_data_creation}`.

**Cluster C — open-ended contextual generation (43 tasks, all 8 capabilities).**
Everything else: grounding conflicts, social/ToM reasoning, prose planning, revision,
evaluation, ambiguity handling.

The sharpest evidence that demand beats topic: **theory of mind splits.** Its two
structured-output tasks (`checklist_generation`, `knowledge_graph`) land in Cluster A
with math and code, while its other eight land in Cluster C. Assigning all 422 ToM atoms
to a single "social inference" skill would be wrong.

At k=4 Cluster C splits into a constrained-generation group and an interpretive/inferential
group, but that split does not replicate well (ARI 0.298) and should not be relied on.

## Where this leaves the candidates

1. **Capability-as-skill (8 dims)** — faithful to the paper, but block-diagonal and
   collinear. Not supported.
2. **Capability-grouped 4 skills** — three of the four are indistinguishable (0.91–0.98).
   Not supported.
3. **General + normative restraint (2 dims)** — supported at every band, safety being the
   only capability that reliably separates.
4. **Demand-based 3 dims (Clusters A/B/C)** — the only option with replicated
   cross-cutting structure, and the only one where a single criterion can genuinely load
   more than one skill. Strongest candidate, pending atom-level confirmation.

Note that 3 and 4 agree about the normative dimension; Cluster B is essentially the safety
dimension with two instruction-following tasks joining it.

## Design requirement: a deliberate mix of single- and multi-loading atoms

The mapping we want is **neither pure simple structure nor maximal cross-loading**. Both
kinds of atom are needed, for different reasons, and an axis that produces only one kind
fails.

**Multi-loading atoms are what make this MIRT** rather than several unrelated
unidimensional tests, and they are often the honest label. "The response reports the
correct level averages: Beginner ~16.5, Intermediate ~20.57" demands both verifiable
computation and fidelity to data supplied in the prompt; forcing it onto one dimension
throws away half of what it measures.

**Single-loading atoms are anchors, and they are what make each dimension identifiable.**
A dimension whose criteria always co-occur with another dimension has no independent
evidence and collapses into it. This is not hypothetical: `source_grounding` was a
conceptually sound third skill that was dropped in `generate_qmatrix.py` v7 precisely
because the sample had no anchor items for it.

The two failure modes are symmetric, and the candidate axes above sit on opposite sides:

- Candidates 1 and 2 fail by being **all anchors** — capability is a scenario-level
  attribute, so every criterion loads exactly one skill and the q-matrix is
  block-diagonal. That is N unidimensional models wearing a MIRT costume.
- A demand-based axis fails in the other direction if nearly every atom loads two or
  three dimensions, leaving nothing to pin them apart.

Practical policy: keep the labeler's conservative 1-placement rule — mark a skill only
when the criterion *cannot* be satisfied without it — which `generate_qmatrix.py` already
implements and which naturally yields mostly single-loading atoms with genuine
cross-loading where warranted. Capability and cluster membership go in as hints, not
labels.

Before committing to an axis, report the loading-pattern distribution from a labeled
sample: how many atoms load 0, 1, 2, or 3 skills; how many pure anchors each dimension
has; and how many all-zero rows appear (a criterion that loads nothing measures nothing —
`generate_qmatrix.py` already counts these). Whether a dimension has enough anchors to be
identifiable is an empirical question and should be answered on the sample rather than
assumed, since v7's lesson is that a reasonable-sounding dimension can fail this test.

## Caveats

- Holistic 1–5 instance scores, not our binary atoms. Atoms within one task can make
  different demands — we found format-compliance atoms inside ToM tasks — so task-level
  clusters are a prior for atom labeling, not a substitute for it.
- 37 models and 70 tasks make residual correlations noisy; ARI 0.447 is moderate.
- Observed correlations are attenuated by measurement error, so true collinearity is
  *higher* than the tables show, which strengthens Finding 1.
- MIRT needs anchor items loading a single dimension. A fully cross-loaded axis is
  unidentifiable — this is how `source_grounding` died in `generate_qmatrix.py` v7.

## To redo after grading

Rerun both scripts against our own model × criterion verdict matrix. The atom-level
equivalents of Findings 1 and 3 are what should actually decide the axis: whether the
Cluster A/B/C structure survives at criterion granularity, and whether the loading mix
meets the requirement above — enough cross-loading atoms to justify a multidimensional
model, and enough pure anchors per dimension to identify one.
