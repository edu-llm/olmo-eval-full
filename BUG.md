# Bug: ATLAS index bridges are permuted for hellaswag, winogrande and gsm8k

> **Status: fixed for the `uni_mcq` CAT style; still open upstream.**
>
> `diagnostics/mcq_cat/styles/uni_mcq/` no longer uses the three generated bridges. It
> reads the leaderboard v1 order directly and commits the result under
> `styles/uni_mcq/bridges/`, built by `scripts/build_leaderboard_bridge.py` and accepted
> on the p-value-against-difficulty test by `scripts/check_bridge_alignment.py`:
> `hellaswag` moved from −0.036 to **−0.538**, `winogrande` from +0.037 to **−0.177**,
> and `gsm8k` has no response data to test on any ref and rests on the ARC control alone.
> The diagnosis below is unchanged and still worth reading; the remedy it proposes has
> been carried out for this style only.
>
> **What is still broken.** Everything under *Consumers* below —
> `src/olmo_eval/adaptive/bank.py` and both CAT entry points on `origin/Research` — still
> reads the permuted `atlas_idx_to_question_id.csv` files, and the generator that produced
> them is still committed there with its premise unmarked. Any theta those paths report
> for the three benchmarks remains meaningless.

**Severity:** any theta or p-IRT accuracy the ATLAS CAT has reported for `hellaswag`,
`winogrande` or `gsm8k` is not meaningful. `arc_challenge` is unaffected.

**Where:** `AdaptiveTesting/Inputs/ATLAS/scripts/build_atlas_idx_bridge.py` on
`origin/Research`, and the three `atlas_idx_to_question_id.csv` files it produced under
`AdaptiveTesting/Inputs/ATLAS/{hellaswag,winogrande,gsm8k}/`.

**Consumers:** `src/olmo_eval/adaptive/bank.py::load_bank` reads
`atlas_idx_to_question_id.csv` by default, so both CAT entry points inherit the fault —
the offline replay in `src/olmo_eval/evals/tasks/atlas.py` and the online eval in
`src/olmo_eval/evals/external/benchmarks/atlas/eval.py`.

---

## The bug

A calibrated bank is keyed only by a positional row label `X<k>`. Those positions are
column positions in the response matrix ATLAS fit against. The bridge exists to map
position to a joinable item id.

`build_atlas_idx_bridge.py` builds that map by pairing ATLAS index *k* with the *k*-th
instance olmo-eval enumerates from the HuggingFace split. That assumes ATLAS's column
order equals HuggingFace split order.

It does not. ATLAS harvested its response grids from the Open LLM Leaderboard v1, whose
harness enumerates examples in its own order. So each bank row is joined to a different
question than the one its parameters describe.

This is now verified rather than inferred — see *Evidence 3*, which reproduces the shipped
ARC bridge from leaderboard v1 records on 1,172 of 1,172 rows.

The repository already contains this knowledge, but it never reached the generator.
`AdaptiveTesting/Experiments/atlas_transfer_published/atlas_diagnostic_validation.py`
says of ARC's bridge, around line 78, that it "is NOT HF test-split order."

`arc_challenge` escapes because its bridge shipped with the ATLAS release carrying native
string ids (`Mercury_7175875`) rather than being generated locally.

---

## Evidence 1: ARC as a control

ARC is the one benchmark where both artifacts exist — the bridge ATLAS shipped, and the
ability to regenerate what `build_atlas_idx_bridge.py` would have produced.

Regenerating it (position *k* → the *k*-th enumerated `arc_challenge` instance's
`metadata["id"]`, 1,172 instances) and comparing against
`AdaptiveTesting/Inputs/ATLAS/arc/atlas_idx_to_question_id.csv`:

| | |
|---|---|
| Positions in agreement | **2 of 1,172 (0.17%)** |
| Shape of disagreement | permutation, not an offset |
| Distinct offsets | 874, modal offset appears 6 times |
| Shipped ids present in the enumeration | 1,170 of 1,172 |
| Absent from the enumeration | `Mercury_7116183`, `TIMSS_2003_8_pg47` |
| Listed twice in the shipped bridge | `Mercury_406639` (idx 249, 738), `Mercury_SC_LBS10597` (idx 331, 644) |

Same item set, different order. This is a direct demonstration on the only benchmark
where the correct answer is independently known.

---

## Evidence 2: empirical difficulty versus calibrated difficulty

This test needs no knowledge of the true ordering. If a bank is correctly joined, an
item's p-value (fraction of models answering correctly) must track its calibrated
difficulty `b = -d / a1` — easy items have low `b`. A permuted join pairs one item's
p-value with another's difficulty, and the correlation collapses.

Spearman rather than Pearson: `b` has heavy tails at small `a1`, and Pearson understates
a correct join (−0.38 where Spearman gives −0.86).

| Join | n items | Spearman(p, b) | Scrambled control |
|---|---|---|---|
| ARC bank × ATLAS train matrix (3,747 models), shipped bridge | 650 | **−0.857** | +0.00 (sd 0.04) |
| ARC bank × ATLAS test matrix (417 models), shipped bridge | 650 | **−0.857** | −0.00 (sd 0.04) |
| ARC bank × 63 other models, shipped bridge | 650 | **−0.542** | +0.00 (sd 0.04) |
| ARC bank × same 63 models, **regenerated positional** bridge | 650 | −0.017 | — |
| ARC bank, shipped bridge shifted by **one position** | 650 | −0.027 | — |
| `hellaswag`, committed bridge, 4 models | 1,057 | **−0.042** | +0.00 (sd 0.03) |
| `winogrande`, committed bridge, 4 models | 865 | **+0.037** | +0.00 (sd 0.04) |

The off-by-one row shows the test is sharp: a single position of slip destroys the signal.
`hellaswag` and `winogrande` sit at the scrambled level.

### The nulls are not a power problem

Only four models' per-question responses were available for those two, against 3,747 for
ARC, so weak power is the obvious alternative explanation. It is ruled out four ways:

- Those **same four checkpoints** on ARC's verified join give **−0.319**.
- Twenty random four-model subsets of the ARC roster average **−0.456** (sd 0.056), every
  draw negative.
- Spearman–Brown reliability of the four-model p-value is 0.833 on ARC and **0.853 on
  hellaswag** — the hellaswag measurement is *more* reliable, not less. WinoGrande's is
  0.540; attenuating ARC's −0.319 for that reliability gap still predicts about −0.26
  against an observed +0.04.
- Position confound excluded: these near-chance checkpoints track difficulty, not option
  position (Spearman 0.00 against `gold == A`; mean p 0.563 versus 0.565).

Response data used: `AdaptiveTesting/Inputs/Open/LLM-Judge/mcq/`, four per-model
per-question CSVs each for hellaswag and winogrande.

---

## Evidence 3: the leaderboard ordering, recovered and confirmed

The v1 per-model detail repositories still exist. `open-llm-leaderboard/details_<org>__<model>`
returns HTTP 307 to **`open-llm-leaderboard-old/details_<org>__<model>`**, which is public
and ungated, carrying one parquet per task per run: `harness|arc:challenge|25`,
`harness|hellaswag|10`, `harness|winogrande|5`, `harness|gsm8k|5`. Row counts match the
bridge row counts exactly — 1,172 / 10,042 / 1,267 / 1,319.

**The ARC control passes on this ordering.** Details row *k*'s `example` stem equals the
shipped ARC bridge's `atlas_idx k+1` question on **1,172 of 1,172 rows**. That converts
"ATLAS numbers its columns by leaderboard example order" from an inference into a measured
fact, and it means the ordering the three broken bridges need is directly recoverable.

**The order is deterministic across models.** Llama-3-8B (Apr 2024) and Mistral-7B-v0.1
(Oct 2023) — six months and a harness generation apart — agree on 1,267 of 1,267 WinoGrande
rows and 1,319 of 1,319 GSM8K rows. One model's details suffice; no cross-model consensus
is needed.

**Measured against this ordering, all three current bridges are refuted:**

| bank | agreement with leaderboard order | item set |
|---|---|---|
| `winogrande` | 1 of 865 | identical (865/865) |
| `gsm8k` | 2 of 1,298 | identical (1,298/1,298) |
| `hellaswag` | 1 of 5,080 | identical |

Pure permutations, the same shape as ARC's generated-versus-shipped diff. Note this
upgrades `gsm8k` from *unverifiable* to *directly refuted* — the ordering test needs no
response data, unlike the correlation test.

### A smaller, separate defect in the shipped ARC bridge

Recovering ARC from leaderboard order matches the shipped bridge on **1,172 of 1,172
question stems** but only **1,170 of 1,172 native ids**, and the two disagreements are
positions where the *shipped* file is wrong.

ATLAS appears to have resolved its ids by matching stem text, and ARC-Challenge contains
two questions whose stems recur under a second id. Both collided onto the wrong twin, which
is why `arc/atlas_idx_to_question_id.csv` lists `Mercury_406639` (idx 249, 738) and
`Mercury_SC_LBS10597` (idx 331, 644) twice each while never listing `TIMSS_2003_8_pg47` or
`Mercury_7116183` at all.

Consequence: one calibrated ARC item carries its twin's difficulty. This is minor next to
the ordering bug — 1 item in 650 rather than all of them — and a stem-collision-aware
rebuild fixes it. We have deliberately *not* switched ARC to the rebuilt bridge, because
every figure in this report was measured through the shipped one and we would rather keep
the reference stable than recover 0.15% of its items.

One practical note for anyone rebuilding: the v1 records carry **no `doc_id` and no native
id**. The schema is lighteval's (`example`, `full_prompt`, `cont_tokens`, `predictions`,
`metrics`), and the `choices`, `gold` and `gold_index` columns are present but empty
null-typed lists. Ordering is row order, and the only item identifier is the prompt text.
That is sufficient — it is how the 1,172/1,172 ARC match above was established — but
`hellaswag` needs a normalization pass, since lighteval and olmo-eval render the stem
differently (activity-label prefix, whitespace) and only about 92% match verbatim.

---

## Why nothing caught this

Every existing guard passes on a permuted bridge, which is what makes the failure silent
rather than loud:

- `build_atlas_idx_bridge.py` validates only that the bank's maximum index equals the
  split size. A permutation preserves both.
- `load_bank`'s join reports full overlap. A permutation still maps every position to a
  valid id.
- The CAT converges normally: Fisher-information selection proceeds, EAP tightens, the
  stopping rule fires.
- The standard error looks healthy, because SE reflects how much *information* the
  administered items carry, not whether their parameters belong to them.

The output is therefore a confident theta with a tight error bar, assembled from
difficulty parameters belonging to other questions. It is not biased in a direction that
could be corrected for.

---

## A second, independent defect in the hellaswag bridge

`hellaswag`'s bridge keys on the dataset's native `ind`, which is **not unique**. The
validation split concatenates the `zeroshot` and `indomain` sub-splits, whose `ind` values
restart, so 10,042 rows carry only 9,609 distinct values and 433 ids are each claimed by
two positions.

For example `ind == 180` is both:

- position 8, `split_type: zeroshot`, activity "Sharpening knives"
- position 3260, `split_type: indomain`, activity "How to become a sports announcer"

`_load_idx_map` builds a dict keyed by `question_id`, so for each of those 433 ids the
later row silently overwrites the earlier one, and the surviving parameters may belong to
either question. This is independent of the ordering bug and would persist even if the
ordering were correct.

Separately, `winogrande`'s and `gsm8k`'s bridges are **pure identity maps**
(`atlas_idx - 1 == question_id`, verified on all 1,267 and 1,319 rows), so they carry no
information beyond the refuted assumption.

---

## Not affected

The Open LLM Leaderboard v2 banks under `AdaptiveTesting/Experiments/openlm_atlas_3pl/`
key on a self-describing `<subtask>|<doc_id>` composite recovered from the leaderboard's
own per-example records, rather than an assumed positional identity. They pass the same
test:

| bank | Spearman(p, b) |
|---|---|
| `ifeval` | −0.888 |
| `math` | −0.732 |
| `gpqa` | −0.747 |

`gsm8k`'s bridge is refuted by the ordering test in *Evidence 3*. What remains true is that
the *correlation* test cannot be run for it at all — no response matrix,
`actual_accuracy.csv`, item-selection frequency or per-item statistic for it exists on any
of the 113 refs. That matters for the repair rather than the diagnosis: a rebuilt `gsm8k`
bridge could not be confirmed by measurement, only by analogy to ARC.

This section originally concluded that we should therefore not ship a rebuilt `gsm8k`
bridge, on the grounds that the current one is also trusted on unverified reasoning and a
second of those is no improvement. **The `uni_mcq` style shipped one anyway, and the
reasoning changed rather than being overruled.** The two bridges are not symmetric: the
old one rests on an assumption ARC *disproves*, while the new one rests on a procedure ARC
*confirms* — the same script, reading the same pinned leaderboard commit, reproduces the
shipped ARC bridge on 1,172 of 1,172 stems and matched GSM8K's 1,319 examples to the split
as an exact bijection on question text. Preferring a confirmed procedure to a refuted
assumption is a real improvement even where neither can be checked against responses.
Harvesting per-item GSM8K responses for a handful of models is still the only thing that
would make this a measurement, and it remains worth doing.

---

## Reproducing this

Two independent checks, neither needing a GPU or a model.

**The ARC control.** Enumerate the `arc_challenge` task, take `metadata["id"]` in order,
pair index *k* (1-based) with the *k*-th id, and diff against
`AdaptiveTesting/Inputs/ATLAS/arc/atlas_idx_to_question_id.csv`. Expect ~0.17% agreement.

**The correlation.** For a bank and a per-model response grid over the same items:
compute each item's p-value across models, take `b = -d / a1` from
`irt_item_parameters_combined.csv`, join through the bridge, and take the Spearman
correlation. Filter to `a1 > 0` and finite, as `load_bank` does. Compare against a
control with the join deliberately shuffled. A correct join is strongly negative; a
permuted one sits at zero.

---

## Suggested fix

Rebuild the three bridges from Open LLM Leaderboard v1 example order, which is the ordering
ATLAS's columns actually follow. The source is `open-llm-leaderboard-old/details_<org>__<model>`
as described in *Evidence 3*; the ordering is deterministic, so one model's details suffice.

**Validate the procedure on ARC first.** ARC's shipped bridge is a known-correct answer, and
the recovery above already reproduces it at 1,172/1,172 — so any rebuild script should be
required to hit that before being trusted on the three benchmarks that have no answer key.

Note the Dataset Viewer is disabled (HTTP 501) on the v1 repos, so parquet files must be
fetched directly rather than through the viewer API.

For `hellaswag`, also choose a join key that is genuinely unique. Native `ind` is not, per
above; position within the split, or a stable content hash, would be.

Until the bridges are rebuilt, `hellaswag`, `winogrande` and `gsm8k` should not be used
to produce reported numbers through either CAT entry point.
