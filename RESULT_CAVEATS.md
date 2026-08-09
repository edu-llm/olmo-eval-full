# Result caveats

Ways a number in a `cat_report.json` can be read wrong. Each entry names the run it was
found in, the evidence, and what to do about it.

## `gpqa` theta is not an ability estimate for a base checkpoint

Found in `run_019fe33f-d5f9-709e-98d6-4990467d53b0` (native OLMo-core MCQ sweep,
Preston's `step305176`), cell 4. Reported `ability.theta = -0.250`, `se_online = 0.276`,
12 items administered, 6 correct.

**The model answered "A" on all twelve items.** `chosen_index` was 0 on every one, each
time by a margin of +0.45 to +0.88 over the runner-up. So the 50% accuracy is not
knowledge; it is how often gold happened to sit at A among the twelve items the CAT
selected.

This is a property of the prompt style, not a bug. `gpqa` is the only bank scored by
`inference.LetteredChoicesStyle`, which puts the option block in the prompt and ranks the
four *letters* rather than the option text. That is deliberate and matches calibration --
GPQA's options are bare quantities, frequently digit-permutations of one another, so
ranking their text measures which numeral looks likely rather than which one answers the
question, and Open LLM Leaderboard v2 ranked letters too. Ask a 135M base model to rank
"A" against "B", "C", "D" and it returns its letter prior every time.

**The bank itself is verified correct; do not re-investigate the vendoring.** Checked
against `bridges/gpqa.choice_order.json`: `gold_index` points at the leaderboard-correct
text on 383 of 395 items, and all 12 exceptions are the leaderboard's own bracket-stripping
preprocessing mangling its record (`[c]pentalene` -> `pentalene`, SMILES chirality tags
deleted) where this bank matches GPQA's raw source. `choices[0]` is the raw Correct Answer
on 103 of 395, i.e. 26%, exactly what a fair shuffle gives. Gold positions are near-uniform
at 26.1 / 22.0 / 27.8 / 24.1 percent.

Quantitatively, an always-A responder should get 26.1% of items right. Twelve items gives
an expectation of 3.1; this run got 6, which has probability about 0.066. Lucky, not
impossible.

This also explains the result that looks most surprising. `gpqa` ranks *highest* of the
five benchmarks (-0.250 against -2.1 to -3.9) because **guessing beat trying**: constant-A
lands near the 25% chance floor, while the four text-ranked banks let the model engage
with the content and it scored below chance -- 12.5% on `arc_challenge`, 11.1% on
`hellaswag`, 0.0% on `bbh`.

What to do: exclude `gpqa` from any cross-benchmark comparison of this checkpoint. The
number is on a different footing from the other four. Switching the bank to text-ranking
is **not** the fix -- it would deviate from the calibration the difficulties came from,
and `LetteredChoicesStyle`'s docstring argues text-ranking is the worse measurement for
GPQA regardless.

Nothing in the report flags this. `chosen_index` collapsing to one value across a whole
session is trivially detectable from data already recorded in `responses[]`, and a guard
was proposed and not built. The modal-choice share across the five cells was: `gpqa` 100%,
`bbh` 67% (16 of 24 at index 0, worth its own look), `musr` 63% (binary, so unremarkable),
`hellaswag` 56%, `arc_challenge` 38%. Any such guard must compare against the tail
probability for that item's choice count rather than a raw share, since the banks differ
in how many options they offer.

## `ability.standard_error` is not the number the stopping rule used

Applies to every run submitted with `--ability-estimator batch_eap+mwle`.

The report's own `metadata.se_mwle_definition` says it plainly: the reported SE is
`1/sqrt(I(theta))` over the administered items, an asymptotic standard error of a
likelihood estimate, while `se_batch` and `se_online` are posterior standard deviations.
The `se_threshold` stopping rule was applied to `se_online`. **The two are not on one
scale**, so reading `ability.standard_error` against the 0.3 threshold compares quantities
that were never compared during the run.

In `run_019fe33f`, this inverts the apparent precision ordering. On the reported
`se_mwle`, `musr` looks anomalously sharp at 0.060 -- tempting to read as implausibly high
discrimination in that bank. On `se_online`, the quantity that actually governed stopping,
`musr` is 0.172 and `bbh` is the tightest at 0.108, which simply tracks `bbh` having
administered 24 items to everyone else's 8 to 12. There is no bank anomaly.

What to do: quote `se_online` when discussing precision or stopping, and label `se_mwle`
as the MWLE asymptotic SE wherever it appears beside a theta. A trajectory plot that draws
the reported SE against the 0.3 line will show a threshold the curve never had to cross.

## An `ifeval` theta can be high, tight, and produced by a model that said nothing

**Found before the first generative run, by simulation rather than by being burned.** Read
this before quoting any IFEval number.

An IFEval prompt states its own constraint, and for a large family of verifiers the
statement contains the token that satisfies it. The verifier is a substring or format check
over the response; the prompt quotes the required substring. So a model that echoes its
prompt back passes.

Measured through this repo's own grading path over the vendored 511-item bank: **a verbatim
echo passes 129 items, 25.2%.** A content-free loop passes 45, 8.8%. The worst offenders:

| instruction id | echo passes | why |
|---|---|---|
| `detectable_format:constrained_response` | 10/10 | the prompt lists the exact allowed phrases |
| `detectable_content:postscript` | 16/26 | the prompt says "starts with P.S." |
| `detectable_format:title` | 20/34 | the prompt shows `<<title>>` as its example |
| `combination:repeat_prompt` | 20/35 | the constraint *is* to echo |
| `combination:two_responses` | 12/24 | the prompt names the `******` separator |
| `keywords:existence` | 15/38 | the prompt names the keywords in quotes |

This is not a floor with a caveat. Simulated against this repo's own Fisher selection and
EAP, an echo reports **theta +0.190 (se_online 0.227) stopping on precision after 9 items**,
and a lowercase echo **+1.106 (se_mwle 0.110) after 8**. The MCQ sweep's five cells ran
-0.250 to -3.9, so the cell that measured nothing would be the highest and by far the most
precisely estimated number in the report, with `ungradable.rate` 0.0,
`stop_reason: precision_reached`, and no alert of any kind.

The discriminations make it bimodal rather than gradual: median `a` is 2.79 and the maximum
19.4, so 2 accidental passes leaves theta at -2.93 while **3 passes puts it at +0.994 with
se 0.129**, stopped at the 8-item floor. There is no recognisable middle.

`TORCH_TODOS.md` records that this checkpoint's IFEval decode *echoed its own instruction*
on a 64-token probe. The exploit is a property of the bank and its verifiers rather than of
any checkpoint, so it is live for every future submitter whose base model recites.

What to do, in order of cost:

1. **Cheapest and unaided:** `stop_reason: precision_reached` at 8 or 9 items is itself the
   alarm. Over the best 40 items at floor ability the bank's information gives an SE floor
   of 0.572, so it *cannot* reach se <= 0.3 on a floor-level checkpoint. Early precision on
   a model you expect at the floor is proof something scored above it.
2. **The guard worth building:** stamp each item offline with whether an echo passes it --
   129 of 511, computed in seconds with no checkpoint -- then report the share of a
   session's passes that an echo would also have produced. Above roughly half, the theta is
   a constraint-checking artifact. This is the direct analogue of the modal-choice-share
   guard that would have caught gpqa, with a computable baseline instead of a raw share.
3. **Already recorded and never read:** `grader_detail.strict` carries the per-instruction
   pass list. Concentration in the table above with zero passes elsewhere is the signature.

## A `leaderboard_math` theta cannot resolve a weak checkpoint at all

Separate from the scale question below, and more fundamental. MATH's difficulties run
median 2.78 and 30% of the bank has |b| > 4, outside the EAP quadrature grid entirely. At
theta = -2 the 40 most informative items in all 1,183 carry total Fisher information 0.97 --
**an SE floor of 1.01**. At theta = -3 it is 0.23, an SE floor of 2.07.

An all-wrong session reports roughly -1.65 +/- 0.55 after all 40 items, and that is very
nearly the prior conditioned on "everything was wrong": a statement about where this bank's
difficulties sit, not about the checkpoint. Debating whether a prompt deviation moved the
scale by tenths of a logit is not meaningful when the instrument's resolution at that
ability is +/- 1.0.

MATH is, unlike IFEval, robust to degeneracy. The grader extracts only from `\boxed{}` or
the taught `Final Answer:` line, so a looping model produces neither and matches nothing,
and the exemplar answers match 0, 2, 7 and 1 of the 1,183 golds, so copying them buys
almost nothing.

## A `leaderboard_math` theta on a weak checkpoint is the prior, not a measurement

Measured, not predicted from theory: run `run_019fe78d-13f9` on Preston's `step305176`
returned **theta -1.6523 with se 0.5475**, having administered all 40 items and got 0 of 40
correct, stopping on `max_items_reached`.

**That number was computed before the run, from the bank alone, as -1.652 +/- 0.547.** The
agreement to four significant figures is the finding: the theta carries no information about
the checkpoint beyond "everything was wrong". It is the standard-normal prior conditioned on
an all-wrong response pattern, which is a statement about where this bank's difficulties sit
rather than about the model.

The mechanism is the SE floor recorded in the entry below -- at theta -2 the 40 most
informative of 1,183 items carry Fisher information 0.97. Any all-wrong session on this bank
returns approximately this pair of numbers, for any checkpoint, however bad.

What to do: on a checkpoint scoring 0, quote nothing. Two checkpoints that both score 0 will
return the same theta and it will not order them.

## Nothing distinguishes a wrong answer from an unreadable one

Found by the same run, and it is why the theta above is doubly uninformative.

`MathLatexEquivalence` reads exactly two forms: a `\boxed{}` expression, or the Minerva
`Final Answer: The final answer is $X$.` line. **Zero of the 40 completions contained
either.** The decode looped -- "The area of the triangle is 12 inches." repeated until the
budget ran out -- so there was never an answer to grade, and every item scored 0 for a
formatting reason rather than a mathematical one.

The report cannot tell you that. `extract_math_answer` falls through to normalizing the whole
completion, so `extracted_answer` reads `'Thefollowingissimplifiedsolutiontothe'` and sits in
the same field, with the same shape, as a genuine `\boxed{}` read. A session that is 100%
extraction failure is indistinguishable from one that is 100% wrong answers.

What to do: add an `extraction_path` field recording `boxed`, `final_answer` or
`whole_completion`. A session that is entirely `whole_completion` is a prompt-convention
failure, and the runner already warns about it per item -- "Repeated across a run this is the
prompt convention failing to take, not a weak checkpoint" -- but nothing aggregates that
warning into the report.

## A `leaderboard_math` theta can average several prompt regimes -- MEASURED INERT at 2048

Applies to any generative MATH run on a checkpoint whose context window is small, which
includes Preston's `step305176`.

MATH sends four Minerva exemplars, 680 tokens, in front of a stem that reaches 1,527 --
MATH stems are dense LaTeX, and `\frac{2}{5}` costs far more tokens than its word count
suggests. Against this checkpoint's **trained** `sequence_length` of 2048, the longest
items do not fit, so the harness drops whole exemplars until they do. That is deliberate:
`hf_config_patch.DEFAULT_MAX_POSITION_EMBEDDINGS` is 2048 because the validated node
configuration exported it, and raising it would push RoPE past the length the model was
trained at rather than buy room.

The consequence is that **one session can administer 4-shot, 2-shot and 1-shot items**,
by stem length. The difficulties were all estimated behind a 4-shot prompt, so the
resulting theta averages across prompt regimes -- which is worse than a uniform deviation,
because a uniform one at least shifts every item the same way. Measured: at 2048 the
ladder settles between 4 and 1 shots with nothing ungradable; at 4096 and above it never
fires and every item runs 4-shot.

**Settled by measurement 2026-08-09, and the hazard did not materialise.** Two agents
disagreed on how many items would ladder at a 2048 window -- one measured with the real
tokenizer and predicted 4-, 2- and 1-shot mixing, another used a character proxy and found a
single qualifying item. Run `run_019fe78d-13f9` administered 40 items at a 2048 window and
`num_fewshot` was **4 for every one of them**. The ladder never fired; the character-proxy
estimate was right.

What did fire, once, was the budget clamp: a 1,645-token prompt had its generation budget
reduced to 403 rather than losing an exemplar, which is the designed order of operations --
shrink the budget first, drop exemplars only when the prompt itself will not fit.

So this caveat is inert for a 2048-context checkpoint and the concern was theoretical. It is
retained because it is a property of the configuration rather than of the harness: a bank
with longer stems, or a smaller window, would reach the ladder.

What to do: read `num_fewshot` per response before comparing a MATH theta with anything.
They are per item rather than per session precisely because a session-level field would
average any mixing away.

## A high `ungradable` rate is recorded but never warned about

`cat_report.json` carries an `ungradable` block with a count, a rate, the item ids and the
reasons, and **nothing enforces a threshold on it**. A run in which most items failed to
grade returns a normal-looking theta computed from the few that did, with the exclusions
visible only to someone who opens that field.

This has never bitten: every cell of `run_019fe33f` reported a rate of 0.0. It becomes
live with the generative banks, where an item can be refused before it is ever sent to the
model -- a prompt that will not fit the context window is the case that exists today.

What to do: check `ungradable.rate` before quoting any theta. Past roughly a fifth of the
administered items, the number is a floor produced by grading failing rather than a
measurement of the model.
