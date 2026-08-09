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

## A `leaderboard_math` theta can average several prompt regimes

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

What to do: read `num_fewshot` per response and the `context_fit` block before comparing a
MATH theta with anything. They are per item rather than per session precisely because a
session-level field would average the mixing away. A MATH theta from a >=4096-context
model is comparable with the bank; one from a 2048-context model is not straightforwardly
comparable with it or with another 2048 model of different stem luck.

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
