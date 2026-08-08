# Deviations from the Research / olmo-eval implementation

Per dataset, where this harness differs from the `olmo_eval` task on this branch (and
from Research's CAT where one exists), and where it does not.

The governing policy is that **the task wins**. An item's difficulty was estimated under
a particular presentation and grading convention, EAP treats that difficulty as fixed
truth, and any gap between the convention the bank was fit under and the one a run uses
has nowhere to go except theta — arriving with a healthy standard error and no other
sign. So a divergence here is not a style preference; it is a shift in the scale a theta
is reported on. Published leaderboard conventions are not the target.

Three entries below record deviations that were deliberate and are still open. The rest
record either parity or a fact about the calibration that could not be recovered from
what was vendored.

**One entry records a deviation that was none of those and has been corrected.** `gpqa`
was scored generatively, by chain of thought, against difficulties fit behind a
multiple-choice log-likelihood ranking; on 2026-08-08 it was re-vendored as MCQ. That was
not a preference between two defensible conventions but a mismatch nobody had noticed,
because the task's default was mistaken for the only available statement about how the
bank was calibrated. The harvest turned out to say otherwise, and to be readable.

**All nine are runnable, and one of them is not trustworthy.** `gpqa` was blocked on a
HuggingFace token scope until that scope was granted; `hellaswag`, `winogrande` and
`gsm8k` were blocked until their bridges were rebuilt from Open LLM Leaderboard v1
example order, and their first bullet now records how well that repair is attested rather
than a presentation difference — which is the right emphasis, because all three matched
their tasks throughout and were refused anyway. How strong the evidence is differs
sharply between them; the long form is in `styles/uni_mcq/datasets.py`. `bbh` completes
the Open LLM Leaderboard v2 set and is the exception to the pattern: nothing about its
join or its presentation is in doubt, and its theta still does not recover, so it ships
labelled rather than repaired. Every `bbh` run carries a `bank_caveat` saying which of
its outputs to read.

**Every bank's join has now been measured rather than assumed.** Seven are keyed by a
content hash; `gpqa` and `bbh` keep a composite because a hash cannot separate their
duplicated items, and both were checked document by document against the leaderboard's
own per-example records. One of those checks changed a bank: `leaderboard_math`'s
composite named the right question on 8 of its 1,324 positions, so anything computed
from that bank before 2026-08-07 was measured against difficulties belonging to other
problems.

**This file is the reading, not the enforcement.** Each `manifest.json` carries a
`scoring_convention` block whose `runtime` half is the convention its bank is to be
scored under and whose `calibration` half is what the parameters were fit against, with
`"unrecorded"` wherever nothing upstream says. The run-time half is compared against the
settings a run actually resolved, before the checkpoint is fetched, and a difference is
an error rather than a warning — so an edit to `config.yaml` that would put a bank
behind a prompt it was not vendored for now stops the run instead of moving theta. The
calibration half is never compared to anything, because a difference there is a property
of the bank rather than a misconfiguration; the two deviations below are exactly that.

Prompt formats live in `MCQ_PROMPT_STYLES` (`diagnostics/mcq_cat/common/inference.py`)
and `PROMPT_TEMPLATES` (`common/generative.py`), selected per dataset from
`styles/uni_mcq/config.yaml`. The guard is
`convention.check_runtime_convention` (`styles/uni_mcq/convention.py`). Bank provenance
is in each `manifest.json` and the long form of every note below is in
`styles/uni_mcq/datasets.py`.

## `arc_challenge`

- 0-shot here, as Research's CAT; bank calibrated 25-shot.
- Prompt, unnormalized-sum metric and choice text now match the task.

## `hellaswag`

- Bridge rebuilt from leaderboard v1 order; verified at −0.538.
- No prompt deviation: bare-context, unnormalized sum, 0-shot.

## `winogrande`

- Bridge rebuilt; corroboration weak at −0.177, below predicted −0.27.
- No deviation: Trinh and Le partial evaluation, per task.

## `gsm8k`

- Bridge rebuilt; no per-item responses exist, so never verified.
- No deviation in prompt, sampling or last-number grading.

## `leaderboard_math`

- Graded by Minerva, not `math_verify`; Windows skips sympy.
- Re-keyed by content; old composite misnamed 1,178 of 1,183.
- Stops at `Problem:` alone, as lm-eval; the task's `\n\n` cut the answer off 40.9% of solutions.

## `ifeval`

- Completion format here, `RequestType.CHAT` in the task; harvest has both.
- Re-keyed by content; all 541 positions confirmed already correct.

The one live deviation whose calibration side is known to be *mixed* rather than
unrecorded, so it is worth the extra lines. Open LLM Leaderboard v2 applied a chat
template per submission — automatically for chat models, not for pretrained ones — and
the bank was fit over 1,102 of its models, so both framings sit in the response matrix
these difficulties came from. Its per-example records show the split on a single
document (key 1000): `meta-llama/Meta-Llama-3-8B` was sent the prompt verbatim,
`microsoft/Phi-3-mini-4k-instruct` was sent `<|user|>\n…<|end|>\n<|assistant|>\n`, under
identical generation kwargs.

No run-time value matches the whole bank, so this one matches the pretrained half, which
is also lm-evaluation-harness's `leaderboard_ifeval` unmodified (`doc_to_text` is the
bare `prompt`, `until: []`, greedy, `max_gen_toks: 1280`) and the only setting under
which a checkpoint with no chat template can be scored at all. The gap against the
templated half lands in theta with a healthy standard error beside it, and chat models
gain heavily from the template on this benchmark specifically — so a theta from this
bank is on the completion scale and is not comparable with a chat-format one.

`gpqa` was deliberately **not** flipped with it, and then stopped being generative
altogether; see its own entry below.

## `gpqa`

- Multiple choice since 2026-08-08; generative before, on the wrong convention.
- No Research CAT precedent; 3PL keeps 579 of 1,192.
- Nested subsets deduplicated to 395; extended over main over diamond.

The one bank here whose *modality* has been corrected rather than its prompt, so it is
worth the extra lines too.

It was vendored as generative until 2026-08-08, taking the registered olmo-eval task's
default — an expert-scientist system turn asking for step-by-step reasoning ending in
`ANSWER: X`, then letter extraction. That was chosen because Research never wired GPQA
into its CAT and the task's default looked like the only statement available. It was not.
The harvest is itself a record of the convention: each cell of the fit's own response
matrix is the Open LLM Leaderboard v2 `acc_norm` outcome of lm-evaluation-harness's
`leaderboard_gpqa` for that document, which agrees with the leaderboard's per-example
records on 1,192 of 1,192 columns for both `microsoft/Phi-3-mini-4k-instruct` and
`01-ai/Yi-1.5-6B-Chat`. That task is `output_type: multiple_choice`, ranking `(A)`–`(D)`
by log-likelihood at 0-shot with no system prompt for any submission. So the difficulties
were fit behind a four-way ranking of option letters, chain-of-thought grading was a
convention mismatch EAP absorbed entirely into theta, and it also put the bank out of
reach of the base checkpoints it is wanted for, since a chain of thought needs a chat
template. The run-time convention is now `prompt_style: gpqa` under
`continuation_logprob_per_character`, which is `acc_norm`, and it reproduces the harvest's
recorded `arg_0` and `arg_1` byte for byte for a pretrained submission.

**The option ordering is the risk this carried and it is the part to check.**
`GPQATask.process_doc` shuffles the options per question, so `gold_index` indexes one
permutation and a rotated choice list is not a broken bank but a working one measuring
the wrong thing — every count matches, the CAT converges, the standard error collapses on
schedule and the theta is noise. The generative stems had frozen that ordering as a
lettered block, so it was *transferred* rather than re-derived:
`scripts/freeze_choice_order.py` parsed all 395 stems back into `(question, choices)`
pairs, demanding a unique decomposition and a byte-identical round trip for each, into
`styles/uni_mcq/bridges/gpqa.choice_order.json`, and `vendor_bank.check_choice_order`
holds every re-vendored item to it. Today's fresh enumeration matched that file on all
395 items — texts, order, gold, and a digest of the whole stem.

Because both halves of a transfer are this repo's, two outside checks sit beside it, and
the control freezes the evidence for both so the tests can re-run them offline. Our four
options equal GPQA's own `Correct Answer` and `Incorrect Answer` columns as a set on
395 of 395, and our `gold_index` names the `Correct Answer` on 395 of 395; rotating the
gold breaks that on 390, the five survivors being questions whose source data repeats an
option. And replaying `Phi-3-mini-4k-instruct`'s own per-option log-likelihoods through
our choice list and gold reproduces the leaderboard's `acc_norm` on 388 of 388 replayable
items and the calibration matrix cell each difficulty was fit from on 388 of 388, against
178 of 388 with the gold rotated. The seven it cannot reach are the ones lm-eval's own
preprocessing empties by deleting bracketed spans.

`scripts/check_bridge_alignment.py` reads −0.754 against a scrambled control of
+0.002 ± 0.047, which attests the parameter join and says nothing about option order:
its p-values come from the calibration matrix rather than from re-scoring, so a permuted
choice list would leave it unchanged. The two checks above are what cover the ordering.

## `musr`

- No deviation: 0-shot `leaderboard_musr` layout, `acc_norm` as calibrated.
- Our per-character divisor is one longer than lm-eval's, matching the repo.

## `bbh`

- Composite key verified on 5,761; duplicate items block content hashing.
- Theta unusable: `theta_mae` 0.684 flat; 3-shot prefix frozen.
