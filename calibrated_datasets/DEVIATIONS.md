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
- Token budget is per item, from each item's own declared constraints; every other bank has one flat number.
- Truncates at the checkpoint's own leaked end-token text; both upstream harnesses send no stop strings.

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

The second change, made 2026-08-08, is the generation budget. Two things happened to it at
once and they are worth separating, because one is a deviation being *removed* and the
other is a deviation being added.

**The cap now matches the harness.** It was a flat 1536, derived locally as "the largest
output this bank explicitly asks for plus a third". Upstream generates IFEval at 1280 —
`src/olmo_eval/evals/tasks/ifeval.py` line 66, and lm-eval's `leaderboard_ifeval` sends
`max_gen_toks: 1280` — so the 1,102-model harvest these difficulties were fit over was
produced at 1280 and 1536 was a local invention. It is now 1280 and this is no longer a
deviation. Note for anyone tracing the number to the paper: **1280 is a harness convention
and not a paper value.** arXiv 2311.07911 specifies no generation cap at all, and the
google-research reference implementation consumes a JSONL of prompt/response pairs, so
generation was always the caller's business. Matching 1280 is calibration parity, not
fidelity to a specification.

**A per-item budget below that ceiling is the deviation.** No upstream harness varies the
budget per document; this one does, and 1280 is now a ceiling the cascade may only lower an
item beneath. The reason is that a single integer is wrong for most of this bank in one
direction or the other. 272 of the 511 items declare no size constraint and a prior
measurement put their typical demand near 52 tokens, while `ab5ca590f2d20f37` — four
sections of at least 100 sentences — converts to about 3,400. `generative.ifeval_token_budget`
holds the derivation: a cascade over every length signal an item declares with the largest
demand winning, `combination:repeat_prompt` additive, `combination:two_responses` doubling,
a 50-token headroom so an over-run is visible rather than prevented, and a 208-token floor
for the 84 items that declare no size *and* carry only constraints that survive being cut
short.

That last split is the part that matters for correctness. Truncating a response fails the
length, section and ending constraints it would otherwise have satisfied, so a budget set
too low makes the harness manufacture constraint violations — the same failure class as the
`leaderboard_math` blank-line stop that deleted 40.9% of answers. So 263 items whose grade
depends on the response *finishing* keep the full ceiling
(`generative.TRUNCATION_FRAGILE_IDS`: ending-dependent ones like `startend:end_checker`,
count-dependent ones like `keywords:frequency`), as do the 31 `language:response_language`
items, because every conversion constant is an English measurement and a non-English
response costs 2x to 22x more tokens per word — Thai 158x, being unspaced. The median
budget over the bank is therefore *at* the ceiling; the saving comes from the other half.

Seven items declare constraints converting to more than 1280 and are truncated at it,
deliberately, which is the same point the calibration population was truncated. Letting the
cascade raise them instead would score them on a scale no fitted model was measured on.

Two parts of the policy are resolved at run time and so are invisible to the startup
convention guard, which runs before the checkpoint is fetched. Words are converted to
tokens by the **evaluated checkpoint's own tokenizer**, making the budget
checkpoint-dependent; the manifest records that in words rather than pinning a ratio, since
pinning one would mean re-stamping the bank for every model. And the result is clamped to
the checkpoint's context window, which is inert for this bank above a 1,636-token window
and exists for future small-context submitters. If no tokenizer can be resolved the run is
**refused** rather than run degraded, because every item would silently take the ceiling
while the report looked normal; `cat_report.json` carries a `generation_budget` block
recording the tokenizer, whether the cascade was active, whether the clamp fired, and the
budget distribution, so a reader can tell computed budgets from defaulted ones. Everything
else about the policy is pinned in the manifest's
`scoring_convention.runtime.per_item_token_budget`, so the guard still refuses a run whose
budget policy differs from the one the bank was stamped under. The cost accounting is in
`config.yaml`.

Whether this changes comparability with the calibrated difficulties: the ceiling change
strictly improves it, and the per-item reduction should be neutral, since it only shortens
budgets for items whose constraints cannot be broken by shortening them. That neutrality is
an argument rather than a measurement — nothing upstream records per-document generation
lengths, so it cannot be checked directly. Where a budget could still truncate a compliant
answer is named in `ifeval_token_budget`.

The third live deviation, added alongside the budget, is a stop sequence — and it is a
deviation from upstream rather than from a default, which was checked rather than assumed.
olmo-eval's `ifeval` task builds `SamplingParams(max_tokens=1280, temperature=0.0,
do_sample=False)` with no stop strings, and lm-eval's `leaderboard_ifeval` sends
`until: []`. Neither harness cuts an IFEval response anywhere, so the 1,102-model harvest
these difficulties were fit over was never cut either. So the two halves of the generation
convention now sit on opposite sides of upstream: the cap matches it exactly, and the stop
list deliberately does not.

What is added is not a benchmark string. `config.yaml` still declares an empty stop list
and the manifest still records one; at run time `generative.eos_stop_sequences` appends the
*evaluated checkpoint's own end-token text*, read from its live tokenizer. Keeping it out
of both artifacts is the point: which string ends a generation is a property of the model,
and a benchmark convention that named one would refuse every model spelling it differently.

It buys no compute — `truncate_at_stop` runs on a finished completion — and exists for one
correctness reason. A model can emit the literal characters of its end marker as ordinary
vocabulary tokens rather than as the special token; `skip_special_tokens=True` does not
remove those, so `<|endoftext|>` lands inside the span the verifiers read. That is not
cosmetic here, because 93 of the 511 items are decided by how the response *ends*:
`startend:quotation` (40) requires the stripped response to open and close with `"`,
`startend:end_checker` (26) requires it to end with a given phrase, and
`detectable_format:json_format` (17) and `detectable_format:constrained_response` (10)
parse the whole span. A trailing marker fails all four on a response that satisfied them,
and cutting at it restores the ending — both of the first two strip whitespace, so the cut
leaves nothing they object to.

It is **conditional**, and the exception is what makes it safe. The 24
`combination:two_responses` items are excluded, because `TwoResponsesChecker` splits on
`******` and demands exactly two non-empty parts: a checkpoint leaking its marker between
the two responses would lose the second to a first-occurrence cut and fail the constraint
it was being judged on — the same class of harness-manufactured failure as the
`leaderboard_math` blank-line truncation above. Excluding them costs nothing, since 0 of
the 24 carry any of the four end-anchored constraints. The residual, stated rather than
hidden: on a non-two-responses item, a leak strictly before content that would have
satisfied a constraint loses that content. That ordering is the rarer one, and the
alternative loses the 93.

Worth knowing when reading either deviation: IFEval currently has no other stopping
mechanism at all. It is 0-shot so no exemplar teaches EOS, `_HFCompleter.__call__` passes
no `eos_token_id` to `generate`, and `hf_config_patch._llama_config` writes
`eos_token_id=None` for a converted checkpoint. Until that plumbing is fixed the per-item
cap is literally the only thing deciding where an IFEval generation ends, which is why it
is sized to err long.

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
