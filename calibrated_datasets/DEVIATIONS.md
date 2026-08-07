# Deviations from the Research / olmo-eval implementation

Per dataset, where this harness differs from the `olmo_eval` task on this branch (and
from Research's CAT where one exists), and where it does not.

The governing policy is that **the task wins**. An item's difficulty was estimated under
a particular presentation and grading convention, EAP treats that difficulty as fixed
truth, and any gap between the convention the bank was fit under and the one a run uses
has nowhere to go except theta — arriving with a healthy standard error and no other
sign. So a divergence here is not a style preference; it is a shift in the scale a theta
is reported on. Published leaderboard conventions are not the target.

Two entries below record deviations that were deliberate and are still open. The rest
record either parity or a fact about the calibration that could not be recovered from
what was vendored.

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

## `ifeval`

- No deviation; grading needs `ifbench` plus its NLTK corpora.
- Re-keyed by content; all 541 positions confirmed already correct.

## `gpqa`

- No Research CAT precedent; 3PL keeps 579 of 1,192.
- Nested subsets deduplicated to 395; extended over main over diamond.

## `musr`

- No deviation: 0-shot `leaderboard_musr` layout, `acc_norm` as calibrated.
- Our per-character divisor is one longer than lm-eval's, matching the repo.

## `bbh`

- Composite key verified on 5,761; duplicate items block content hashing.
- Theta unusable: `theta_mae` 0.684 flat; 3-shot prefix frozen.
